#!/usr/bin/env python3
"""
QLoRA fine-tuning for the filthy-clanker worker model.

Trains a LoRA adapter on the SFT dataset produced by build_sft_dataset.py / ingest_ctfdojo.py
(one OpenAI chat-with-tools object per JSONL line: {"messages": [...], "tools": [...]}). Loss is
masked to ASSISTANT tokens only (text + tool_calls) via the tokenizer's chat template, so the
model learns to PRODUCE agent turns, not to echo tool output.

    pip install -r training/requirements-train.txt
    python training/train_qlora.py --config training/qlora_config.yaml --data training/data/sft.jsonl

VRAM reality (see README): a 32B QLoRA at 16k context needs ~24GB+ — it will NOT fit a single
16GB card. Options: a 14B base (fits 16GB), CPU/disk offload (slow), shorter max_seq_len, or a
bigger/rented GPU. The base model must be HF weights (fp16/bf16), NOT an Ollama GGUF.

Masking requires a chat template that supports `return_assistant_tokens_mask` (has {% generation %}
blocks) — Qwen3 and most recent instruct templates do. If the base lacks it, this errors early
with a clear message.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

# Heavy ML deps are imported inside main() so --help / config parsing work without them installed.


def load_config(path: str | None, overrides: dict) -> dict:
    cfg = {
        "base_model": "Qwen/Qwen3-32B",   # swap to your abliterated HF variant; must be HF weights
        "output_dir": "training/out/clanker-qlora",
        "max_seq_len": 16384,             # median trajectory ~12k tok; drop to 8192 if VRAM-bound
        # LoRA
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        # Optim / schedule
        "learning_rate": 2.0e-4,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "lr_scheduler_type": "cosine",
        "logging_steps": 5,
        "save_steps": 100,
        "seed": 3407,
        "bf16": True,
    }
    if path:
        cfg.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def build_dataset(data_path: str, tokenizer, max_seq_len: int):
    """Read the SFT JSONL and tokenize with assistant-only label masking.

    Returns a datasets.Dataset with input_ids / attention_mask / labels (non-assistant = -100).
    """
    from datasets import Dataset

    rows, dropped_len, dropped_mask = [], 0, 0
    for line in Path(data_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        ex = json.loads(line)
        messages, tools = ex.get("messages", []), ex.get("tools") or None
        try:
            enc = tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=True, return_dict=True,
                return_assistant_tokens_mask=True, add_generation_prompt=False,
            )
        except TypeError as exc:
            raise SystemExit(
                "This tokenizer's chat template does not support return_assistant_tokens_mask "
                "(no {% generation %} blocks). Use a base whose template marks assistant spans "
                f"(e.g. Qwen3). Underlying error: {exc}"
            )
        input_ids = enc["input_ids"]
        amask = enc.get("assistant_masks") or enc.get("assistant_tokens_mask")
        if not amask or sum(amask) == 0:
            dropped_mask += 1          # no assistant tokens found — nothing to train on
            continue
        if len(input_ids) > max_seq_len:
            dropped_len += 1           # skip rather than truncate mid-trajectory
            continue
        labels = [tok if m else -100 for tok, m in zip(input_ids, amask)]
        rows.append({"input_ids": input_ids,
                     "attention_mask": enc["attention_mask"],
                     "labels": labels})
    print(f"[data] {len(rows)} training examples "
          f"(dropped {dropped_len} over {max_seq_len} tok, {dropped_mask} with no assistant span)")
    if not rows:
        raise SystemExit("No usable examples — check the dataset and max_seq_len.")
    return Dataset.from_list(rows)


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, batch):
        import torch
        maxlen = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [self.pad_token_id] * pad)
            out["attention_mask"].append(b["attention_mask"] + [0] * pad)
            out["labels"].append(b["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="QLoRA fine-tune the worker model on SFT trajectories.")
    ap.add_argument("--config", default="training/qlora_config.yaml")
    ap.add_argument("--data", required=True, help="SFT JSONL (messages+tools) from build_sft_dataset.py.")
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--num-train-epochs", type=float, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true", help="Build+tokenize the dataset and exit (no training).")
    args = ap.parse_args()

    cfg = load_config(args.config, {
        "base_model": args.base_model, "output_dir": args.output_dir,
        "max_seq_len": args.max_seq_len, "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
    })
    print("[cfg]", json.dumps(cfg, indent=2))

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(args.data, tokenizer, cfg["max_seq_len"])
    if args.dry_run:
        lens = [len(r["input_ids"]) for r in dataset]
        print(f"[dry-run] {len(dataset)} examples, token len min={min(lens)} "
              f"max={max(lens)} mean={sum(lens)//len(lens)}. Stopping before training.")
        return

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg["bf16"] else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16 if cfg["bf16"] else torch.float16, trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"], num_train_epochs=cfg["num_train_epochs"],
        warmup_ratio=cfg["warmup_ratio"], weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"], logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"], save_total_limit=3, bf16=cfg["bf16"],
        gradient_checkpointing=True, optim="paged_adamw_8bit", seed=cfg["seed"],
        report_to="none",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=dataset,
                      data_collator=PadCollator(tokenizer.pad_token_id))
    trainer.train()

    out = Path(cfg["output_dir"])
    trainer.save_model(str(out))            # LoRA adapter
    tokenizer.save_pretrained(str(out))
    print(f"[done] adapter saved to {out}. Next: merge_lora.py -> GGUF -> ollama (see README).")


if __name__ == "__main__":
    main()
