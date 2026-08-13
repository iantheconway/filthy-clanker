#!/usr/bin/env python3
"""
Merge a trained LoRA adapter into its base model and save fp16 HF weights.

This is the bridge from training (adapter) toward serving (GGUF -> Ollama). It loads the base in
fp16 (NOT 4-bit — merging into a quantized base loses precision), applies the adapter, merges, and
saves a standalone HF model that llama.cpp's convert_hf_to_gguf.py can consume.

    python training/merge_lora.py --base Qwen/Qwen3-32B \
        --adapter training/out/clanker-qlora --out training/out/clanker-merged

Then (llama.cpp, external):
    python convert_hf_to_gguf.py training/out/clanker-merged --outfile clanker.gguf --outtype f16
    llama-quantize clanker.gguf clanker-q4_k_m.gguf Q4_K_M
    printf 'FROM ./clanker-q4_k_m.gguf\n' > Modelfile && ollama create clanker-sft -f Modelfile
Then run the eval with:  --worker-model clanker-sft
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge a LoRA adapter into its base (fp16 HF output).")
    ap.add_argument("--base", required=True, help="HF base model id/path (same as training).")
    ap.add_argument("--adapter", required=True, help="Path to the trained adapter (train_qlora output_dir).")
    ap.add_argument("--out", required=True, help="Output dir for the merged fp16 HF model.")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[merge] loading base {args.base} in fp16 ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True,
    )
    print(f"[merge] applying adapter {args.adapter} ...")
    model = PeftModel.from_pretrained(base, args.adapter)
    print("[merge] merging ...")
    model = model.merge_and_unload()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base, trust_remote_code=True).save_pretrained(str(out))
    print(f"[merge] merged fp16 model saved to {out}. Next: convert_hf_to_gguf.py -> quantize -> ollama create.")


if __name__ == "__main__":
    main()
