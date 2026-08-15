# Fine-tuning: hardware, data, and commands

Straight answers to "can I train a ~30B on my box, and how do I do it locally or in the cloud."

## TL;DR

1. **No ~30B trains on your hardware.** QLoRA of a 30–32B needs ~18 GB just for the 4-bit base
   (more with activations/optimizer), so it won't fit the 16 GB RTX 5070 Ti. The GTX 1080 Ti
   (Pascal) can't help — no bf16, no modern bitsandbytes 4-bit — so treat training as **single
   16 GB GPU**. Local ceiling is **~14B** (fits) / 7–8B (easy). 30B+ ⇒ **cloud**.
2. **Your Together key can fine-tune** (`GET /v1/fine-tunes` → 200). Together fine-tunes a
   curated open-model list; confirm the exact 32B base + data schema before spending (below).
3. **The real blocker is DATA, not compute.** We currently have **7 ToS-clean / 28 all / 56
   balanced** positive (solved) examples. That is nowhere near enough — you'd memorize 28
   examples, not learn. **Get bulk positives first** (CTF-Dojo, below) before any training run.

## What fits where

| Base | 4-bit weights | QLoRA @ 8k | QLoRA @ 16k | Where |
|------|--------------|-----------|------------|-------|
| 7–8B  | ~4–5 GB | ✅ 16 GB | ✅ 16 GB | local (5070 Ti) |
| 14B   | ~8 GB  | ✅ 16 GB | ~14 GB, tight | local (5070 Ti) — **best local target** |
| 30–32B (dense or 30B-A3B MoE) | ~18 GB | ✗ (>16 GB) | ✗ (~24 GB) | **cloud only** |

MoE doesn't help for training: all experts are quantized/resident, so 30B-A3B ≈ dense-32B in
QLoRA memory.

## Step 0 — get enough data (the gate)

We have far too few positives to train. Two ways to fix that, in order of leverage:

- **CTF-Dojo** — 486 execution-verified CTF trajectories (Amazon, CC-BY-NC). Not downloaded yet;
  the ingester exists: `python training/ingest_ctfdojo.py --inspect` then ingest → `data/sft/`.
  This is the fastest path to a real training set.
- **Harvest more** of our own: more challenges / benchmarks (InterCode gives easy positives),
  and keep `--capture-sft` on. Our solve rates are low, so this is slow.

Rule of thumb: don't launch a training run under a few hundred positives.

## Step 1 — build the dataset (filtering + balancing are wired)

Run from the repo root. Solved sessions are auto-unioned across **all** `evals/**/run-*.json`.

```bash
# ToS-clean positives only (open-weight models: gpt-oss / Qwen / local)
python training/build_sft_dataset.py --only-solved --out training/data/sft_positives_clean.jsonl

# include frontier (Claude/GPT) trajectories too — ToS-gated, for a non-competing/specialized model
python training/build_sft_dataset.py --only-solved --include-frontier --yes-i-accept-tos-risk \
    --out training/data/sft_positives_all.jsonl

# balanced set: positives + an equal share of sampled negatives (contrastive signal)
python training/build_sft_dataset.py --only-solved --neg-ratio 1.0 --include-frontier \
    --yes-i-accept-tos-risk --out training/data/sft_balanced.jsonl
```

Useful knobs: `--neg-ratio` (0 = positives only, 1.0 ≈ 50/50), `--agents exploit,reversing`
(train a specialist), `--min-assistant-turns`, `--report <file>` (pin specific runs instead of
auto-glob), `--strip-htb-hint`.

## Step 2a — train locally (≤14B)

```bash
pip install -r training/requirements-train.txt
# always dry-run first: tokenizes with assistant-only masking, reports lengths, no training
python training/train_qlora.py --base-model Qwen/Qwen3-14B --max-seq-len 8192 \
    --data training/data/sft_positives_all.jsonl --dry-run
# real run
python training/train_qlora.py --base-model Qwen/Qwen3-14B --max-seq-len 8192 \
    --data training/data/sft_positives_all.jsonl
# then: merge_lora.py -> GGUF -> ollama create  (see training/README.md)
```

The base must be **HF weights** (not an Ollama GGUF) and have a chat template with
`{% generation %}` spans (Qwen3 does) so loss masks to assistant tokens only.

## Step 2b — train in the cloud (for 30–32B)

**Option A — rented GPU (most reliable for our tool-use format).** Our `train_qlora.py` already
handles the OpenAI chat+tools JSONL via the chat template.
- Rent 1× A100/H100 **80 GB** (RunPod / Vast / Lambda). 32B QLoRA @ 8–16k fits comfortably.
- Push repo + `training/data/*.jsonl`, `pip install -r training/requirements-train.txt`, run
  `train_qlora.py --base-model Qwen/Qwen3-32B --data …` (config default is already 32B).
- ~1–4 GPU-hours for a small set → **~$3–12**. Pull the adapter back, `merge_lora.py`, serve in Ollama.

**Option B — Together managed (uses your key; simplest infra).** Together hosts the training and
serves the result, which plugs straight back into our `OpenAIClient`.
```bash
pip install together
export TOGETHER_API_KEY=...              # already in .env
together files check training/data/sft_positives_all.jsonl   # validate schema first
together files upload training/data/sft_positives_all.jsonl  # -> file-id
together fine-tuning create --training-file <file-id> \
    --model <a fine-tunable base, e.g. Qwen/Qwen2.5-32B-Instruct> --lora --n-epochs 3
together fine-tuning list        # watch status; then deploy/serve the checkpoint
```
Two things to **verify before spending**: (1) Together fine-tunes a *curated* base list — confirm
your 32B target is on it (`together fine-tuning models`, or the models API `pricing.finetune`
field); (2) our OpenAI chat+tools JSONL may need conversion to Together's fine-tuning schema
(tool-call support is newer) — run `together files check` first. If either is a hurdle, Option A
sidesteps both.

## ToS posture

Open-weight positives (gpt-oss Apache-2.0, Qwen, local, Llama) are clean to train on. Frontier
(Claude/GPT/Gemini) trajectories are gated behind `--include-frontier --yes-i-accept-tos-risk`
and intended only for a **specialized, non-competing** CTF tool-use model, not a general chatbot.
