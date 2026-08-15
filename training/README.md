# Fine-tuning data pipeline

Turns trajectories into a training set for fine-tuning the local worker model. Design and
decisions live in [`../docs/SFT_PLAN.md`](../docs/SFT_PLAN.md); this is the code.

All three bootstrap paths produce the **same** output format via `sft_common.py`:
one OpenAI chat-with-tools object per JSONL line (system + user + assistant`(tool_calls)` +
tool turns, plus the `tools` schema). Train with a tool-aware chat template and
**assistant-only loss masking** (axolotl / unsloth / trl).

```
 CTF-Dojo / Cyber-Zero release ──▶ ingest_ctfdojo.py ─────────────┐
 HTB walkthrough self-distill ──▶ generate_htb_trajectories.py ─┐ │
 frontier / strong open model ─▶ generate_frontier_trajectories ┤ ├─▶ *.jsonl ─▶ train
                                  (both capture to data/sft/) ───┴─┴▶ build_sft_dataset.py
```

## The three paths

| Path | Script | Generates positives by | ToS / license |
|---|---|---|---|
| **CTF-Dojo reuse** | `ingest_ctfdojo.py` | converting released trajectories | data is **CC-BY-NC-4.0** (non-commercial, attribute) |
| **HTB self-distill** | `generate_htb_trajectories.py` | your *local* model solving HTB with a walkthrough hint (stripped at build time) | no API ToS (local model); don't redistribute HTB walkthrough text |
| **Frontier / strong** | `generate_frontier_trajectories.py` | a stronger model solving in our harness, filtered to solves | frontier APIs restrict training competing models — script gates this; prefer strong **open-weight** models |

### 1. CTF-Dojo / Cyber-Zero (fastest — thousands of vetted positives)
```bash
python training/ingest_ctfdojo.py --inspect --in <downloaded_trajectories>   # confirm schema first
python training/ingest_ctfdojo.py --in <downloaded_trajectories> --out training/data/ctfdojo.jsonl
```
Repos: `amazon-science/CTF-Dojo` (2508.18370), `amazon-science/Cyber-Zero` (2508.00910). The
ingester auto-detects OpenAI-chat / ShareGPT / SWE-agent shapes; if yours differs, run
`--inspect` and share the keys.

### 2. HTB walkthrough self-distillation (targets the web/pwn gap)
```bash
export HTB_TOKEN=...   # HTB VPN must be connected
python training/generate_htb_trajectories.py --machine Lame --walkthrough walkthroughs/lame.md
python training/build_sft_dataset.py --sft-dir data/sft --strip-htb-hint --out training/data/htb.jsonl
```
The walkthrough is injected as a capture-only hint and **stripped** from the training prompt
so the model learns to solve unaided.

### 3. Frontier / strong open-weight distillation
```bash
# ToS-clean: strong open-weight model, permissive license
python training/generate_frontier_trajectories.py --provider ollama --model deepseek-v3 --subset ctftiny
# frontier API (your ToS call — gated behind an explicit flag)
python training/generate_frontier_trajectories.py --provider anthropic --model claude-opus-5 \
    --subset ctftiny --yes-i-accept-tos-risk
```

## Shared builder
`build_sft_dataset.py` turns anything captured to `data/sft/` (HTB + frontier paths) into the
training set, filtered to **solved** sessions:
```bash
python training/build_sft_dataset.py --sft-dir data/sft \
    --report evals/nyu_bench/results/run-XXXXX.json --only-solved \
    --out training/data/sft.jsonl
```
`--only-solved` is strongly recommended (negatives teach failure). `--agents exploit,reversing`
narrows to specific agents; `--strip-htb-hint` removes injected walkthrough hints.

## ToS — one-line summary
You were right to be cautious: **all three frontier APIs restrict using outputs to train
competing models.** Anthropic permits narrow *specialized tools* but not general/open-ended
models; fine-tuning a general open-weight LLM is in/near the prohibited zone. Clean routes:
HTB self-distill (local model), CTF-Dojo data (CC-BY-NC), or distilling a strong **open-weight**
model. Full analysis in `../docs/SFT_PLAN.md`.

## Training the model (QLoRA)

Once you have a POSITIVE dataset, fine-tune a LoRA adapter and serve it in Ollama:

```bash
pip install -r training/requirements-train.txt
# 1. train — assistant-only loss, tool-aware chat template (masking via the base's chat template)
python training/train_qlora.py --config training/qlora_config.yaml --data training/data/sft.jsonl
#    (--dry-run builds + tokenizes the dataset and reports lengths without training)
# 2. merge adapter -> fp16 HF weights
python training/merge_lora.py --base Qwen/Qwen3-32B --adapter training/out/clanker-qlora \
    --out training/out/clanker-merged
# 3. GGUF + quantize + register in Ollama (llama.cpp, external clone)
python convert_hf_to_gguf.py training/out/clanker-merged --outfile clanker.gguf --outtype f16
llama-quantize clanker.gguf clanker-q4_k_m.gguf Q4_K_M
printf 'FROM ./clanker-q4_k_m.gguf\n' > Modelfile && ollama create clanker-sft -f Modelfile
# 4. evaluate the fine-tune with the existing harness (same --worker-model swap as the bake-off)
python evals/intercode/run_intercode.py --worker-model clanker-sft --max-tasks 20 ...
```

Check dataset readiness anytime: `python training/check_dataset.py`.

### VRAM reality (read before picking a base)
`base_model` must be **HF weights (fp16/bf16)**, NOT an Ollama GGUF. A **32B QLoRA @ 16k ctx needs
~24GB+** — it will **not** fit the 16GB RTX 5070 Ti. Practical options on this box: a **14B base**
(fits 16GB — best first target), a shorter `max_seq_len`, `device_map="auto"` CPU/disk offload
(slow), or a rented GPU. The GTX 1080 Ti (Pascal) can't do bf16/modern bitsandbytes, so treat this
as single-GPU training on the 5070 Ti. Captured trajectories are large (median ~12k tok), so
context length is the dominant VRAM cost.

### Current data status (from `check_dataset.py`)
**106 trajectories, 2,770 assistant turns, but 0 solved sessions** — all failures. The capture
format is correct and converts cleanly, but training needs POSITIVES (the three paths above)
first, or it just learns the weak model's failure patterns.
