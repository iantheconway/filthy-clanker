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

## Not built yet (next, once a path is chosen)
The trainer itself (QLoRA config), the GGUF/Ollama export, and the CTFTiny A/B eval loop — see
`../docs/SFT_PLAN.md §4`. Target model to fine-tune is still open (dense abliterated-32B is the
recommended default, pending the out-of-the-box model bake-off).
