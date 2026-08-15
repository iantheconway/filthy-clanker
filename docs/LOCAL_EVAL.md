# Local eval: base Qwen3.5-35B-A3B vs. our fine-tuned LoRA

Runs the held-out set (`evals/nyu_bench/eval_holdout.txt`, 15 web-weighted NYU-dev challenges)
against the base model and the fine-tuned adapter on the **RTX 5070 Ti**, and compares. Nothing
here touches the system CUDA, the GPU driver, ComfyUI, or games — llama.cpp is a self-contained
prebuilt binary on `D:\`, and the GPU is used only while `llama-server` is running (which you
control).

## What's staged (all under `D:\clanker-models\`)

| Artifact | Path | Notes |
|---|---|---|
| Base model (Q4_K_M, 20.5 GB) | `Qwen3.5-35B-A3B-Q4_K_M.gguf` | unsloth, Apache-2.0, matches the fine-tune base |
| Our LoRA (GGUF f16, 6.6 MB) | `clanker-ctf-v1-lora-f16.gguf` | r=16, attention-only, **layers 3,7,…39 only** (Together's sparse-layer default) |
| llama.cpp b10430 (CUDA 13.3) | `llamacpp/bin/llama-server.exe` | Blackwell-native; bundled cudart, no toolkit needed |
| Eval profile | `profiles/local-clanker.yaml` | all worker agents → `http://127.0.0.1:8080/v1` |

The model was fine-tuned by Together (job `ft-72428a74-09ae`, 3 epochs, 42 steps, loss 1.44→~0.4).

## 1. Launch `llama-server` (needs the free GPU)

CUDA 13.3 supports the 5070 Ti (Blackwell/sm_120) but **not** the 1080 Ti (Pascal/sm_61, dropped
in CUDA 13), so this runs on the **5070 Ti alone** with the MoE experts offloaded to system RAM
(128 GB — trivially enough). Only ~3B params are active per token, so it stays usable.

```bash
cd /d/clanker-models/llamacpp/bin
# BASE model:
./llama-server.exe -m /d/clanker-models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 --alias clanker \
  -ngl 99 --n-cpu-moe 40 -c 16384 -fa auto --jinja
```

`--n-cpu-moe 40` keeps every layer's experts on CPU (safe first launch, ~4–5 GB VRAM). To go
faster, lower N (pushes more experts onto the GPU) until VRAM is ~90% full, or let
`llama-fit-params.exe` auto-size it. `--jinja` is on by default and enables the tool-calling
template the harness needs.

For the **LoRA** run, add the adapter (same base, so it's a clean A/B toggle):

```bash
./llama-server.exe -m /d/clanker-models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --lora /d/clanker-models/clanker-ctf-v1-lora-f16.gguf \
  --host 127.0.0.1 --port 8080 --alias clanker \
  -ngl 99 --n-cpu-moe 40 -c 16384 -fa auto --jinja
```

## 2. Run the eval (in the repo, separate terminal)

```bash
cd /d/filthy-clanker
source venv/Scripts/activate            # the app venv, NOT the train venv
export LLAMA_LOCAL_KEY=sk-local         # llama-server ignores it; the OpenAI client needs one
export OLLAMA_HOST=127.0.0.1:1          # dead port: the rare Ollama refusal path fails fast

# BASE (llama-server running WITHOUT --lora):
EVAL_RESULTS_DIR=evals/nyu_bench/results/base-35b \
  python evals/nyu_bench/run_eval.py --provider openai \
    --profile nyu-ctf,local-clanker,cloud-haiku-summ \
    --challenge-list evals/nyu_bench/eval_holdout.txt

# then Ctrl-C llama-server, relaunch it WITH --lora, and:
EVAL_RESULTS_DIR=evals/nyu_bench/results/lora-35b \
  python evals/nyu_bench/run_eval.py --provider openai \
    --profile nyu-ctf,local-clanker,cloud-haiku-summ \
    --challenge-list evals/nyu_bench/eval_holdout.txt
```

`cloud-haiku-summ` runs the summarizer on Anthropic Haiku (needs `ANTHROPIC_API_KEY`, ~cents for
15 challenges) so the local GPU only serves the worker model. To stay fully local instead, point
the summarizer at the same endpoint — but that competes for the one GPU model.

## 3. Compare

Each run writes `run-*.json` + a markdown report to its `EVAL_RESULTS_DIR`. Compare pass@1 per
challenge, base vs LoRA, split by category — flag both new solves and regressions.

```bash
python - <<'PY'
import json, glob
def load(d):
    f=sorted(glob.glob(f"evals/nyu_bench/results/{d}/run-*.json"))[-1]
    r=json.load(open(f))["results"]
    return {x["session_id"].split("-",1)[-1].rsplit("-",1)[0]: x["solved"] for x in r}
base, lora = load("base-35b"), load("lora-35b")
for c in sorted(set(base)|set(lora)):
    b,l=base.get(c),lora.get(c)
    tag="=" if b==l else ("＋LoRA-only" if l and not b else "－regressed" if b and not l else "?")
    print(f"  base={int(bool(b))} lora={int(bool(l))}  {tag:12} {c}")
print(f"base solved {sum(base.values())}/{len(base)} | lora solved {sum(lora.values())}/{len(lora)}")
PY
```

## Reading it

- Expect the most movement in **rev / misc / crypto** (categories with real training signal).
  `2013q-cry-slurp`, `2013q-msc-networking_2`, `2014f-rev-odd` are the regression probes there.
- **Web is the stretch**: training had zero web positives, and the adapter is a thin
  every-4th-layer attention LoRA, so large web gains are unlikely; the 3 held-out-solved web
  challenges mainly test that base web ability wasn't broken.
