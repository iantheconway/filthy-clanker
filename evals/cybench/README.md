# Cybench eval for Filthy-Clanker (in progress)

[Cybench](https://github.com/andyzorigin/cybench) — 40 professional CTF tasks (HackTheBox,
SekaiCTF, Glacier, HKCert) with subtask decomposition — run through the **same** multi-agent
graph as the NYU eval. Unlike CTFTiny (a subset of NYU CTF that drops straight into
`run_eval.py`), Cybench is a **separate dataset with its own on-disk format**, so it needs its
own loader (`cybench_dataset.py`) and runner (`run_cybench.py`).

> **Status: scaffold, not yet run.** The parser + prompt builder are written from Cybench's
> docs and are unit-testable; the runner reuses the NYU harness core. One integration seam
> (target-host networking) must be validated on a real clone + free GPU before the numbers
> mean anything. See "Open seam" below.

## Setup

```bash
git clone https://github.com/andyzorigin/cybench /path/to/cybench
export CYBENCH_DIR=/path/to/cybench
# Cybench tasks each ship a start_docker.sh that does `docker-compose up`; Docker must be up.
```

## Run

```bash
python evals/cybench/run_cybench.py --max-tasks 3 --timeout 1200 --capture-sft \
  --worker-model huihui_ai/gemma-4-abliterated:26b
```

Reuses `--worker-model` / `--model-override` / `--capture-sft` exactly like `run_eval.py`, so
model A/B runs and SFT capture work identically.

## How it maps to Cybench's format

| Cybench | Filthy-Clanker |
|---|---|
| `benchmark/<comp>/…/<task>/metadata/metadata.json` | `CybenchTask` (parsed) |
| `subtasks[-1].answer` | the flag we score against (`flags_match`) |
| final `subtasks[-1].question` | the agent's objective (unguided — we don't feed the subtask ladder) |
| `init_script.sh <dir>` | `task.stage_files()` → handout files given to the agent |
| `start_docker.sh` / `stop_docker.sh` | `task.start_docker()` / `task.stop_docker()` |
| `target_host` (e.g. `otp:80`) | the live-service target — **see open seam** |
| `metadata.categories` | `challenge_category` (drives RE/web routing) |

We run **unguided** (final objective only), matching Cybench's `unguided_mode`. Subtask-level
partial credit (Cybench's headline feature) is a **future add** — we'd score each
`subtasks[i].answer` as the agent surfaces it, giving partial-credit signal on the multi-step
web/pwn tasks where a binary solved/failed is too coarse.

## Open seam — target_host networking (must fix before trusting results)

Cybench's `target_host` is a **docker-compose service hostname** (`otp:80`). Our agent runs
*inside* the eval container and reaches services via `host.docker.internal:<published_port>`,
so it cannot resolve `otp`. Two ways to bridge (decide when we can test):

1. **Join the network** — attach the eval/agent container to the network the task's
   `start_docker.sh` creates, so `otp:80` resolves as-is. Cleanest if the compose network is
   predictable.
2. **Publish + rewrite** — ensure the task compose publishes its port to the host, parse the
   published port (reuse NYU's `_get_exposed_port` logic), and hand the agent
   `host.docker.internal:<published_port>`.

Until one is wired, `CYBENCH_TARGET=host:port` overrides the target manually for a single-task
smoke test. File-only Cybench tasks (no `target_host`) don't hit this seam and can be validated
first.

## Validation checklist (when GPU is free + clone exists)

- [ ] `python -c "from cybench_dataset import find_tasks; print(len(find_tasks('$CYBENCH_DIR/benchmark')))"` finds ~40 tasks.
- [ ] `CybenchTask(...).flag` returns the right answer for a couple of known tasks.
- [ ] Run one **file-only** task end-to-end (no docker seam) and confirm scoring.
- [ ] Wire one of the two target-host bridges; run one **service** task.
- [ ] Confirm `--capture-sft` writes `data/sft/<session>.jsonl` for Cybench runs too.
