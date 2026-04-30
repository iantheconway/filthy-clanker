# NYU CTF Bench Evaluation

Runs Filthy-Clanker against the [NYU_CTF_Bench](https://github.com/NYU-LLM-CTF/NYU_CTF_Bench) benchmark dataset.

## Setup

Install the extra packages into the project venv:

```bash
source /home/kali/filthy-clanker/venv/bin/activate
pip install nyuctf docker
```

The dataset is downloaded automatically on first run (~1 GB git clone from GitHub).  
To pre-download manually:

```bash
python -m nyuctf --version v20250206
```

## Usage

```bash
cd /home/kali/filthy-clanker
source venv/bin/activate

# Run all development challenges (default timeout 10 min each)
python evals/nyu_bench/run_eval.py

# Only web challenges, 5-minute timeout
python evals/nyu_bench/run_eval.py --category web --timeout 300

# First 10 challenges, force Anthropic for all agents
python evals/nyu_bench/run_eval.py --max-chals 10 --provider anthropic

# Skip Docker (static/file-only challenges only)
python evals/nyu_bench/run_eval.py --no-docker --category crypto
```

## Output

Results are written to `evals/nyu_bench/results/`:

- `run-<timestamp>-<id>.json` — Full structured report with per-challenge results
- `run-<timestamp>-<id>.md`  — Human-readable Markdown summary table
- `logs/<session-id>.log`    — Per-challenge agent logs (same format as normal sessions)

Results are flushed after **every challenge** so a crash mid-run doesn't lose data.

## How it works

1. `CTFDataset(split="development")` loads the challenge index (auto-downloaded if missing)
2. For each challenge a `CTFChallenge` object is constructed, giving access to name, category, flag, files, and docker-compose path
3. If `challenge.container` is set, `docker compose up -d` is called and the exposed port is resolved via the Docker SDK
4. A task prompt is built from the category, description, flag format, target address, and file paths
5. A fresh LangGraph session is initialised (unique `session_id` per challenge)
6. The graph runs headlessly — HITL interrupts are auto-answered with "Continue with best effort."
7. `asyncio.wait_for` enforces the per-challenge timeout
8. Docker is torn down in a `finally` block regardless of outcome
9. Flags found in `knowledge_base["flags"]` are compared against the correct flag (exact + case-insensitive)

## Notes

- Hexstrike must be cloned to `HEXSTRIKE_DIR` (default `/home/kali/hexstrike-ai`) and its Flask server is started automatically
- The eval uses a separate SQLite checkpoint DB (`eval_checkpoints.db`) so it never pollutes normal session history
- Trajectory data is appended to the same `data/training/` JSONL files as normal sessions, giving additional fine-tuning signal
