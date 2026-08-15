# InterCode-CTF eval for Filthy-Clanker

[InterCode-CTF](https://github.com/princeton-nlp/intercode) — **100 picoCTF tasks**, the
**easiest** benchmark we run — through the same multi-agent graph as the NYU eval.

**Why this one:** NYU CTF / CTFTiny are past the local models' ceiling (a3b scored **0/6**,
mostly timeouts), so a bake-off there is all zeros — no resolution to rank models. picoCTF's
beginner-to-intermediate tasks should put the local models in the tens-of-percent range, where
solve-rate differences between models are actual signal.

Most tasks are **file-based**, so no per-task Docker is needed — our eval container already
ships the full Kali+Hexstrike toolset. We only need the picoCTF task files on disk.

> **Status: built, not yet run** (GPU in use). Loader + prompt builder are straightforward and
> unit-testable; the runner reuses the tested NYU core. Validate on a clone + free GPU.

## Setup

```bash
git clone https://github.com/princeton-nlp/intercode /path/to/intercode
export INTERCODE_DIR=/path/to/intercode
# Optional: fetch the large `setup`-wget assets ON THE HOST (the eval sandbox has no internet):
python evals/intercode/fetch_assets.py
```

## Run

```bash
# smoke test — 10 file-based tasks
python evals/intercode/run_intercode.py --max-tasks 10 --timeout 600 --capture-sft
# one category, one model (bake-off style)
python evals/intercode/run_intercode.py --tag crypto --worker-model huihui_ai/gemma-4-abliterated:26b
# specific task ids
python evals/intercode/run_intercode.py --only 0,2,5
```

Reuses `--worker-model` / `--model-override` / `--capture-sft` exactly like `run_eval.py`, so
model A/B runs and SFT capture work identically. Reports (per-task JSONL + JSON with a
per-category breakdown) land in `evals/intercode/results/`.

## Format & mapping

`data/ctf/ic_ctf.json` — 100 tasks: `{task_id, query, gold:"picoCTF{...}", source, tags, setup?, hint?}`.
Files live in `data/ctf/task_assets/<task_id>/`. The flag is `gold`, scored by normalised match.

| picoCTF tag | our `challenge_category` |
|---|---|
| Reverse Engineering | rev |
| Binary Exploitation | pwn |
| Cryptography | crypto |
| Forensics | forensics |
| Web Exploitation | web |
| General Skills | misc |

## Two internet caveats (the eval sandbox blocks egress)

1. **`nc`-service tasks (~20-25).** Their query embeds `nc <host> <port>` to a **live picoCTF
   server** — unrunnable offline. **Skipped by default** (`needs_service`); `--include-service`
   forces them (they'll fail under the sandbox unless you also allowlist + the server is up).
2. **`setup`-wget assets.** Some tasks download a large asset from picoCTF at setup. Fetch them
   **on the host** first with `fetch_assets.py`; the runner **skips** any task whose local assets
   are still missing (`--allow-missing-assets` to override).

So a plain clone runs the file-based tasks that ship their assets in-repo immediately; running
`fetch_assets.py` adds the wget-asset tasks; the `nc` tasks are out of scope offline.
