# Running the NYU CTF Bench eval (Dockerized)

The eval runs in the **`filthy-clanker-nyu-bench`** container, which extends the
harness image with the `nyuctf` dataset loader and a Docker client. It drives the
multi-agent graph headlessly against local Ollama, brings up each challenge's
containers as **siblings** on the host daemon (via the mounted Docker socket), and
writes JSONL + JSON + Markdown reports to a results volume.

Ollama is **not** in the container — it runs on the host with the GPUs and is
reached over the network at `host.docker.internal:11434`.

## Prerequisites (one-time)

- **Docker Desktop** with its disk image on `D:` (not `C:` — the 12 GB images +
  pulled challenge images fill a small C: drive). Settings → Resources → Advanced →
  Disk image location.
- **Ollama running on the host**, listening on all interfaces
  (`OLLAMA_HOST=0.0.0.0:11434`), with the models from `agents.yaml` pulled:
  - `huihui_ai/qwen3-abliterated:30b-a3b-q4_K_M` (workers)
  - `huihui_ai/qwen3.5-abliterated:9b` (summariser / refusal specialist)
  Verify: `curl -s http://localhost:11434/api/tags`.
- **Images built** (`docker images | grep clanker`):
  - `filthy-clanker:latest` (harness base)
  - `filthy-clanker-nyu-bench:latest` (eval runner)
  Build both with: `docker compose --profile eval build`
- **Volumes** (created on first `compose up`, persist across runs):
  - `filthy-clanker_clanker-eval-results` → `/results` (reports)
  - `filthy-clanker_clanker-nyuctf-data` → `/root/.nyuctf` (cached ~1 GB dataset)

> On Windows Git Bash, put Docker on PATH first:
> `export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"`

## Run it — two ways

### A. Compose (uses the code baked into the image)

Rebuilds pick up host source changes. Good for a clean, reproducible run.

```bash
docker compose --profile eval run --rm nyu-bench \
  python3 evals/nyu_bench/run_eval.py --max-chals 3 --timeout 1200
```

### B. `docker run` with a live source mount (iterate WITHOUT rebuilding)

The eval image **bakes the code in** (unlike the `harness` service, which mounts
`./`). Rebuilding to test a prompt/flag tweak busts the base image's expensive
Hexstrike-venv layers (~10+ min). Instead, bind-mount the host repo over
`/home/kali/filthy-clanker` — the same pattern the `harness` service uses — so the
container runs your edited `src/`, `agents.yaml`, `profiles/`, and
`evals/nyu_bench/` directly:

```bash
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"
MSYS_NO_PATHCONV=1 docker run -d --name clanker-sq \
  --add-host host.docker.internal:host-gateway \
  --cap-add NET_ADMIN --cap-add NET_RAW \
  --env-file .env \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -e EVAL_RESULTS_DIR=/results \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v D:/filthy-clanker:/home/kali/filthy-clanker \
  -v filthy-clanker_clanker-eval-results:/results \
  -v filthy-clanker_clanker-nyuctf-data:/root/.nyuctf \
  filthy-clanker-nyu-bench:latest \
  python3 evals/nyu_bench/run_eval.py --max-chals 3 --timeout 1200 --rerun --profile nyu-ctf
```

- `MSYS_NO_PATHCONV=1` stops Git Bash mangling the container-side `/…` paths.
- `--cap-add NET_ADMIN --cap-add NET_RAW` are **required** for the egress sandbox
  (firewall) and nmap SYN scans. Without NET_ADMIN the sandbox fails closed and the
  container exits — pass them, or set `CLANKER_ALLOW_INTERNET=1` to skip sandboxing.
- `--env-file .env` passes `LANGSMITH_*` / `BRAVE_API_KEY` / `CLANKER_ALLOW_INTERNET`.
- `-d` = detached. Watch it: `docker logs -f clanker-sq`.
- Drop the mount line to run the baked-in code instead.
- The sandbox needs `tinyproxy`/`iptables` baked into the image — **rebuild the eval
  image once** after pulling these changes: `docker compose --profile eval build nyu-bench`.

## Watching a run

```bash
docker logs -f clanker-sq
```

Signals worth grepping for:
- `MCP pool ready — 150 tools available.` — Hexstrike toolset loaded.
- `[<agent>] → <tool>(…)` — the agent is **actually calling tools** (the whole
  point; the failure mode this eval exists to catch is 0 tool calls + flag guesses).
- `[<agent>] Prompt size (est. tokens): system=… tool_schema=… total=…` — per-turn
  context measurement (exp. 3).
- `[submit_flag] Rejected — gated …` — the gate is holding back a premature guess.
- `Result: SOLVED|FAILED|TIMEOUT | submitted=… | correct=…`
- `EVAL COMPLETE: N / M solved`

## Reading results

Reports land in the `clanker-eval-results` volume as `run-<ts>.{jsonl,json,md}`
plus per-challenge logs under `logs/`. Dump them from the host:

```bash
docker run --rm -v filthy-clanker_clanker-eval-results:/results alpine \
  sh -c 'cat /results/*.md; echo; cat /results/*.jsonl'
```

Per-challenge JSONL fields of interest: `solved`, `solved_via_tool`,
`submitted_flags`, `failure_reason` (`no_flag|timeout|error|refusal`),
`refusal_fired`, `has_files`, `has_container`.

## Internet sandbox (agent isolation)

By default the eval container **blocks the agent from the public internet** so it
can't attack real hosts. Implemented in `sandbox/entrypoint.sh` (baked into the
image): an in-container allowlisting proxy (`tinyproxy`) is the *only* process
permitted to egress, and `iptables` drops all other public-bound traffic while
allowing:
- **loopback** + established connections,
- **the Docker host** (`host.docker.internal`) — local Ollama and every sibling
  challenge container's published port live here,
- **DNS**,
- **only** the domains in `sandbox/proxy-allowlist.txt` (LangSmith, Brave, npm),
  via the proxy.

The agent's scanners/`curl` therefore cannot reach arbitrary public IPs, but
Ollama, the challenge targets, and (opt-in) tracing/Brave still work. It is
**fail-closed**: if the firewall can't be applied the container exits rather than
run wide open. Requirements: `--cap-add NET_ADMIN` (+ `NET_RAW` for nmap).

**Opt in to full internet** (e.g. to let the agent do live web research):
```bash
# docker run: add -e CLANKER_ALLOW_INTERNET=1     |     compose: CLANKER_ALLOW_INTERNET=1 in .env
```
To widen the allowlist, add regex lines to `sandbox/proxy-allowlist.txt` and
rebuild the eval image.

> Note: the interactive HTB harness (`main.py`) is intentionally **not**
> sandboxed — it attacks remote HTB machines by design. This isolation is
> eval-only.

## LangSmith tracing

Fully wired (`@traceable` on all LLM clients + the MCP tool dispatch, incl. the
local Ollama calls). Enable by putting your key in `.env`:
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=filthy-clanker
```
Compose reads `.env` automatically; for `docker run` add `--env-file .env`. The
sandbox allowlists `*.smith.langchain.com`, so traces flow even when the agent is
isolated from the rest of the internet.

## Solve-quality experiment flags

(See spec `filthy-clanker-agent-solve-quality`.) These levers make the four
experiments A/B-testable; defaults are the recommended settings.

| Lever | Flag / setting | Default | A/B |
|-------|----------------|---------|-----|
| **1. submit_flag gating** | `--submit-flag-mode always\|gated\|off` `--gate-after N` | `gated`, `N=3` | `gated` vs `off` vs `always` |
| **2. name-as-hint** | (baked into `build_task`) | softened | — |
| **3. tool-desc trim** | `settings.max_tool_description_chars` in `agents.yaml` | `240` | `240` vs `0` (full) |
| **4. NYU-CTF prompts** | `--profile nyu-ctf` | off (base = HTB prompts) | base vs `--profile nyu-ctf` |

**Resume vs re-run:** by default a run **skips** challenges already scored in any
`/results/*.jsonl` (crash-safe resume). Pass **`--rerun`** to redo them — required
when A/B-testing the same challenges with new code/flags.

### Example A/Bs

```bash
# Exp 1 — gating effect on tool-calls-per-challenge and solve rate
... run_eval.py --max-chals 5 --rerun --submit-flag-mode gated
... run_eval.py --max-chals 5 --rerun --submit-flag-mode off
```
```bash
# Exp 4 — NYU-CTF prompts vs base HTB prompts
... run_eval.py --max-chals 5 --rerun                     # base
... run_eval.py --max-chals 5 --rerun --profile nyu-ctf   # CTF prompts + recon gets execute_command
```

## Full development split (long)

57 challenges; at 20 min/challenge worst case this is many hours. Run detached
with restart-on-failure and check back:

```bash
MSYS_NO_PATHCONV=1 docker run -d --name clanker-full --restart on-failure:10 \
  --add-host host.docker.internal:host-gateway \
  -e OLLAMA_HOST=http://host.docker.internal:11434 -e EVAL_RESULTS_DIR=/results \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v D:/filthy-clanker:/home/kali/filthy-clanker \
  -v filthy-clanker_clanker-eval-results:/results \
  -v filthy-clanker_clanker-nyuctf-data:/root/.nyuctf \
  filthy-clanker-nyu-bench:latest \
  python3 evals/nyu_bench/run_eval.py --timeout 1200 --profile nyu-ctf
```

An interrupted full run resumes automatically on restart (already-scored
challenges are skipped unless `--rerun`).
