# Experiment: Frontier Hinting / Guidance Agent

**Branch:** `experiment/frontier-hint` (isolated git worktree)
**Status:** Phase 1–2 buildout (offline, GPU-free, no API spend by default)
**Date:** 2026-08-20

## The idea

A local model grinds through a CTF. At sparse decision points where it is
**stuck**, the system compacts the current status and routes it to a **frontier
model** for a hint on the next steps. Local model does the cheap grinding;
frontier injects high-value judgment only when it matters.

Two payoffs:
1. **Cost-asymmetric labor** — local does the 200-step slog (cheap tokens),
   frontier reasons only at the 3–5 moments per box that decide the outcome.
   This targets the gap in the memory (local ~12–25% vs Sonnet ~33–50% on
   NYU-dev) without paying frontier prices for the whole run.
2. **Data flywheel** — every `(stuck-state → hint → measured-progress)` triple
   is a labeled distillation target ("expert next move given this state"), the
   exact supervision the SFT/LoRA thread is starved for (~28 positives). As the
   LoRA learns to self-hint, frontier calls can be retired.

## Why this experiment, why now

The live integration (a `guidance` node in the LangGraph) needs GPU + would
disturb the currently-running experiment. So we **validate the hard parts
offline first**, against trajectories already captured in
`data/training/*.jsonl`:

- **Q1 (detection):** Can we reliably detect stuck states from trajectory
  signals *without* firing constantly on runs that are about to succeed?
- **Q2 (compaction):** Is the compacted status rich enough — and grounded in
  raw tool output, not the local model's paraphrase — that a frontier model
  could give an actionable hint?
- **Q3 (cost/frequency):** How often would we fire, and what does that cost at
  frontier prices? Is the economics sane?

Only Q1 and Q3-projection need running here; Q2 produces inspectable payloads.
The actual frontier calls (Q2 confirmation) are **staged but gated** behind an
explicit `--live` flag + a hard `--max-cost` cap so nothing auto-spends.

## Corpus

`data/training/` in the main tree (read-only). Join key: `session_id`.

- **Action records** (`type` absent, has `action`/`success_score`/
  `knowledge_base_before|after`): the per-step trajectory. 29 per-session files,
  22 with ≥8 steps.
- **`session_end` records** (in `all_trajectories.jsonl`): outcome labels
  (`session_success`, `flags_captured`).

Ground-truth stuck label used here: a run that **captured no flag**
(`success_score` never hits 1.0 and no flag in any `submit_flag`). Of the 22
multi-step runs, **8 solved / 14 ground out** — the 14 (incl. 59-, 57-, 35-step
runs that never flagged) are the positives detection must catch.

## Signals (see `signals.py`)

Computed per-step from the trajectory. `success_score` already encodes the
finding heuristic (1.0 flag / 0.85 cred / 0.70 surface / 0.55 port / … / 0.0
error), so it is the backbone.

| Signal | Meaning | Source |
|---|---|---|
| `kb_stall` | consecutive steps with `success_score ≤ 0.05` (no new finding) | score run-length |
| `repeat` | near-duplicate tool calls in a sliding window (normalized cmd) | action args |
| `error_rate` | fraction of recent steps with `success_score == 0.0` | scores |
| `stale_progress` | steps since last "real finding" (`score ≥ 0.55`) | scores |
| `oscillation` | agent bouncing (e.g. recon↔exploit) with no score gain | `agent` field |
| `step_budget` | raw step count vs a per-category soft budget | index |
| `self_id` *(live-only)* | model says "I'm stuck" / refusal phrases | message stream — **not in captured trajectories**, documented but not backtested |

These combine into a `stuckness ∈ [0,1]` (weighted sum, `config.yaml`), with a
**min-steps warmup** and a **cooldown** so it can't fire every step or during
the opening enumeration.

## Files

- `config.yaml` — signal weights, window, threshold, cooldown, budgets (tune
  without touching code).
- `signals.py` — per-step signal extraction (pure, stdlib).
- `detector.py` — `StuckDetector`: streaming `update(action)` + batch
  `evaluate(actions)`. The `update` API is written to drop straight into a live
  LangGraph node later.
- `compactor.py` — builds the frontier-hint payload: objective + KB snapshot +
  **tried-and-failed ledger** + **raw recent evidence** (ground truth, not
  paraphrased) + which signals fired. Renders prompt string + structured JSON +
  token estimate.
- `frontier_hint.py` — gated frontier caller. **Dry-run by default** (returns
  payload + projected cost, no call). `--live` requires `ANTHROPIC_API_KEY` and
  stays under `--max-cost`. Pricing mirrored from `src/llms/cost.py`.
- `backtest.py` — runs the detector over the corpus, scores detection against
  the solved/stuck labels, projects frontier cost, and dumps per-fire payloads
  to `out/payloads/` + a report to `out/backtest_report.md`.

## Run

```bash
# Phase 1+2 — offline, no GPU, no API spend. Uses main-tree data by default.
python experiments/frontier_hint/backtest.py

# inspect what we'd send the frontier at each fire:
ls experiments/frontier_hint/out/payloads/

# Phase 2 confirmation (OPTIONAL, costs money — gated):
python experiments/frontier_hint/frontier_hint.py \
    --payload experiments/frontier_hint/out/payloads/<file>.json \
    --live --model claude-sonnet-5 --max-cost 0.50
```

## What "good" looks like (success criteria)

- **Detection:** fires on ≥ ~10/14 stuck runs, and on a stuck run fires with
  enough steps left that a hint could plausibly change the outcome (report the
  "steps remaining at first fire" distribution). On solved runs, either does not
  fire, or fires-then-recovers (report both — a fire that precedes a solve is
  not automatically a false alarm, but a *frequent* one erodes the economics).
- **Frequency/cost:** projected frontier cost per box is a small fraction of
  what running the *whole* box on the frontier would cost (the whole point).
- **Payloads:** spot-check shows the tried-ledger prevents re-suggesting dead
  ends and the raw evidence exposes what the local model misread.

## Known limitations / honesty

- `self_id` and true agent-oscillation are richer live (full message stream);
  the captured trajectory is tool-call-level, so the backtest under-uses them.
  Results here are a **lower bound** on detectability.
- Outcome labels come from flag capture; a run marked "stuck" that was actually
  an unsolvable/misconfigured challenge is noise — spot-checked in the report.
- `success_score` is itself a heuristic; the detector inherits its blind spots.
- This does **not** validate that a hint *causes* recovery — that needs the live
  loop (Phase 3). It validates detection, compaction quality, and economics.

## Next (Phase 3, needs GPU + a run)

Wire a `guidance` node mirroring `compaction_node`: supervisor detects stuck →
`guidance` calls frontier → hint written into `messages`/`knowledge_base` →
back to supervisor. Gate with the existing `cost.py` cap (cooldown, max
hints/session). Measure KB-delta / `success_score` over the K steps after a
hint to (a) tune thresholds and (b) harvest the distillation triples.
