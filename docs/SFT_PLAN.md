# Fine-tuning Filthy-Clanker's local model — plan & options

Status: **draft for discussion** (2026-08-12). Nothing here is committed; the training
pipeline is deliberately *not* built yet. This documents where the training data really
stands, what was just fixed, and the candidate paths to a fine-tuned model — with the
trade-offs and the licensing/ToS questions called out so we can pick a direction.

---

## 1. Where the training data actually stands (reality check)

The claim that "we're already sitting on training data" was **wrong**, and the code proves
it. Two problems with what `data/training/` holds today (306 files from prior eval runs):

1. **Wrong shape for SFT.** `data_capture.TrajectoryLogger` records a *derived* view —
   parsed `{tool_name, arguments}` + a 10 KB-truncated `result_snippet` + KB diffs. SFT of a
   tool-calling agent needs the **literal prompt** (system + full message history + tool JSON
   schema) paired with the **literal assistant completion** (its text + `tool_calls`). Those
   are not stored.
2. **Broken provenance/scoring in eval mode.** `run_eval._log_trajectories` called the logger
   with an **empty `state_before`** (`knowledge_base={}`, `task=""`, `session_id=""`), so the
   `success_score` diff is meaningless and records were unattributable. Grep confirms the
   corpus has **zero** successful trajectories (`session_success:true` → 0 files;
   `success_score:1.0` → 0 records) — consistent with the ~3–6/57 solve rate. **No positives.**

So the legacy corpus is analytics telemetry, not a fine-tuning set. It is not worth
harvesting.

### What was fixed in this pass

- **New `src/sft_capture.py`** — opt-in, full-fidelity capture. One record per *agent
  invocation* holding the exact `system_prompt`, the tool schema as shown (post-trim), and the
  complete message list (every assistant turn with `tool_calls` + the tool results it saw).
  Enable with `run_eval.py --capture-sft` (sets `CLANKER_CAPTURE_SFT=1`); writes
  `data/sft/<session>.jsonl`. Hooked at `agents.py` end-of-`_run_agent_loop`. Never raises
  into the loop. `CLANKER_SFT_TOOLS=names` shrinks files if the schema is constant.
- **`_log_trajectories`** now threads real `session_id`/`task` and is documented as
  analytics-only (its score stays unreliable — sft_capture is the SFT source of truth).

**This only fixes *capture*. It does not create positives** — the base model still rarely
solves, so self-play capture yields mostly negative trajectories. Positives must be
*bootstrapped* (Section 3).

---

## 2. Capture route: local JSONL vs LangSmith

Both now exist; they capture the same underlying thing (the LLM call's inputs+output).

| | `sft_capture.py` (local) | LangSmith (already wired) |
|---|---|---|
| Source | end of `_run_agent_loop` | `@traceable` on `OllamaClient.generate_response` (`ollama_client.py:90`) |
| Fidelity | system + messages + tool schema + all assistant turns | same inputs/outputs, per LLM call |
| Granularity | one record per agent invocation | one run per LLM call (finer) |
| Outcome tagging | join to run report by `session_id` | needs run metadata/tags to filter by solved |
| Export | read JSONL directly | `langsmith` SDK `client.list_runs(project=…, filter=…)` |
| Dependency | none (files) | LangSmith backend + `LANGSMITH_TRACING=true` |
| Best for | building the SFT set | debugging + a backup capture already running |

**Recommendation:** use **`sft_capture.py` as the primary** dataset source (self-contained,
trivially filterable by joining `data/sft/<session>.jsonl` to the eval report's `solved`
field). LangSmith is a **viable fallback/cross-check** — the data is already flowing there —
but correlating runs → solved-challenge and exporting token-exact completions is more work
than reading our own JSONL. Concretely for LangSmith export we'd: tag each LLM run with
`session_id` + `challenge` (add to the `@traceable` metadata), then `list_runs` filtered to
sessions the report marked solved. Worth doing only if we want the finer per-call granularity
or we lose the local files.

---

## 3. The real problem: getting *positive* trajectories

The base abliterated 30B-A3B rarely solves, so its own runs can't teach it to solve. Every
bootstrapping option below is a way to manufacture **successful** trajectories in *our* tool
space (so the tool-call format matches deployment — no distribution shift). Three candidates:

### Option A — Guided self-distillation from HTB walkthroughs  *(your idea 1)*
Feed the official HTB walkthrough for a machine to the model as a hint, let it drive the real
MCP tools to the flag, keep the trajectory, then **strip the walkthrough from the training
prompt** so the model learns to produce the trajectory unaided. (This is rejection-sampling /
STaR-style "hindsight" bootstrapping.)
- **Pros:** generates positives in our exact harness + tool space; targets the recon→exploit
  →privesc, multi-step, and web/service skills that are precisely the a3b's weak spot (and the
  documented failure mode across TrustedSec/Purpleshift). Uses assets you already pay for.
- **Cons:** semi-manual and slow (spawn each box, feed walkthrough, capture); **domain
  mismatch** — HTB is pentest/host-compromise, so it won't help NYU-CTF crypto/rev/forensics
  much (but those are already the a3b's *strong* categories). Needs the interactive harness
  (`main.py`), not the sandboxed eval.
- **Licensing:** HTB walkthroughs are HTB IP. Using them privately as a solving aid to
  generate *your own* trajectories is different from redistributing the walkthrough text —
  keep only the generated tool-call trajectories, don't redistribute walkthrough content, and
  confirm against current HTB ToS. Prefer official HTB walkthroughs over third-party writeups.

### Option B — Distill a stronger model through the same harness  *(your idea 2)*
Point `--worker-model`/`--provider` at a stronger model, run it over CTF challenges, capture
its trajectories, SFT the local model on them.
- **Pros:** highest-quality, scalable, covers all categories; same MCP tools ⇒ format matches.
- **Cons / the ToS question you raised:** frontier **API** terms generally restrict using
  outputs to *develop competing models*. Anthropic, OpenAI, and Google Gemini all carry a
  clause of this kind. For a **private, non-distributed research artifact** the risk is lower
  than for anything shipped/commercial, but it is against the letter of most ToS — this is a
  judgment call for whoever owns the account, not something to hand-wave. **Check the current
  terms before doing this**; don't rely on this doc's summary.
- **ToS-clean variant (recommended if we go this way):** distill from a **strong open-weight**
  model whose license permits training on outputs — e.g. DeepSeek-V3, Qwen3-235B, or even the
  **dense `qwen3-abliterated:32b` you already have locally** — driving the same harness. No
  API, no competing-model clause. Lower ceiling than Claude/Gemini but zero legal ambiguity.

### Option C — Reuse published execution-verified trajectories  *(fastest)*
Cyber-Zero and CTF-Dojo (both Amazon-science, Aug 2025) **released** their training
trajectories and code. SFT directly on those.
- **Pros:** thousands of vetted positives immediately; the exact recipe that lifted open Qwen3
  to 13.5% / 10.4% on NYU-CTF; no generation step.
- **Cons:** their trajectories use *their* scaffold/tool schema, not our Hexstrike MCP tool
  names — so either (a) accept the format gap (the model learns CTF reasoning/tool-use
  patterns generally, then our harness prompts adapt it), or (b) reformat their steps into our
  tool schema. **Check each repo's data license** (`amazon-science/Cyber-Zero`,
  `amazon-science/CTF-Dojo`) before use.

**Likely best sequence:** start with **C** (cheap, proven, gets a first fine-tune on the
board), measure on CTFTiny, then add **A** (HTB) to target our specific web/pwn gap, and only
consider **B-frontier** if the ToS answer is acceptable to you.

---

## 4. Training recipe (once we have positives)

### Which model to fine-tune
- **Recommended:** the **dense `huihui_ai/qwen3-abliterated:32b`** you already have. Dense is
  what Cyber-Zero/CTF-Dojo fine-tuned; it's easier to LoRA than an MoE and already abliterated
  (keeps the no-refusal property for offensive work). Trade-off vs the current a3b worker:
  slower per token, more VRAM, but far more per-step reasoning.
- **Alternative:** the a3b MoE (matches current serving, fast) — but **MoE SFT is fiddly**
  (router/expert-load balance, expert dropout) and lower per-step reasoning caps the ceiling.
  Not recommended for the first attempt.
- Re-check refusal rate after SFT (the eval already logs `refusal_fired` per challenge; a
  CTFTiny run gives an immediate abliteration-intact check).

### Method
- **QLoRA** (4-bit base + LoRA adapters) via unsloth or axolotl — fits a 32B on a single 24 GB
  card for training; cheap to iterate. Full fine-tune only if a LoRA plateaus.
- **Chat + tools format:** convert each captured turn to `messages=[system, user,
  assistant(tool_calls), tool, …]` and apply the **Qwen3 chat template** (it supports tool
  role + tool_calls). **Mask the loss to assistant tokens only** (including tool-call tokens);
  do not train on system/user/tool-result tokens.
- **Filter to positives:** keep trajectories from sessions the report marked `solved:true`
  (join `data/sft/<session>.jsonl` ↔ run report). Optionally reward-weight by
  `tool_calls`/efficiency. Consider trimming to the *minimal* successful path per challenge.

### Serving bridge (SFT → runs in the existing harness)
HF/PEFT training → merge adapters → **convert to GGUF** (llama.cpp) → quantize (Q4_K_M to
match current footprint) → `ollama create clanker-sft -f Modelfile` → point the harness at it
with `--worker-model clanker-sft`. Document the exact conversion commands when we build it.

### Eval loop & overfitting guards
- Iterate on **CTFTiny (50)** with `--worker-model` for fast comparison vs the baseline; use
  the full dev split for milestone checks.
- **Hold out** a slice of challenges from training and never train on them; watch train-vs-
  heldout solve-rate gap. Keep every harness fix general (the standing overfitting rule).
- Track solve rate **and** refusal rate **and** avg tool-calls/challenge (all already in the
  report) so we see capability, alignment, and efficiency together.

---

## 5. Decisions to make together (blocking the build)

1. **Bootstrap source:** C (reuse Cyber-Zero/CTF-Dojo data) first, or straight to A (HTB) /
   B (distill)? — affects licensing and effort most.
2. **Frontier distillation (B):** in or out, given the ToS? If out, confirm the open-weight
   distill variant is the fallback.
3. **Target model:** dense abliterated-32B (recommended) vs the a3b MoE (matches serving).
4. **HTB capture ergonomics:** is semi-manual walkthrough-guided capture acceptable, or do we
   want to script machine spawn + walkthrough injection first?
5. **Training stack:** unsloth vs axolotl (both fine; unsloth is faster/simpler for a single
   card).

Once 1–3 are decided, the build is: dataset-builder script (`data/sft/*` + report → masked
chat/tools examples) → QLoRA config → GGUF/Ollama export → CTFTiny A/B. Est. ~1–2 days to a
first measurable fine-tune via Option C.
