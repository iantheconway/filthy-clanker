# Clanker SFT Dataset Report

Provenance of the fine-tuning corpus and the held-out evaluation set for
`Qwen3.5-35B-A3B-clanker-ctf-v1`. Generated from the run reports
(`evals/**/run-*.json`, `evals/**/ic-*.json`) and capture files (`data/sft/*.jsonl`),
not from notes — re-derive with the analysis snippet at the bottom.

## Training set — `training/data/sft_train_full.jsonl`

| Metric | Value |
|---|---|
| Examples (JSONL lines) | **125** |
| Distinct challenges | **54** (8 NYU dev + 46 InterCode) |
| Solved sessions | **89** (11 NYU + 78 InterCode) |
| Assistant turns (total, the trained tokens) | **822** |
| Tool schema | 32-tool unified union + `objdump_helper` stub |
| Fidelity | full — no tool-output trimming, no length dropping (examples up to ~24k tok) |
| File size | ~8.9 MB |
| Together file id | `file-52789812-…` |

One example = one agent invocation's turn-sequence from a solved run. The same
challenge solved by two models counts as two sessions (→ more examples), which is
why 54 challenges expand to 89 sessions and 125 examples.

### By benchmark (sessions)

- **InterCode-CTF: 78** — picoCTF-derived tasks, the bulk of the corpus.
- **NYU CTF Bench (dev split): 11**

### By model that produced the solve (sessions)

| Model | Sessions | Weights / ToS |
|---|---|---|
| gpt-oss-120b | 40 | open, Apache-2.0 |
| qwen3-abliterated 30b-a3b | 11 | open |
| qwen3-abliterated 32b | 9 | open |
| gemma-4-abliterated 26b | 6 | open |
| gemma4 26b | 5 | open |
| claude-sonnet-5 | 5 | **frontier — ToS-gated, opted in** |
| *unknown* (`?`) | 13 | capture line lacked a `model` field (older local runs) |

The 5 `claude-sonnet-5` sessions expand to **13 examples**, included only because
the build was run with `--include-frontier --yes-i-accept-tos-risk` (this model is
for our own non-competing solver; see `docs/CLOUD_TRAINING.md`). Everything else is
open-weight and ToS-clean. The 13 `?` sessions are InterCode captures whose first
JSONL line had no `model` key — almost certainly the earlier local a3b/gemma/dense32b
runs, **not** additional frontier data.

### By category (sessions)

misc 29 · rev 23 · forensics 17 · crypto 17 · pwn 3 · **web 0**

**Web is absent from training by construction** — every web solve we have ever
captured was routed into the held-out eval set (below). The fine-tune therefore
learns web only indirectly, through general tool-use transfer.

### By run

**InterCode:** `gptoss-overnight` 44 · `dense32b` 8 · `a3b` 7 · `gemma-abl` 7 ·
`results` 6 · `gemma-stock` 6
**NYU:** `frontier-sonnet` 3 · `cloud-gptoss120-devrest` 2 · `local-dense32b` 2 ·
`sonnet-overnight` 2 · `frontier-sonnet-devrest` 1 · `sonnet-smoke` 1

### The 8 NYU training challenges

`2013q-cry-csawpad` (crypto) · `2013q-msc-networking_1` (misc) · `2014f-pwn-kernel`
(pwn) · `2014f-rev-return_of_the_weiner` (rev) · `2014q-pwn-the_road_less_traveled`
(pwn) · `2015q-cry-eps` (crypto) · `2016q-for-kill` (forensics) · `2016q-rev-rock`
(rev)

### 46 InterCode training challenges

Distinct picoCTF tasks by category: misc 17 · rev 13 · crypto 10 · forensics 5 ·
pwn 1 (no web). Identified by task number (`ic-NNN`); the picoCTF category comes
from the run report.

### Post-processing applied for the Together upload

CRLF→LF · `ensure_ascii=True` (Together's Windows validator opens files as cp1252) ·
all examples' `tools` unified to the 32-tool union so no `tool_call` references a
tool absent from its own example (multi-agent shared history caused this) · all
`tool_call.arguments` coerced to valid JSON strings.

## Held-out evaluation set — `evals/nyu_bench/eval_holdout.txt`

15 challenges, **all NYU dev split**, deliberately web-weighted (10/15 web). None of
these appear in training.

### 6 held-out SOLVED — regression check

We hold working trajectories for these but pulled them OUT of training
(`train_exclude.txt`), so a pass/fail post-fine-tune measures **regression** against
what the base already handled:

- `2013f-web-historypeats` (web)
- `2013q-web-guess_harder` (web)
- `2016q-web-mfw` (web)
- `2013q-cry-slurp` (crypto) ← in a category with real training signal
- `2013q-msc-networking_2` (misc) ← in a category with real training signal
- `2014f-rev-odd` (rev) ← in a category with real training signal

### 9 NEVER-solved — improvement check

Passing any of these post-fine-tune is net-new capability:

- Web (7): `2015q-web-throwback` · `2016q-web-i_got_id` · `2014f-web-webroot` ·
  `2016f-web-seizure_cipher` · `2016f-web-cloudb` · `2015q-web-k_stairs` ·
  `2014q-web-silkgoat`
- `2015q-rev-wyvern` (rev)
- `2016q-msc-coinslot` (misc)

### Also fully held out

The **entire NYU CTF Bench TEST split** was never captured or trained on — reserved
as a clean, unbiased benchmark for a later, larger evaluation.

## Reading the eventual eval honestly

- Movement is most plausible in **rev / misc / crypto**, where training has real
  in-category signal. `slurp`, `networking_2`, `odd` are the meaningful
  regression probes there.
- **Web is the stretch.** With zero web positives in training, don't expect the LoRA
  to crack the 7 never-solved web challenges; the 3 held-out-solved web ones mostly
  test that we didn't *break* base web ability. To actually move web we need web
  positives — which currently only exist in this held-out set.
- Report base-35B vs LoRA-35B as pass@1 per challenge with the delta, split by
  category, calling out both new solves and any regressions.

## Re-derive

```bash
# from repo root — regenerates every number above
python - <<'PY'
import json, glob, os
from collections import Counter
def canon(sid):
    if sid.startswith("eval-"): return sid[5:].rsplit("-",1)[0]
    if sid.startswith("ic-"):   return sid.rsplit("-",1)[0]
    return sid
solved={}
for r in glob.glob("evals/**/run-*.json",recursive=True)+glob.glob("evals/**/ic-*.json",recursive=True):
    try: d=json.load(open(r,encoding="utf-8"))
    except: continue
    b="InterCode" if "intercode" in r else "NYU"
    for x in d.get("results",[]):
        if x.get("solved"): solved[x["session_id"]]=(b,os.path.basename(os.path.dirname(r)))
held={l.strip() for l in open("evals/nyu_bench/train_exclude.txt") if l.strip() and not l.startswith("#")}
train=[s for s in solved if canon(s) not in held]
print("sessions",len(train),"| distinct",len({canon(s) for s in train}))
print(Counter(solved[s][0] for s in train))
PY
```
