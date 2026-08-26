# Frontier-Hint Backtest Report

Offline detection + compaction study over captured trajectories. No GPU, no API spend.

## Summary

- **n_sessions**: 25
- **n_stuck**: 14
- **n_solved**: 11
- **stuck_detection_recall**: 9/14
- **avg_steps_left_at_fire_on_stuck**: 18.8
- **solved_runs_that_fired**: 1/11
- **solved_fired_that_recovered**: 1/1
- **total_fires**: 32
- **avg_fires_per_session**: 1.28
- **projected_cost_all_fires_sonnet**: 0.5397
- **projected_cost_all_fires_opus**: 2.6986
- **projected_cost_per_session_sonnet**: 0.02159

### How to read it

- `stuck_detection_recall` — of runs that captured NO flag, how many the detector flagged as stuck (higher = catches more salvageable runs).
- `avg_steps_left_at_fire_on_stuck` — headroom a hint would have had (higher = more chance to change the outcome).
- `solved_runs_that_fired` — potential false alarms; `solved_fired_that_recovered` shows how many of those were followed by a real finding anyway (a fire-then-recover is benign, a frequent one erodes economics).
- cost lines project the frontier spend if EVERY fire made one hint call.

## Config used
```yaml
window: 6   min_steps: 5   cooldown: 5   fire_threshold: 0.4
weights: {"kb_stall": 0.18, "repeat": 0.35, "error_rate": 0.2, "stale_progress": 0.15, "oscillation": 0.15, "step_budget": 0.3}
```

## Per-session

| session | cat | steps | solved | flag@ | fires | 1st fire@ | steps left | recovered | top signals |
|---|---|--:|:--:|--:|--:|--:|--:|:--:|---|
| 2014f-web-webroot-d8dfd173 | web | 59 | ❌ | — | 8 | 18 | 41 | no | repeat:0.21, stale_progress:0.15, step_budget:0.09 |
| 2016q-for-watchword-7f7e284b | for | 57 | ❌ | — | 9 | 14 | 43 | no | kb_stall:0.18, stale_progress:0.15, step_budget:0.05, error_rate:0.03 |
| 2013f-cry-stfu-a2c4d498 | cry | 35 | ❌ | — | 5 | 6 | 29 | no | repeat:0.21, kb_stall:0.18, stale_progress:0.11 |
| 2013q-pwn-csaw_diary-80bb3321 | pwn | 35 | ❌ | — | 1 | 19 | 16 | yes | repeat:0.35, step_budget:0.06, error_rate:0.03 |
| 2014q-rev-aerosol_can-bff9bcb8 | rev | 27 | ❌ | — | 2 | 8 | 19 | no | repeat:0.35, stale_progress:0.15 |
| 2013q-cry-csawpad-f515d3fa | cry | 26 | ❌ | — | 2 | 18 | 8 | no | kb_stall:0.18, stale_progress:0.15, step_budget:0.09 |
| 2014q-rev-aerosol_can-03f0b723 | rev | 23 | ❌ | — | 0 | — | — | — | — |
| 2013q-cry-slurp-c8566fd3 | cry | 22 | ❌ | — | 2 | 17 | 5 | no | repeat:0.21, stale_progress:0.15, step_budget:0.06 |
| 2013q-pwn-csaw_diary-66f8aaed | pwn | 22 | ❌ | — | 1 | 16 | 6 | yes | repeat:0.35, kb_stall:0.04, error_rate:0.03, stale_progress:0.02 |
| 2016q-pwn-warmup-3eed5665 | pwn | 22 | ❌ | — | 0 | — | — | — | — |
| 2013q-cry-onlythisprogram-7aaf84ff | cry | 20 | ❌ | — | 1 | 18 | 2 | no | kb_stall:0.18, stale_progress:0.15, step_budget:0.09 |
| 2013q-msc-life-037c777a | msc | 20 | ❌ | — | 0 | — | — | — | — |
| 2013q-msc-life-b61ef326 | msc | 20 | ❌ | — | 0 | — | — | — | — |
| 2016q-msc-regexpire-92544872 | msc | 19 | ❌ | — | 0 | — | — | — | — |
| 2013f-web-historypeats-53454031 | web | 35 | ✅ | 17 | 0 | — | — | — | — |
| 2016q-rev-rock-83904aca | rev | 22 | ✅ | 1 | 0 | — | — | — | — |
| 2016q-web-mfw-4ecd54d0 | web | 18 | ✅ | 1 | 0 | — | — | — | — |
| 2014f-rev-odd-d3b6b859 | rev | 16 | ✅ | 14 | 1 | 9 | 7 | yes | repeat:0.21, kb_stall:0.18, stale_progress:0.15, error_rate:0.07 |
| 2014f-rev-return_of_the_weiner-46fc2d41 | rev | 16 | ✅ | 1 | 0 | — | — | — | — |
| 2016q-rev-rock-cbc7919e | rev | 16 | ✅ | 1 | 0 | — | — | — | — |
| 2013f-cry-stfu-e476c6af | cry | 15 | ✅ | 1 | 0 | — | — | — | — |
| 2014f-pwn-kernel-71a45375 | pwn | 8 | ✅ | 7 | 0 | — | — | — | — |
| 2013q-msc-networking_1-12aad63e | msc | 5 | ✅ | 1 | 0 | — | — | — | — |
| 2013q-msc-networking_1-1fe43450 | msc | 5 | ✅ | 1 | 0 | — | — | — | — |
| 2013q-web-guess_harder-91a52874 | web | 5 | ✅ | 1 | 0 | — | — | — | — |

_Payloads for every fire are in `out/payloads/`. Inspect one with `frontier_hint.py --payload <f>` (dry-run) to see exactly what the frontier would receive._
