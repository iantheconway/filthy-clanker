"""
StuckDetector — combines per-step signals into a stuckness score and fire events.

Two entry points:
  * evaluate(actions)  -> list[FireEvent]   (batch, used by the backtest)
  * update(action)     -> FireEvent | None  (streaming, drops into a live node)

The streaming API keeps internal history so a future LangGraph `guidance` node
can call detector.update(normalized_action) after each tool result and fire a
hint request the moment stuckness crosses threshold (respecting warmup/cooldown).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import signals

# Defaults mirror config.yaml so the module runs even without pyyaml / the file.
DEFAULTS = {
    "window": 6,
    "min_steps": 5,
    "cooldown": 5,
    "fire_threshold": 0.60,
    "weights": {
        "kb_stall": 0.10, "repeat": 0.25, "error_rate": 0.30,
        "stale_progress": 0.08, "oscillation": 0.15, "step_budget": 0.20,
    },
    "kb_stall_cap": 4,
    "stale_progress_cap": 8,
    "step_budgets": {"default": 12, "cry": 14, "rev": 16, "pwn": 16,
                     "web": 14, "for": 12, "msc": 12},
    "repeat_similarity": 0.90,
}


def load_config(path: str | None = None) -> dict:
    """Overlay config.yaml onto DEFAULTS if pyyaml + the file are available."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.yaml")
    try:
        import yaml  # optional
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        for k, v in loaded.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    except Exception:
        pass  # DEFAULTS are a complete, valid config on their own
    return cfg


@dataclass
class FireEvent:
    idx: int                     # step index that tripped the detector
    stuckness: float
    contributions: dict          # signal -> weighted contribution
    raw: dict                    # signal -> raw magnitude


def _budget_for(cfg: dict, category: str) -> int:
    b = cfg.get("step_budgets", {})
    return b.get(category, b.get("default", 12))


def step_stuckness(actions, i, cfg, category="default"):
    """Return (stuckness, contributions, raw_magnitudes) for step i."""
    w = cfg["weights"]
    win = cfg["window"]
    raw = {
        "kb_stall": signals.kb_stall(actions, i, cfg["kb_stall_cap"]),
        "repeat": signals.repeat(actions, i, win, cfg["repeat_similarity"]),
        "error_rate": signals.error_rate(actions, i, win),
        "stale_progress": signals.stale_progress(actions, i, cfg["stale_progress_cap"]),
        "oscillation": signals.oscillation(actions, i, win),
        "step_budget": signals.step_budget(actions, i, _budget_for(cfg, category)),
    }
    contrib = {k: w.get(k, 0.0) * v for k, v in raw.items()}
    stuckness = min(1.0, sum(contrib.values()))
    return stuckness, contrib, raw


class StuckDetector:
    def __init__(self, cfg: dict | None = None, category: str = "default"):
        self.cfg = cfg or load_config()
        self.category = category
        self._actions: list[dict] = []
        self._last_fire = -10 ** 9

    # ---- batch (backtest) -------------------------------------------------
    def evaluate(self, actions: list[dict]) -> list[FireEvent]:
        fires: list[FireEvent] = []
        last_fire = -10 ** 9
        for i in range(len(actions)):
            if i + 1 < self.cfg["min_steps"]:
                continue
            if i - last_fire < self.cfg["cooldown"]:
                continue
            s, contrib, raw = step_stuckness(actions, i, self.cfg, self.category)
            if s >= self.cfg["fire_threshold"]:
                fires.append(FireEvent(i, round(s, 4), contrib, raw))
                last_fire = i
        return fires

    # ---- streaming (live node) -------------------------------------------
    def update(self, action: dict) -> FireEvent | None:
        """Feed one normalized action; return a FireEvent if it should hint now."""
        action = dict(action)
        action.setdefault("idx", len(self._actions))
        self._actions.append(action)
        i = len(self._actions) - 1
        if i + 1 < self.cfg["min_steps"]:
            return None
        if i - self._last_fire < self.cfg["cooldown"]:
            return None
        s, contrib, raw = step_stuckness(self._actions, i, self.cfg, self.category)
        if s >= self.cfg["fire_threshold"]:
            self._last_fire = i
            return FireEvent(i, round(s, 4), contrib, raw)
        return None
