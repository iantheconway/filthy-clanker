"""
Token-cost accounting + a hard budget guard for PAID (API) runs.

The API clients (Anthropic, OpenAI-compatible) call ``add_usage(model, usage)`` after every
response; ``total_cost()`` returns the running USD spend, which run_eval checks between
challenges against ``--max-cost`` and stops the run when exceeded. Ollama (local) is free and
falls through to $0.

Prices are USD per 1M tokens (input, output, cache_read). They are APPROXIMATE — update as
provider pricing changes — and the guard is a SAFETY bound, not an invoice. run_eval also caps
by challenge count (--max-chals / the slice size), so even if a price here is stale the spend
is doubly bounded. When in doubt, prices here err slightly HIGH so the cap trips early.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger("filthy_clanker")

# model-substring -> (input_per_1M, output_per_1M, cache_read_per_1M). First substring match wins.
_PRICES: dict[str, tuple[float, float, float]] = {
    # Anthropic (approx; cache_write ≈ 1.25x input, handled below)
    "claude-opus-5":    (15.0, 75.0, 1.50),
    "claude-opus":      (15.0, 75.0, 1.50),
    "claude-sonnet-5":  (3.0, 15.0, 0.30),
    "claude-sonnet":    (3.0, 15.0, 0.30),
    "claude-haiku":     (1.0,  5.0, 0.10),
    # OpenAI-compatible open models (Together/DeepInfra-ish, cheap; no prompt-cache pricing)
    "qwen3-235b":       (0.20, 0.60, 0.0),
    "qwen":             (0.20, 0.60, 0.0),
    "gpt-oss":          (0.15, 0.60, 0.0),   # OpenAI OPEN-weight (Apache-2.0); serverless on Together
    "deepseek":         (0.60, 1.70, 0.0),
    "llama":            (1.05, 1.05, 0.0),   # e.g. Llama-3.3-70B-Turbo on Together ($1.04/$1.04)
}

_lock = threading.Lock()
# Paid (cloud) usage in input/output/…; FREE/local-model usage (e.g. the served "clanker"
# worker, which has no price) is tracked separately in local_* so efficiency metrics can see
# the worker's real token volume even though it costs $0.
_state = {"cost": 0.0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "calls": 0,
          "local_input": 0, "local_output": 0, "local_calls": 0}
_warned_unknown: set = set()


def _price(model: str) -> tuple[float, float, float] | None:
    m = (model or "").lower()
    for key, price in _PRICES.items():
        if key in m:
            return price
    return None


def add_usage(model: str, usage: dict) -> None:
    """Accumulate the USD cost of one API response. No-op for free/local (unknown) models."""
    if not usage:
        return
    price = _price(model)
    it = int(usage.get("input_tokens", 0) or 0)
    ot = int(usage.get("output_tokens", 0) or 0)
    cr = int(usage.get("cache_read_input_tokens", 0) or 0)
    cw = int(usage.get("cache_creation_input_tokens", 0) or 0)
    if price is None:
        # Local/unknown model (e.g. the served "clanker" worker, or an Ollama tag) → $0, but
        # still record its token volume in local_* for efficiency metrics.
        with _lock:
            _state["local_input"] += it
            _state["local_output"] += ot
            _state["local_calls"] += 1
        # Warn once for anything that *looks* like it should cost (vendor prefix) so a mispriced
        # paid model shows up.
        if model and ("/" in model or model.lower().startswith("claude")) and model not in _warned_unknown:
            _warned_unknown.add(model)
            logger.warning("[cost] no price for %r — counting as $0 (add it to cost._PRICES if paid)", model)
        return
    pin, pout, pcache = price
    cost = (it / 1e6) * pin + (ot / 1e6) * pout + (cr / 1e6) * pcache + (cw / 1e6) * pin * 1.25
    with _lock:
        _state["cost"] += cost
        _state["input"] += it
        _state["output"] += ot
        _state["cache_read"] += cr
        _state["cache_write"] += cw
        _state["calls"] += 1


# LLM-call FAILURES by kind (e.g. BUG-B 400s, context overflow, connection). A run can exit 0
# with results while most calls silently failed — this lets the runner self-report that.
_errors: dict = {}


def add_error(kind: str) -> None:
    with _lock:
        _errors[kind] = _errors.get(kind, 0) + 1


def errors_summary() -> dict:
    with _lock:
        return dict(_errors)


def total_cost() -> float:
    with _lock:
        return _state["cost"]


def summary() -> dict:
    with _lock:
        return dict(_state)


def reset() -> None:
    with _lock:
        for k in _state:
            _state[k] = 0.0 if k == "cost" else 0
        _errors.clear()
