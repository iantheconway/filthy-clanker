"""
Gated frontier-hint caller.

DRY-RUN BY DEFAULT: given a payload JSON, prints the rendered prompt + a projected
cost and exits WITHOUT calling any API. Real calls require BOTH:
  * --live
  * ANTHROPIC_API_KEY in the environment
and stay under --max-cost (projected input+output). Nothing auto-spends.

Pricing mirrors src/llms/cost.py (USD per 1M tokens). Kept local so this module
does not import the running project source.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import compactor

# (input_per_1M, output_per_1M) — mirror of src/llms/cost.py _PRICES, first match wins.
_PRICES = {
    "claude-opus-5": (15.0, 75.0),
    "claude-opus": (15.0, 75.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (1.0, 5.0),
}
_DEFAULT_MAX_OUTPUT = 700  # hints are short structured JSON


def price(model: str):
    m = (model or "").lower()
    for k, v in _PRICES.items():
        if k in m:
            return v
    return None


def project_cost(payload, model: str, max_output=_DEFAULT_MAX_OUTPUT) -> float:
    p = price(model)
    if p is None:
        return 0.0
    pin, pout = p
    tin = compactor.estimate_tokens(payload)
    return (tin / 1e6) * pin + (max_output / 1e6) * pout


def call_frontier(payload, model="claude-sonnet-5", max_output=_DEFAULT_MAX_OUTPUT):
    """Real API call. Requires the `anthropic` SDK + ANTHROPIC_API_KEY."""
    import anthropic  # imported lazily so dry-run needs no dependency
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_output,
        system=compactor.SYSTEM_PROMPT,
        messages=[{"role": "user", "content": compactor.render_prompt(payload)}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    usage = getattr(resp, "usage", None)
    try:
        hint = json.loads(text)
    except json.JSONDecodeError:
        # models sometimes wrap JSON in prose/fences — salvage the object
        s, e = text.find("{"), text.rfind("}")
        hint = json.loads(text[s:e + 1]) if s >= 0 and e > s else {"_raw": text}
    return {
        "hint": hint,
        "usage": {"input": getattr(usage, "input_tokens", None),
                  "output": getattr(usage, "output_tokens", None)} if usage else None,
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # avoid cp1252 mojibake on Windows consoles
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Gated frontier-hint caller (dry-run by default).")
    ap.add_argument("--payload", required=True, help="path to a payload JSON from backtest out/payloads/")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--max-cost", type=float, default=0.25, help="hard cap on projected USD for a live call")
    ap.add_argument("--max-output", type=int, default=_DEFAULT_MAX_OUTPUT)
    ap.add_argument("--live", action="store_true", help="actually call the API (else dry-run)")
    args = ap.parse_args()

    with open(args.payload, encoding="utf-8") as f:
        payload = json.load(f)

    projected = project_cost(payload, args.model, args.max_output)
    print(f"# payload: {args.payload}")
    print(f"# model: {args.model}   projected cost: ${projected:.4f}   "
          f"(~{compactor.estimate_tokens(payload)} in + {args.max_output} out tok)")

    if not args.live:
        print("\n# DRY-RUN — rendered prompt below. Pass --live to actually call.\n")
        print("=== SYSTEM ===\n" + compactor.SYSTEM_PROMPT)
        print("\n=== USER ===\n" + compactor.render_prompt(payload))
        return

    if projected > args.max_cost:
        sys.exit(f"REFUSING live call: projected ${projected:.4f} > --max-cost ${args.max_cost:.4f}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("REFUSING live call: ANTHROPIC_API_KEY not set")

    print(f"\n# LIVE call (under ${args.max_cost} cap)…\n")
    result = call_frontier(payload, args.model, args.max_output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
