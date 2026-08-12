"""
Shared helpers for building SFT datasets from captured trajectories.

All three bootstrap paths (CTF-Dojo ingest, HTB self-distillation, frontier distill)
funnel through this module to produce ONE training format: OpenAI-style chat with tools,
one JSON object per line —

    {"messages": [{"role":"system",...},{"role":"user",...},
                  {"role":"assistant","content":"...","tool_calls":[{"id","type","function":{"name","arguments"}}]},
                  {"role":"tool","tool_call_id":"...","content":"..."}, ...],
     "tools": [ {"type":"function","function":{"name","description","parameters"}}, ... ]}

Trainers (axolotl / unsloth / trl) apply the model's chat template and mask the loss to
assistant tokens. This module does the format normalisation; it does NOT train.

Our capture (`src/sft_capture.py`) stores provider-native messages — Ollama-style
(str content + `tool_calls`) or Anthropic-style (list-of-blocks content with tool_use /
tool_result). `normalize_conversation` handles both.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


# ── Tool schema ──────────────────────────────────────────────────────────────

def mcp_tools_to_openai(tools: Any) -> list[dict]:
    """MCP tool schema list → OpenAI function-tool list. Tolerates a names-only capture
    (CLANKER_SFT_TOOLS=names) by emitting name-only stubs."""
    out: list[dict] = []
    for t in tools or []:
        if isinstance(t, str):
            out.append({"type": "function", "function": {"name": t, "parameters": {"type": "object", "properties": {}}}})
            continue
        if not isinstance(t, dict):
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("inputSchema") or {"type": "object", "properties": {}},
            },
        })
    return out


def _stringify(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        # Anthropic tool_result content is often a list of {"type":"text","text":...}
        parts = []
        for b in x:
            if isinstance(b, dict):
                parts.append(b.get("text") or b.get("content") or json.dumps(b))
            else:
                parts.append(str(b))
        return "\n".join(parts)
    if x is None:
        return ""
    return json.dumps(x, ensure_ascii=False, default=str)


# ── Message normalisation (Ollama- and Anthropic-shaped → OpenAI-shaped) ──────

def normalize_conversation(system_prompt: str, messages: list) -> list[dict]:
    """Return a clean OpenAI-style message list with tool_call ids paired to tool results."""
    out: list[dict] = []
    if system_prompt:
        out.append({"role": "system", "content": system_prompt})

    id_counter = 0
    pending_ids: list[str] = []  # tool_call ids awaiting their tool-result messages (Ollama has none inline)

    def new_id() -> str:
        nonlocal id_counter
        id_counter += 1
        return f"call_{id_counter}"

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Anthropic block-style content (list of {type: text|tool_use|tool_result})
        if isinstance(content, list):
            text_parts, tool_calls, tool_results = [], [], []
            for b in content:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    text_parts.append(b.get("text", ""))
                elif bt == "tool_use":
                    tid = b.get("id") or new_id()
                    tool_calls.append({"id": tid, "type": "function",
                                       "function": {"name": b.get("name", ""),
                                                    "arguments": json.dumps(b.get("input", {}), ensure_ascii=False)}})
                elif bt == "tool_result":
                    tool_results.append((b.get("tool_use_id") or (new_id()), _stringify(b.get("content", ""))))
            if role == "assistant":
                am: dict = {"role": "assistant", "content": "\n".join(text_parts)}
                if tool_calls:
                    am["tool_calls"] = tool_calls
                out.append(am)
            else:
                # tool results arrive on a user turn in Anthropic format
                for tid, tcontent in tool_results:
                    out.append({"role": "tool", "tool_call_id": tid, "content": tcontent})
                if text_parts and any(p.strip() for p in text_parts):
                    out.append({"role": "user", "content": "\n".join(text_parts)})
            continue

        # String content (Ollama / plain)
        if role == "assistant":
            am = {"role": "assistant", "content": content or ""}
            raw_tcs = msg.get("tool_calls") or []
            norm_tcs = []
            pending_ids = []
            for tc in raw_tcs:
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                tid = tc.get("id") or new_id()
                pending_ids.append(tid)
                norm_tcs.append({"id": tid, "type": "function", "function": {"name": name, "arguments": args}})
            if norm_tcs:
                am["tool_calls"] = norm_tcs
            out.append(am)
        elif role == "tool":
            tid = msg.get("tool_call_id") or (pending_ids.pop(0) if pending_ids else new_id())
            out.append({"role": "tool", "tool_call_id": tid, "content": _stringify(content)})
        elif role in ("user", "system"):
            out.append({"role": role, "content": _stringify(content)})
        # unknown roles dropped
    return out


def record_to_example(rec: dict, include_tools: bool = True) -> Optional[dict]:
    """Convert one captured agent-invocation record → one SFT example, or None if it has
    no assistant turn (nothing to train on)."""
    msgs = normalize_conversation(rec.get("system_prompt", ""), rec.get("messages", []))
    if not any(m.get("role") == "assistant" and (m.get("content") or m.get("tool_calls")) for m in msgs):
        return None
    ex: dict = {"messages": msgs}
    if include_tools:
        ex["tools"] = mcp_tools_to_openai(rec.get("tools"))
    return ex


# ── Loading / filtering / writing ────────────────────────────────────────────

def iter_captured_records(sft_dir: str | Path, sessions: Optional[set] = None) -> Iterator[dict]:
    """Yield capture records from data/sft/*.jsonl, optionally restricted to `sessions`."""
    d = Path(sft_dir)
    for jf in sorted(d.glob("*.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if sessions is not None and rec.get("session_id") not in sessions:
                continue
            yield rec


def solved_sessions(report_path: str | Path) -> set:
    """Session ids marked solved in a run report JSON (results[].session_id where solved)."""
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    return {r.get("session_id") for r in data.get("results", []) if r.get("solved")}


def strip_hint(text: str, start_marker: str, end_marker: str) -> str:
    """Remove a sentinel-delimited hint block (e.g. an injected HTB walkthrough) from a
    prompt so the trained model never sees the crutch it was given at capture time."""
    if not text or start_marker not in text:
        return text
    out, s, e = text, start_marker, end_marker
    while s in out and e in out:
        i, j = out.index(s), out.index(e)
        if j < i:
            break
        out = out[:i] + out[j + len(e):]
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
