"""
SummarizerAgent node — uses a local Ollama model to condense large tool outputs
before they are passed to the expensive primary LLM provider.

Two entry points:
  - `maybe_summarize(text, config)` — conditionally summarize if text exceeds threshold.
  - `compaction_node(state)`        — LangGraph node that compresses full message history.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict

import requests

from .state import TeamState


# ---------------------------------------------------------------------------
# Low-level Ollama HTTP helper
# ---------------------------------------------------------------------------

def _ollama_generate(prompt: str, model: str, host: str, system: str = "") -> str:
    """
    Call the Ollama /api/generate endpoint synchronously.
    Returns the generated text or an error string.
    """
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return f"[Summarizer unavailable — Ollama not reachable at {host}]"
    except Exception as e:
        return f"[Summarizer error: {e}]"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def maybe_summarize(text: str, config: Dict[str, Any]) -> str:
    """
    If `text` exceeds the configured character threshold, pass it through the
    Ollama summarizer and return a "Condensed Report". Otherwise return as-is.

    Args:
        text:   Raw tool output string.
        config: The `settings` + `agents.summarizer` config dict from agents.yaml.

    Returns:
        Original text or a condensed summary prefixed with [Condensed Report].
    """
    threshold: int = config.get("settings", {}).get("tool_output_threshold", 4000)
    if len(text) <= threshold:
        return text

    agent_cfg: dict = config.get("agents", {}).get("summarizer", {})
    model: str = agent_cfg.get("model", "llama3.2")
    host: str = agent_cfg.get("host", "http://10.0.2.2:11434")
    system: str = agent_cfg.get("system_prompt", "Summarize this security tool output concisely.")

    prompt = (
        f"The following is raw output from a security tool. "
        f"Produce a concise Condensed Report highlighting only security-relevant findings.\n\n"
        f"--- RAW OUTPUT ---\n{text}\n--- END ---"
    )

    summary = _ollama_generate(prompt, model, host, system)
    char_reduction = len(text) - len(summary)
    return (
        f"[Condensed Report — original {len(text):,} chars → {len(summary):,} chars "
        f"(saved {char_reduction:,})]\n\n{summary}"
    )


def compaction_node(state: TeamState) -> dict:
    """
    LangGraph node: compress the conversation history into a summary when the
    estimated context approaches the token limit.

    Triggered by the supervisor when `context_token_estimate` exceeds threshold.
    Replaces the bulk of the message history with a single summary message while
    keeping the last few turns for continuity.

    Returns a partial state update.
    """
    config = state.get("config", {})
    agent_cfg = config.get("agents", {}).get("summarizer", {})
    model: str = agent_cfg.get("model", "llama3.2")
    host: str = agent_cfg.get("host", "http://10.0.2.2:11434")
    system: str = agent_cfg.get("system_prompt", "Summarize this CTF hacking session.")

    messages = state.get("messages", [])
    kb = state.get("knowledge_base", {})

    # Keep last 6 messages intact for continuity; summarize everything before
    keep_tail = 6
    messages_to_compress = messages[:-keep_tail] if len(messages) > keep_tail else messages
    messages_to_keep = messages[-keep_tail:] if len(messages) > keep_tail else []

    if not messages_to_compress:
        return {}  # Nothing to compact

    # Build a text digest of the messages to compress
    digest_parts = []
    for msg in messages_to_compress:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Extract text from content blocks
            text_parts = [
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = " ".join(text_parts)
        if content:
            digest_parts.append(f"[{role.upper()}]: {str(content)[:1000]}")

    digest = "\n\n".join(digest_parts)
    kb_summary = json.dumps(kb, indent=2)

    prompt = (
        f"Summarize this CTF hacking session history. Include:\n"
        f"- What was discovered (ports, services, directories, credentials)\n"
        f"- What was attempted and whether it succeeded\n"
        f"- Current status and next logical step\n\n"
        f"KNOWLEDGE BASE:\n{kb_summary}\n\n"
        f"SESSION HISTORY:\n{digest}"
    )

    summary_text = _ollama_generate(prompt, model, host, system)

    # Build a replacement summary message
    summary_message = {
        "role": "user",
        "content": (
            f"[SESSION COMPACTION — {len(messages_to_compress)} messages compressed]\n\n"
            f"{summary_text}"
        ),
    }

    new_messages = [summary_message] + messages_to_keep
    new_estimate = sum(len(str(m)) for m in new_messages) // 4

    print(f"\n[Compaction] Compressed {len(messages_to_compress)} messages → summary. "
          f"Tokens: ~{state.get('context_token_estimate', 0):,} → ~{new_estimate:,}")

    return {
        "messages": new_messages,
        "context_token_estimate": new_estimate,
    }
