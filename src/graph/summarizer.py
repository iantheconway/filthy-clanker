"""
SummarizerAgent node — condenses large tool outputs and compacts message history.

Two entry points:
  - `maybe_summarize(text, config)` — conditionally summarize if text exceeds threshold.
  - `compaction_node(state)`        — LangGraph node that compresses full message history.

Provider routing
----------------
The summarizer honours the `provider` field in agents.yaml under `agents.summarizer`.
Supported values: 'anthropic', 'gemini', 'ollama'.

  - anthropic: uses the Anthropic Messages API (sync client).  The `host` key in the
    summarizer config is IGNORED — Anthropic always uses the official API endpoint.
  - gemini:    uses the Google Generative AI SDK (sync client).  `host` is ignored.
  - ollama:    uses the local Ollama /api/generate HTTP endpoint.  `host` is required
    (defaults to OLLAMA_HOST env var or http://host.docker.internal:11434).

If `provider` is not 'ollama' and something would cause a fall-through to the Ollama
path, a ValueError is raised immediately rather than silently calling the wrong backend.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Callable, Dict, Optional

import requests

from .state import TeamState

logger = logging.getLogger("filthy_clanker")


# ---------------------------------------------------------------------------
# Per-provider low-level generate helpers (all synchronous)
# ---------------------------------------------------------------------------

def _ollama_generate(prompt: str, model: str, host: str, system: str = "") -> str:
    """Call the Ollama /api/generate endpoint synchronously."""
    url = f"{host.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    logger.debug("[Ollama] POST %s  model=%s  prompt_chars=%d", url, model, len(prompt))
    try:
        resp = requests.post(url, json=payload, timeout=90)
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        if not text:
            raise ValueError("Ollama returned an empty response")
        return text
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Ollama not reachable at {host}")
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP {exc.response.status_code} for model '{model}' at {url}") from exc
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


def _anthropic_generate(prompt: str, model: str, api_key: str, system: str = "") -> str:
    """Call the Anthropic Messages API synchronously. Raises RuntimeError on failure."""
    import anthropic as _anthropic
    try:
        client = _anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system or "Summarize the provided content concisely.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if not text:
            raise RuntimeError("Anthropic summarizer returned an empty response")
        return text
    except Exception as exc:
        raise RuntimeError(f"Anthropic summarizer error: {exc}") from exc


def _gemini_generate(prompt: str, model: str, api_key: str, system: str = "") -> str:
    """Call the Gemini Generative AI API synchronously. Raises RuntimeError on failure."""
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        resp = client.models.generate_content(model=model, contents=full_prompt)
        text = (resp.text or "").strip()
        if not text:
            raise RuntimeError("Gemini summarizer returned an empty response")
        return text
    except Exception as exc:
        raise RuntimeError(f"Gemini summarizer error: {exc}") from exc


# ---------------------------------------------------------------------------
# Flag / ASCII-art detection helpers
# ---------------------------------------------------------------------------

# Matches common CTF flag formats: flag{...}, key{...}, HTB{...}, picoCTF{...}, etc.
_FLAG_RE = re.compile(
    r'(?:flag|key|ctf|htb|thm|picoctf|csaw|crypto|web|pwn|misc|rev|forensics'
    r'|rtcp|ductf|darkctf|bucket|nite|jctf|cyber)\{[^}]{1,200}\}',
    re.IGNORECASE,
)
# Raw 32-char hex (MD5-sized) strings — common HTB user/root flag format.
_HEX_FLAG_RE = re.compile(r'\b([0-9a-f]{32})\b', re.IGNORECASE)


def _detect_flag_content(text: str) -> bool:
    """
    Return True if the text contains a flag-like pattern or dense ASCII art.

    ASCII art detection: >= 3 lines where > 60 % of printable characters are
    non-alphanumeric (typical of banner/art-rendered flags).
    """
    if _FLAG_RE.search(text):
        return True
    lines = text.splitlines()
    art_count = 0
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 6:
            continue
        non_alnum = sum(1 for c in stripped if not c.isalnum() and c != ' ')
        if non_alnum / len(stripped) > 0.60:
            art_count += 1
            if art_count >= 3:
                return True
    return False


def _build_raw_clip(text: str, cap: int = 2_000) -> str:
    """
    Extract a compact clip of the most flag-relevant lines for the RAW_CLIP block.

    Includes:
      - Lines containing a flag pattern (± 2 context lines)
      - Runs of ≥ 3 consecutive dense ASCII-art lines
    """
    lines = text.splitlines()
    keep: set[int] = set()

    # Flag-matching lines + context
    for i, line in enumerate(lines):
        if _FLAG_RE.search(line):
            for j in range(max(0, i - 2), min(len(lines), i + 3)):
                keep.add(j)

    # ASCII-art runs
    art_run: list[int] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) >= 6:
            non_alnum = sum(1 for c in stripped if not c.isalnum() and c != ' ')
            if non_alnum / len(stripped) > 0.60:
                art_run.append(i)
                continue
        if len(art_run) >= 3:
            keep.update(art_run)
        art_run = []
    if len(art_run) >= 3:
        keep.update(art_run)

    if not keep:
        # No specific lines matched — return the first `cap` chars as-is
        return text[:cap]

    clip_lines = [lines[i] for i in sorted(keep)]
    return "\n".join(clip_lines)[:cap]


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------

def _build_generate_fn(agent_cfg: dict) -> Callable[[str, str], str]:
    """
    Return a (prompt, system) -> str callable for the configured summarizer provider.

    Raises ValueError if:
      - provider is 'anthropic' or 'gemini' but the required API key is absent
      - provider is unknown
      - provider is not 'ollama' but something would cause an Ollama call (belt-and-suspenders)
    """
    provider: str = agent_cfg.get("provider", "ollama").lower().strip()

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "summarizer.provider is 'anthropic' but ANTHROPIC_API_KEY is not set. "
                "Set the key in .env or switch summarizer.provider to 'ollama'."
            )
        model: str = agent_cfg.get("model", "claude-haiku-4-5-20251001")
        # Explicitly reject any host key — Anthropic uses the official endpoint only.
        if "host" in agent_cfg:
            logger.warning(
                "[Summarizer] 'host' key found in summarizer config but provider is "
                "'anthropic' — ignoring host. Anthropic always uses the official API endpoint."
            )
        logger.info("[Summarizer] provider=anthropic model=%s", model)
        return lambda prompt, system="": _anthropic_generate(prompt, model, api_key, system)

    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "summarizer.provider is 'gemini' but GEMINI_API_KEY is not set."
            )
        model = agent_cfg.get("model", "gemini-2.5-flash")
        if "host" in agent_cfg:
            logger.warning(
                "[Summarizer] 'host' key found in summarizer config but provider is "
                "'gemini' — ignoring host."
            )
        logger.info("[Summarizer] provider=gemini model=%s", model)
        return lambda prompt, system="": _gemini_generate(prompt, model, api_key, system)

    elif provider == "ollama":
        model = agent_cfg.get("model", "llama3.2")
        host: str = agent_cfg.get("host", os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434"))
        logger.info("[Summarizer] provider=ollama model=%s host=%s", model, host)
        # _ollama_generate now raises RuntimeError on failure; callers must handle it.
        return lambda prompt, system="": _ollama_generate(prompt, model, host, system)

    else:
        raise ValueError(
            f"Unknown summarizer provider: {provider!r}. "
            "Set summarizer.provider to 'anthropic', 'gemini', or 'ollama' in agents.yaml."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def maybe_summarize(text: str, config: Dict[str, Any]) -> str:
    """
    If `text` exceeds the configured character threshold, pass it through the
    summarizer and return a condensed version.  Otherwise return as-is.

    Flag protection:
      If the raw text contains a recognisable flag pattern or dense ASCII art,
      the LLM summary is augmented with a verbatim RAW_CLIP section so the
      agent never loses the actual flag value through lossy summarisation.

    Fallback:
      On any summariser failure the agent receives `summarizer_fallback_chars`
      (default 6 000) characters of raw text rather than an opaque error string,
      preventing "blind retry" loops.

    Args:
        text:   Raw tool output string.
        config: Full agents.yaml config dict (settings + agents sections).
    """
    settings = config.get("settings", {})
    threshold: int = settings.get("tool_output_threshold", 4000)
    fallback_chars: int = settings.get("summarizer_fallback_chars", 6000)

    if len(text) <= threshold:
        return text

    # ------------------------------------------------------------------
    # Flag / ASCII-art guard — detect BEFORE handing to the LLM.
    # ------------------------------------------------------------------
    _has_flags = _detect_flag_content(text)
    _raw_clip: Optional[str] = None
    if _has_flags:
        _raw_clip = _build_raw_clip(text)
        logger.info(
            "[Summarizer] Flag content / ASCII art detected in %d-char output — "
            "RAW_CLIP will be appended to the condensed report.", len(text)
        )
        # If the whole text is short enough to include almost verbatim, skip LLM.
        if len(text) <= fallback_chars * 2:
            logger.info("[Summarizer] Text short enough — bypassing LLM, returning raw fallback")
            return _raw_fallback(text, fallback_chars)

    agent_cfg: dict = config.get("agents", {}).get("summarizer", {})
    system: str = agent_cfg.get("system_prompt", "Summarize this security tool output concisely.")

    try:
        generate = _build_generate_fn(agent_cfg)
    except ValueError as exc:
        logger.error("[Summarizer] Configuration error — returning truncated raw text: %s", exc)
        return _raw_fallback(text, fallback_chars)

    prompt = (
        "The following is raw output from a security tool. "
        "Produce a concise Condensed Report highlighting only security-relevant findings.\n\n"
        f"--- RAW OUTPUT ---\n{text}\n--- END ---"
    )

    try:
        summary = generate(prompt, system)
    except Exception as exc:
        logger.error("[Summarizer] Generate failed (%s) — returning truncated raw text", exc)
        return _raw_fallback(text, fallback_chars)

    # Append RAW_CLIP when flag content was present so the agent always sees it.
    if _raw_clip:
        summary = (
            f"{summary}\n\n"
            f"--- RAW_CLIP (flag content / ASCII art preserved verbatim) ---\n"
            f"{_raw_clip}\n"
            f"--- END RAW_CLIP ---"
        )

    char_reduction = len(text) - len(summary)
    return (
        f"[Condensed Report — original {len(text):,} chars → {len(summary):,} chars "
        f"(saved {char_reduction:,})]\n\n{summary}"
    )


def _raw_fallback(text: str, cap: int) -> str:
    """Return the first `cap` chars of raw text with a truncation notice."""
    if len(text) <= cap:
        return text
    return (
        f"[Summarizer unavailable — showing first {cap:,} of {len(text):,} chars]\n\n"
        f"{text[:cap]}\n…[truncated]"
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
    agent_cfg: dict = config.get("agents", {}).get("summarizer", {})
    system: str = agent_cfg.get("system_prompt", "Summarize this CTF hacking session.")

    messages = state.get("messages", [])
    kb = state.get("knowledge_base", {})

    # Keep only the last 2 messages for continuity — large tool outputs in the
    # tail are the main reason compaction barely reduces token count.
    keep_tail = 2
    messages_to_compress = messages[:-keep_tail] if len(messages) > keep_tail else messages
    messages_to_keep = messages[-keep_tail:] if len(messages) > keep_tail else []

    if not messages_to_compress:
        # Nothing to compact — still reset the estimate so the supervisor doesn't
        # immediately re-trigger compaction on the next pass.
        current_estimate = sum(len(str(m)) for m in messages) // 4
        return {"context_token_estimate": current_estimate}

    # Build a text digest of the messages to compress.
    # Cap each message at 800 chars and the total digest at 40k chars.
    digest_parts = []
    digest_len = 0
    _DIGEST_CAP = 40_000
    _MSG_CAP = 800
    for msg in messages_to_compress:
        if digest_len >= _DIGEST_CAP:
            break
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        if content:
            snippet = str(content)[:_MSG_CAP]
            digest_parts.append(f"[{role.upper()}]: {snippet}")
            digest_len += len(snippet)

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

    try:
        generate = _build_generate_fn(agent_cfg)
        summary_text = generate(prompt, system)
    except (ValueError, RuntimeError, Exception) as exc:
        logger.error("[Compaction] Summarizer failed — skipping compaction: %s", exc)
        current_estimate = sum(len(str(m)) for m in messages) // 4
        return {"context_token_estimate": current_estimate}

    summary_message = {
        "role": "user",
        "content": (
            f"[SESSION COMPACTION — {len(messages_to_compress)} messages compressed]\n\n"
            f"{summary_text}"
        ),
    }

    # Truncate tail messages so they don't dominate the new estimate.
    _TAIL_CAP = 3_000
    truncated_tail = []
    for msg in messages_to_keep:
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > _TAIL_CAP:
            msg = dict(msg)
            msg["content"] = content[:_TAIL_CAP] + "\n…[truncated for compaction]"
        truncated_tail.append(msg)

    new_messages = [summary_message] + truncated_tail
    new_estimate = sum(len(str(m)) for m in new_messages) // 4

    print(f"\n[Compaction] Compressed {len(messages_to_compress)} messages → summary. "
          f"Tokens: ~{state.get('context_token_estimate', 0):,} → ~{new_estimate:,}")

    return {
        "messages": {"__replace__": new_messages},
        "context_token_estimate": new_estimate,
    }
