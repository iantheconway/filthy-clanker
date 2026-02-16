import json
import os
from datetime import datetime


SESSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sessions")


def _ensure_sessions_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def default_session_name() -> str:
    return datetime.now().strftime("session-%Y-%m-%d-%H%M%S")


def save_session_json(messages: list[dict], provider: str, filepath: str):
    """Write raw messages list + metadata to a JSON file."""
    _ensure_sessions_dir()
    data = {
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "messages": messages,
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _sanitize_anthropic_messages(messages: list[dict]) -> list[dict]:
    """Ensure every assistant tool_use block has a matching tool_result.

    If the session was saved mid-tool-loop, the final assistant message may
    contain tool_use blocks with no corresponding tool_result in the next
    message.  This adds stub results so the API won't reject the history.
    """
    sanitized = list(messages)
    i = 0
    while i < len(sanitized):
        msg = sanitized[i]
        if msg.get("role") != "assistant":
            i += 1
            continue

        # Collect tool_use ids in this assistant message
        tool_use_ids = [
            block["id"]
            for block in msg.get("content", [])
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        if not tool_use_ids:
            i += 1
            continue

        # Check the next message for matching tool_results
        next_msg = sanitized[i + 1] if i + 1 < len(sanitized) else None
        answered_ids = set()
        if next_msg and next_msg.get("role") == "user" and isinstance(next_msg.get("content"), list):
            for block in next_msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    answered_ids.add(block.get("tool_use_id"))

        missing_ids = [tid for tid in tool_use_ids if tid not in answered_ids]
        if missing_ids:
            stub_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": "[Tool call not completed — session was saved before execution]",
                }
                for tid in missing_ids
            ]
            if answered_ids and next_msg:
                # Some results exist; append stubs to the existing message
                next_msg["content"].extend(stub_results)
            else:
                # No result message at all; insert one
                sanitized.insert(i + 1, {"role": "user", "content": stub_results})

        i += 1

    return sanitized


def load_session_json(filepath: str) -> tuple[list[dict], str]:
    """Read a session JSON file. Returns (messages, provider)."""
    with open(filepath) as f:
        data = json.load(f)
    messages = data["messages"]
    provider = data["provider"]
    if provider == "anthropic":
        messages = _sanitize_anthropic_messages(messages)
    return messages, provider


async def save_summary(messages, llm_client, tools, system_prompt, filepath: str):
    """Ask the LLM to summarize the conversation and write to a .md file."""
    _ensure_sessions_dir()
    summary_prompt = (
        "Summarize this conversation concisely. Include:\n"
        "- Key findings and discoveries\n"
        "- Tools run and their notable results\n"
        "- Current progress and state\n"
        "- Suggested next steps\n\n"
        "Format as markdown with clear headings."
    )
    summary_messages = messages + [{"role": "user", "content": summary_prompt}]
    try:
        response = await llm_client.generate_response(
            summary_messages, tools, system_prompt
        )
        summary_text = response.get("text", "")
    except Exception as e:
        summary_text = f"*Summary generation failed: {e}*"

    with open(filepath, "w") as f:
        f.write(summary_text)


def load_summary(filepath: str) -> str:
    """Read a .md summary file and return its contents."""
    with open(filepath) as f:
        return f.read()


def list_sessions() -> list[dict]:
    """List saved sessions by scanning for .json/.md files in sessions dir.

    Returns a list of dicts with keys: name, has_json, has_md.
    """
    if not os.path.isdir(SESSIONS_DIR):
        return []

    names = set()
    for fname in os.listdir(SESSIONS_DIR):
        if fname.endswith(".json") or fname.endswith(".md"):
            names.add(os.path.splitext(fname)[0])

    sessions = []
    for name in sorted(names):
        sessions.append({
            "name": name,
            "has_json": os.path.exists(os.path.join(SESSIONS_DIR, f"{name}.json")),
            "has_md": os.path.exists(os.path.join(SESSIONS_DIR, f"{name}.md")),
        })
    return sessions
