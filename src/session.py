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


def load_session_json(filepath: str) -> tuple[list[dict], str]:
    """Read a session JSON file. Returns (messages, provider)."""
    with open(filepath) as f:
        data = json.load(f)
    return data["messages"], data["provider"]


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
