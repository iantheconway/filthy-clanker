"""Tests for session.py"""
import json
import os
import tempfile

import pytest

from session import (
    SessionLogger,
    _sanitize_anthropic_messages,
    default_session_name,
    get_latest_session,
    list_sessions,
    load_session_json,
    load_summary,
    save_session_json,
)


# ---------------------------------------------------------------------------
# default_session_name
# ---------------------------------------------------------------------------

def test_default_session_name_format():
    name = default_session_name()
    assert name.startswith("session-")
    # Should match session-YYYY-MM-DD-HHMMSS
    parts = name.split("-")
    assert len(parts) == 5
    assert len(parts[1]) == 4  # year
    assert len(parts[2]) == 2  # month
    assert len(parts[3]) == 2  # day


# ---------------------------------------------------------------------------
# save_session_json / load_session_json
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip(tmp_path):
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    fpath = str(tmp_path / "test.json")
    save_session_json(messages, "anthropic", fpath)

    loaded_messages, provider = load_session_json(fpath)
    assert provider == "anthropic"
    assert loaded_messages == messages


def test_save_creates_parent_dir(tmp_path):
    fpath = str(tmp_path / "subdir" / "session.json")
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    save_session_json([], "gemini", fpath)
    assert os.path.exists(fpath)


def test_load_session_json_includes_timestamp(tmp_path):
    fpath = str(tmp_path / "session.json")
    save_session_json([{"role": "user", "content": "x"}], "gemini", fpath)
    with open(fpath) as f:
        data = json.load(f)
    assert "timestamp" in data
    assert data["provider"] == "gemini"


# ---------------------------------------------------------------------------
# _sanitize_anthropic_messages
# ---------------------------------------------------------------------------

def _make_tool_use_msg(tool_id: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tool_id, "name": "foo", "input": {}}],
    }


def _make_tool_result_msg(tool_id: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}],
    }


def test_sanitize_no_tool_use():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]
    result = _sanitize_anthropic_messages(messages)
    assert result == messages


def test_sanitize_complete_pair_unchanged():
    messages = [
        _make_tool_use_msg("id1"),
        _make_tool_result_msg("id1"),
    ]
    result = _sanitize_anthropic_messages(messages)
    assert len(result) == 2
    # The tool_result message should still have exactly one block
    assert len(result[1]["content"]) == 1


def test_sanitize_inserts_stub_when_result_missing():
    messages = [_make_tool_use_msg("id1")]
    result = _sanitize_anthropic_messages(messages)
    assert len(result) == 2
    stub_msg = result[1]
    assert stub_msg["role"] == "user"
    assert stub_msg["content"][0]["type"] == "tool_result"
    assert stub_msg["content"][0]["tool_use_id"] == "id1"


def test_sanitize_appends_stub_to_partial_results():
    """When some tool_results exist but one is missing, stub is appended."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "id1", "name": "a", "input": {}},
                {"type": "tool_use", "id": "id2", "name": "b", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "id1", "content": "done"}],
        },
    ]
    result = _sanitize_anthropic_messages(messages)
    assert len(result) == 2
    result_blocks = result[1]["content"]
    assert len(result_blocks) == 2
    tool_use_ids = {b["tool_use_id"] for b in result_blocks}
    assert tool_use_ids == {"id1", "id2"}


def test_sanitize_does_not_mutate_original():
    """Sanitization must not modify the original messages list or its dicts."""
    original_result_content = [{"type": "tool_result", "tool_use_id": "id1", "content": "ok"}]
    messages = [
        _make_tool_use_msg("id1"),
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "id2", "name": "b", "input": {}}],
        },
        # id1 answered, id2 not answered
        {"role": "user", "content": list(original_result_content)},
    ]
    original_len = len(messages[2]["content"])
    _sanitize_anthropic_messages(messages)
    # Original should be unchanged
    assert len(messages[2]["content"]) == original_len


# ---------------------------------------------------------------------------
# load_summary
# ---------------------------------------------------------------------------

def test_load_summary(tmp_path):
    md_path = str(tmp_path / "summary.md")
    content = "# Summary\n\nFound RCE on port 8080."
    with open(md_path, "w") as f:
        f.write(content)
    assert load_summary(md_path) == content


# ---------------------------------------------------------------------------
# list_sessions
# ---------------------------------------------------------------------------

def test_list_sessions_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("session.SESSIONS_DIR", str(tmp_path))
    result = list_sessions()
    assert result == []


def test_list_sessions_nonexistent_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("session.SESSIONS_DIR", str(tmp_path / "nonexistent"))
    result = list_sessions()
    assert result == []


def test_list_sessions_finds_files(tmp_path, monkeypatch):
    monkeypatch.setattr("session.SESSIONS_DIR", str(tmp_path))
    (tmp_path / "session-2024-01-01-120000.json").write_text("{}")
    (tmp_path / "session-2024-01-01-120000.md").write_text("# summary")
    (tmp_path / "session-2024-01-02-090000.json").write_text("{}")

    sessions = list_sessions()
    names = [s["name"] for s in sessions]
    assert "session-2024-01-01-120000" in names
    assert "session-2024-01-02-090000" in names

    s = next(s for s in sessions if s["name"] == "session-2024-01-01-120000")
    assert s["has_json"] is True
    assert s["has_md"] is True

    s2 = next(s for s in sessions if s["name"] == "session-2024-01-02-090000")
    assert s2["has_json"] is True
    assert s2["has_md"] is False


# ---------------------------------------------------------------------------
# get_latest_session
# ---------------------------------------------------------------------------

def test_get_latest_session_none_when_no_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("session.SESSIONS_DIR", str(tmp_path / "nodir"))
    assert get_latest_session("anthropic") is None


def test_get_latest_session_returns_most_recent(tmp_path, monkeypatch):
    monkeypatch.setattr("session.SESSIONS_DIR", str(tmp_path))
    older = {"timestamp": "2024-01-01T10:00:00", "provider": "anthropic", "messages": [{"role": "user", "content": "hi"}]}
    newer = {"timestamp": "2024-06-01T10:00:00", "provider": "anthropic", "messages": [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "ok"}]}

    (tmp_path / "old.json").write_text(json.dumps(older))
    (tmp_path / "new.json").write_text(json.dumps(newer))

    result = get_latest_session("anthropic")
    assert result is not None
    name, fpath, count = result
    assert name == "new"
    assert count == 2


def test_get_latest_session_filters_by_provider(tmp_path, monkeypatch):
    monkeypatch.setattr("session.SESSIONS_DIR", str(tmp_path))
    gemini_session = {"timestamp": "2024-06-01T10:00:00", "provider": "gemini", "messages": [{"role": "user", "content": "x"}]}
    (tmp_path / "gemini.json").write_text(json.dumps(gemini_session))

    assert get_latest_session("anthropic") is None
    assert get_latest_session("gemini") is not None


def test_get_latest_session_skips_corrupt_files(tmp_path, monkeypatch):
    monkeypatch.setattr("session.SESSIONS_DIR", str(tmp_path))
    (tmp_path / "corrupt.json").write_text("not valid json{{{")
    assert get_latest_session("anthropic") is None


# ---------------------------------------------------------------------------
# SessionLogger
# ---------------------------------------------------------------------------

def test_session_logger_append_and_persist(tmp_path):
    fpath = str(tmp_path / "session.json")
    logger = SessionLogger("anthropic", fpath)
    logger.append({"role": "user", "content": "hello"})
    logger.append({"role": "assistant", "content": "hi"})

    assert len(logger) == 2
    loaded, provider = load_session_json(fpath)
    assert provider == "anthropic"
    assert len(loaded) == 2


def test_session_logger_pop(tmp_path):
    fpath = str(tmp_path / "session.json")
    logger = SessionLogger("anthropic", fpath, [{"role": "user", "content": "msg"}])
    popped = logger.pop()
    assert popped == {"role": "user", "content": "msg"}
    assert len(logger) == 0


def test_session_logger_clear(tmp_path):
    fpath = str(tmp_path / "session.json")
    logger = SessionLogger("gemini", fpath, [{"role": "user", "content": "x"}])
    logger.clear()
    assert len(logger) == 0
    loaded, _ = load_session_json(fpath)
    assert loaded == []


def test_session_logger_extend(tmp_path):
    fpath = str(tmp_path / "session.json")
    logger = SessionLogger("anthropic", fpath)
    logger.extend([
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ])
    assert len(logger) == 2


def test_session_logger_bool(tmp_path):
    fpath = str(tmp_path / "session.json")
    logger = SessionLogger("anthropic", fpath)
    assert not logger
    logger.append({"role": "user", "content": "x"})
    assert logger


def test_session_logger_add_operator(tmp_path):
    fpath = str(tmp_path / "session.json")
    logger = SessionLogger("anthropic", fpath, [{"role": "user", "content": "x"}])
    combined = logger + [{"role": "user", "content": "extra"}]
    assert len(combined) == 2
    # Original logger should be unchanged
    assert len(logger) == 1


def test_session_logger_setitem(tmp_path):
    fpath = str(tmp_path / "session.json")
    logger = SessionLogger("anthropic", fpath, [{"role": "user", "content": "original"}])
    logger[0] = {"role": "user", "content": "updated"}
    assert logger[0]["content"] == "updated"
    # Persisted
    loaded, _ = load_session_json(fpath)
    assert loaded[0]["content"] == "updated"


def test_session_logger_initial_messages(tmp_path):
    fpath = str(tmp_path / "session.json")
    initial = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    logger = SessionLogger("anthropic", fpath, initial)
    # Should not share references with initial list
    initial.append({"role": "user", "content": "should not appear"})
    assert len(logger) == 2
