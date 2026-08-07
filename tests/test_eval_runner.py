"""Tests for the NYU CTF eval runner's scoring/resume logic (evals/nyu_bench/run_eval.py).

These pin the SPEC-03 additions — the distinct `refusal` failure reason, the
refusal-rate metric in the report, per-challenge JSONL output, and the --rerun
resume filter — without needing Docker, Ollama, or the nyuctf dataset.
"""
import json
import sys
import types
from pathlib import Path

import pytest

# The runner imports nyuctf at module load; stub it so we can import the pure
# scoring functions without the dataset package installed.
for _name, _attr in (("nyuctf", None),
                     ("nyuctf.dataset", "CTFDataset"),
                     ("nyuctf.challenge", "CTFChallenge")):
    _mod = sys.modules.setdefault(_name, types.ModuleType(_name))
    if _attr:
        setattr(_mod, _attr, object)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals" / "nyu_bench"))
import run_eval  # noqa: E402


def _chal(name="chalA", category="web"):
    return types.SimpleNamespace(
        name=name, canonical_name=f"2024-q-{name}", category=category, year=2024,
        event="quals", flag_format="flag{...}", flag="flag{real}",
        container=True, has_files=False,
    )


# --- failure_reason -----------------------------------------------------------

def test_refusal_is_a_distinct_failure_reason():
    r = run_eval._make_result(_chal(), "s1", [], solved=False, target=None,
                              timed_out=False, refusal_fired=True)
    assert r["refusal_fired"] is True
    assert r["failure_reason"] == "refusal"


def test_refusal_takes_precedence_over_timeout():
    r = run_eval._make_result(_chal(), "s1", [], solved=False, target=None,
                              timed_out=True, refusal_fired=True)
    assert r["failure_reason"] == "refusal"  # a refusal that also ran out the clock is still a refusal


def test_solved_has_no_failure_reason():
    r = run_eval._make_result(_chal(), "s1", ["flag{real}"], solved=True, target=None,
                              timed_out=False, refusal_fired=False)
    assert r["failure_reason"] is None


def test_plain_failure_is_no_flag():
    r = run_eval._make_result(_chal(), "s1", [], solved=False, target=None,
                              timed_out=False, refusal_fired=False)
    assert r["failure_reason"] == "no_flag"


# --- report: refusal rate -----------------------------------------------------

def _args():
    return types.SimpleNamespace(split="development", category=None, max_chals=None,
                                 timeout=600, provider=None, version="v20250206")


def test_report_computes_refusal_rate():
    results = [
        run_eval._make_result(_chal("a"), "s", ["flag{real}"], solved=True, target=None, timed_out=False),
        run_eval._make_result(_chal("b"), "s", [], solved=False, target=None, timed_out=False, refusal_fired=True),
        run_eval._make_result(_chal("c"), "s", [], solved=False, target=None, timed_out=True),
        run_eval._make_result(_chal("d"), "s", [], solved=False, target=None, timed_out=False, refusal_fired=True),
    ]
    report, md = run_eval.generate_report(results, "run-x", _args())
    s = report["summary"]
    assert s["total"] == 4 and s["solved"] == 1
    assert s["refusals"] == 2
    assert s["refusal_rate"] == 0.5
    assert report["by_category"]["web"]["refusals"] == 2
    assert "Refusals | 2" in md


# --- JSONL + resume -----------------------------------------------------------

def test_append_jsonl_and_scored_names_roundtrip(tmp_path):
    p = tmp_path / "run-1.jsonl"
    run_eval._append_jsonl(run_eval._make_result(_chal("a"), "s", [], solved=False,
                                                 target=None, timed_out=False), p)
    run_eval._append_jsonl(run_eval._make_result(_chal("b"), "s", ["flag{real}"], solved=True,
                                                 target=None, timed_out=False), p)
    # Two JSON lines, each parseable.
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and all(json.loads(l) for l in lines)
    # Both canonical names are collected for the resume skip-set.
    scored = run_eval._scored_canonical_names(tmp_path)
    assert scored == {"2024-q-a", "2024-q-b"}


def test_scored_names_tolerates_corrupt_lines(tmp_path):
    (tmp_path / "run-2.jsonl").write_text(
        '{"canonical_name": "2024-q-a"}\n'
        'not-json-garbage\n'          # e.g. a half-written line from a crash
        '{"canonical_name": "2024-q-b"}\n',
        encoding="utf-8",
    )
    assert run_eval._scored_canonical_names(tmp_path) == {"2024-q-a", "2024-q-b"}


def test_scored_names_empty_when_no_jsonl(tmp_path):
    assert run_eval._scored_canonical_names(tmp_path) == set()
