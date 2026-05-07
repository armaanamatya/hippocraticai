"""Unit tests for the trace.py SQLite layer.

We never hit the real OpenAI API here — record_call is fed a stubbed
response object that mimics openai's pre-1.0 OpenAIObject (dict access for
`usage`, attribute access for `choices`)."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import trace as trace_mod  # noqa: E402


# ---------- response stub ----------

class _StubMessage(dict):
    pass


class _StubChoice:
    def __init__(self, content: str):
        self.message = _StubMessage(content=content)


class StubResponse(dict):
    """Mimics pre-1.0 openai response (dict-like with .choices attr)."""
    def __init__(self, content: str, prompt_tokens: int, completion_tokens: int):
        super().__init__(usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })
        self.choices = [_StubChoice(content)]


# ---------- enable / disable ----------

def test_enabled_when_path_set(monkeypatch, tmp_path):
    monkeypatch.setenv("STORY_DB_PATH", str(tmp_path / "x.db"))
    assert trace_mod.enabled() is True


def test_disabled_when_empty(monkeypatch):
    monkeypatch.setenv("STORY_DB_PATH", "")
    assert trace_mod.enabled() is False


def test_start_run_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("STORY_DB_PATH", "")
    assert trace_mod.start_run("hello") is None


# ---------- schema + write paths ----------

def test_start_run_inserts_row(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    monkeypatch.setenv("STORY_DB_PATH", str(db))
    rid = trace_mod.start_run("uplifting")
    assert rid is not None
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT request, ended_at FROM runs WHERE run_id=?", (rid,)).fetchone()
    conn.close()
    assert row[0] == "uplifting"
    assert row[1] is None  # not finished yet


def test_record_call_writes_row_with_cost(tmp_path, monkeypatch):
    monkeypatch.setenv("STORY_DB_PATH", str(tmp_path / "t.db"))
    rid = trace_mod.start_run("x")
    resp = StubResponse("hello", prompt_tokens=200, completion_tokens=50)

    content = trace_mod.record_call(
        run_id=rid, role="categorizer", iteration=-1,
        model="gpt-3.5-turbo", temperature=0.0, fn=lambda: resp,
    )
    assert content == "hello"

    conn = sqlite3.connect(trace_mod.db_path())
    row = conn.execute(
        "SELECT role, prompt_tokens, completion_tokens, cost_usd, ok "
        "FROM calls WHERE run_id=?", (rid,)
    ).fetchone()
    conn.close()
    role, pt, ct, cost, ok = row
    assert role == "categorizer"
    assert pt == 200 and ct == 50
    # 200/1000 * 0.0005 + 50/1000 * 0.0015 = 0.0001 + 0.000075 = 0.000175
    assert cost == pytest.approx(0.000175, rel=1e-6)
    assert ok == 1


def test_record_call_writes_row_on_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("STORY_DB_PATH", str(tmp_path / "t.db"))
    rid = trace_mod.start_run("x")

    def boom():
        raise RuntimeError("api went poof")

    with pytest.raises(RuntimeError, match="poof"):
        trace_mod.record_call(
            run_id=rid, role="judge", iteration=0,
            model="gpt-3.5-turbo", temperature=0.2, fn=boom,
        )

    conn = sqlite3.connect(trace_mod.db_path())
    row = conn.execute(
        "SELECT ok, error FROM calls WHERE run_id=?", (rid,)
    ).fetchone()
    conn.close()
    assert row[0] == 0
    assert "poof" in row[1]


def test_record_judge_inserts_scores(tmp_path, monkeypatch):
    monkeypatch.setenv("STORY_DB_PATH", str(tmp_path / "t.db"))
    rid = trace_mod.start_run("x")
    judge_result = {
        "scores": {
            "age_fit": 4, "safety": 5, "narrative_arc": 4,
            "engagement": 3, "calmness": 5,
        },
        "weakest": "engagement",
        "overall": 4.2,
        "critique": "more sensory detail",
    }
    trace_mod.record_judge(rid, iteration=0, judge_result=judge_result)
    conn = sqlite3.connect(trace_mod.db_path())
    row = conn.execute(
        "SELECT iteration, overall, weakest, engagement FROM judge_scores WHERE run_id=?",
        (rid,)
    ).fetchone()
    conn.close()
    assert row == (0, 4.2, "engagement", 3.0)


def test_finish_run_aggregates_cost(tmp_path, monkeypatch):
    monkeypatch.setenv("STORY_DB_PATH", str(tmp_path / "t.db"))
    rid = trace_mod.start_run("x")
    for _ in range(3):
        resp = StubResponse("ok", prompt_tokens=100, completion_tokens=20)
        trace_mod.record_call(rid, "judge", 0, "gpt-3.5-turbo", 0.2, lambda r=resp: r)

    trace_mod.finish_run(rid, category="ADVENTURE", iterations=1, final_overall=4.4, passed=True)

    conn = sqlite3.connect(trace_mod.db_path())
    row = conn.execute(
        "SELECT category, iterations, final_overall, passed, total_prompt_tokens, "
        "total_completion_tokens, total_cost_usd FROM runs WHERE run_id=?",
        (rid,)
    ).fetchone()
    conn.close()
    cat, iters, overall, passed, pt, ct, cost = row
    assert cat == "ADVENTURE"
    assert iters == 1
    assert overall == 4.4
    assert passed == 1
    assert pt == 300 and ct == 60
    # 3 calls × (100/1000 × 0.0005 + 20/1000 × 0.0015) = 3 × 0.00008 = 0.00024
    assert cost == pytest.approx(0.00024, rel=1e-3)


def test_record_call_no_op_when_disabled(monkeypatch):
    monkeypatch.setenv("STORY_DB_PATH", "")
    resp = StubResponse("hi", 10, 5)
    out = trace_mod.record_call(
        run_id=None, role="x", iteration=0,
        model="gpt-3.5-turbo", temperature=0.0, fn=lambda: resp,
    )
    assert out == "hi"  # still strips, still returns content


# ---------- cost prefix-match fallback ----------

def test_cost_unknown_model_returns_zero():
    assert trace_mod._cost_usd("totally-fake-model", 100, 100) == 0.0


def test_cost_prefix_match():
    # "gpt-3.5-turbo-0125-some-suffix" should match "gpt-3.5-turbo" prefix.
    cost = trace_mod._cost_usd("gpt-3.5-turbo-extra", 1000, 1000)
    # Either matched gpt-3.5-turbo (0.0005, 0.0015) or 0125 (same rates).
    assert cost == pytest.approx(0.002, rel=1e-3)
