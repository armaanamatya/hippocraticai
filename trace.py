"""SQLite trace layer for the bedtime-story pipeline.

Logs every pipeline run (categorize → … → final) plus every individual
LLM call (role, latency, tokens, cost) to a single SQLite file. The
aggregate report lives at `python -m evals report`.

DB path: STORY_DB_PATH env var, default `traces.db` in cwd.
Set STORY_DB_PATH="" to disable tracing entirely (no-op).

Why SQLite + stdlib only: the takehome is meant to be a small, runnable
script. Pulling in Phoenix / Langfuse / LangSmith would be overkill and
would obscure the design choices. SQLite + sqlite3 module gives us
durable, queryable traces with zero extra deps.
"""
from __future__ import annotations

import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Callable, Optional

DEFAULT_DB_PATH: str = "traces.db"

# gpt-3.5-turbo-0125 pricing (verified 2026-01 via openai.com/api/pricing).
# Tuple = ($/1k prompt tokens, $/1k completion tokens).
PRICING: dict[str, tuple[float, float]] = {
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "gpt-3.5-turbo-0125": (0.0005, 0.0015),
    "gpt-3.5-turbo-1106": (0.001, 0.002),
    "gpt-3.5-turbo-instruct": (0.0015, 0.002),
}

SCHEMA: str = """
CREATE TABLE IF NOT EXISTS runs (
  run_id              TEXT PRIMARY KEY,
  started_at          REAL NOT NULL,
  ended_at            REAL,
  request             TEXT NOT NULL,
  category            TEXT,
  iterations          INTEGER,
  final_overall       REAL,
  passed              INTEGER,
  total_prompt_tokens     INTEGER DEFAULT 0,
  total_completion_tokens INTEGER DEFAULT 0,
  total_cost_usd      REAL DEFAULT 0.0,
  error               TEXT
);

CREATE TABLE IF NOT EXISTS calls (
  call_id           INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id            TEXT NOT NULL REFERENCES runs(run_id),
  role              TEXT NOT NULL,
  iteration         INTEGER NOT NULL,
  model             TEXT NOT NULL,
  temperature       REAL,
  started_at        REAL NOT NULL,
  latency_ms        INTEGER NOT NULL,
  prompt_tokens     INTEGER,
  completion_tokens INTEGER,
  cost_usd          REAL,
  ok                INTEGER NOT NULL,
  error             TEXT
);

CREATE TABLE IF NOT EXISTS judge_scores (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id        TEXT NOT NULL REFERENCES runs(run_id),
  iteration     INTEGER NOT NULL,
  overall       REAL NOT NULL,
  weakest       TEXT,
  age_fit       REAL,
  safety        REAL,
  narrative_arc REAL,
  engagement    REAL,
  calmness      REAL
);

CREATE INDEX IF NOT EXISTS idx_calls_run ON calls(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_cat ON runs(category);
"""


def db_path() -> str:
    """Resolve the SQLite path from STORY_DB_PATH (default: traces.db).
    Empty string means tracing is disabled."""
    return os.environ.get("STORY_DB_PATH", DEFAULT_DB_PATH)


def enabled() -> bool:
    return db_path() != ""


@contextmanager
def _conn():
    """Open + commit + close. Yields None when tracing is disabled so callers
    can `with _conn() as c: if c is None: return` without branching twice."""
    if not enabled():
        yield None
        return
    path = db_path()
    c = sqlite3.connect(path)
    try:
        if path != ":memory:":
            c.execute("PRAGMA journal_mode=WAL;")
        c.executescript(SCHEMA)
        yield c
        c.commit()
    finally:
        c.close()


def _cost_usd(
    model: str,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
) -> float:
    if prompt_tokens is None or completion_tokens is None:
        return 0.0
    rate = PRICING.get(model)
    if rate is None:
        # Prefix-match fallback so model variants don't silently zero-cost.
        for prefix, r in PRICING.items():
            if model.startswith(prefix):
                rate = r
                break
    if rate is None:
        return 0.0
    return (prompt_tokens / 1000.0) * rate[0] + (completion_tokens / 1000.0) * rate[1]


def start_run(request: str) -> Optional[str]:
    """Open a run row; return run_id or None when tracing is disabled."""
    if not enabled():
        return None
    run_id = uuid.uuid4().hex
    with _conn() as c:
        if c is None:
            return None
        c.execute(
            "INSERT INTO runs (run_id, started_at, request) VALUES (?, ?, ?)",
            (run_id, time.time(), request),
        )
    return run_id


def finish_run(
    run_id: Optional[str],
    *,
    category: Optional[str] = None,
    iterations: Optional[int] = None,
    final_overall: Optional[float] = None,
    passed: Optional[bool] = None,
    error: Optional[str] = None,
) -> None:
    """Close the run row. Aggregates token + cost totals from calls."""
    if run_id is None or not enabled():
        return
    with _conn() as c:
        if c is None:
            return
        row = c.execute(
            "SELECT COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0), "
            "COALESCE(SUM(cost_usd), 0.0) FROM calls WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        total_pt, total_ct, total_cost = row
        c.execute(
            "UPDATE runs SET ended_at=?, category=?, iterations=?, final_overall=?, "
            "passed=?, total_prompt_tokens=?, total_completion_tokens=?, "
            "total_cost_usd=?, error=? WHERE run_id = ?",
            (
                time.time(),
                category,
                iterations,
                final_overall,
                int(bool(passed)) if passed is not None else None,
                total_pt,
                total_ct,
                total_cost,
                error,
                run_id,
            ),
        )


def record_call(
    run_id: Optional[str],
    role: str,
    iteration: int,
    model: str,
    temperature: float,
    fn: Callable[[], object],
) -> str:
    """Time fn(), insert a calls row, and return the response content (stripped).

    `fn` should return the raw OpenAI response object. We extract usage
    (prompt/completion tokens) and `.choices[0].message["content"]` here so
    callers don't have to. Failures still write a row (ok=0, error=...) and
    re-raise so the existing retry path in main.py is undisturbed.
    """
    started = time.time()
    try:
        resp = fn()
    except Exception as e:
        latency_ms = int((time.time() - started) * 1000)
        if run_id is not None and enabled():
            with _conn() as c:
                if c is not None:
                    c.execute(
                        "INSERT INTO calls (run_id, role, iteration, model, temperature, "
                        "started_at, latency_ms, prompt_tokens, completion_tokens, "
                        "cost_usd, ok, error) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, "
                        "NULL, 0, ?)",
                        (run_id, role, iteration, model, temperature, started,
                         latency_ms, str(e)[:500]),
                    )
        raise

    latency_ms = int((time.time() - started) * 1000)
    if run_id is not None and enabled():
        usage = _extract_usage(resp)
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        cost = _cost_usd(model, pt, ct)
        with _conn() as c:
            if c is not None:
                c.execute(
                    "INSERT INTO calls (run_id, role, iteration, model, temperature, "
                    "started_at, latency_ms, prompt_tokens, completion_tokens, "
                    "cost_usd, ok) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                    (run_id, role, iteration, model, temperature, started,
                     latency_ms, pt, ct, cost),
                )
    return resp.choices[0].message["content"].strip()


def _extract_usage(resp: object) -> dict:
    """Pre-1.0 openai SDK returns an OpenAIObject (dict-like); usage is in resp['usage']."""
    try:
        u = resp["usage"] if "usage" in resp else None  # type: ignore[index]
    except Exception:
        u = getattr(resp, "usage", None)
    if u is None:
        return {}
    if hasattr(u, "get"):
        return {
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
        }
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", None),
        "completion_tokens": getattr(u, "completion_tokens", None),
    }


def record_judge(
    run_id: Optional[str],
    iteration: int,
    judge_result: dict,
) -> None:
    if run_id is None or not enabled():
        return
    scores = judge_result.get("scores", {}) or {}
    with _conn() as c:
        if c is None:
            return
        c.execute(
            "INSERT INTO judge_scores (run_id, iteration, overall, weakest, "
            "age_fit, safety, narrative_arc, engagement, calmness) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                iteration,
                float(judge_result.get("overall", 0.0)),
                judge_result.get("weakest"),
                scores.get("age_fit"),
                scores.get("safety"),
                scores.get("narrative_arc"),
                scores.get("engagement"),
                scores.get("calmness"),
            ),
        )


# ---------- Read helpers (used by `python -m evals report`) ----------

def open_read_only(path: Optional[str] = None) -> sqlite3.Connection:
    """Open the trace DB read-only-ish (not actually RO at SQLite level — just
    callers should not write). Initializes schema if missing so a fresh DB
    doesn't blow up the report."""
    p = path or db_path()
    if not p:
        raise RuntimeError("STORY_DB_PATH is empty — tracing is disabled.")
    c = sqlite3.connect(p)
    c.executescript(SCHEMA)
    return c
