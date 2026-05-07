"""Aggregate report over the SQLite trace DB written by trace.py.

Usage:
    python -m evals report                 # full report from $STORY_DB_PATH (or traces.db)
    python -m evals report --db custom.db
    python -m evals report --category ADVENTURE
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from collections import Counter
from pathlib import Path

# Project root on path so we can import trace from a sibling.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import trace as trace_mod  # noqa: E402


# ---------- math helpers (no numpy / scipy — stdlib only) ----------

def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def _fmt_secs(ms: float) -> str:
    return f"{ms / 1000.0:5.2f}"


# ---------- queries ----------

def _overview(conn, where: str, params: tuple):
    row = conn.execute(
        f"SELECT COUNT(*), AVG(iterations), "
        f"SUM(CASE WHEN passed=1 THEN 1 ELSE 0 END), "
        f"SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END), "
        f"AVG(final_overall), SUM(total_prompt_tokens), "
        f"SUM(total_completion_tokens), SUM(total_cost_usd), "
        f"SUM(CASE WHEN iterations >= 1 THEN 1 ELSE 0 END) "
        f"FROM runs {where}",
        params,
    ).fetchone()
    return {
        "n_runs": row[0] or 0,
        "mean_iters": row[1] or 0.0,
        "n_passed": row[2] or 0,
        "n_errors": row[3] or 0,
        "mean_overall": row[4] or 0.0,
        "tokens_in": row[5] or 0,
        "tokens_out": row[6] or 0,
        "total_cost": row[7] or 0.0,
        "n_refined": row[8] or 0,
    }


def _e2e_latencies_ms(conn, where: str, params: tuple) -> list[float]:
    rows = conn.execute(
        f"SELECT (ended_at - started_at) * 1000.0 FROM runs "
        f"{where} {'AND' if where else 'WHERE'} ended_at IS NOT NULL",
        params,
    ).fetchall()
    return [r[0] for r in rows if r[0] is not None]


def _per_role_latencies_ms(conn, run_ids_where: str, params: tuple) -> dict[str, list[float]]:
    rows = conn.execute(
        f"SELECT role, latency_ms FROM calls "
        f"WHERE run_id IN (SELECT run_id FROM runs {run_ids_where})",
        params,
    ).fetchall()
    by_role: dict[str, list[float]] = {}
    for role, ms in rows:
        by_role.setdefault(role, []).append(ms)
    return by_role


def _mean_judge_scores(conn, run_ids_where: str, params: tuple) -> dict:
    row = conn.execute(
        f"SELECT AVG(overall), AVG(age_fit), AVG(safety), AVG(narrative_arc), "
        f"AVG(engagement), AVG(calmness) FROM judge_scores "
        f"WHERE run_id IN (SELECT run_id FROM runs {run_ids_where})",
        params,
    ).fetchone()
    return {
        "overall": row[0],
        "age_fit": row[1],
        "safety": row[2],
        "narrative_arc": row[3],
        "engagement": row[4],
        "calmness": row[5],
    }


def _by_category(conn, where: str, params: tuple) -> list[tuple]:
    rows = conn.execute(
        f"SELECT category, COUNT(*), "
        f"AVG(CASE WHEN iterations >= 1 THEN 1.0 ELSE 0.0 END) * 100.0, "
        f"AVG(final_overall), AVG(total_cost_usd) "
        f"FROM runs {where} GROUP BY category ORDER BY category",
        params,
    ).fetchall()
    return rows


def _weakest_distribution(conn, run_ids_where: str, params: tuple) -> list[tuple]:
    return conn.execute(
        f"SELECT weakest, COUNT(*) FROM judge_scores "
        f"WHERE run_id IN (SELECT run_id FROM runs {run_ids_where}) "
        f"AND weakest IS NOT NULL "
        f"GROUP BY weakest ORDER BY COUNT(*) DESC",
        params,
    ).fetchall()


# ---------- rendering ----------

def render_report(db_path: str, category_filter: str | None) -> str:
    conn = trace_mod.open_read_only(db_path)
    try:
        if category_filter:
            where = "WHERE category = ?"
            params: tuple = (category_filter,)
        else:
            where = ""
            params = ()

        ov = _overview(conn, where, params)
        if ov["n_runs"] == 0:
            return (
                f"# Story Pipeline Report\n\n"
                f"DB: `{db_path}` | filter: `{category_filter or 'none'}`\n\n"
                f"**No runs recorded yet.** "
                f"Run the pipeline (e.g. `python main.py`) and try again.\n"
            )

        e2e = _e2e_latencies_ms(conn, where, params)
        per_role = _per_role_latencies_ms(conn, where, params)
        scores = _mean_judge_scores(conn, where, params)
        by_cat = _by_category(conn, where, params)
        weakest = _weakest_distribution(conn, where, params)
    finally:
        conn.close()

    out: list[str] = []
    out.append("# Story Pipeline Report")
    out.append(
        f"_Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M %Z').strip()} | "
        f"DB: `{db_path}` | filter: `{category_filter or 'all'}`_\n"
    )

    pass_pct = 100.0 * ov["n_passed"] / ov["n_runs"]
    refine_pct = 100.0 * ov["n_refined"] / ov["n_runs"]
    out.append("## Overview")
    out.append(f"- Runs: **{ov['n_runs']}**")
    out.append(f"- Errors: {ov['n_errors']}")
    out.append(f"- Mean iterations: {ov['mean_iters']:.2f}")
    out.append(f"- Refinement rate (≥1 refine): **{refine_pct:.1f}%**")
    out.append(f"- Pass rate (overall ≥ 4.0): **{pass_pct:.1f}%**")
    out.append("")

    out.append("## End-to-end latency (seconds)")
    if e2e:
        out.append(
            f"- p50 {_fmt_secs(_percentile(e2e, 50))}   "
            f"p95 {_fmt_secs(_percentile(e2e, 95))}   "
            f"max {_fmt_secs(max(e2e))}\n"
        )
    else:
        out.append("- (no completed runs)\n")

    out.append("## Per-call latency (seconds)")
    out.append("| role | n | p50 | p95 |")
    out.append("|---|---:|---:|---:|")
    for role in sorted(per_role):
        ms = per_role[role]
        out.append(
            f"| {role} | {len(ms)} | {_fmt_secs(_percentile(ms, 50))} | "
            f"{_fmt_secs(_percentile(ms, 95))} |"
        )
    out.append("")

    cost_per_run = ov["total_cost"] / max(1, ov["n_runs"])
    out.append("## Cost")
    out.append(f"- Total tokens: {ov['tokens_in']:,} in / {ov['tokens_out']:,} out")
    out.append(
        f"- Total cost: ${ov['total_cost']:.4f} (${cost_per_run:.4f}/run avg)"
    )
    out.append("")

    out.append("## Mean judge scores")
    if scores["overall"] is not None:
        out.append(
            f"- overall **{scores['overall']:.2f}** | "
            f"age_fit {scores['age_fit'] or 0:.2f} | "
            f"safety {scores['safety'] or 0:.2f} | "
            f"narrative_arc {scores['narrative_arc'] or 0:.2f} | "
            f"engagement {scores['engagement'] or 0:.2f} | "
            f"calmness {scores['calmness'] or 0:.2f}\n"
        )
    else:
        out.append("- (no judge scores recorded)\n")

    if by_cat and not category_filter:
        out.append("## By category")
        out.append("| category | n | refine% | mean_overall | $/run |")
        out.append("|---|---:|---:|---:|---:|")
        for cat, n, ref_pct, mean_o, cost in by_cat:
            out.append(
                f"| {cat or '(none)'} | {n} | {ref_pct or 0:.0f}% | "
                f"{mean_o or 0:.2f} | ${cost or 0:.4f} |"
            )
        out.append("")

    if weakest:
        out.append("## Weakest dimension distribution")
        for dim, n in weakest:
            out.append(f"- {dim}: {n}")
        out.append("")

    return "\n".join(out)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Aggregate report over the SQLite trace DB.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    rep = sub.add_parser("report", help="Render markdown report.")
    rep.add_argument("--db", default=None, help="Trace DB path (default: $STORY_DB_PATH or traces.db).")
    rep.add_argument("--category", default=None, help="Filter to one category.")

    args = parser.parse_args(argv[1:])
    if args.cmd == "report":
        path = args.db or os.environ.get("STORY_DB_PATH") or trace_mod.DEFAULT_DB_PATH
        print(render_report(path, args.category))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
