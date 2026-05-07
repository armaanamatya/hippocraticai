"""Calibration check — run the pipeline N times against the same prompt and
report per-run iteration counts + score histories. Used to verify the
refiner loop actually fires (i.e. is not decorative).

Run from project root:
    python examples/loop_check.py                      # 5 runs of "uplifting"
    python examples/loop_check.py 3 "a sleepy bunny"   # 3 runs, custom prompt
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import run_pipeline, MIN_PASS_SCORE, MAX_REFINE_ITERS  # noqa: E402


def main(argv: list[str]) -> int:
    n = int(argv[1]) if len(argv) > 1 else 5
    prompt = argv[2] if len(argv) > 2 else "uplifting"

    print(f"Calibration: {n} runs of prompt={prompt!r}")
    print(f"  MIN_PASS_SCORE={MIN_PASS_SCORE}, MAX_REFINE_ITERS={MAX_REFINE_ITERS}")
    print("=" * 60)

    summary = []
    refines_fired = 0
    for run_idx in range(1, n + 1):
        print(f"\n--- run {run_idx}/{n} ---")
        try:
            result = run_pipeline(prompt)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            summary.append((run_idx, None, None, None))
            continue

        iters = result["iterations"]
        final_overall = result["final_judge"].get("overall")
        history = result["history"]
        if iters >= 1:
            refines_fired += 1
        summary.append((run_idx, iters, final_overall, history))

    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    for run_idx, iters, overall, history in summary:
        if iters is None:
            print(f"  run {run_idx}: ERROR")
            continue
        hist_str = " -> ".join(f"{o}({w})" for _, o, w in history)
        print(f"  run {run_idx}: iters={iters} final={overall}  history: {hist_str}")
    print(
        f"\nRefiner loop fired in {refines_fired}/{n} runs "
        f"({'OK' if refines_fired > 0 else 'STILL DECORATIVE — tighten further'})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
