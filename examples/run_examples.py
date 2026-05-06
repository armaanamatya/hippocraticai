"""End-to-end demo runner — executes the full pipeline against one prompt
per category and writes each result to examples/outputs/<category>.md.

Requires a real OPENAI_API_KEY in .env. Costs ~$0.01-0.05 for the whole run.
NOT a unit test — this hits the live OpenAI API.

Run from project root:
    python examples/run_examples.py            # all 4 categories
    python examples/run_examples.py ADVENTURE  # just one
"""
import sys
from pathlib import Path

# Project root on path so we can import main + prompts + readability.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import run_pipeline  # noqa: E402

EXAMPLES: dict[str, str] = {
    "ADVENTURE": "a story about a brave little knight who learns that asking for help is a kind of courage",
    "FRIENDSHIP": "a story about Alice and her best friend Bob, who happens to be a cat, and a misunderstanding they fix together",
    "CURIOSITY": "a story about why the rain falls, told through a small girl noticing puddles",
    "CALMING": "a sleepy story about a tired bunny finding the softest place in the meadow to rest",
}


def render_markdown(prompt: str, result: dict) -> str:
    judge = result["final_judge"]
    scores = judge["scores"]
    score_line = " ".join(f"{k}={v}" for k, v in scores.items())
    history = "\n".join(
        f"- iter {i}: overall={overall} (weakest={weakest})"
        for i, overall, weakest in result["history"]
    )
    return f"""# {result['category']} example

**Prompt:** {prompt}

**Iterations:** {result['iterations']}
**Final overall score:** {judge['overall']}
**Per-criterion:** {score_line}

## Iteration history
{history}

## Final critique from judge
> {judge.get('critique', '(none)')}

## Story

{result['story']}
"""


def main(argv: list[str]) -> int:
    selected = [a.upper() for a in argv[1:]] or list(EXAMPLES.keys())
    invalid = [s for s in selected if s not in EXAMPLES]
    if invalid:
        print(f"ERROR: unknown category/categories {invalid}. Valid: {list(EXAMPLES.keys())}")
        return 1

    out_dir = ROOT / "examples" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for cat in selected:
        prompt = EXAMPLES[cat]
        print(f"\n{'=' * 60}\nRunning category={cat}\nPrompt: {prompt}\n{'=' * 60}")
        try:
            result = run_pipeline(prompt)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            failures += 1
            continue

        md = render_markdown(prompt, result)
        path = out_dir / f"{cat.lower()}.md"
        path.write_text(md, encoding="utf-8")
        print(f"  wrote {path}  (overall={result['final_judge']['overall']})")

    print(f"\nDone. {len(selected) - failures}/{len(selected)} succeeded.")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
