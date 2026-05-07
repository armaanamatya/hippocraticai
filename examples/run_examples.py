"""End-to-end demo runner — executes the full pipeline against one prompt
per category and writes each result to examples/outputs/<category>.md +
examples/outputs/<category>.mp3 (TTS audio, default ON).

Requires a real OPENAI_API_KEY in .env. Costs ~$0.05-0.10 for the whole run
including TTS (~$0.04 of that is audio at $15/1M chars × ~2400 chars × 4 cats).
NOT a unit test — this hits the live OpenAI API.

Run from project root:
    python examples/run_examples.py                 # all 4 categories, with audio
    python examples/run_examples.py ADVENTURE       # just one, with audio
    python examples/run_examples.py --no-voice      # skip TTS (cheaper, text only)
    python examples/run_examples.py --voice nova    # try a different voice
"""
import argparse
import sys
from pathlib import Path

# Project root on path so we can import main + prompts + readability.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import run_pipeline  # noqa: E402
from tts_voice import TTSError, extract_story_block, synthesize  # noqa: E402

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "categories",
        nargs="*",
        type=str.upper,
        help="Categories to run (default: all 4). Case-insensitive.",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Skip TTS synthesis (text-only run; saves ~$0.04 in API cost).",
    )
    parser.add_argument(
        "--voice",
        default="shimmer",
        help="TTS voice (shimmer/sage/nova/fable/alloy/echo/onyx). Default: shimmer.",
    )
    args = parser.parse_args(argv[1:])

    selected = args.categories or list(EXAMPLES.keys())
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
        md_path = out_dir / f"{cat.lower()}.md"
        md_path.write_text(md, encoding="utf-8")
        print(f"  wrote {md_path}  (overall={result['final_judge']['overall']})")

        if not args.no_voice:
            mp3_path = out_dir / f"{cat.lower()}.mp3"
            try:
                synthesize(
                    extract_story_block(result["story"]),
                    mp3_path,
                    voice=args.voice,
                )
                print(f"  wrote {mp3_path}")
            except TTSError as e:
                print(f"  TTS warning ({cat}): {e}")

    print(f"\nDone. {len(selected) - failures}/{len(selected)} succeeded.")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
