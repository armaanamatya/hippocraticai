"""Unit tests for prompts.py — verify invariants and message shapes.

No API calls. Run from project root:  pytest tests/test_prompts.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from prompts import (
    CATEGORIES,
    CATEGORIZER_PROMPT,
    CONSTITUTION,
    JSON_FIX_PROMPT,
    MODEL,
    RUBRIC_CRITERIA,
    STORYTELLER_PERSONAS,
    build_judge_prompt,
    build_kid_reaction_prompt,
    build_refiner_prompt,
    build_storyteller_prompt,
)


# ----------------------- module-level invariants ----------------------


def test_model_is_pinned_to_gpt_3_5_turbo():
    # Assignment forbids changing the model. If this ever fails, check git diff.
    assert MODEL == "gpt-3.5-turbo"


def test_categories_count_and_uppercase():
    assert len(CATEGORIES) == 4
    assert all(c.isupper() for c in CATEGORIES)
    assert set(CATEGORIES) == {"ADVENTURE", "FRIENDSHIP", "CURIOSITY", "CALMING"}


def test_personas_cover_all_categories():
    assert set(STORYTELLER_PERSONAS.keys()) == set(CATEGORIES)
    for persona in STORYTELLER_PERSONAS.values():
        # Each persona should be substantial (3 sentences ≈ ≥ 100 chars).
        assert len(persona) > 100


def test_rubric_has_exactly_five_criteria():
    expected = {"age_fit", "safety", "narrative_arc", "engagement", "calmness"}
    assert set(RUBRIC_CRITERIA.keys()) == expected
    for desc in RUBRIC_CRITERIA.values():
        assert isinstance(desc, str) and len(desc) > 10


def test_constitution_is_substantial():
    # We need ≥ 6 explicit safety principles per the design.
    assert len(CONSTITUTION) >= 6
    assert all(isinstance(p, str) and len(p) > 10 for p in CONSTITUTION)


def test_categorizer_prompt_has_request_placeholder():
    assert "{request}" in CATEGORIZER_PROMPT
    # Should also enumerate all categories so the model knows the label set.
    for cat in CATEGORIES:
        assert cat in CATEGORIZER_PROMPT


def test_json_fix_prompt_mentions_schema():
    assert "JSON" in JSON_FIX_PROMPT
    assert "scores" in JSON_FIX_PROMPT


# ----------------------- build_storyteller_prompt ---------------------


@pytest.mark.parametrize("category", ["ADVENTURE", "FRIENDSHIP", "CURIOSITY", "CALMING"])
def test_storyteller_returns_two_message_dicts(category):
    msgs = build_storyteller_prompt(category, "a story about a snail")
    assert isinstance(msgs, list) and len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "snail" in msgs[1]["content"]


def test_storyteller_system_contains_category_persona():
    msgs = build_storyteller_prompt("ADVENTURE", "anything")
    persona = STORYTELLER_PERSONAS["ADVENTURE"]
    assert persona in msgs[0]["content"]


def test_storyteller_invalid_category_raises():
    with pytest.raises(KeyError):
        build_storyteller_prompt("NOT_A_CATEGORY", "anything")


def test_storyteller_mentions_arc_or_outline():
    msgs = build_storyteller_prompt("CALMING", "a sleepy bunny")
    sys_text = msgs[0]["content"].lower()
    # The storyteller must scaffold a beat structure.
    assert "outline" in sys_text or "beat" in sys_text


# ----------------------- build_judge_prompt ---------------------------


def _sample_metrics():
    return {
        "total_words": 247,
        "total_sentences": 18,
        "avg_sentence_len": 13.7,
        "fk_grade": 4.1,
        "hard_vocab_pct": 4.5,
        "longest_sentence_words": 24,
    }


def test_judge_returns_two_messages_with_story_in_user():
    msgs = build_judge_prompt("Once upon a time, a small fox napped.", _sample_metrics())
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "fox napped" in msgs[1]["content"]


def test_judge_system_lists_all_rubric_keys():
    msgs = build_judge_prompt("any story", _sample_metrics())
    system = msgs[0]["content"]
    for key in RUBRIC_CRITERIA.keys():
        assert key in system


def test_judge_system_includes_constitution_and_metrics():
    msgs = build_judge_prompt("any story", _sample_metrics())
    system = msgs[0]["content"]
    # At least one constitutional principle should be present verbatim.
    assert any(rule in system for rule in CONSTITUTION)
    # Metric values should be visible to the judge as evidence.
    assert "247" in system
    assert "4.1" in system


def test_judge_system_demands_json():
    msgs = build_judge_prompt("any story", _sample_metrics())
    system = msgs[0]["content"]
    assert "JSON" in system
    # Should mention the required keys of the JSON schema.
    for key in ("scores", "weakest", "critique", "overall"):
        assert key in system


def test_judge_has_no_persona_language():
    # Judges with personas degrade objective evaluation (arXiv:2311.10054).
    # System prompt should not assign a "you are X expert/teacher/parent" identity.
    msgs = build_judge_prompt("story", _sample_metrics())
    system = msgs[0]["content"].lower()
    # Look for explicit "no persona" framing OR absence of common persona triggers.
    assert "no persona" in system or "evaluator" in system


# ----------------------- build_refiner_prompt -------------------------


def test_refiner_returns_two_messages_with_critique_in_user():
    msgs = build_refiner_prompt(
        "FRIENDSHIP",
        prev_story="The bunny smiled.",
        judge_critique="Add more sensory detail.",
        weakest="engagement",
        kid_reactions=None,
    )
    assert len(msgs) == 2
    assert "sensory detail" in msgs[1]["content"]
    assert "engagement" in msgs[1]["content"]


def test_refiner_includes_kid_reactions_when_provided():
    kid = "I liked the moon.\nI didn't get the part with the cup.\nI wish there was a cat."
    msgs = build_refiner_prompt(
        "FRIENDSHIP",
        prev_story="...",
        judge_critique="fix things",
        weakest="engagement",
        kid_reactions=kid,
    )
    user = msgs[1]["content"]
    assert "didn't get the part with the cup" in user


def test_refiner_uses_same_persona_as_storyteller():
    msgs = build_refiner_prompt(
        "ADVENTURE", prev_story="x", judge_critique="y", weakest="age_fit", kid_reactions=None
    )
    assert STORYTELLER_PERSONAS["ADVENTURE"] in msgs[0]["content"]


# ----------------------- build_kid_reaction_prompt --------------------


def test_kid_reaction_prompt_specifies_seven_year_old_and_ordering():
    msgs = build_kid_reaction_prompt("the story text")
    system = msgs[0]["content"]
    assert "7-year-old" in system or "seven-year-old" in system
    # Three required openings:
    assert '"I liked"' in system or "I liked" in system
    assert "I didn't get" in system
    assert "I wish" in system
    assert "the story text" in msgs[1]["content"]
