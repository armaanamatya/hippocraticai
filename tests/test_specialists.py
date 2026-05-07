"""Unit tests for the specialist refiner constellation (Tier 2).

Pure prompt-shape tests — no LLM calls."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prompts import CONSTITUTION, RUBRIC_CRITERIA, STORYTELLER_PERSONAS  # noqa: E402
from specialists import (  # noqa: E402
    SPECIALIST_INSTRUCTIONS,
    SPECIALIST_KEYS,
    SPECIALIST_ROLES,
    USER_REQUEST_SENTINEL,
    build_specialist_prompt,
    dispatch,
)


# ---------- structural sanity ----------

def test_specialist_keys_match_rubric():
    assert set(SPECIALIST_KEYS) == set(RUBRIC_CRITERIA.keys())


def test_every_key_has_role_and_instructions():
    for key in SPECIALIST_KEYS:
        assert key in SPECIALIST_ROLES
        assert key in SPECIALIST_INSTRUCTIONS
        assert len(SPECIALIST_INSTRUCTIONS[key]) >= 3


# ---------- dispatch ----------

@pytest.mark.parametrize("key", list(SPECIALIST_KEYS))
def test_dispatch_returns_identity_for_rubric_keys(key):
    assert dispatch(key) == key


def test_dispatch_returns_sentinel_for_unknown():
    assert dispatch("user_request") == USER_REQUEST_SENTINEL
    assert dispatch("totally_made_up") == USER_REQUEST_SENTINEL


# ---------- build_specialist_prompt — common shape ----------

@pytest.mark.parametrize("key", list(SPECIALIST_KEYS))
def test_build_specialist_prompt_has_required_blocks(key):
    msgs = build_specialist_prompt(
        weakest=key,
        category="FRIENDSHIP",
        prev_story="Once upon a time, a small mouse...",
        judge_critique="Some critique here.",
        kid_reactions=None,
        edit_log=[],
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"

    sys_text = msgs[0]["content"]
    # System message must mention the dimension, the role descriptor, and
    # the constitution.
    assert key in sys_text
    assert RUBRIC_CRITERIA[key].split(";")[0][:30] in sys_text or RUBRIC_CRITERIA[key][:30] in sys_text
    # First constitution rule must appear (defensive — confirms block is rendered).
    assert CONSTITUTION[0][:30] in sys_text
    # Style anchor (first sentence of category persona) must appear.
    anchor = STORYTELLER_PERSONAS["FRIENDSHIP"].split(".")[0][:40]
    assert anchor in sys_text
    # Output format directive.
    assert "STORY:" in sys_text
    # User message contains prev_story + critique.
    user_text = msgs[1]["content"]
    assert "Once upon a time" in user_text
    assert "Some critique here." in user_text


def test_build_specialist_prompt_rejects_unknown_dim():
    with pytest.raises(ValueError, match="not a rubric dimension"):
        build_specialist_prompt(
            weakest="user_request",
            category="FRIENDSHIP",
            prev_story="...",
            judge_critique="...",
            kid_reactions=None,
            edit_log=[],
        )


# ---------- per-specialist content checks ----------

def test_age_fit_specialist_mentions_vocabulary():
    msgs = build_specialist_prompt(
        "age_fit", "ADVENTURE", "story", "crit", None, []
    )
    sys_text = msgs[0]["content"].lower()
    assert "vocabulary" in sys_text or "word" in sys_text or "grade" in sys_text


def test_safety_specialist_mentions_constitution():
    msgs = build_specialist_prompt(
        "safety", "ADVENTURE", "story", "crit", None, []
    )
    sys_text = msgs[0]["content"].lower()
    assert "safety" in sys_text
    assert "constitution" in sys_text


def test_engagement_specialist_mentions_sensory_or_dialogue():
    msgs = build_specialist_prompt(
        "engagement", "ADVENTURE", "story", "crit", None, []
    )
    sys_text = msgs[0]["content"].lower()
    assert "sensory" in sys_text or "dialogue" in sys_text


def test_calmness_specialist_mentions_pacing_or_closing():
    msgs = build_specialist_prompt(
        "calmness", "CALMING", "story", "crit", None, []
    )
    sys_text = msgs[0]["content"].lower()
    assert "pacing" in sys_text or "closing" in sys_text or "settling" in sys_text


def test_narrative_arc_specialist_mentions_beats():
    msgs = build_specialist_prompt(
        "narrative_arc", "FRIENDSHIP", "story", "crit", None, []
    )
    sys_text = msgs[0]["content"].lower()
    assert "beat" in sys_text or "structure" in sys_text


# ---------- kid_reactions gating ----------

def _all_text(msgs: list[dict]) -> str:
    return msgs[0]["content"] + "\n" + msgs[1]["content"]


def test_kid_reactions_appear_for_engagement():
    msgs = build_specialist_prompt(
        "engagement", "FRIENDSHIP", "story", "crit",
        kid_reactions="I liked the bunny.\nI didn't get the moon.\nI wish more dragons.",
        edit_log=[],
    )
    text = _all_text(msgs)
    assert "7-year-old" in text
    assert "bunny" in text


def test_kid_reactions_appear_for_narrative_arc():
    msgs = build_specialist_prompt(
        "narrative_arc", "FRIENDSHIP", "story", "crit",
        kid_reactions="I liked X.",
        edit_log=[],
    )
    assert "7-year-old" in _all_text(msgs)


def test_kid_reactions_skipped_for_age_fit():
    msgs = build_specialist_prompt(
        "age_fit", "FRIENDSHIP", "story", "crit",
        kid_reactions="I liked X.",
        edit_log=[],
    )
    assert "7-year-old" not in _all_text(msgs)


def test_kid_reactions_skipped_for_safety():
    msgs = build_specialist_prompt(
        "safety", "FRIENDSHIP", "story", "crit",
        kid_reactions="I liked X.",
        edit_log=[],
    )
    assert "7-year-old" not in _all_text(msgs)


# ---------- edit_log threading ----------

def test_edit_log_renders_in_system_message():
    msgs = build_specialist_prompt(
        "calmness", "CALMING", "story", "crit", None,
        edit_log=[(1, "age_fit", "age_fit"), (2, "engagement", "engagement")],
    )
    sys_text = msgs[0]["content"]
    assert "PRIOR EDITS" in sys_text
    assert "age_fit" in sys_text
    assert "engagement" in sys_text
    assert "do NOT undo" in sys_text


def test_empty_edit_log_omits_prior_edits_block():
    msgs = build_specialist_prompt(
        "calmness", "CALMING", "story", "crit", None, edit_log=[],
    )
    assert "PRIOR EDITS" not in msgs[0]["content"]
