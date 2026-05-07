"""Multi-agent specialist refiner constellation (Tier 2 extension).

Replaces the single generic refiner with one of five narrowly-scoped editors,
each focused on a single rubric dimension. Mirrors Hippocratic AI's Polaris
primary + specialist architecture (arXiv:2403.13313): the persona-bearing
storyteller is the "primary" generative agent; specialists are persona-free,
single-task editors invoked by the judge's weakest-dimension diagnosis.

Why persona-free: arXiv:2311.10054 — personas degrade objective, narrowly
scoped tasks. Each specialist performs a focused linguistic/structural edit,
not generative roleplay. The original storyteller persona is passed in as a
STYLE ANCHOR ("preserve this voice") but specialists do NOT adopt it.

Why edit_log: arXiv:2310.01798 ("LLMs Cannot Self-Correct Reasoning Yet")
warned that iterative refinement can ping-pong (one specialist undoes
another). The edit_log lets the current specialist see prior touch points
and is told NOT to undo their work.

Cited:
- Polaris constellation: arXiv:2403.13313 / hippocraticai.com/polaris2
- Persona harms objective tasks: arXiv:2311.10054
- Self-Refine baseline: arXiv:2303.17651
- Iterative refinement regression risk: arXiv:2310.01798
"""
from __future__ import annotations

from typing import Optional

from prompts import CONSTITUTION, RUBRIC_CRITERIA, STORYTELLER_PERSONAS


# Mirrors RUBRIC_CRITERIA.keys() — the 5 dimensions the judge scores.
SPECIALIST_KEYS: tuple[str, ...] = (
    "age_fit", "safety", "narrative_arc", "engagement", "calmness",
)

# Sentinel returned by dispatch() for non-rubric weaknesses (e.g. user
# free-text requests like 'make it sillier'); caller routes those to the
# legacy generic refiner.
USER_REQUEST_SENTINEL: str = "user_request"


SPECIALIST_ROLES: dict[str, str] = {
    "age_fit": (
        "You are a children's-book copy editor specializing in ages 5-10. "
        "You do NOT rewrite stories; you swap words and split clauses to "
        "lower reading difficulty."
    ),
    "safety": (
        "You are a child-safety editor. Your only job is to neutralize "
        "borderline content while preserving the story's shape, characters, "
        "and pacing."
    ),
    "narrative_arc": (
        "You are a story-structure editor. You diagnose missing beats and "
        "add the smallest patch possible — never new characters, never "
        "rewrites of prose that already works."
    ),
    "engagement": (
        "You are a sensory-detail editor. You add texture (sight, sound, "
        "touch) to flat sentences and ensure at least one line of dialogue "
        "exists. You do NOT add plot or events."
    ),
    "calmness": (
        "You are a bedtime-pacing editor. You slow rhythm in the back half "
        "of the story without changing the plot, soften consonants, and "
        "ensure the closing two sentences are settling, eyes-heavy."
    ),
}


SPECIALIST_INSTRUCTIONS: dict[str, list[str]] = {
    "age_fit": [
        "Replace any word above ~Grade 3 with a concrete, high-frequency synonym.",
        "Split sentences longer than 12 words at clause boundaries.",
        "Prefer concrete nouns over abstract concepts.",
        "Keep all named characters, plot beats, and dialogue lines unchanged.",
        "Target Flesch-Kincaid grade <= 3.5.",
    ],
    "safety": [
        "Re-read each Constitution rule above and rewrite any sentence that brushes against it.",
        "Replace any threat / scary imagery with a gentle equivalent ('growling shadow' -> 'soft rustling').",
        "Ensure conflict resolves through kindness or cleverness, never punishment.",
        "Reinforce a safe, warm closing line.",
        "Leave non-flagged sentences untouched.",
    ],
    "narrative_arc": [
        "Identify which of {setup, small problem, kind resolution, wind-down} is weak or missing.",
        "Add or tighten 1-3 sentences for that beat ONLY.",
        "Do not introduce new characters.",
        "Keep the problem tiny and the resolution kind.",
        "Maintain total length 250-400 words.",
    ],
    "engagement": [
        "Inject sight / sound / touch / smell details into 2-4 sentences that currently feel flat.",
        "Ensure at least one line of dialogue exists; add one in-character if missing.",
        "Do NOT add new events or change pacing.",
        "Prefer soft sensory words (whisper, shimmer, hush) to keep bedtime tone.",
        "Preserve all existing dialogue verbatim.",
    ],
    "calmness": [
        "Lengthen and soften the final third.",
        "Replace hard consonants (k / t / p) with softer ones where it reads naturally.",
        "Remove or downshift any spike of excitement in the last 5 sentences.",
        "Ensure the closing two sentences depict the character settling, eyes growing heavy.",
        "Keep the opening intact.",
    ],
}


def dispatch(weakest: str) -> str:
    """Return the specialist key for a rubric dimension; return the
    USER_REQUEST_SENTINEL for anything outside the rubric so the caller can
    branch to the legacy generic refiner."""
    if weakest in SPECIALIST_KEYS:
        return weakest
    return USER_REQUEST_SENTINEL


def build_specialist_prompt(
    weakest: str,
    category: str,
    prev_story: str,
    judge_critique: str,
    kid_reactions: Optional[str],
    edit_log: list[tuple[int, str, str]],
) -> list[dict]:
    """Build messages for the specialist that targets `weakest`."""
    if weakest not in SPECIALIST_KEYS:
        raise ValueError(
            f"build_specialist_prompt: '{weakest}' is not a rubric dimension. "
            f"Route non-rubric requests through the legacy generic refiner."
        )

    role = SPECIALIST_ROLES[weakest]
    instructions = SPECIALIST_INSTRUCTIONS[weakest]
    target_desc = RUBRIC_CRITERIA[weakest]
    style_anchor = STORYTELLER_PERSONAS[category].split(".")[0] + "."
    constitution_block = "\n".join(
        f"  {i + 1}. {rule}" for i, rule in enumerate(CONSTITUTION)
    )
    instructions_block = "\n".join(f"  - {ins}" for ins in instructions)

    log_block = ""
    if edit_log:
        prior = "\n".join(
            f"  - iter {it}: specialist '{key}' targeted {dim}"
            for it, key, dim in edit_log
        )
        log_block = (
            "\n\nPRIOR EDITS (do NOT undo — these dimensions are already "
            "improved):\n" + prior
        )

    kid_block = ""
    # Kid reactions are most actionable for engagement + narrative beats only.
    if kid_reactions and weakest in ("engagement", "narrative_arc"):
        kid_block = (
            f"\n\nA 7-year-old listener said:\n{kid_reactions}\n"
            "Use these as signal for what to fix."
        )

    system = f"""{role}

CONSTITUTION (must always hold — every edit must preserve these):
{constitution_block}

YOUR TARGET DIMENSION: {weakest} — {target_desc}

EDIT INSTRUCTIONS (apply ALL):
{instructions_block}

STYLE ANCHOR (preserve this voice; do NOT adopt it as your own identity):
  {style_anchor}

OUTPUT FORMAT: emit only the full revised STORY (250-400 words), labeled "STORY:".
Do not include OUTLINE, commentary, or explanation. Edit surgically — keep
every part of the story that is not directly related to your target
dimension exactly as it is.{log_block}"""

    user = f"""Judge critique (focused on {weakest}):
{judge_critique}
{kid_block}
Previous story:
{prev_story}

Now emit the revised STORY: that improves only the {weakest} dimension."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
