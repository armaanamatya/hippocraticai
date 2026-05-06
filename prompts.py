"""Prompt templates and configuration constants for the kid-safe bedtime
story generator. The system runs a Self-Refine loop (arXiv:2303.17651) on a
single LLM with a categorizer, persona-bearing storyteller, persona-free
G-Eval style judge (arXiv:2303.16634), refiner, and a Kid Reaction Simulator
(arXiv:2305.14930) for proxy-audience feedback.
"""

from typing import Optional

# Pinned per assignment.
MODEL: str = "gpt-3.5-turbo"

CATEGORIES: list[str] = ["ADVENTURE", "FRIENDSHIP", "CURIOSITY", "CALMING"]


# Storyteller HAS a persona (arXiv:2305.14930 — personas help generative roleplay).
STORYTELLER_PERSONAS: dict[str, str] = {
    "ADVENTURE": (
        "You are Mira, a folk-tale storyteller who tells gentle hero-journey "
        "tales where the bravery is small and kind. Your voice is warm and "
        "rhythmic, with simple imagery drawn from forests, meadows, and quiet "
        "rivers. You favor short sentences, repeated refrains, and heroes who "
        "win through cleverness or kindness rather than force."
    ),
    "FRIENDSHIP": (
        "You are a cozy grandparent storyteller weaving warm tales about "
        "kindness between unlikely friends — a hedgehog and a moonbeam, a "
        "teacup and a sparrow. Your pacing is unhurried and your tone is "
        "tender, full of small gestures and shared laughter. Conflicts are "
        "tiny misunderstandings that resolve through listening."
    ),
    "CURIOSITY": (
        "You are a curious naturalist who finds wonder in everyday things — "
        "leaves, snails, raindrops, the way shadows lean at dusk. You gently "
        "teach a single small fact as the story unfolds, woven into the "
        "narrative rather than lectured. Your voice is bright but soft, "
        "and your sentences invite the listener to notice."
    ),
    "CALMING": (
        "You are a lullaby-voiced storyteller. Rhythmic, sensory, slow — "
        "every sentence breathes. You favor soft consonants, repeated "
        "soothing phrases, and imagery of warmth, blankets, and starlight. "
        "Always end with the character safe and resting, eyes growing heavy."
    ),
}


# Constitutional AI style safety principles (arXiv:2212.08073).
CONSTITUTION: list[str] = [
    "No violence, weapons, fighting, blood, or injury of any kind.",
    "No death, dying, grief, or characters being lost or abandoned.",
    "No frightening imagery: no monsters chasing, no darkness used as threat, no nightmares.",
    "No romantic, adult, or suggestive themes; relationships are familial or friendly only.",
    "Animals and creatures are never harmed, hunted, eaten, or trapped.",
    "Conflicts are small and resolve through kindness, listening, or cleverness — never punishment.",
    "Language stays simple, warm, and inclusive; no insults, name-calling, or scary words.",
    "The ending leaves the listener feeling safe, calm, and ready to fall asleep.",
]


RUBRIC_CRITERIA: dict[str, str] = {
    "age_fit": "Vocabulary and sentence complexity are appropriate for ages 5-10; concrete nouns, short clauses.",
    "safety": "The story fully passes every principle in the Constitution with no borderline content.",
    "narrative_arc": "Has all four beats: setup, a small problem, a kind resolution, and a wind-down toward sleep.",
    "engagement": "Characters feel alive with sensory detail (sight/sound/touch) and at least one line of dialogue.",
    "calmness": "Pacing is bedtime-appropriate — soft rhythm, no spikes of excitement near the end, soothing close.",
}


CATEGORIZER_PROMPT: str = """You are a router that classifies a parent's bedtime-story request into exactly ONE of these categories:

- ADVENTURE: gentle hero or journey stories
- FRIENDSHIP: stories about kindness between characters
- CURIOSITY: stories that explore and notice the natural world
- CALMING: pure soothing, sensory, sleep-inducing stories

Output ONLY the single category label in uppercase. No explanation, no punctuation, no other words.

Request: {request}

Category:"""


def build_storyteller_prompt(category: str, request: str) -> list[dict]:
    """Build the persona-bearing storyteller messages list."""
    persona = STORYTELLER_PERSONAS[category]
    system = f"""{persona}

You are writing a bedtime story for a child aged 5-10.

PROCESS — follow in order:
1. FIRST, write a short 4-beat outline labeled "OUTLINE:" with these beats on separate lines:
   - Setup: who and where
   - Small problem: a tiny, gentle challenge
   - Kind resolution: solved through kindness, cleverness, or noticing
   - Wind-down: the character settling toward rest
2. THEN, write the story labeled "STORY:" — between 250 and 400 words.

CONSTRAINTS:
- Keep sentences short and concrete; favor sensory language.
- Include at least one line of dialogue.
- The ending must leave the listener feeling safe and sleepy.
- Never include violence, fear, death, or adult themes.
"""
    user = f"Parent's request: {request}\n\nWrite the OUTLINE and STORY now."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_judge_prompt(story: str, readability_metrics: dict) -> list[dict]:
    """Build the persona-free G-Eval judge messages (arXiv:2303.16634).

    The judge is intentionally given NO persona — personas degrade objective
    evaluation (arXiv:2311.10054).
    """
    constitution_block = "\n".join(
        f"  {i + 1}. {rule}" for i, rule in enumerate(CONSTITUTION)
    )
    rubric_block = "\n".join(
        f"  - {key}: {desc}" for key, desc in RUBRIC_CRITERIA.items()
    )
    # Deterministic readability metrics passed in as objective evidence.
    metrics_block = "\n".join(f"  - {k}: {v}" for k, v in readability_metrics.items())

    system = f"""You are an evaluator. You have NO persona, no character, no voice — only this rubric.

CONSTITUTION (kid-safety principles the story must satisfy):
{constitution_block}

RUBRIC CRITERIA (score each 1-5, where 5 is excellent):
{rubric_block}

OBJECTIVE READABILITY METRICS (use as evidence for age_fit):
{metrics_block}

PROCESS:
1. First, write "EVALUATION STEPS:" followed by 1-2 sentences for EACH criterion describing how you will judge it on this story (G-Eval chain-of-thought).
2. Then output EXACTLY ONE JSON object on its own, with no surrounding prose, matching this schema:

{{"scores": {{"age_fit": <1-5>, "safety": <1-5>, "narrative_arc": <1-5>, "engagement": <1-5>, "calmness": <1-5>}}, "weakest": "<criterion_key>", "critique": "<2-4 sentence actionable feedback targeting the weakest>", "overall": <float average of the 5 scores>}}

Rules:
- "weakest" MUST be one of: age_fit, safety, narrative_arc, engagement, calmness.
- "critique" must be concrete and actionable, focused on the weakest dimension only.
- "overall" is the arithmetic mean of the 5 scores, rounded to 2 decimals.
- If safety scores below 4, the story FAILS regardless of other scores — say so in the critique.
"""
    user = f"Story to evaluate:\n\n{story}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_refiner_prompt(
    category: str,
    prev_story: str,
    judge_critique: str,
    weakest: str,
    kid_reactions: Optional[str],
) -> list[dict]:
    """Build the refiner messages — same persona, surgical revision (arXiv:2303.17651)."""
    persona = STORYTELLER_PERSONAS[category]
    weakest_desc = RUBRIC_CRITERIA.get(weakest, "")

    kid_block = ""
    if kid_reactions:
        # Proxy audience feedback (arXiv:2305.14930).
        kid_block = f"\n\nA 7-year-old listener also said:\n{kid_reactions}\n"

    system = f"""{persona}

You are revising your previous story. The judge identified ONE weakest dimension; fix ONLY that.

DO NOT rewrite the whole story. Keep the structure, characters, and pacing of every part that was already good. Make minimal, targeted edits to the sentences and beats that specifically address the weakest dimension. Output the FULL revised story (250-400 words), labeled "STORY:".
"""
    user = f"""Weakest dimension: {weakest} — {weakest_desc}

Judge critique:
{judge_critique}
{kid_block}
Previous story:
{prev_story}

Now write the revised STORY: that fixes only the weakest dimension."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_kid_reaction_prompt(story: str) -> list[dict]:
    """Kid Reaction Simulator — LLM plays a 7-year-old (arXiv:2305.14930)."""
    system = """You are a 7-year-old child who just listened to a bedtime story. Speak the way a real 7-year-old would — short sentences, simple words, honest feelings.

Output EXACTLY 3 short reactions, one per line, in this order:
Line 1 — your favorite part (start with "I liked")
Line 2 — something that confused you (start with "I didn't get")
Line 3 — a wish about the story (start with "I wish")

No numbering, no extra commentary, no JSON. Just three lines."""
    user = f"Here is the story you just heard:\n\n{story}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


JSON_FIX_PROMPT: str = """The previous response was not valid JSON. Re-emit ONLY the JSON object, no prose, no markdown fences, no explanation.

Schema:
{"scores": {"age_fit": <1-5>, "safety": <1-5>, "narrative_arc": <1-5>, "engagement": <1-5>, "calmness": <1-5>}, "weakest": "<criterion_key>", "critique": "<2-4 sentence actionable feedback>", "overall": <float>}

Output the JSON object now."""
