# Implementation Plan — Bedtime Story Generator (Hippocratic AI Takehome)

> **Constraints:** OpenAI `gpt-3.5-turbo` only · ≤ 2–3 hours of work · key in `.env` (gitignored) · ages 5–10 · LLM-as-judge required.

---

## 1. Architecture Overview

A single-model **Constitutional Self-Refine** pipeline with a category-routed storyteller, a rubric-driven judge, and a deterministic local readability check. Every design choice below is grounded in a specific research finding from the indexed NotebookLM corpus.

```
                    ┌─────────────────────────────┐
                    │   User: "tell me a story…"  │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
   GPT-3.5  ───────►│  1. Categorizer             │  → arc ∈ {adventure,
                    │     (zero-shot classify)    │     friendship, curiosity,
                    └──────────────┬──────────────┘     calming-bedtime}
                                   ▼
                    ┌─────────────────────────────┐
   GPT-3.5  ───────►│  2. Storyteller (PERSONA)   │  ← arc-specific persona
                    │     "warm grandparent who…" │     + outline scaffolding
                    └──────────────┬──────────────┘
                                   │  draft v_n
                                   ▼
                    ┌─────────────────────────────┐
   (local Python) ─►│  3. Readability Gate        │  Flesch-Kincaid,
                    │     deterministic check     │  sentence-len, hard-vocab
                    └──────────────┬──────────────┘
                                   │  metrics_n
                                   ▼
                    ┌─────────────────────────────┐
   GPT-3.5  ───────►│  4. Rubric Judge (NO        │  G-Eval style CoT
                    │     PERSONA) + Constitution │  5 criteria × 1–5 scale
                    └──────────────┬──────────────┘  → score + critique
                                   ▼
                          ┌────────┴────────┐
                          │  Pass gate?     │
                          │  score ≥ 4.0    │
                          │  AND iter ≤ 2   │
                          └───┬─────────┬───┘
                          NO  │         │  YES
                              ▼         ▼
            ┌──────────────────────┐  ┌──────────────────────┐
   GPT-3.5 ►│ 5. Refiner           │  │ 6. (BONUS) Kid       │
            │    same model + crit │  │    Reaction Simulator│ ← surprise
            └──────────┬───────────┘  └──────────┬───────────┘
                       │ v_{n+1}                 │ kid_feedback
                       └────► loop ◄─────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │  7. Final story → user      │
                    │     + optional feedback turn│
                    └─────────────────────────────┘
```

(Final submission will render this as Mermaid in `README.md` for crisp display.)

---

## 2. Component-by-Component Design

### 2.1 Categorizer — *zero-shot, single LLM call*

- **Why:** README explicitly suggests "categorize the request and use a tailored generation strategy." Lets us swap in a category-specific persona and outline.
- **Prompt:** "Classify this story request into exactly one category: ADVENTURE | FRIENDSHIP | CURIOSITY | CALMING. Output only the label."
- **Cost:** ~30 tokens, one call.

### 2.2 Storyteller — *persona-driven generation*

- **Why persona helps here:** Salewski et al. (arXiv 2305.14930) — expert/role personas measurably help open-ended creative tasks; "Two Tales of Persona" (2406.01171) catalogs the role-play branch positively for creative output.
- **Persona by category** (each ~3 sentences):
  - Adventure → *"Mira, a folk-tale storyteller who tells gentle hero-journey tales where the bravery is small and kind."*
  - Friendship → *"Grandparent storyteller who weaves cozy stories about kindness between unlikely friends."*
  - Curiosity → *"Curious naturalist who shows wonder in everyday things — leaves, snails, raindrops."*
  - Calming → *"Lullaby-voiced storyteller. Rhythmic, sensory, slow. Always ends with the character safe and resting."*
- **Scaffolded outline:** Storyteller is asked to produce a 4-beat outline first (setup → small problem → kind resolution → wind-down), then write the story. Story-arc structure was an explicit README hint.
- **Length:** 250–400 words target. Brief but complete.

### 2.3 Readability Gate — *deterministic, no LLM*

- **Why deterministic:** arXiv 2510.24250 finds LLMs systematically generate text *too linguistically advanced* for ages 5–10 and **fail to self-recognize this**. So the readability check must not be an LLM.
- **Computed in Python** (zero extra deps — pure stdlib):
  - Avg sentence length (target ≤ 12 words for 5-yr feel, ≤ 16 for 10-yr feel)
  - Approximate Flesch-Kincaid grade (vowel-cluster syllable count) — target ≤ 4.0
  - Hard-vocab count: tokens with > 3 syllables (target ≤ 3% of words)
- **Output:** structured metrics dict, fed into the judge's context as objective evidence.

### 2.4 Rubric Judge — *G-Eval style, no persona*

- **Why no persona:** "When 'A Helpful Assistant' Is Not Really Helpful" (arXiv 2311.10054) — personas don't improve and often hurt objective evaluation. Keep the judge plain and rubric-focused.
- **Why CoT + form-filling:** G-Eval (2303.16634) — Spearman 0.514 with humans, beats reference metrics handily. The judge first writes evaluation steps for each criterion, then fills a JSON form with scores.
- **5 criteria, 1–5 each:**
  1. **Age-fit** — vocabulary + sentence complexity match ages 5–10 (uses readability metrics from step 2.3 as evidence)
  2. **Safety** — no scary, violent, inappropriate, or distressing content (Constitutional principles, see 2.5)
  3. **Narrative arc** — clear setup, gentle stakes, satisfying resolution
  4. **Engagement** — characters feel alive, sensory details, ≥ 1 line of dialogue
  5. **Calmness** — bedtime-appropriate pacing, soothing close
- **Output:** strict JSON `{scores: {...}, weakest: "...", critique: "...", overall: float}`.
- **Constraint:** no pairwise comparison. Score the current draft alone — sidesteps position bias (arXiv 2406.07791) and self-enhancement bias.

### 2.5 Constitution (in judge prompt)

A short list of explicit principles, embedded in the judge's system prompt — Anthropic-CAI style (arXiv 2212.08073), but used at *inference* not training:

- The story must not contain violence, weapons, death, or characters in lasting peril.
- No frightening imagery (monsters that aren't gentle, darkness as menacing, abandonment).
- No romantic or adult themes.
- Animals should not be harmed; conflicts resolve with kindness, cleverness, or help.
- The ending must leave the listener feeling safe and ready to sleep.

The judge is told to scan the draft for each principle and dock the Safety score for any violation.

### 2.6 Refiner — *same model, applies critique*

- **Why Self-Refine and not multi-agent debate:** Madaan et al. (arXiv 2303.17651) report ~20% gain from a single critique–refine pass with one model. ChatEval-style debate (2308.07201) needs *heterogeneous* models — with only `gpt-3.5-turbo`, simulated debate among identical models yields diminishing returns and burns tokens.
- **Loop control:** stop when `overall ≥ 4.0` OR `iter == 2`. Hard cap at 2 to keep cost/latency sane.
- **Prompt:** receives previous draft + judge critique + readability metrics, asked to revise *only the weakest criterion's issues* without rewriting the rest.

### 2.7 Bonus — *Kid Reaction Simulator* (the "surprise us" piece)

- **Idea:** a separate LLM call adopts the persona of a 7-year-old hearing the story, and produces 3 short reactions ("My favorite part was…", "I didn't get…", "I wish there was more…").
- **Why this is grounded, not gimmicky:** Salewski et al. specifically show that prompting an LLM to be *a child* recovers human-like developmental exploration patterns. This is one of the few legitimate uses of persona for evaluation — not as a judge of quality, but as a *proxy audience* whose reactions surface engagement gaps the rubric judge can't see (e.g., "I didn't get why the bunny was sad").
- **How it feeds in:** kid-reactions are appended to the next-iteration refiner prompt. Cheap, surprising, and well-cited.

### 2.8 Optional feedback turn

After the final story, prompt the user: *"Want me to make it more [adventurous | calmer | sillier], or change anything?"* If they reply, restart at the Refiner with their wish as critique.

---

## 3. File Layout

```
.
├── main.py                  # Orchestrator — keeps existing call_model signature
├── prompts.py               # All prompts: personas, rubric, constitution, judge schema
├── readability.py           # Pure-stdlib readability metrics
├── requirements.txt         # openai, python-dotenv  (plus existing constraints)
├── .env                     # OPENAI_API_KEY=...   (gitignored)
├── .env.example             # template, committed
├── .gitignore               # excludes .env, venvs, pycache
├── README.md                # Replaces or extends — design notes + Mermaid diagram
└── IMPLEMENTATION_PLAN.md   # this file
```

---

## 4. Build Order (matches the 2–3 hour budget)

| # | Task                                                                  | Est. |
|---|-----------------------------------------------------------------------|------|
| 1 | `requirements.txt` + load `.env` via `python-dotenv` in `main.py`     | 10m  |
| 2 | `prompts.py` — personas, judge rubric, constitution, JSON schema      | 30m  |
| 3 | `readability.py` — sentence/syllable/hard-vocab counters              | 20m  |
| 4 | Categorizer + Storyteller calls in `main.py`                          | 20m  |
| 5 | Judge call with JSON parsing + retry on bad JSON                      | 25m  |
| 6 | Refiner loop + stopping condition                                     | 20m  |
| 7 | Kid Reaction Simulator (bonus)                                        | 15m  |
| 8 | `README.md` Mermaid diagram + how-to-run + design rationale           | 20m  |
| 9 | Manual smoke tests with 4 prompts (one per category) + tune thresholds| 30m  |

**Buffer:** ~20 min for the inevitable surprises.

---

## 5. Token / Cost Sketch (back-of-envelope)

Per story (typical, 2 refine iters):
- Categorizer: ~80 tokens
- Storyteller (×2): ~1,500 tokens
- Judge (×2): ~1,200 tokens
- Refiner (×1): ~700 tokens
- Kid simulator: ~300 tokens
- **Total ~3,800 tokens** × `gpt-3.5-turbo` rate → fractions of a cent. Well within takehome budget.

---

## 6. Risks & Mitigations

| Risk                                                | Mitigation                                                                 |
|-----------------------------------------------------|----------------------------------------------------------------------------|
| Judge returns malformed JSON                        | Strict schema in prompt + one parse-retry call asking it to fix only JSON  |
| `gpt-3.5-turbo` writes for ages 12+ (known issue)   | Deterministic readability gate forces explicit metrics into judge context  |
| Score never reaches 4.0 → infinite loop             | Hard cap at 2 refine iterations; return best-scoring draft                 |
| User key leaks                                      | `.env` gitignored from the start; `.env.example` is the only committed one |
| Pre-`v1.0.0` `openai` SDK in skeleton (`ChatCompletion`) | Pin in `requirements.txt`: `openai<1.0.0`                              |

---

## 7. What I'd Build Next (the docstring at top of `main.py`)

Placeholder content for the existing module docstring — to be polished after the build:

> *With 2 more hours I'd add (a) a tiny SQLite-backed memory of past stories per child so a returning request like "another story about Pip" continues the same world, (b) replace the deterministic readability gate with a calibrated regression head that learns from a small set of teacher-rated stories, and (c) add an audio output via TTS with paragraph-level pause hints generated by the storyteller — turning bedtime mode into a true wind-down experience.*

---

## 8. Submission Checklist

- [ ] `main.py` runs end-to-end on a single prompt with `OPENAI_API_KEY` set in `.env`
- [ ] `.env` not committed (verify `git status` shows it ignored)
- [ ] Block diagram present in `README.md` (Mermaid)
- [ ] "What I'd build next" docstring filled in at top of `main.py`
- [ ] OpenAI model unchanged (`gpt-3.5-turbo`)
- [ ] Tested with at least one prompt per category
- [ ] All paper citations referenced in code comments where relevant (so reviewers can trace design choices)
