# Sources

Research underpinning the design choices in `IMPLEMENTATION_PLAN.md`. All papers were indexed into a NotebookLM notebook ("Bedtime Story LLM — Judge and Persona Research") and synthesized in `briefing-doc.md`.

---

## LLM-as-a-Judge — foundational

| # | Paper | arXiv | What we took from it |
|---|-------|-------|----------------------|
| 1 | A Survey on LLM-as-a-Judge | [2411.15594](https://arxiv.org/abs/2411.15594) | Catalogs position, verbosity, self-enhancement, authority biases. Mitigations: rubrics > free-form, CoT before scoring, reference-free metrics. |
| 2 | G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment | [2303.16634](https://arxiv.org/abs/2303.16634) | The judge design we use: model writes its own evaluation steps from criteria, then form-fills scores. Spearman 0.514 with humans, beats BLEU/ROUGE. |
| 3 | Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge | [2406.07791](https://arxiv.org/abs/2406.07791) | Position bias is largest when quality gap is small. Mitigation: avoid pairwise; score absolute. We do absolute-only scoring. |

## Iterative refinement (single-model)

| # | Paper | arXiv | What we took from it |
|---|-------|-------|----------------------|
| 4 | Self-Refine: Iterative Refinement with Self-Feedback (Madaan et al.) | [2303.17651](https://arxiv.org/abs/2303.17651) | The core loop: same model generates → critiques → refines. ~20% absolute lift, no extra training. Perfect fit for the `gpt-3.5-turbo`-only constraint. |
| 5 | Constitutional AI: Harmlessness from AI Feedback (Anthropic) | [2212.08073](https://arxiv.org/abs/2212.08073) | We use the *critique-revise* mechanism at inference time (not training). Codified safety principles into the judge prompt as a kid-safe "constitution". |

## Multi-agent / debate alternatives (considered, rejected)

| # | Paper | arXiv | What we took from it |
|---|-------|-------|----------------------|
| 6 | ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate | [2308.07201](https://arxiv.org/abs/2308.07201) | Debate works when agents are *heterogeneous*. With a single model, simulated debate yields diminishing returns and burns tokens. → We do not implement debate. |

## Persona / role-play

| # | Paper | arXiv | What we took from it |
|---|-------|-------|----------------------|
| 7 | Two Tales of Persona in LLMs: A Survey of Role-Playing and Personalization | [2406.01171](https://arxiv.org/abs/2406.01171) | Frames the role-play vs. personalization split. Validates persona for creative open-ended generation. |
| 8 | In-Context Impersonation Reveals Large Language Models' Strengths and Biases (Salewski et al.) | [2305.14930](https://arxiv.org/abs/2305.14930) | Expert/role personas measurably help open-ended tasks. Crucially: child personas recover developmental exploration patterns → justifies our **Kid Reaction Simulator**. |
| 9 | When "A Helpful Assistant" Is Not Really Helpful: Personas in System Prompts Do Not Improve Performances of LLMs | [2311.10054](https://arxiv.org/abs/2311.10054) | Personas don't help — and often hurt — *objective* evaluation. → Our judge has **no persona**, only a rubric. |

## Age-appropriate content for children

| # | Paper | arXiv | What we took from it |
|---|-------|-------|----------------------|
| 10 | Evaluating LLMs on Generating Age-Appropriate Child-Like Conversations | [2510.24250](https://arxiv.org/abs/2510.24250) | LLMs systematically generate text **too linguistically advanced** for ages 5–10, and **fail to self-detect this**. → Our readability gate is **deterministic Python**, not an LLM call. |

---

## Generated artifacts (in this repo)

- **`briefing-doc.md`** — NotebookLM-generated synthesis, 8 KB. Executive summary + sections on judge utility, bias mitigations, persona trade-offs, age rubric, and end-to-end recipe.
- **NotebookLM saved note** *"FAQ: Building a Kid-Safe Bedtime Story Generator"* — Q1–Q9 walkthrough inside the notebook (not in repo, lives at notebooklm.google.com).

---

## How the sources map to the design

| Design decision in `IMPLEMENTATION_PLAN.md` | Driven by source(s) |
|---------------------------------------------|---------------------|
| Self-Refine loop instead of multi-agent debate | #4, #6 |
| Persona on storyteller, none on judge | #7, #8, #9 |
| G-Eval style CoT + form-filling judge | #2, #1 |
| Absolute scoring, no pairwise comparison | #3 |
| Kid-safe Constitution embedded in judge prompt | #5 |
| Deterministic readability gate (not LLM-based) | #10 |
| Kid Reaction Simulator (proxy audience) | #8 |
