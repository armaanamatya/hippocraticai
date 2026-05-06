"""Deterministic readability metrics for English text (pure stdlib).
Used because arXiv:2510.24250 found LLMs generate text too linguistically advanced for ages 5-10 and fail to self-detect this, so the age-fit check must not be an LLM call.
"""

import re
from typing import Dict, List

TARGET_FK_GRADE_MAX: float = 4.0
TARGET_AVG_SENTENCE_LEN_MAX: float = 14.0
TARGET_HARD_VOCAB_PCT_MAX: float = 5.0

_VOWELS = set("aeiouy")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_WORD_RE = re.compile(r"[A-Za-z']+")


def count_syllables(word: str) -> int:
    """Estimate syllable count via vowel-cluster heuristic with silent-e adjustment."""
    word = word.lower()
    if not word:
        return 1
    clusters = 0
    in_vowel = False
    for ch in word:
        if ch in _VOWELS:
            if not in_vowel:
                clusters += 1
                in_vowel = True
        else:
            in_vowel = False
    if word.endswith("e") and len(word) > 1 and clusters > 1:
        clusters -= 1
    return max(1, clusters)


def split_sentences(text: str) -> List[str]:
    """Split text on runs of .!? and return non-empty stripped sentences."""
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def tokenize_words(text: str) -> List[str]:
    """Extract alpha-only word tokens, lowercased."""
    if not text:
        return []
    return [w.lower() for w in _WORD_RE.findall(text)]


def flesch_kincaid_grade(text: str) -> float:
    """Compute the Flesch-Kincaid grade level for the given text."""
    words = tokenize_words(text)
    sentences = split_sentences(text)
    if not words or not sentences:
        return 0.0
    total_syllables = sum(count_syllables(w) for w in words)
    return 0.39 * (len(words) / len(sentences)) + 11.8 * (total_syllables / len(words)) - 15.59


def compute_metrics(text: str) -> Dict[str, float]:
    """Compute the full readability metrics dict for the given text."""
    empty = {
        "total_words": 0,
        "total_sentences": 0,
        "avg_sentence_len": 0.0,
        "fk_grade": 0.0,
        "hard_vocab_pct": 0.0,
        "longest_sentence_words": 0,
    }
    if not text or not text.strip():
        return empty
    sentences = split_sentences(text)
    words = tokenize_words(text)
    if not sentences or not words:
        return empty
    total_words = len(words)
    total_sentences = len(sentences)
    avg_sentence_len = total_words / total_sentences
    hard_words = sum(1 for w in words if count_syllables(w) > 3)
    hard_vocab_pct = 100.0 * hard_words / total_words
    longest_sentence_words = max(len(tokenize_words(s)) for s in sentences)
    return {
        "total_words": total_words,
        "total_sentences": total_sentences,
        "avg_sentence_len": round(avg_sentence_len, 2),
        "fk_grade": round(flesch_kincaid_grade(text), 2),
        "hard_vocab_pct": round(hard_vocab_pct, 2),
        "longest_sentence_words": longest_sentence_words,
    }


def format_metrics_for_judge(m: Dict[str, float]) -> str:
    """Render metrics as a single human-readable paragraph for a judge prompt."""
    return (
        f"Readability metrics - {m['total_words']} words across "
        f"{m['total_sentences']} sentences "
        f"(avg {m['avg_sentence_len']} words/sentence; "
        f"longest {m['longest_sentence_words']} words). "
        f"Approx. Flesch-Kincaid grade {m['fk_grade']}. "
        f"Hard-vocab (>3 syllables) is {m['hard_vocab_pct']}% of words."
    )


if __name__ == "__main__":
    sample = "The cat sat on the mat. It was a sunny day. The little bird sang a happy song."
    print(compute_metrics(sample))
