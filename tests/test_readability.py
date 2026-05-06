"""Unit tests for readability.py — pure-stdlib, no API key needed.

Run from project root:  pytest tests/test_readability.py -v
"""
import sys
from pathlib import Path

# Make project root importable when pytest is invoked from elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from readability import (
    TARGET_AVG_SENTENCE_LEN_MAX,
    TARGET_FK_GRADE_MAX,
    TARGET_HARD_VOCAB_PCT_MAX,
    compute_metrics,
    count_syllables,
    flesch_kincaid_grade,
    format_metrics_for_judge,
    split_sentences,
    tokenize_words,
)


# ----------------------- count_syllables --------------------------------


@pytest.mark.parametrize(
    "word,expected",
    [
        ("cat", 1),
        ("the", 1),
        ("happy", 2),
        ("banana", 3),
        ("elephant", 3),
        ("cake", 1),       # silent-e correction
        ("strength", 1),
        ("", 1),           # minimum guard
        ("rhythm", 1),     # no aeiouy clusters → falls back to min 1
    ],
)
def test_count_syllables(word, expected):
    assert count_syllables(word) == expected


def test_count_syllables_is_case_insensitive():
    assert count_syllables("Happy") == count_syllables("happy") == 2


# ----------------------- split_sentences --------------------------------


def test_split_sentences_basic():
    assert split_sentences("Hi. Hello! Why?") == ["Hi", "Hello", "Why"]


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_sentences_no_terminator():
    # Single chunk with no .!? still counts as one sentence.
    assert split_sentences("just one sentence") == ["just one sentence"]


def test_split_sentences_collapses_runs():
    # Consecutive .!? characters collapse into one boundary.
    assert split_sentences("Wait... what?!") == ["Wait", "what"]


# ----------------------- tokenize_words ---------------------------------


def test_tokenize_words_basic():
    assert tokenize_words("Hello, world!") == ["hello", "world"]


def test_tokenize_words_keeps_apostrophes():
    assert tokenize_words("Don't can't won't") == ["don't", "can't", "won't"]


def test_tokenize_words_strips_digits_and_symbols():
    assert tokenize_words("hello 123 world #!") == ["hello", "world"]


def test_tokenize_words_empty():
    assert tokenize_words("") == []


# ----------------------- flesch_kincaid_grade ---------------------------


def test_fk_grade_empty_is_zero():
    # No division-by-zero on empty input.
    assert flesch_kincaid_grade("") == 0.0


def test_fk_grade_simple_text_is_low():
    # Simple text for kids should land in single-digit grade.
    text = "The cat sat on the mat. The dog ran fast. The sun is hot."
    assert flesch_kincaid_grade(text) < 4.0


def test_fk_grade_complex_text_is_higher_than_simple():
    simple = "The cat sat. The dog ran. The bird sang."
    complex_ = (
        "The metropolitan administration announced unprecedented infrastructural "
        "developments encompassing transportation, communication, and educational "
        "facilities throughout the surrounding municipalities."
    )
    assert flesch_kincaid_grade(complex_) > flesch_kincaid_grade(simple)


# ----------------------- compute_metrics --------------------------------


def test_compute_metrics_empty():
    m = compute_metrics("")
    assert m["total_words"] == 0
    assert m["total_sentences"] == 0
    assert m["avg_sentence_len"] == 0.0
    assert m["fk_grade"] == 0.0
    assert m["hard_vocab_pct"] == 0.0
    assert m["longest_sentence_words"] == 0


def test_compute_metrics_whitespace_only():
    assert compute_metrics("   \n\t  ")["total_words"] == 0


def test_compute_metrics_basic_shape():
    text = "The cat sat on the mat. It was a sunny day. The little bird sang a happy song."
    m = compute_metrics(text)
    assert m["total_words"] == 18
    assert m["total_sentences"] == 3
    assert m["avg_sentence_len"] == pytest.approx(6.0, abs=0.01)
    assert m["longest_sentence_words"] == 7  # "The little bird sang a happy song"
    assert isinstance(m["fk_grade"], float)
    assert m["hard_vocab_pct"] == 0.0


def test_compute_metrics_flags_long_sentence():
    short = "Cat sits. Dog runs. Bird flies."
    long_one = "The remarkably curious cat slowly sat on the comfortable mat near the open door."
    assert (
        compute_metrics(long_one)["longest_sentence_words"]
        > compute_metrics(short)["longest_sentence_words"]
    )


def test_compute_metrics_returns_floats_rounded_to_2dp():
    text = "Hello there. How are you today my dear friend."
    m = compute_metrics(text)
    # avg_sentence_len rounded to 2 decimals — make sure no extra precision sneaks in.
    assert round(m["avg_sentence_len"], 2) == m["avg_sentence_len"]
    assert round(m["fk_grade"], 2) == m["fk_grade"]
    assert round(m["hard_vocab_pct"], 2) == m["hard_vocab_pct"]


# ----------------------- format_metrics_for_judge ----------------------


def test_format_metrics_string_contains_key_numbers():
    m = {
        "total_words": 247,
        "total_sentences": 18,
        "avg_sentence_len": 13.7,
        "fk_grade": 4.1,
        "hard_vocab_pct": 4.5,
        "longest_sentence_words": 24,
    }
    s = format_metrics_for_judge(m)
    assert "247" in s
    assert "18" in s
    assert "13.7" in s
    assert "4.1" in s
    assert "4.5" in s
    assert "24" in s


# ----------------------- module constants ------------------------------


def test_target_thresholds_are_sane():
    # These are public constants used by the judge — sanity-bound them.
    assert 0 < TARGET_FK_GRADE_MAX <= 6.0
    assert 0 < TARGET_AVG_SENTENCE_LEN_MAX <= 20.0
    assert 0 < TARGET_HARD_VOCAB_PCT_MAX <= 15.0
