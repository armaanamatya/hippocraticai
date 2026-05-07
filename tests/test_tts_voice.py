"""Unit tests for tts_voice — all HTTP is mocked; no API calls."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Project root on sys.path so we can import tts_voice.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tts_voice import (  # noqa: E402
    MAX_CHARS_PER_REQUEST,
    TTS_ENDPOINT,
    TTSError,
    chunk_text,
    extract_story_block,
    synthesize,
)


# ---------- extract_story_block ----------

def test_extract_strips_outline_block():
    full = (
        "OUTLINE:\n"
        "- Setup: forest\n"
        "- Problem: lost acorn\n\n"
        "STORY:\n"
        "Once upon a time in a cozy clearing..."
    )
    assert extract_story_block(full) == "Once upon a time in a cozy clearing..."


def test_extract_falls_back_when_no_marker():
    full = "Just a story with no marker."
    assert extract_story_block(full) == "Just a story with no marker."


def test_extract_handles_extra_whitespace():
    full = "OUTLINE: x\nSTORY:\n\n   Hello.   \n\n"
    assert extract_story_block(full) == "Hello."


# ---------- chunk_text ----------

def test_chunk_short_returns_single():
    assert chunk_text("short") == ["short"]


def test_chunk_long_paragraphs():
    para = "A " * 2500  # ~5000 chars, well over the 4096 cap
    long_text = para + "\n\n" + para
    chunks = chunk_text(long_text)
    assert len(chunks) > 1
    assert all(len(c) <= MAX_CHARS_PER_REQUEST for c in chunks)


def test_chunk_at_paragraph_boundary():
    # Two paragraphs, each well under cap; should split cleanly into 1 chunk
    # (or possibly 2 depending on lengths) — assert no truncation.
    text = ("A " * 500) + "\n\n" + ("B " * 500)
    chunks = chunk_text(text)
    rejoined = "".join(c.replace("\n\n", "") for c in chunks)
    # Rejoined content should preserve the same letter set.
    assert rejoined.count("A") == 500
    assert rejoined.count("B") == 500


# ---------- synthesize: payload shape ----------

def test_synthesize_posts_correct_payload(tmp_path):
    fake_mp3 = b"\xff\xfb" + b"\x00" * 100
    mock_resp = MagicMock(status_code=200, content=fake_mp3)

    with patch("tts_voice.requests.post", return_value=mock_resp) as mock_post:
        out = synthesize("Hello.", tmp_path / "test.mp3", api_key="sk-test")

    assert out == tmp_path / "test.mp3"
    assert out.read_bytes() == fake_mp3
    args, kwargs = mock_post.call_args
    assert args[0] == TTS_ENDPOINT
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
    payload = kwargs["json"]
    assert payload["model"] == "tts-1"
    assert payload["voice"] == "shimmer"
    assert payload["input"] == "Hello."
    assert payload["response_format"] == "mp3"
    assert payload["speed"] == 0.9


# ---------- synthesize: error paths ----------

def test_synthesize_missing_key_raises_before_http(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("tts_voice.requests.post") as mock_post:
        with pytest.raises(TTSError, match="OPENAI_API_KEY"):
            synthesize("Hello.", tmp_path / "test.mp3")
        assert mock_post.call_count == 0


def test_synthesize_placeholder_key_raises(tmp_path):
    with patch("tts_voice.requests.post") as mock_post:
        with pytest.raises(TTSError, match="OPENAI_API_KEY"):
            synthesize("Hello.", tmp_path / "test.mp3", api_key="sk-REPLACE_ME-xxx")
        assert mock_post.call_count == 0


def test_synthesize_401_raises_clear_error(tmp_path):
    mock_resp = MagicMock(status_code=401, text="invalid")
    with patch("tts_voice.requests.post", return_value=mock_resp):
        with pytest.raises(TTSError, match="auth"):
            synthesize("Hello.", tmp_path / "test.mp3", api_key="sk-test")


def test_synthesize_retries_once_on_500_then_succeeds(tmp_path):
    mock_500 = MagicMock(status_code=500, text="server error")
    mock_200 = MagicMock(status_code=200, content=b"audio")
    with patch("tts_voice.requests.post", side_effect=[mock_500, mock_200]) as mock_post:
        with patch("tts_voice.time.sleep"):  # don't actually sleep in tests
            synthesize("Hello.", tmp_path / "test.mp3", api_key="sk-test")
    assert mock_post.call_count == 2


def test_synthesize_empty_text_raises(tmp_path):
    with pytest.raises(TTSError, match="Empty"):
        synthesize("   \n\n  ", tmp_path / "test.mp3", api_key="sk-test")


def test_synthesize_refuses_to_chunk_non_mp3(tmp_path):
    long_text = "A " * 5000  # >4096
    with pytest.raises(TTSError, match="mp3"):
        synthesize(
            long_text,
            tmp_path / "test.wav",
            api_key="sk-test",
            response_format="wav",
        )
