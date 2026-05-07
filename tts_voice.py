"""Bedtime-mode TTS — turns the final story into an MP3 via OpenAI's TTS-1.

We POST directly to https://api.openai.com/v1/audio/speech with `requests`
because the project pins `openai<1.0.0`; the `audio.speech` SDK namespace
was added in v1.x. Defaults are tuned for bedtime listening:
  - voice='shimmer' (soft, intimate timbre)
  - speed=0.9 (slightly slower for sleepy pacing)
  - model='tts-1' ($15/1M chars, ~0.5s latency, sufficient for spoken word)

Why this matters for the takehome: Hippocratic AI's product (Polaris) is
voice-first. Adding bedtime narration is the highest-signal alignment with
how they actually deploy LLMs in production.

OpenAI TTS API reference:
https://platform.openai.com/docs/api-reference/audio/createSpeech
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

import requests

TTS_ENDPOINT: str = "https://api.openai.com/v1/audio/speech"
MAX_CHARS_PER_REQUEST: int = 4096  # OpenAI hard cap on `input` field

# Voices ranked subjectively for bedtime calmness (shimmer first).
# Source: own listening + community comparisons cited in README.
BEDTIME_VOICES: tuple[str, ...] = ("shimmer", "sage", "nova", "fable", "alloy", "echo", "onyx")


class TTSError(RuntimeError):
    """Raised when synthesis fails. Caller is expected to print + continue,
    not crash — the text story is the primary deliverable; audio is a bonus."""


def extract_story_block(full_output: str) -> str:
    """Return only text after the 'STORY:' marker; fall back to full text.

    The storyteller prompt enforces an 'OUTLINE:' / 'STORY:' structure;
    we only want the story read aloud, not the structural outline.
    """
    if "STORY:" in full_output:
        return full_output.split("STORY:", 1)[1].strip()
    return full_output.strip()


def _word_split(text: str, max_chars: int) -> list[str]:
    """Last-resort hard split at word boundaries when a single 'sentence'
    exceeds max_chars (i.e. text has no punctuation)."""
    words = text.split(" ")
    out: list[str] = []
    buf = ""
    for w in words:
        if len(buf) + len(w) + 1 > max_chars and buf:
            out.append(buf)
            buf = ""
        buf += (" " if buf else "") + w
    if buf:
        out.append(buf)
    return out


def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_REQUEST) -> list[str]:
    """Split on paragraph then sentence then word boundaries; never mid-word.

    MP3 frames are concat-safe so multiple chunks can be glued together as
    raw bytes. Other formats (wav/flac) are NOT concat-safe — synthesize()
    refuses to chunk in those cases.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    buf = ""

    for para in paragraphs:
        if len(para) > max_chars:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            pieces: list[str] = []
            for sent in sentences:
                if len(sent) <= max_chars:
                    pieces.append(sent)
                else:
                    pieces.extend(_word_split(sent, max_chars))
            for piece in pieces:
                if len(buf) + len(piece) + 1 > max_chars and buf:
                    chunks.append(buf.strip())
                    buf = ""
                buf += (" " if buf else "") + piece
            continue
        if len(buf) + len(para) + 2 > max_chars:
            if buf:
                chunks.append(buf.strip())
            buf = para
        else:
            buf += ("\n\n" if buf else "") + para

    if buf:
        chunks.append(buf.strip())
    return chunks


def synthesize(
    text: str,
    out_path: Path,
    voice: str = "shimmer",
    model: str = "tts-1",
    response_format: str = "mp3",
    speed: float = 0.9,
    api_key: Optional[str] = None,
    timeout: float = 60.0,
) -> Path:
    """POST text to OpenAI TTS, write audio bytes to out_path, return out_path.

    Raises TTSError on missing key, empty input, or HTTP failure (after one
    retry on 429/5xx). Caller should catch + warn, not crash.
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key or key.startswith("sk-REPLACE_ME"):
        raise TTSError("OPENAI_API_KEY missing — cannot synthesize audio.")

    text = text.strip()
    if not text:
        raise TTSError("Empty text — nothing to synthesize.")

    chunks = chunk_text(text)
    if len(chunks) > 1 and response_format != "mp3":
        raise TTSError(
            f"Chunking >{MAX_CHARS_PER_REQUEST} chars only supported for mp3; "
            f"got format={response_format!r}."
        )

    audio_bytes = b""
    for chunk in chunks:
        audio_bytes += _request_one(
            chunk, voice, model, response_format, speed, key, timeout
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio_bytes)
    return out_path


def _request_one(
    text: str,
    voice: str,
    model: str,
    response_format: str,
    speed: float,
    api_key: str,
    timeout: float,
) -> bytes:
    """Single POST with one retry on 429/5xx; mirrors call_model retry pattern."""
    payload = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": response_format,
        "speed": speed,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    for attempt in (0, 1):
        try:
            resp = requests.post(
                TTS_ENDPOINT, json=payload, headers=headers, timeout=timeout
            )
        except requests.RequestException as e:
            if attempt == 0:
                time.sleep(2)
                continue
            raise TTSError(f"TTS network error: {e}") from e

        if resp.status_code == 200:
            return resp.content
        if resp.status_code in (429, 500, 502, 503, 504) and attempt == 0:
            time.sleep(2)
            continue
        if resp.status_code == 401:
            raise TTSError("TTS auth failed — check OPENAI_API_KEY.")
        raise TTSError(f"TTS HTTP {resp.status_code}: {resp.text[:200]}")

    raise TTSError("TTS retries exhausted.")
