from __future__ import annotations

import io
import struct
import wave

import pytest

from seshat.core.utils.audio import audio_duration_seconds, audio_duration_seconds_ceil


def _make_wav_bytes(duration_seconds: int = 1) -> bytes:
    sample_rate = 16000
    n_frames = sample_rate * duration_seconds
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
    return buf.getvalue()


def test_audio_duration_seconds_returns_float():
    result = audio_duration_seconds(_make_wav_bytes(duration_seconds=2))
    assert result == pytest.approx(2.0, abs=0.05)


def test_audio_duration_seconds_returns_none_for_invalid():
    assert audio_duration_seconds(b"\x00" * 64) is None


def test_audio_duration_seconds_ceil_rounds_up():
    result = audio_duration_seconds_ceil(_make_wav_bytes(duration_seconds=1))
    assert result == 1


def test_audio_duration_seconds_ceil_returns_none_for_invalid():
    assert audio_duration_seconds_ceil(b"\x00" * 64) is None
