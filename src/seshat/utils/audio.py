from __future__ import annotations

import io
import math

import mutagen


def audio_duration_seconds(audio_bytes: bytes) -> float | None:
    """Return audio duration in seconds, or None if mutagen cannot parse the format."""
    try:
        audio_file = mutagen.File(io.BytesIO(audio_bytes))
    except Exception:
        return None

    if audio_file is None or not hasattr(audio_file.info, "length"):
        return None
    return audio_file.info.length


def audio_duration_seconds_ceil(audio_bytes: bytes) -> int | None:
    """Return audio duration rounded up to the nearest second, or None if unparseable."""
    duration = audio_duration_seconds(audio_bytes)
    if duration is None:
        return None
    return math.ceil(duration)
