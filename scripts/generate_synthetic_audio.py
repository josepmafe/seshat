"""Generate a synthetic multi-speaker meeting audio file using ElevenLabs text_to_dialogue.

Uses the text_to_dialogue API which processes the full script in a single call,
giving natural cross-turn prosody and pacing (vs. concatenating separate TTS calls).
Optionally mixes in office background noise generated via text_to_sound_effects.

Usage:
    uv run --group scripts python scripts/generate_synthetic_audio.py
    uv run --group scripts python scripts/generate_synthetic_audio.py --output-path data/fixtures/audio/my_meeting.mp3
    uv run --group scripts python scripts/generate_synthetic_audio.py --script scripts/meeting_script.txt
    uv run --group scripts python scripts/generate_synthetic_audio.py --no-background-noise

The meeting script is a plain-text file where each line follows the format:
    <voice_name>: <line of dialogue>

Lines starting with '#' are treated as comments and skipped.
If --script is not provided, a short built-in sample meeting is used.
If --output-path is not provided, a UUID filename is generated under data/fixtures/audio/.

Voice names must match the VOICE_IDS mapping below.
Max 10 unique voices and ~2000 characters per request (ElevenLabs API limit).

Requires:
    ELEVENLABS_API_KEY in .env or environment.
    ffmpeg on PATH (required by pydub for MP3 mixing).
"""

from __future__ import annotations

import io
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import lameenc
import miniaudio
import numpy as np
from elevenlabs import DialogueInput, ElevenLabs
from elevenlabs.types import ModelSettingsResponseModel
from mutagen.mp3 import MP3

if TYPE_CHECKING:
    from argparse import Namespace


_SAMPLE_SCRIPT = """\
LiamNarrator: We need to pick a database for the new service. Options are Postgres or MySQL.
Joseff: Postgres has native JSONB and better full-text search. I'd go with it.
Anika: Agreed. I'd go with Postgres too. It also has a more active community.
LiamNarrator: It's settled then. Anika, can you set up the schema migrations by end of week?
Anika: Sure, I'll have it done by Friday.
Joseff: We also need to decide on the caching layer — Redis or Memcached?
Riley: We don't have the load numbers yet. I'd defer that decision. There's also a risk with the migration timeline; if staging is still down on Thursday we can't validate before Friday's deadline.
Joseff: That's a real concern. I haven't got a fixed date from the infra team yet. I'll ping them again.
"""

_DEFAULT_OUTPUT_DIR = Path("data/fixtures/audio")
_BACKGROUND_NOISE_DB = -10  # dB relative to speech
_MAX_SOUND_EFFECT_DURATION = 30.0

NOISE_PROFILES: dict[str, str] = {
    "quiet": "quiet office background noise, air conditioning hum, distant keyboard typing, occasional footsteps in corridor",
    "busy": "busy open-plan office, multiple people talking in background, keyboard typing, occasional laughter, chair movements",
    "meeting": "meeting room ambience, air conditioning, muffled conversation through walls, occasional laugh, coffee cup sounds",
}
_DEFAULT_NOISE_PROFILE = "quiet"

# Premade voices available on all ElevenLabs plans (as of 2026-06-17).
# Add entries here to use a voice by name in meeting scripts.
# All voices available on the account (as of 2026-06-18, Creator tier).
# Run `client.voices.get_all(show_legacy=False)` to refresh.
# For voices with duplicate first names, use the full key shown below.
VOICE_IDS: dict[str, str] = {
    # professional
    "Alex": "17bSMslPF4HPyQrGIXAG",
    "Anika": "ecp3DWciuUyW7BYM7II1",
    "Joseff": "3TStB8f3X3To0Uj5R7RK",
    "LiamNarrator": "bu5eKETbFKC8G702EAU4",
    "LiamStoryteller": "VCgLBmBjldJmfphyB8sZ",
    "Riley": "hA4zGnmTwX2NQiTRMt7o",
    # premade
    "Adam": "pNInz6obpgDQGcFmaJgB",
    "Alice": "Xb7hH8MSUJpSbSDYk0k2",
    "Bella": "hpp4J3VqNfWAUOO0d1Us",
    "Bill": "pqHfZKP75CvOlQylNhV4",
    "Brian": "nPczCjzI2devNBz1zQrb",
    "Callum": "N2lVS1w4EtoT3dr4eOWO",
    "Charlie": "IKne3meq5aSn9XLyUdCD",
    "Chris": "iP95p4xoKVk53GoZ742B",
    "Daniel": "onwK4e9ZLuTAKqWW03F9",
    "Eric": "cjVigY5qzO86Huf0OWal",
    "George": "JBFqnCBsd6RMkjVDRZzb",
    "Harry": "SOYHLrjzK2X1ezoPC6cr",
    "Jessica": "cgSgspJ2msm6clMCkdW9",
    "Laura": "FGY2WhTYpPnrIDTdsKH5",
    "Liam": "TX3LPaxmHKxFdv7VOQHJ",
    "Lily": "pFZP5JQG7iQjIQuC4Bku",
    "Matilda": "XrExE9yKIg1WjnnlVkGX",
    "River": "SAz9YHcvj6GT2YYXdXww",
    "Roger": "CwhRBWXzGAHq8TQ4Fs17",
    "Sarah": "EXAVITQu4vr4xnSDxMaL",
    "Will": "bIHbv24MWmeRgasZH58o",
}


def main(args: Namespace) -> None:
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")

    out_path = Path(args.output_path) if args.output_path else _DEFAULT_OUTPUT_DIR / f"{uuid.uuid4()}.mp3"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = ElevenLabs(api_key=api_key, httpx_client=httpx.Client(verify=False))

    if args.dialogue_path:
        print(f"Loading dialogue from {args.dialogue_path}...")
        dialogue_bytes = Path(args.dialogue_path).read_bytes()
    else:
        script_text = Path(args.script).read_text(encoding="utf-8") if args.script else _SAMPLE_SCRIPT
        turns = _parse_script(script_text)
        print(f"Generating {len(turns)}-turn dialogue...")
        for i, (name, inp) in enumerate(turns, 1):
            input_str = inp.text[:60] + ("..." if len(inp.text) > 60 else "")
            print(f"  [{i}/{len(turns)}] {name}: {input_str}")
        dialogue_bytes = _generate_dialogue(client, turns)
        dialogue_raw_path = out_path.with_name(out_path.stem + "_dialogue.mp3")
        dialogue_raw_path.write_bytes(dialogue_bytes)
        print(f"Raw dialogue saved to {dialogue_raw_path}")

    if args.background_noise:
        if args.noise_path:
            print(f"Loading background noise from {args.noise_path}...")
            noise_bytes = Path(args.noise_path).read_bytes()
        else:
            duration = MP3(io.BytesIO(dialogue_bytes)).info.length
            noise_prompt = NOISE_PROFILES[args.noise_profile]
            print(f"Generating background noise ({duration:.1f}s, profile: {args.noise_profile})...")
            noise_bytes = _generate_background_noise(client, duration, noise_prompt)
            noise_path = out_path.with_name(out_path.stem + "_noise.mp3")
            noise_path.write_bytes(noise_bytes)
            print(f"Background noise saved to {noise_path}")

        print("Mixing dialogue and background noise...")
        audio_bytes = _mix_audio_tracks(dialogue_bytes, noise_bytes, noise_db=args.noise_db)
    else:
        audio_bytes = dialogue_bytes

    out_path.write_bytes(audio_bytes)
    print(f"\nSaved to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def _parse_script(text: str) -> list[tuple[str, DialogueInput]]:
    turns = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Each line must be '<voice>: <text>', got: {line!r}")
        name, _, dialogue = line.partition(":")
        name = name.strip()
        voice_id = VOICE_IDS.get(name)
        if voice_id is None:
            raise ValueError(f"Unknown voice {name!r}. Add it to VOICE_IDS or use one of: {', '.join(VOICE_IDS)}")
        turns.append((name, DialogueInput(text=dialogue.strip(), voice_id=voice_id)))
    return turns


def _generate_dialogue(client: ElevenLabs, turns: list[tuple[str, DialogueInput]]) -> bytes:
    chunks = client.text_to_dialogue.convert(
        inputs=[inp for _, inp in turns],
        model_id="eleven_v3",
        language_code="en",
        output_format="mp3_44100_128",
        settings=ModelSettingsResponseModel(stability=0.5),
    )
    return b"".join(chunks)


def _generate_background_noise(client: ElevenLabs, duration_seconds: float, prompt: str) -> bytes:
    # API caps duration_seconds at 30s — generate in segments and concatenate.
    remaining = duration_seconds
    segments: list[bytes] = []
    while remaining > 0:
        seg_duration = min(remaining, _MAX_SOUND_EFFECT_DURATION)
        chunks = client.text_to_sound_effects.convert(
            text=prompt,
            duration_seconds=seg_duration,
            loop=True,
            output_format="mp3_44100_128",
        )
        segments.append(b"".join(chunks))
        remaining -= seg_duration
    return b"".join(segments)


def _mix_audio_tracks(dialogue_bytes: bytes, noise_bytes: bytes, noise_db: float = _BACKGROUND_NOISE_DB) -> bytes:
    speech, sample_rate = _mp3_to_pcm_f32(dialogue_bytes)
    noise, noise_sample_rate = _mp3_to_pcm_f32(noise_bytes)

    # Resample noise if sample rates differ.
    if noise_sample_rate != sample_rate:
        new_len = int(len(noise) * sample_rate / noise_sample_rate)
        noise = np.interp(np.linspace(0, len(noise) - 1, new_len), np.arange(len(noise)), noise)

    # Loop noise to match speech length, then trim.
    while len(noise) < len(speech):
        noise = np.concatenate([noise, noise])
    noise = noise[: len(speech)]

    # Scale noise by dB gain.
    noise = noise * (10 ** (noise_db / 20))

    mixed = np.clip(speech + noise, -1.0, 1.0)
    return _pcm_f32_to_mp3(mixed, sample_rate)


def _mp3_to_pcm_f32(mp3_bytes: bytes) -> tuple[np.ndarray, int]:
    decoded = miniaudio.decode(mp3_bytes, output_format=miniaudio.SampleFormat.FLOAT32)
    samples = np.frombuffer(decoded.samples, dtype=np.float32)
    if decoded.nchannels > 1:
        samples = samples.reshape(-1, decoded.nchannels).mean(axis=1)
    return samples, decoded.sample_rate


def _pcm_f32_to_mp3(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm_i16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(2)
    return encoder.encode(pcm_i16.tobytes()) + encoder.flush()


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate synthetic meeting audio via ElevenLabs.")
    parser.add_argument("--output-path", default=None, help="Output MP3 path (default: data/fixtures/audio/<uuid>.mp3)")
    parser.add_argument("--script", default=None, help="Path to meeting script file (voice: text lines)")
    parser.add_argument(
        "--dialogue-path", default=None, help="Reuse an existing dialogue MP3 instead of calling the API"
    )
    parser.add_argument(
        "--no-background-noise",
        dest="background_noise",
        action="store_false",
        default=True,
        help="Skip background noise generation and mixing",
    )
    parser.add_argument("--noise-path", default=None, help="Reuse an existing noise MP3 instead of calling the API")
    parser.add_argument(
        "--noise-profile",
        choices=list(NOISE_PROFILES),
        default=_DEFAULT_NOISE_PROFILE,
        help=f"Background noise profile (default: {_DEFAULT_NOISE_PROFILE})",
    )
    parser.add_argument(
        "--noise-db",
        type=float,
        default=_BACKGROUND_NOISE_DB,
        help=f"Background noise volume in dB relative to speech (default: {_BACKGROUND_NOISE_DB})",
    )
    args = parser.parse_args()

    main(args)
