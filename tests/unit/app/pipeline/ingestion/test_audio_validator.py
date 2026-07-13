from unittest.mock import patch

import pytest

from seshat.app.pipeline.ingestion.audio_validator import AudioValidationError, AudioValidator


class TestAudioValidator:
    def test_valid_mp3_id3(self):
        data = b"ID3" + b"\x00" * 30
        ext = AudioValidator.validate_magic_bytes(data)
        assert ext == "mp3"

    def test_valid_mp3_sync(self):
        data = b"\xff\xfb" + b"\x00" * 30
        ext = AudioValidator.validate_magic_bytes(data)
        assert ext == "mp3"

    def test_valid_wav(self):
        data = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 10
        ext = AudioValidator.validate_magic_bytes(data)
        assert ext == "wav"

    def test_valid_m4a(self):
        data = b"\x00\x00\x00\x08ftyp" + b"M4A " + b"\x00" * 10
        ext = AudioValidator.validate_magic_bytes(data)
        assert ext == "m4a"

    def test_invalid_format_raises(self):
        data = b"\x00\x01\x02\x03" * 8
        with pytest.raises(AudioValidationError, match="Unsupported audio format"):
            AudioValidator.validate_magic_bytes(data)

    def test_alleged_ext_match_passes(self):
        data = b"ID3" + b"\x00" * 30
        assert AudioValidator.validate_magic_bytes(data, alleged_ext="mp3") == "mp3"

    def test_alleged_ext_with_dot_passes(self):
        data = b"ID3" + b"\x00" * 30
        assert AudioValidator.validate_magic_bytes(data, alleged_ext=".mp3") == "mp3"

    def test_alleged_ext_mismatch_raises(self):
        data = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 10  # WAV bytes
        with pytest.raises(AudioValidationError, match="Extension mismatch"):
            AudioValidator.validate_magic_bytes(data, alleged_ext="mp3")

    def test_size_check_raises_on_exceed(self):
        with pytest.raises(AudioValidationError, match="exceeds maximum"):
            AudioValidator.check_size(600 * 1024 * 1024, max_bytes=500 * 1024 * 1024)

    def test_size_check_passes(self):
        AudioValidator.check_size(100 * 1024 * 1024, max_bytes=500 * 1024 * 1024)

    def test_duration_check_raises_on_exceed(self):
        with pytest.raises(AudioValidationError, match="exceeds maximum"):
            AudioValidator.check_duration(7300, max_seconds=7200)

    def test_duration_check_passes(self):
        AudioValidator.check_duration(3600, max_seconds=7200)

    def test_duration_check_passes_at_exact_limit(self):
        AudioValidator.check_duration(7200, max_seconds=7200)

    def test_check_size_at_exact_maximum_passes(self):
        max_bytes = 500 * 1024 * 1024
        AudioValidator.check_size(max_bytes, max_bytes=max_bytes)

    def test_get_duration_seconds_none_raises(self):
        with (
            patch("seshat.app.pipeline.ingestion.audio_validator.audio_duration_seconds", return_value=None),
            pytest.raises(AudioValidationError, match="Unable to determine audio duration"),
        ):
            AudioValidator.get_duration_seconds(b"some bytes")
