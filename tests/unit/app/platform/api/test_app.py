from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seshat.app.platform.api.app import _check_eval_gate, _emit_config_warnings, _ping_llms


class TestCheckEvalGate:
    def test_skip_eval_gate_returns_immediately(self, caplog):
        config = MagicMock()
        config.skip_eval_gate = True
        _check_eval_gate(config)  # must not raise

    def test_missing_gate_file_raises_system_exit(self, tmp_path):
        config = MagicMock()
        config.skip_eval_gate = False
        config.eval_gate_path = tmp_path / "nonexistent.json"
        with pytest.raises(SystemExit):
            _check_eval_gate(config)

    def test_gate_not_passed_raises_system_exit(self, tmp_path):
        gate_path = tmp_path / "eval_gate.json"
        gate_path.write_text(json.dumps({"passed": False}))
        config = MagicMock()
        config.skip_eval_gate = False
        config.eval_gate_path = gate_path
        with pytest.raises(SystemExit):
            _check_eval_gate(config)

    def test_gate_passed_does_not_raise(self, tmp_path):
        gate_path = tmp_path / "eval_gate.json"
        gate_path.write_text(json.dumps({"passed": True}))
        config = MagicMock()
        config.skip_eval_gate = False
        config.eval_gate_path = gate_path
        _check_eval_gate(config)  # must not raise


class TestEmitConfigWarnings:
    def test_no_grounding_emits_warning(self, caplog):
        import logging

        config = MagicMock()
        config.extraction.grounding = None
        with caplog.at_level(logging.WARNING):
            _emit_config_warnings(config)
        assert any("grounding=None" in r.message for r in caplog.records)

    def test_with_grounding_no_warning(self, caplog):
        import logging

        config = MagicMock()
        config.extraction.grounding = MagicMock()
        with caplog.at_level(logging.WARNING):
            _emit_config_warnings(config)
        assert not any("grounding" in r.message for r in caplog.records)


class TestPingLlms:
    async def test_skip_llm_ping_returns_immediately(self):
        config = MagicMock()
        config.api.skip_llm_ping = True
        await _ping_llms(config)  # must not raise

    async def test_all_providers_reachable_does_not_raise(self):
        config = MagicMock()
        config.api.skip_llm_ping = False
        with (
            patch("seshat.app.platform.api.app._ping_chat_models", new=AsyncMock(return_value=[])),
            patch("seshat.app.platform.api.app._ping_embedding_models", new=AsyncMock(return_value=[])),
            patch("seshat.app.platform.api.app._ping_transcription_models", new=AsyncMock(return_value=[])),
        ):
            await _ping_llms(config)  # must not raise

    async def test_faulty_provider_raises_system_exit(self):
        config = MagicMock()
        config.api.skip_llm_ping = False
        with (
            patch("seshat.app.platform.api.app._ping_chat_models", new=AsyncMock(return_value=["anthropic"])),
            patch("seshat.app.platform.api.app._ping_embedding_models", new=AsyncMock(return_value=[])),
            patch("seshat.app.platform.api.app._ping_transcription_models", new=AsyncMock(return_value=[])),
            pytest.raises(SystemExit),
        ):
            await _ping_llms(config)
