from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from seshat.cli.app import app
from seshat.core.config.eval_settings import EvalConfig

# `seshat.cli.__init__` binds the name `app` to the Typer object, shadowing the
# `seshat.cli.app` submodule on attribute access; fetch the real module from sys.modules.
cli_app = sys.modules["seshat.cli.app"]

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.fixture
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point EvalConfig's cache dir at a tmp tree with one seeded file per harness."""
    monkeypatch.setattr(EvalConfig, "_cache_dir", tmp_path)
    for harness in ("identification", "resolution", "retrieval", "grounding", "grouping"):
        subdir = tmp_path / harness
        subdir.mkdir()
        (subdir / "seed.json").write_text("{}")

    return tmp_path


class TestClearCacheCommand:
    def test_clears_single_harness_only(self, cache_root: Path) -> None:
        result = runner.invoke(app, ["eval", "clear-cache", "retrieval"])

        assert result.exit_code == 0
        assert list((cache_root / "retrieval").glob("*.json")) == []
        assert (cache_root / "grouping" / "seed.json").exists()

    def test_no_argument_clears_all_harnesses(self, cache_root: Path) -> None:
        result = runner.invoke(app, ["eval", "clear-cache"])

        assert result.exit_code == 0
        for harness in ("identification", "resolution", "retrieval", "grounding", "grouping"):
            assert list((cache_root / harness).glob("*.json")) == []

    def test_unknown_harness_exits_nonzero(self, cache_root: Path) -> None:
        result = runner.invoke(app, ["eval", "clear-cache", "bogus"])

        assert result.exit_code == 1
        assert "Unknown harness" in result.output


class TestHarnessClearCacheFlag:
    def test_flag_clears_cache_before_running(self, cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ran = False

        def _fake_run_async(coro: object) -> None:
            nonlocal ran
            ran = True
            coro.close()  # type: ignore[attr-defined]

        monkeypatch.setattr(cli_app, "_run_async", _fake_run_async)

        result = runner.invoke(app, ["eval", "harness", "retrieval", "--clear-cache"])

        assert result.exit_code == 0
        assert list((cache_root / "retrieval").glob("*.json")) == []
        assert ran


class TestHarnessAllFlag:
    @pytest.fixture(autouse=True)
    def _all_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Pin every run_<harness> flag so tests do not depend on a local .env.
        for flag in ("IDENTIFICATION", "RESOLUTION", "RETRIEVAL", "GROUNDING", "GROUPING"):
            monkeypatch.setenv(f"EVAL__RUN_{flag}", "true")

    def test_all_runs_each_enabled_harness(self, cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ran: list[str] = []
        monkeypatch.setattr(cli_app, "_run_single_harness", lambda harness, tags: ran.append(harness))

        result = runner.invoke(app, ["eval", "harness", "--all"])

        assert result.exit_code == 0
        assert ran == ["identification", "resolution", "retrieval", "grounding", "grouping"]

    def test_all_respects_disabled_flags(self, cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EVAL__RUN_RESOLUTION", "false")
        monkeypatch.setenv("EVAL__RUN_GROUNDING", "false")
        ran: list[str] = []
        monkeypatch.setattr(cli_app, "_run_single_harness", lambda harness, tags: ran.append(harness))

        result = runner.invoke(app, ["eval", "harness", "--all"])

        assert result.exit_code == 0
        assert ran == ["identification", "retrieval", "grouping"]

    def test_all_with_clear_cache_clears_each_run_harness(
        self, cache_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EVAL__RUN_RESOLUTION", "false")
        monkeypatch.setattr(cli_app, "_run_single_harness", lambda harness, tags: None)

        result = runner.invoke(app, ["eval", "harness", "--all", "--clear-cache"])

        assert result.exit_code == 0
        assert list((cache_root / "retrieval").glob("*.json")) == []
        # resolution is disabled, so its cache is left alone
        assert list((cache_root / "resolution").glob("*.json")) != []

    def test_name_and_all_together_errors(self, cache_root: Path) -> None:
        result = runner.invoke(app, ["eval", "harness", "retrieval", "--all"])

        assert result.exit_code == 1
        assert "both" in result.output.lower()

    def test_neither_name_nor_all_errors(self, cache_root: Path) -> None:
        result = runner.invoke(app, ["eval", "harness"])

        assert result.exit_code == 1

    def test_all_with_nothing_enabled_errors(self, cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for flag in ("IDENTIFICATION", "RESOLUTION", "RETRIEVAL", "GROUNDING", "GROUPING"):
            monkeypatch.setenv(f"EVAL__RUN_{flag}", "false")

        result = runner.invoke(app, ["eval", "harness", "--all"])

        assert result.exit_code == 1

    def test_all_continues_past_a_failing_harness(self, cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ran: list[str] = []

        def _fake(harness: str, tags: object) -> None:
            ran.append(harness)
            if harness == "resolution":
                raise RuntimeError("boom")

        monkeypatch.setattr(cli_app, "_run_single_harness", _fake)

        result = runner.invoke(app, ["eval", "harness", "--all"])

        # every harness is attempted despite resolution raising
        assert ran == ["identification", "resolution", "retrieval", "grounding", "grouping"]
        # a failure makes the overall run exit non-zero
        assert result.exit_code == 1
        # the failed harness is named in the summary
        assert "resolution" in result.output

    def test_all_exits_zero_when_all_succeed(self, cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli_app, "_run_single_harness", lambda harness, tags: None)

        result = runner.invoke(app, ["eval", "harness", "--all"])

        assert result.exit_code == 0


class TestHarnessSingleFailHard:
    def test_named_harness_failure_propagates(self, cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(harness: str, tags: object) -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr(cli_app, "_run_single_harness", _boom)

        result = runner.invoke(app, ["eval", "harness", "retrieval"])

        assert result.exit_code != 0


class TestCalibrateClearCacheFlag:
    def test_flag_clears_component_cache_before_running(
        self, cache_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ran = False

        def _fake_run_async(coro: object) -> None:
            nonlocal ran
            ran = True
            coro.close()  # type: ignore[attr-defined]

        monkeypatch.setattr(cli_app, "_run_async", _fake_run_async)

        result = runner.invoke(app, ["eval", "calibrate", "retrieval", "--clear-cache"])

        assert result.exit_code == 0
        assert list((cache_root / "retrieval").glob("*.json")) == []
        assert list((cache_root / "grouping").glob("*.json")) != []
        assert ran
