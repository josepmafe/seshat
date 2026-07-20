from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from seshat.eval.cache import clear_cache_dir, read_or_run, sweep_stale_entries

if TYPE_CHECKING:
    from pathlib import Path


class _Dummy(BaseModel):
    value: int


class _Text(BaseModel):
    text: str


async def _coro(value: int) -> _Dummy:
    return _Dummy(value=value)


class TestCacheEncoding:
    """Cache files must round-trip non-ASCII regardless of the platform default encoding.

    See project_seshat_eval_mlflow_hang follow-up: read_text/write_text with no explicit
    encoding used cp1252 on Windows; an em-dash written then read under a different default
    (e.g. forced UTF-8) raised UnicodeDecodeError.
    """

    async def test_non_ascii_round_trips_through_cache(self, tmp_path: Path):
        cache_fp = tmp_path / "entry.json"
        payload = "decision — narrows scope, café, 你好"

        # miss: writes the file
        written, _, was_cached = await read_or_run(cache_fp, _Text, _make_text(payload))
        assert was_cached is False
        assert written.text == payload

        # hit: reads it back — must decode identically
        read_back, _, was_cached = await read_or_run(cache_fp, _Text, _make_text("unused"))
        assert was_cached is True
        assert read_back.text == payload

    async def test_cache_file_is_utf8_on_disk(self, tmp_path: Path):
        cache_fp = tmp_path / "entry.json"
        await read_or_run(cache_fp, _Text, _make_text("café — 你好"))

        # explicit utf-8 read must succeed (fails if written as cp1252)
        raw = cache_fp.read_text(encoding="utf-8")
        assert "café — 你好" in raw


async def _make_text(text: str) -> _Text:
    return _Text(text=text)


class TestReadOrRun:
    async def test_returns_coroutine_result_and_writes_cache_fp(self, tmp_path: Path):
        cache_fp = tmp_path / "entry.json"
        result, _, _ = await read_or_run(cache_fp, _Dummy, _coro(42))
        assert result.value == 42
        assert cache_fp.exists()

    async def test_returns_cached_result_without_running_coroutine(self, tmp_path: Path):
        cache_fp = tmp_path / "entry.json"
        cache_fp.write_text(_Dummy(value=99).model_dump_json())

        called = False

        async def _never_called() -> _Dummy:
            nonlocal called
            called = True
            return _Dummy(value=0)

        result, _, _ = await read_or_run(cache_fp, _Dummy, _never_called())
        assert result.value == 99
        assert not called

    async def test_returns_path_on_both_miss_and_hit(self, tmp_path: Path):
        cache_fp = tmp_path / "entry.json"
        _, used_miss, _ = await read_or_run(cache_fp, _Dummy, _coro(1))
        assert used_miss == cache_fp
        _, used_hit, _ = await read_or_run(cache_fp, _Dummy, _coro(0))
        assert used_hit == cache_fp


class TestSweepStaleEntries:
    def test_deletes_unvisited_files_matching_corpus_ids(self, tmp_path: Path):
        (tmp_path / "abc_aaa_bbb.json").write_text("{}")
        (tmp_path / "abc_old_old.json").write_text("{}")  # stale variant
        touched = {tmp_path / "abc_aaa_bbb.json"}

        sweep_stale_entries(tmp_path, corpus_ids=["abc"], touched=touched)

        assert (tmp_path / "abc_aaa_bbb.json").exists()
        assert not (tmp_path / "abc_old_old.json").exists()

    def test_does_not_touch_files_for_out_of_scope_corpus_ids(self, tmp_path: Path):
        other_file = tmp_path / "xyz_aaa_bbb.json"
        other_file.write_text("{}")
        touched: set[Path] = set()

        sweep_stale_entries(tmp_path, corpus_ids=["abc"], touched=touched)

        assert other_file.exists()

    def test_no_op_when_cache_dir_is_empty(self, tmp_path: Path):
        sweep_stale_entries(tmp_path, corpus_ids=["abc"], touched=set())

    def test_all_touched_files_are_kept(self, tmp_path: Path):
        f1 = tmp_path / "abc_v1_v2.json"
        f2 = tmp_path / "abc_v3_v4.json"
        f1.write_text("{}")
        f2.write_text("{}")
        touched = {f1, f2}

        sweep_stale_entries(tmp_path, corpus_ids=["abc"], touched=touched)

        assert f1.exists()
        assert f2.exists()


class TestClearCacheDir:
    def test_deletes_all_json_files(self, tmp_path: Path):
        for name in ("a.json", "b.json", "c.json"):
            (tmp_path / name).write_text("{}")

        clear_cache_dir(tmp_path)

        assert list(tmp_path.glob("*.json")) == []

    def test_leaves_non_json_files_untouched(self, tmp_path: Path):
        other = tmp_path / "notes.txt"
        other.write_text("hello")
        (tmp_path / "a.json").write_text("{}")

        clear_cache_dir(tmp_path)

        assert other.exists()
