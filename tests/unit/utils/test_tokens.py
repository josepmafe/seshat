from __future__ import annotations

from seshat.utils.tokens import count_tokens


def test_count_tokens_empty_string():
    assert count_tokens("") == 0


def test_count_tokens_nonempty_string():
    # cl100k_base encodes "hello world" as ["hello", " world"] → 2 tokens
    assert count_tokens("hello world") == 2


def test_count_tokens_known_openai_model_returns_positive_int():
    result = count_tokens("some text", model="gpt-4")
    assert isinstance(result, int)
    assert result > 0


def test_count_tokens_unknown_model_falls_back_without_raising():
    # "claude-opus-4-8" is not a tiktoken-known model; should fall back silently
    result = count_tokens("some text", model="claude-opus-4-8")
    assert isinstance(result, int)
    assert result > 0
