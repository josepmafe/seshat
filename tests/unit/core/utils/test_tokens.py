from __future__ import annotations

from seshat.core.utils.tokens import count_tokens


def test_count_tokens_nonempty_string():
    # cl100k_base encodes "hello world" as ["hello", " world"] → 2 tokens
    assert count_tokens("hello world") == 2


def test_count_tokens_known_openai_model():
    # cl100k_base (gpt-4) encodes "some text" as ["some", " text"] → 2 tokens
    assert count_tokens("some text", model="gpt-4") == 2


def test_count_tokens_unknown_model_falls_back_without_raising():
    # "claude-opus-4-8" is not a tiktoken-known model; should fall back silently
    result = count_tokens("some text", model="claude-opus-4-8")
    assert isinstance(result, int)
    assert result > 0
