from __future__ import annotations

import tiktoken

_FALLBACK_ENCODING = "cl100k_base"


def count_tokens(text: str, model: str | None = None) -> int:
    """Return the token count for text.

    Uses the model's native tiktoken encoding when known (OpenAI models),
    falls back to cl100k_base for unknown models (Claude, Bedrock).
    """
    if model is not None:
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding(_FALLBACK_ENCODING)
    else:
        enc = tiktoken.get_encoding(_FALLBACK_ENCODING)
    return len(enc.encode(text))
