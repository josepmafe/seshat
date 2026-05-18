import re
import unicodedata
from typing import ClassVar, Self

from pydantic import BaseModel


class QuoteAnchor(BaseModel):
    transcript_file: str
    char_start: int
    char_end: int

    _FUZZY_ANCHOR_SIZES: ClassVar[tuple[int, ...]] = (10, 20, 30, 40, 50)

    @classmethod
    def from_transcript_quote(cls, quote: str, transcript: str, transcript_file: str) -> Self | None:
        norm_transcript = unicodedata.normalize("NFKC", transcript)
        norm_quote = unicodedata.normalize("NFKC", quote)

        norm_quote = re.sub(r"\n\.\.\.\n", "", norm_quote)

        if norm_quote in norm_transcript:
            char_start = norm_transcript.index(norm_quote)
            return cls(
                transcript_file=transcript_file,
                char_start=char_start,
                char_end=char_start + len(norm_quote),
            )

        # Fallback: find shortest prefix/suffix that appears only once.
        norm_transcript_rev = norm_transcript[::-1]
        fuzzy_start = cls._unique_anchor(norm_quote, norm_transcript)
        fuzzy_end_rev = cls._unique_anchor(norm_quote[::-1], norm_transcript_rev)
        if fuzzy_start is None or fuzzy_end_rev is None:
            return None

        fuzzy_end = fuzzy_end_rev[::-1].lstrip(".")

        try:
            char_start = norm_transcript.index(fuzzy_start.rstrip("."))
            char_end = norm_transcript.rindex(fuzzy_end) + len(fuzzy_end)
        except ValueError:
            return None

        return cls(
            transcript_file=transcript_file,
            char_start=char_start,
            char_end=char_end,
        )

    @classmethod
    def _unique_anchor(cls, text: str, norm_transcript: str) -> str | None:
        for size in cls._FUZZY_ANCHOR_SIZES:
            anchor = text[:size]
            if norm_transcript.count(anchor) == 1:
                return f"{anchor}..."
        return None
