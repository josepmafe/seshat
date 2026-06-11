from __future__ import annotations

from collections import defaultdict
from typing import Any

type CorpusTagFilter = dict[str, str | list[str]]

_DEFAULT_TAGS_BLACKLIST = ("detail",)


def matches_tags(tags: dict[str, Any], tag_filter: CorpusTagFilter) -> bool:
    for key, wanted in tag_filter.items():
        value = tags.get(key)
        if isinstance(wanted, list):
            if not (isinstance(value, list) and set(wanted) <= set(value)):
                return False
        else:
            if value != wanted:
                return False
    return True


def corpus_tag_summary(examples: list, *, tags_blacklist: tuple[str, ...] | None = None) -> dict[str, str]:
    """Return one MLflow tag per corpus tag key, value = sorted comma-joined unique values.

    Each example must have a `tags` attribute of type dict[str, Any].
    """
    if tags_blacklist is None:
        tags_blacklist = _DEFAULT_TAGS_BLACKLIST

    seen: dict[str, set[str]] = defaultdict(set)
    for ex in examples:
        for k, v in ex.tags.items():
            if k in tags_blacklist:
                continue

            seen[k].add(str(v))

    return {f"corpus.{k}": ",".join(sorted(vs)) for k, vs in seen.items()}
