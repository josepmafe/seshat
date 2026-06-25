from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from seshat.observability.usage_tracker import get_run_tracker
from seshat.utils.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.language_models import BaseChatModel

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are extracting search keywords for a knowledge base of meeting notes.
The KB contains four node types: decisions, risks, action items, and open questions.

Given a query node, extract the most discriminating search terms — words that would \
uniquely identify semantically related nodes in the KB. Prioritise:
- Proper nouns and named tools/systems (e.g. "Flyway", "PagerDuty", "Terraform")
- Domain-specific technical terms (e.g. "schema drift", "rollback", "SLO breach")
- The specific subject or object of the relationship (what was decided/risked/asked)

Avoid:
- Generic words that appear in most nodes ("service", "team", "issue", "change")
- Node-type words ("decision", "risk", "action item", "question")
- Stop words and filler

Return 3-6 space-separated keywords or short phrases. No explanation, no punctuation.
"""


async def _extract(llm: BaseChatModel, query: str) -> str:
    messages = [SystemMessage(_SYSTEM_PROMPT), HumanMessage(query)]
    callback = get_run_tracker()
    config = RunnableConfig(callbacks=[callback]) if callback is not None else None
    response = await llm.ainvoke(messages, config=config)
    keywords = str(response.content).strip()
    logger.debug("keyword_extractor: %r -> %r", query[:60], keywords[:60])
    return keywords


def build_keyword_extractor(llm: BaseChatModel) -> Callable[[str], Awaitable[str]]:
    async def extractor(query: str) -> str:
        return await _extract(llm, query)

    return extractor
