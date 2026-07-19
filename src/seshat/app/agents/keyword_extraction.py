from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from seshat.app.agents.base import RetryExhaustedError, _BaseAgent
from seshat.core.utils.log import get_logger

logger = get_logger(__name__)

_KEYWORD_EXTRACTION_PROMPT = """\
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

Return 3-6 keywords or short phrases as a JSON object with a "keywords" array.
"""


class _Keywords(BaseModel):
    keywords: list[str]


class KeywordExtractionRetryExhaustedError(RetryExhaustedError):
    pass


class KeywordAgent(_BaseAgent):
    @property
    def _system_prompt(self) -> str:
        return _KEYWORD_EXTRACTION_PROMPT

    async def extract(self, query: str) -> str:
        messages = [SystemMessage(_KEYWORD_EXTRACTION_PROMPT), HumanMessage(query)]
        result: _Keywords = await self._retryable_structured_ainvoke(
            messages,
            _Keywords,
            raise_on_exhaustion=KeywordExtractionRetryExhaustedError("keyword extraction exhausted retries"),
        )
        keywords = " ".join(result.keywords)
        logger.debug("keyword extraction: %r -> %r", query[:60], keywords[:60])
        return keywords
