from typing import Any

from langchain_core.language_models import BaseChatModel

from seshat.agents.identification.action_item import ActionItemIdentificationAgent
from seshat.agents.identification.base import _BaseIdentificationAgent
from seshat.agents.identification.decision import DecisionIdentificationAgent
from seshat.agents.identification.open_question import OpenQuestionIdentificationAgent
from seshat.agents.identification.risk import RiskIdentificationAgent
from seshat.config.settings import ExtractionConfig
from seshat.models.enums import ConceptType


class IdentificationAgentRegistry:
    def __init__(self, llm: BaseChatModel, config: ExtractionConfig) -> None:
        kwargs: dict[str, Any] = {
            "llm": llm,
            "config": config.identification,
            "grouped_identification_types": config.grouped_identification_types,
        }
        self._agents: dict[ConceptType, _BaseIdentificationAgent] = {
            ConceptType.DECISION: DecisionIdentificationAgent(**kwargs),
            ConceptType.RISK: RiskIdentificationAgent(**kwargs),
            ConceptType.OPEN_QUESTION: OpenQuestionIdentificationAgent(**kwargs),
            ConceptType.ACTION_ITEM: ActionItemIdentificationAgent(**kwargs),
        }

    def get(self, concept_type: ConceptType) -> _BaseIdentificationAgent:
        agent = self._agents.get(concept_type)
        if agent is None:
            raise KeyError(f"No agent registered for {concept_type}")
        return agent
