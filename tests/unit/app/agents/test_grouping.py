import logging

from seshat.app.agents.identification.grouping import GroupingAgent, _GroupingSchema, _GroupSchema
from seshat.core.config.settings import IdentificationLLMConfig
from seshat.core.models.enums import ConceptType
from tests.helpers import make_anchored_concept, make_structured_llm


def _make_agent(return_value=None, side_effect=None, max_retries: int = 2) -> GroupingAgent:
    return GroupingAgent(
        llm=make_structured_llm(return_value=return_value, side_effect=side_effect),
        config=IdentificationLLMConfig(max_retries=max_retries),
    )


class TestGroupingAgent:
    async def test_empty_input_returns_empty_without_calling_llm(self):
        llm = make_structured_llm()
        agent = GroupingAgent(llm=llm, config=IdentificationLLMConfig())

        result = await agent.group(items=[], concept_type=ConceptType.DECISION)

        assert result == []
        llm.with_structured_output.assert_not_called()

    async def test_groups_assembled_from_llm_response(self):
        concepts = [make_anchored_concept("Use PostgreSQL"), make_anchored_concept("Use Redis")]
        schema = _GroupingSchema(
            groups=[
                _GroupSchema(group_title="Storage", group_description="DB choices", member_ids=["D01", "D02"]),
            ]
        )
        agent = _make_agent(return_value=schema)

        result = await agent.group(items=concepts, concept_type=ConceptType.DECISION)

        assert len(result) == 1
        assert result[0].group_title == "Storage"
        assert len(result[0].members) == 2
        assert result[0].members[0].item.title == "Use PostgreSQL"
        assert result[0].members[1].item.title == "Use Redis"

    async def test_unknown_member_ids_are_silently_dropped(self):
        concepts = [make_anchored_concept("Use PostgreSQL")]
        schema = _GroupingSchema(
            groups=[
                _GroupSchema(group_title="Storage", group_description="DB choices", member_ids=["D01", "UNKNOWN"]),
            ]
        )
        agent = _make_agent(return_value=schema)

        result = await agent.group(items=concepts, concept_type=ConceptType.DECISION)

        assert len(result[0].members) == 1

    async def test_falls_back_to_singletons_after_all_retries_fail(self):
        concepts = [make_anchored_concept("Use PostgreSQL"), make_anchored_concept("Use Redis")]
        agent = _make_agent(side_effect=Exception("LLM error"), max_retries=3)

        result = await agent.group(items=concepts, concept_type=ConceptType.DECISION)

        assert len(result) == 2
        titles = {g.group_title for g in result}
        assert titles == {"Use PostgreSQL", "Use Redis"}
        for group in result:
            assert len(group.members) == 1

    async def test_llm_returns_empty_groups_list_preserves_items_as_singletons(self, caplog):
        concepts = [make_anchored_concept("Use PostgreSQL")]
        schema = _GroupingSchema(groups=[])
        agent = _make_agent(return_value=schema)

        with caplog.at_level(logging.WARNING, logger="seshat.app.agents.identification.grouping"):
            result = await agent.group(items=concepts, concept_type=ConceptType.DECISION)

        assert len(result) == 1
        assert result[0].group_title == "Use PostgreSQL"
        assert result[0].members[0].item.title == "Use PostgreSQL"
        assert any("singletons" in r.message for r in caplog.records)

    async def test_group_with_all_unknown_ids_emits_warning_and_preserves_item(self, caplog):
        """A group whose every member_id is hallucinated is dropped, but the unassigned item becomes a singleton."""
        concepts = [make_anchored_concept("Use PostgreSQL")]
        schema = _GroupingSchema(
            groups=[
                _GroupSchema(group_title="Ghost", group_description="No valid members", member_ids=["UNKNOWN"]),
            ]
        )
        agent = _make_agent(return_value=schema)

        with caplog.at_level(logging.WARNING, logger="seshat.app.agents.identification.grouping"):
            result = await agent.group(items=concepts, concept_type=ConceptType.DECISION)

        assert len(result) == 1
        assert result[0].members[0].item.title == "Use PostgreSQL"
        assert any("singletons" in r.message for r in caplog.records)

    async def test_unassigned_items_become_singletons(self, caplog):
        """Items not assigned to any group are added as singleton groups with a warning."""
        concepts = [make_anchored_concept("Use PostgreSQL"), make_anchored_concept("Use Redis")]
        schema = _GroupingSchema(
            groups=[
                _GroupSchema(group_title="Storage", group_description="DB choices", member_ids=["D01"]),
            ]
        )
        agent = _make_agent(return_value=schema)

        with caplog.at_level(logging.WARNING, logger="seshat.app.agents.identification.grouping"):
            result = await agent.group(items=concepts, concept_type=ConceptType.DECISION)

        assert len(result) == 2
        titles = {g.group_title for g in result}
        assert "Storage" in titles
        assert "Use Redis" in titles
        assert any("singletons" in r.message for r in caplog.records)

    async def test_item_in_multiple_groups_emits_warning(self, caplog):
        """When the LLM violates the one-group-per-item rule, the item still appears in both groups
        and a warning is logged."""
        concepts = [make_anchored_concept("Use PostgreSQL")]
        schema = _GroupingSchema(
            groups=[
                _GroupSchema(group_title="Storage", group_description="DB choices", member_ids=["D01"]),
                _GroupSchema(group_title="Infra", group_description="Infrastructure", member_ids=["D01"]),
            ]
        )
        agent = _make_agent(return_value=schema)

        with caplog.at_level(logging.WARNING, logger="seshat.app.agents.identification.grouping"):
            result = await agent.group(items=concepts, concept_type=ConceptType.DECISION)

        assert len(result) == 2
        assert result[0].members[0].item.title == "Use PostgreSQL"
        assert result[1].members[0].item.title == "Use PostgreSQL"
        assert any("multiple groups" in r.message for r in caplog.records)
