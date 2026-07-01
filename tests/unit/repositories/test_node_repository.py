from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from seshat.models.enums import NodeState, NodeStatus, RelationshipType
from seshat.models.nodes import ExtractionResult
from seshat.repositories.node_repository import NodeRepository
from tests.helpers import make_node
from tests.integration.helpers import make_relationship


def _make_repo() -> tuple[NodeRepository, MagicMock, MagicMock]:
    kb_store = MagicMock()
    kb_store.write_node = AsyncMock()
    kb_store.write_relationship = AsyncMock()
    kb_store.update_node = AsyncMock()
    kb_store.update_node_state = AsyncMock()
    kb_store.delete_node = AsyncMock()
    kb_store.delete_relationships_for_node = AsyncMock()

    @asynccontextmanager
    async def _fake_transaction():
        yield MagicMock()

    kb_store.transaction = _fake_transaction

    vs = MagicMock()
    vs.upsert = AsyncMock()
    vs.delete = AsyncMock()

    return NodeRepository(kb_store, vs), kb_store, vs


class TestWriteBatch:
    async def test_approved_node_written(self):
        repo, kb, vs = _make_repo()
        node = make_node()
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        await repo.write_batch(result)

        kb.write_node.assert_called_once()
        vs.upsert.assert_called_once()

    async def test_rejected_node_not_written(self):
        repo, kb, vs = _make_repo()
        node = make_node(status=NodeStatus.REJECTED)
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        await repo.write_batch(result)

        kb.write_node.assert_not_called()
        vs.upsert.assert_not_called()

    async def test_supersedes_triggers_state_transition(self):
        repo, kb, _vs = _make_repo()
        existing_node = make_node("existing")
        new_node = make_node("new")
        rel = make_relationship(new_node, existing_node, rel_type=RelationshipType.SUPERSEDES)

        result = ExtractionResult(job_id="job-1", nodes=[new_node], relationships=[rel])
        await repo.write_batch(result)

        args, kwargs = kb.update_node_state.call_args
        assert args == (str(existing_node.id), NodeState.SUPERSEDED)
        assert "conn" in kwargs

    async def test_relationship_not_written_if_source_rejected(self):
        repo, kb, _vs = _make_repo()
        source = make_node("source", status=NodeStatus.REJECTED)
        target = make_node("target")
        rel = make_relationship(source, target)

        result = ExtractionResult(job_id="job-1", nodes=[source, target], relationships=[rel])
        await repo.write_batch(result)

        kb.write_relationship.assert_not_called()

    async def test_supersedes_missing_target_logs_warning(self):
        repo, kb, _vs = _make_repo()
        new_node = make_node("new")
        existing_node = make_node("existing")
        rel = make_relationship(new_node, existing_node, rel_type=RelationshipType.SUPERSEDES)

        kb.update_node_state = AsyncMock(side_effect=KeyError("not found"))

        result = ExtractionResult(job_id="job-1", nodes=[new_node], relationships=[rel])
        written = await repo.write_batch(result)
        assert written == 1


class TestWriteNode:
    async def test_writes_kb_and_vs(self):
        repo, kb, vs = _make_repo()
        node = make_node()
        await repo.write_node(node)

        kb.write_node.assert_called_once()
        vs.upsert.assert_called_once()

    async def test_writes_relationships_in_transaction(self):
        repo, kb, _vs = _make_repo()
        node = make_node()
        other = make_node("other")
        rel = make_relationship(node, other)
        await repo.write_node(node, relationships=[rel])

        kb.write_relationship.assert_called_once()


class TestDeleteNode:
    async def test_deletes_from_kb_and_vs(self):
        repo, kb, vs = _make_repo()
        await repo.delete_node("node-1")

        kb.delete_relationships_for_node.assert_called_once()
        kb.delete_node.assert_called_once()
        vs.delete.assert_called_once_with("node-1")
