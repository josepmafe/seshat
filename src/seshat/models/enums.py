from enum import StrEnum, auto


class ConceptType(StrEnum):
    DECISION = auto()
    RISK = auto()
    ACTION_ITEM = auto()
    OPEN_QUESTION = auto()


class RelationshipType(StrEnum):
    MITIGATES = auto()
    BLOCKS = auto()
    CONFLICTS_WITH = auto()
    DEPENDS_ON = auto()
    SUPERSEDES = auto()
    AMENDS = auto()
    RESOLVES = auto()


class NodeStatus(StrEnum):
    APPROVED = auto()
    PENDING_REVIEW = auto()
    REJECTED = auto()


class NodeState(StrEnum):
    CURRENT = auto()
    AMENDED = auto()
    SUPERSEDED = auto()


class ApprovalMethod(StrEnum):
    INDIVIDUAL = auto()
    BULK = auto()
    AUTO = auto()
    THRESHOLD = auto()


class IngestionSource(StrEnum):
    JOB = auto()
    INIT = auto()


class JobStatus(StrEnum):
    PENDING = auto()
    TRANSCRIBING = auto()
    EXTRACTING = auto()
    AWAITING_REVIEW = auto()
    WRITING = auto()
    DONE = auto()
    FAILED = auto()


class LLMProvider(StrEnum):
    OPENAI = auto()
    ANTHROPIC = auto()
    AZURE_OPENAI = auto()
    BEDROCK_CONVERSE = auto()


class TranscriptionProvider(StrEnum):
    ASSEMBLYAI = auto()
    OPENAI = auto()
    DEEPGRAM = auto()


class VectorStoreProvider(StrEnum):
    PGVECTOR = auto()


class EmbeddingProvider(StrEnum):
    OPENAI = auto()
    AZURE_OPENAI = auto()
    ANTHROPIC = auto()


class SecretsProvider(StrEnum):
    ENV = auto()
    AWS = auto()


class DocumentLoaderProvider(StrEnum):
    MARKDOWN = auto()


class CallType(StrEnum):
    LLM_INPUT = auto()
    LLM_OUTPUT = auto()
    EMBEDDING = auto()
    TRANSCRIPTION = auto()


class GraphDirection(StrEnum):
    INBOUND = auto()
    OUTBOUND = auto()
    BOTH = auto()
