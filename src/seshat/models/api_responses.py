from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, computed_field

from seshat.models.enums import HealthStatus, UserRole
from seshat.models.nodes import KBNode


class HealthResponse(BaseModel):
    status: HealthStatus
    components: dict[str, HealthStatus] | None = None


class NodeListResponse(BaseModel):
    nodes: list[KBNode]


class NodeDetailResponse(BaseModel):
    node: KBNode
    neighbours: list[KBNode]


class ImpactNode(BaseModel):
    node: KBNode
    traversal_depth: int


class ImpactResponse(BaseModel):
    nodes: list[ImpactNode]


class JobSubmitResponse(BaseModel):
    job_id: str


class JobActionResponse(BaseModel):
    status: str


class ApiKeyResponse(BaseModel):
    id: int
    user_id: str
    role: UserRole
    created_at: datetime
    revoked_at: datetime | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_active(self) -> bool:
        return self.revoked_at is None


class CreateApiKeyRequest(BaseModel):
    user_id: str
    role: UserRole


class CreateApiKeyResponse(BaseModel):
    api_key: str
    user_id: str
    role: UserRole
