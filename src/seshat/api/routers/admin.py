from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status

from seshat.api.dependencies import get_app_state
from seshat.api.state import AppState
from seshat.models.api_responses import ApiKeyResponse, CreateApiKeyRequest, CreateApiKeyResponse
from seshat.secrets.factory import get_secrets_resolver
from seshat.services.admin_service import ApiKeyAlreadyRevokedError, ApiKeyNotFoundError
from seshat.utils.concurrency import run_in_thread


async def _get_root_key(state: Annotated[AppState, Depends(get_app_state)]) -> str:
    resolver = get_secrets_resolver(state.config)
    secret_key = state.config.api.root_api_key_secret_key
    return await run_in_thread(resolver.get_secret, secret_key)


async def _require_root_key(
    root_key: Annotated[str, Depends(_get_root_key)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="X-API-Key required")
    if not secrets.compare_digest(x_api_key, root_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid root key")


router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(_require_root_key)])


@router.get(
    "/api-keys",
    response_model=list[ApiKeyResponse],
    summary="List all API keys",
    responses={
        200: {"description": "API keys returned"},
        401: {"description": "Missing or invalid root key"},
    },
)
async def list_api_keys(
    state: Annotated[AppState, Depends(get_app_state)],
) -> list[ApiKeyResponse]:
    rows = await state.admin_service.list_api_keys()
    return [ApiKeyResponse.model_validate(dict(row)) for row in rows]


@router.delete(
    "/api-keys/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke an API key",
    responses={
        204: {"description": "Key revoked"},
        401: {"description": "Missing or invalid root key"},
        404: {"description": "Key not found"},
        409: {"description": "Key already revoked"},
    },
)
async def revoke_api_key(
    key_id: int,
    state: Annotated[AppState, Depends(get_app_state)],
) -> None:
    try:
        await state.admin_service.revoke_api_key(key_id)
    except ApiKeyNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")
    except ApiKeyAlreadyRevokedError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="API key already revoked")


@router.post(
    "/api-keys",
    response_model=CreateApiKeyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new API key",
    responses={
        201: {"description": "API key created; plaintext returned once"},
        401: {"description": "Missing or invalid root key"},
    },
)
async def create_api_key(
    body: CreateApiKeyRequest,
    state: Annotated[AppState, Depends(get_app_state)],
) -> CreateApiKeyResponse:
    plaintext, user_id, role = await state.admin_service.create_api_key(body.user_id, body.role)
    return CreateApiKeyResponse(api_key=plaintext, user_id=user_id, role=role)
