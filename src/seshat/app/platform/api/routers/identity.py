from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from seshat.app.platform.api.dependencies import CurrentUser, _get_current_user

router = APIRouter(tags=["identity"])


@router.get("/me", summary="Resolve API key to user identity")
async def me(user: Annotated[CurrentUser, Depends(_get_current_user)]) -> dict:
    return {"user_id": user.user_id, "role": user.role}
