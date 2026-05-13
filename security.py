"""Authentication dependencies for the RBAC RAG API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import Client, create_client


bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    """Authenticated user context available to protected routes."""

    id: str
    role: str


def load_env_file(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs from .env without overriding real env vars."""

    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} must be configured")
    return value


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Create a cached Supabase client from environment configuration."""

    load_env_file()
    return create_client(
        require_env("SUPABASE_URL"),
        require_env("SUPABASE_KEY"),
    )


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "dict"):
        return value.dict()

    return {}


def _extract_user_data(user_response: Any) -> dict[str, Any]:
    user = getattr(user_response, "user", None)
    if user is not None:
        return _as_dict(user)

    response_data = getattr(user_response, "data", None)
    if response_data is not None:
        data = _as_dict(response_data)
        if isinstance(data.get("user"), dict):
            return data["user"]
        return data

    response_dict = _as_dict(user_response)
    if isinstance(response_dict.get("user"), dict):
        return response_dict["user"]
    if isinstance(response_dict.get("data"), dict):
        data = response_dict["data"]
        if isinstance(data.get("user"), dict):
            return data["user"]
        return data

    return response_dict


def _extract_role(user_data: dict[str, Any]) -> str | None:
    user_metadata = user_data.get("user_metadata")
    if isinstance(user_metadata, dict):
        role = user_metadata.get("role")
        if isinstance(role, str) and role.strip():
            return role.strip()

    return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> CurrentUser:
    """Verify a Supabase JWT Bearer token and return the user's RBAC context."""

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        user_response = get_supabase_client().auth.get_user(token)
        user_data = _extract_user_data(user_response)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user_id = user_data.get("id")
    role = _extract_role(user_data)

    if not isinstance(user_id, str) or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token user",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if role is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is missing an assigned role",
        )

    return CurrentUser(id=user_id, role=role)
