"""Persistent conversation memory backed by a Unity Catalog memory store.

This module uses the Databricks managed memory REST API through workspace-native
credentials. Each user/session is stored as one JSON memory entry. The application
fails open: if the preview feature or permissions are unavailable, the RAG answer can
still run, but the response metadata reports that memory was unavailable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import quote

import requests
from databricks.sdk import WorkspaceClient


@dataclass
class SessionMemory:
    user_id: str
    session_id: str
    active_patient_id: str | None = None
    recent_messages: list[dict[str, str]] = field(default_factory=list)
    summary: str = ""
    last_tool: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "SessionMemory":
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Memory entry must contain a JSON object")
        messages = value.get("recent_messages") or []
        clean_messages = []
        for message in messages:
            if isinstance(message, dict):
                clean_messages.append(
                    {
                        "role": str(message.get("role", "unknown")),
                        "content": str(message.get("content", "")),
                    }
                )
        return cls(
            user_id=str(value.get("user_id", "anonymous")),
            session_id=str(value.get("session_id", "default")),
            active_patient_id=(
                str(value["active_patient_id"])
                if value.get("active_patient_id")
                else None
            ),
            recent_messages=clean_messages,
            summary=str(value.get("summary", "")),
            last_tool=(str(value["last_tool"]) if value.get("last_tool") else None),
        )


class CatalogMemoryStore:
    """Read and write session state in a Unity Catalog memory store."""

    def __init__(
        self,
        full_name: str,
        *,
        enabled: bool = True,
        fail_open: bool = True,
        timeout_seconds: int = 20,
    ):
        self.full_name = full_name
        self.enabled = bool(enabled and full_name)
        self.fail_open = bool(fail_open)
        self.timeout_seconds = int(timeout_seconds)
        self.workspace = WorkspaceClient() if self.enabled else None
        self.last_error: str | None = None

    @property
    def encoded_name(self) -> str:
        return quote(self.full_name, safe=".")

    def ensure_store(self, description: str) -> None:
        if not self.enabled:
            return
        self.last_error = None
        try:
            response = self._request(
                "GET",
                f"/api/2.1/unity-catalog/memory-stores/{self.encoded_name}",
                allow_not_found=True,
            )
            if response is not None:
                return
            parts = self.full_name.split(".")
            if len(parts) != 3:
                raise ValueError(
                    "memory_store_name must be a three-part Unity Catalog name"
                )
            catalog_name, schema_name, name = parts
            self._request(
                "POST",
                "/api/2.1/unity-catalog/memory-stores",
                body={
                    "name": name,
                    "catalog_name": catalog_name,
                    "schema_name": schema_name,
                    "description": description,
                },
            )
        except Exception as exc:
            self._handle_error(exc)

    def load(self, user_id: str, session_id: str) -> SessionMemory:
        default = SessionMemory(user_id=user_id, session_id=session_id)
        if not self.enabled:
            return default
        self.last_error = None
        try:
            response = self._request(
                "GET",
                (
                    f"/api/2.1/unity-catalog/memory-stores/"
                    f"{self.encoded_name}/entries:get"
                ),
                params={
                    "scope": user_id,
                    "path": self._entry_path(session_id),
                },
                allow_not_found=True,
            )
            if response is None or not response.get("contents"):
                return default
            memory = SessionMemory.from_json(str(response["contents"]))
            memory.user_id = user_id
            memory.session_id = session_id
            return memory
        except Exception as exc:
            self._handle_error(exc)
            return default

    def save(self, memory: SessionMemory) -> None:
        if not self.enabled:
            return
        self.last_error = None
        path = self._entry_path(memory.session_id)
        try:
            existing = self._request(
                "GET",
                (
                    f"/api/2.1/unity-catalog/memory-stores/"
                    f"{self.encoded_name}/entries:get"
                ),
                params={"scope": memory.user_id, "path": path},
                allow_not_found=True,
            )
            if existing is None:
                self._request(
                    "POST",
                    (
                        f"/api/2.1/unity-catalog/memory-stores/"
                        f"{self.encoded_name}/entries"
                    ),
                    params={"scope": memory.user_id},
                    body={
                        "path": path,
                        "contents": memory.to_json(),
                        "description": (
                            "Patient History Navigator persistent conversation state"
                        ),
                    },
                )
            else:
                self._request(
                    "PATCH",
                    (
                        f"/api/2.1/unity-catalog/memory-stores/"
                        f"{self.encoded_name}/entries"
                    ),
                    body={
                        "scope": memory.user_id,
                        "path": path,
                        "replace_all": {"contents": memory.to_json()},
                    },
                )
        except Exception as exc:
            self._handle_error(exc)

    def _entry_path(self, session_id: str) -> str:
        safe = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in session_id
        )[:120]
        return f"/memories/patient-history/{safe or 'default'}.json"

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> dict[str, Any] | None:
        if self.workspace is None:
            return None
        host = str(self.workspace.config.host).rstrip("/")
        headers = dict(self.workspace.config.authenticate())
        headers["Content-Type"] = "application/json"
        response = requests.request(
            method,
            f"{host}{path}",
            headers=headers,
            params=params,
            json=body,
            timeout=self.timeout_seconds,
        )
        if allow_not_found and response.status_code == 404:
            return None
        response.raise_for_status()
        if not response.content:
            return {}
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError("Unexpected Databricks memory API response")
        return value

    def _handle_error(self, exc: Exception) -> None:
        self.last_error = f"{type(exc).__name__}: {exc}"
        if not self.fail_open:
            raise exc
