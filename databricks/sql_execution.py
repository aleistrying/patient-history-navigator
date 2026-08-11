"""Safe, small wrapper around Databricks SQL Statement Execution.

The deterministic tools use a SQL warehouse to read the full-document Delta table.
All table names are validated and all dynamic values are escaped as SQL literals.
"""

from __future__ import annotations

import re
import time
from typing import Any

from databricks.sdk import WorkspaceClient

_FQN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")


def validate_fqn(value: str) -> str:
    if not _FQN.fullmatch(value):
        raise ValueError(f"Unsafe or invalid Unity Catalog name: {value!r}")
    return value


def sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


class DatabricksSQLExecutor:
    """Execute small inline queries through a configured SQL warehouse."""

    def __init__(self, warehouse_id: str, timeout_seconds: int = 50):
        self.warehouse_id = warehouse_id.strip()
        self.timeout_seconds = int(timeout_seconds)
        self.workspace = WorkspaceClient() if self.is_configured else None

    @property
    def is_configured(self) -> bool:
        return bool(
            self.warehouse_id
            and "REPLACE" not in self.warehouse_id.upper()
            and "WAREHOUSE" not in self.warehouse_id.upper()
        )

    def query(self, statement: str) -> list[dict[str, Any]]:
        if not self.is_configured or self.workspace is None:
            raise RuntimeError(
                "A Databricks SQL warehouse ID is required for deterministic tools. "
                "Set databricks_resources.sql_warehouse_id in navigator_config.yaml."
            )
        response = self.workspace.statement_execution.execute_statement(
            warehouse_id=self.warehouse_id,
            statement=statement,
            wait_timeout=f"{min(self.timeout_seconds, 50)}s",
        )

        statement_id = getattr(response, "statement_id", None)
        deadline = time.monotonic() + self.timeout_seconds
        while self._state(response) in {"PENDING", "RUNNING"}:
            if not statement_id or time.monotonic() >= deadline:
                raise TimeoutError("Databricks SQL statement did not finish in time")
            time.sleep(0.5)
            response = self.workspace.statement_execution.get_statement(statement_id)

        state = self._state(response)
        if state != "SUCCEEDED":
            error = getattr(getattr(response, "status", None), "error", None)
            raise RuntimeError(f"Databricks SQL statement failed ({state}): {error}")

        manifest = getattr(response, "manifest", None)
        schema = getattr(manifest, "schema", None)
        columns = [
            getattr(column, "name", str(column))
            for column in (getattr(schema, "columns", None) or [])
        ]
        result = getattr(response, "result", None)
        rows = getattr(result, "data_array", None) or []
        return [dict(zip(columns, row, strict=False)) for row in rows]

    @staticmethod
    def _state(response: Any) -> str:
        status = getattr(response, "status", None)
        state = getattr(status, "state", "UNKNOWN")
        value = getattr(state, "value", state)
        return str(value).upper().split(".")[-1]
