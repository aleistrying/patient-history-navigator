"""Create preview resources that are not created by the ingestion job.

At present this script creates the Unity Catalog managed memory store used for
persistent session memory. AI Search endpoint/index creation lives in
ingest_records.py, because the embedding dimension is only known after the first
batch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from session_memory import CatalogMemoryStore


def _runtime_directory() -> Path:
    module_file = globals().get("__file__")
    if module_file:
        return Path(str(module_file)).resolve().parent
    return Path.cwd()


ROOT = _runtime_directory()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-file",
        type=Path,
        default=ROOT / "navigator_config.yaml",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping")
    return value


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config_file)
    resources = config.get("databricks_resources", {})
    memory_config = config.get("memory_config", {})
    store_name = str(resources.get("memory_store_name", "")).strip()

    if not bool(memory_config.get("enabled", True)):
        print("Persistent memory is disabled; no memory store was created.")
        return
    if not store_name:
        raise ValueError("databricks_resources.memory_store_name is required")

    store = CatalogMemoryStore(
        full_name=store_name,
        enabled=True,
        fail_open=False,
        timeout_seconds=int(memory_config.get("timeout_seconds", 20)),
    )
    store.ensure_store(
        description=(
            "Persistent session state for the Patient History Navigator academic "
            "RAG project. Contains synthetic-data workflow context only."
        )
    )
    if store.last_error:
        raise RuntimeError(store.last_error)
    print(f"Managed Unity Catalog memory store is ready: {store_name}")
    print(
        "The Model Serving identity must receive READ MEMORY STORE and WRITE MEMORY "
        "STORE on this securable before persistent memory works in production."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Resource setup failed: {exc}", file=sys.stderr)
        raise
