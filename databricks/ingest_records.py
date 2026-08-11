"""Patient History Navigator ingestion and Databricks AI Search preprocessing.

This Databricks job:
1. discovers PDF, Markdown, text, and Synthea FHIR JSON files in a Unity Catalog volume;
2. tries selectable PDF extraction with pypdf;
3. automatically falls back to Databricks ai_parse_document when pypdf text is unreadable;
4. converts each source into patient-scoped document records;
5. splits text with LangChain RecursiveCharacterTextSplitter;
6. creates embeddings through a Databricks Model Serving embedding endpoint;
7. writes full documents and chunks to Unity Catalog Delta tables; and
8. creates or synchronizes the Databricks AI Search index.

The previous manual linear character chunker is preserved as commented history near
`split_record_with_langchain`, but it is no longer executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml
from databricks.ai_search.client import AISearchClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import current_timestamp
from pyspark.sql.types import (
    ArrayType,
    DateType,
    DoubleType,
    FloatType,
    StringType,
    StructField,
    StructType,
)


def _runtime_directory() -> Path:
    module_file = globals().get("__file__")
    if module_file:
        return Path(str(module_file)).resolve().parent
    if sys.argv and sys.argv[0] and sys.argv[0] not in {"-c", "-m", ""}:
        candidate = Path(sys.argv[0])
        if candidate.exists():
            return candidate.resolve().parent
    return Path.cwd()


ROOT = _runtime_directory()
DEFAULT_CONFIG_PATH = ROOT / "navigator_config.yaml"


@dataclass(frozen=True)
class Settings:
    volume_path: str
    file_glob: str
    supported_extensions: tuple[str, ...]
    sidecar_suffix: str
    default_patient_id: str
    ocr_mode: str
    ocr_page_batch_size: int
    min_total_chars: int
    min_chars_per_page: int
    min_readable_page_ratio: float
    min_alpha_ratio: float
    parsed_cache_table: str
    documents_table: str
    processed_documents_table: str
    source_table: str
    embedding_endpoint_name: str
    vector_search_endpoint_name: str
    vector_search_index: str
    chunk_size: int
    chunk_overlap: int
    chunk_separators: tuple[str, ...]
    embedding_batch_size: int
    embedding_request_interval_seconds: float
    embedding_retry_timeout_seconds: int
    embedding_max_attempts: int
    embedding_backoff_seconds: float
    embedding_max_backoff_seconds: float
    create_index_if_missing: bool
    sync_index: bool
    force: bool


@dataclass
class DocumentRecord:
    document_id: str
    patient_id: str
    document_type: str
    document_date: date | None
    section: str
    source_path: str
    source_doc: str
    full_text: str
    extraction_method: str
    event_name: str = ""
    numeric_value: float | None = None
    unit: str = ""
    status: str = ""
    content_hash: str = ""


@dataclass
class ChunkRecord:
    chunk_id: str
    patient_id: str
    document_id: str
    document_type: str
    document_date: date | None
    section: str
    source_path: str
    source_doc: str
    text_chunk: str
    retrieval_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--volume-path")
    parser.add_argument("--file-glob")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--create-index-if-missing", action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--sync-index", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return value


def _bool_override(value: bool | None, default: bool) -> bool:
    return default if value is None else value


def build_settings(args: argparse.Namespace) -> Settings:
    config = load_yaml(args.config_file)
    preprocessing = config.get("preprocessing_config", {})
    resources = config.get("databricks_resources", {})
    retriever = config.get("retriever_config", {})
    quality = preprocessing.get("pypdf_quality", {})
    separators = preprocessing.get(
        "chunk_separators", ["\n\n", "\n", ". ", "; ", ", ", " ", ""]
    )
    if not isinstance(separators, list):
        raise ValueError("chunk_separators must be a YAML list")
    extensions = preprocessing.get(
        "supported_extensions", [".pdf", ".txt", ".md", ".json"]
    )
    settings = Settings(
        volume_path=args.volume_path
        or preprocessing.get(
            "volume_path", "/Volumes/workspace/default/rag/patient_history"
        ),
        file_glob=args.file_glob or preprocessing.get("file_glob", "**/*"),
        supported_extensions=tuple(str(item).lower() for item in extensions),
        sidecar_suffix=str(preprocessing.get("sidecar_suffix", ".metadata.json")),
        default_patient_id=str(
            preprocessing.get("default_patient_id", "demo-patient")
        ),
        ocr_mode=str(preprocessing.get("ocr_mode", "auto")).lower(),
        ocr_page_batch_size=int(preprocessing.get("ocr_page_batch_size", 500)),
        min_total_chars=int(quality.get("min_total_chars", 200)),
        min_chars_per_page=int(quality.get("min_chars_per_page", 30)),
        min_readable_page_ratio=float(
            quality.get("min_readable_page_ratio", 0.60)
        ),
        min_alpha_ratio=float(quality.get("min_alpha_ratio", 0.20)),
        parsed_cache_table=str(
            preprocessing.get(
                "parsed_cache_table",
                "workspace.default.patient_history_parsed_documents",
            )
        ),
        documents_table=str(
            preprocessing.get(
                "documents_table", "workspace.default.patient_history_documents"
            )
        ),
        processed_documents_table=str(
            preprocessing.get(
                "processed_documents_table",
                "workspace.default.patient_history_processed_documents",
            )
        ),
        source_table=str(
            preprocessing.get(
                "source_table", "workspace.default.patient_history_rag_chunks"
            )
        ),
        embedding_endpoint_name=str(
            resources.get("embedding_endpoint_name", "databricks-qwen3-embedding-0-6b")
        ),
        vector_search_endpoint_name=str(
            resources.get("vector_search_endpoint_name", "rag-vs-endpoint")
        ),
        vector_search_index=str(
            retriever.get(
                "vector_search_index",
                "workspace.default.patient_history_chunks_index",
            )
        ),
        chunk_size=int(preprocessing.get("chunk_size", 900)),
        chunk_overlap=int(preprocessing.get("chunk_overlap", 150)),
        chunk_separators=tuple(str(item) for item in separators),
        embedding_batch_size=int(preprocessing.get("embedding_batch_size", 1)),
        embedding_request_interval_seconds=float(
            preprocessing.get("embedding_request_interval_seconds", 5.0)
        ),
        embedding_retry_timeout_seconds=int(
            preprocessing.get("embedding_retry_timeout_seconds", 20)
        ),
        embedding_max_attempts=int(
            preprocessing.get("embedding_max_attempts", 8)
        ),
        embedding_backoff_seconds=float(
            preprocessing.get("embedding_backoff_seconds", 15.0)
        ),
        embedding_max_backoff_seconds=float(
            preprocessing.get("embedding_max_backoff_seconds", 120.0)
        ),
        create_index_if_missing=_bool_override(
            args.create_index_if_missing,
            bool(preprocessing.get("create_index_if_missing", True)),
        ),
        sync_index=_bool_override(
            args.sync_index, bool(preprocessing.get("sync_index", True))
        ),
        force=bool(args.force),
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    if not settings.volume_path.startswith("/Volumes/"):
        raise ValueError("volume_path must be a Unity Catalog /Volumes/... path")
    if settings.ocr_mode not in {"auto", "always", "never"}:
        raise ValueError("ocr_mode must be auto, always, or never")
    if settings.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= settings.chunk_overlap < settings.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if settings.ocr_page_batch_size <= 0:
        raise ValueError("ocr_page_batch_size must be positive")
    if settings.embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size must be positive")


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def content_hash(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def stable_id(*parts: str) -> str:
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Unity Catalog tables
# ---------------------------------------------------------------------------


def ensure_tables(spark: SparkSession, settings: Settings) -> None:
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {settings.parsed_cache_table} (
          source_path STRING NOT NULL,
          source_hash STRING NOT NULL,
          page_range STRING NOT NULL,
          parsed VARIANT,
          parsed_at TIMESTAMP NOT NULL
        )
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {settings.processed_documents_table} (
          source_path STRING NOT NULL,
          source_hash STRING NOT NULL,
          processing_signature STRING NOT NULL,
          document_count INT NOT NULL,
          chunk_count INT NOT NULL,
          processed_at TIMESTAMP NOT NULL
        )
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {settings.documents_table} (
          document_id STRING NOT NULL,
          patient_id STRING NOT NULL,
          document_type STRING NOT NULL,
          document_date DATE,
          section STRING,
          source_path STRING NOT NULL,
          source_doc STRING NOT NULL,
          full_text STRING NOT NULL,
          extraction_method STRING NOT NULL,
          event_name STRING,
          numeric_value DOUBLE,
          unit STRING,
          status STRING,
          content_hash STRING NOT NULL,
          processed_at TIMESTAMP NOT NULL
        )
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {settings.source_table} (
          chunk_id STRING NOT NULL,
          patient_id STRING NOT NULL,
          document_id STRING NOT NULL,
          document_type STRING NOT NULL,
          document_date DATE,
          section STRING,
          source_path STRING NOT NULL,
          source_doc STRING NOT NULL,
          text_chunk STRING NOT NULL,
          retrieval_text STRING NOT NULL,
          embedding ARRAY<FLOAT> NOT NULL,
          processed_at TIMESTAMP NOT NULL
        )
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """
    )
    spark.sql(
        f"ALTER TABLE {settings.source_table} "
        "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )
    _ensure_primary_key(spark, settings.source_table, "chunk_id", "patient_history_chunks_pk")


def _ensure_primary_key(
    spark: SparkSession, table_name: str, column_name: str, constraint_name: str
) -> None:
    parts = table_name.split(".")
    if len(parts) != 3:
        return
    catalog, schema, table = parts
    rows = spark.sql(
        f"""
        SELECT constraint_name
        FROM {catalog}.information_schema.table_constraints
        WHERE table_schema = {sql_string(schema)}
          AND table_name = {sql_string(table)}
          AND constraint_type = 'PRIMARY KEY'
        """
    ).collect()
    if rows:
        return
    spark.sql(
        f"ALTER TABLE {table_name} ADD CONSTRAINT {constraint_name} "
        f"PRIMARY KEY ({column_name}) NOT ENFORCED"
    )


# ---------------------------------------------------------------------------
# File discovery and metadata
# ---------------------------------------------------------------------------


def discover_files(settings: Settings) -> list[Path]:
    root = Path(settings.volume_path)
    if not root.exists():
        raise FileNotFoundError(f"Unity Catalog volume path does not exist: {root}")
    files = []
    for path in root.glob(settings.file_glob):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if lower.endswith(settings.sidecar_suffix.lower()):
            continue
        if path.suffix.lower() in settings.supported_extensions:
            files.append(path)
    return sorted(files)


def load_sidecar(path: Path, settings: Settings) -> dict[str, Any]:
    sidecar = Path(str(path) + settings.sidecar_suffix)
    if not sidecar.exists():
        return {}
    with sidecar.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Metadata sidecar must contain an object: {sidecar}")
    return value


def metadata_from_filename(path: Path) -> dict[str, Any]:
    # Recommended convention:
    # patient-id__2025-03-01__lab_result__cbc.pdf
    parts = path.stem.split("__")
    result: dict[str, Any] = {}
    if len(parts) >= 1 and parts[0]:
        result["patient_id"] = parts[0]
    if len(parts) >= 2 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", parts[1]):
        result["document_date"] = parts[1]
    if len(parts) >= 3 and parts[2]:
        result["document_type"] = parts[2]
    if len(parts) >= 4 and parts[3]:
        result["document_id"] = parts[3]
    return result


def parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    text = str(value)[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# PDF extraction: pypdf first, Databricks ai_parse_document fallback
# ---------------------------------------------------------------------------


def extract_pdf_with_pypdf(path: Path) -> tuple[str, int, dict[str, Any]]:
    import pypdf

    pages: list[str] = []
    failures = 0
    with path.open("rb") as handle:
        reader = pypdf.PdfReader(handle)
        page_count = len(reader.pages)
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                pages.append(page.extract_text() or "")
            except Exception as exc:
                failures += 1
                print(f"pypdf could not extract page {page_number}: {exc}")
                pages.append("")
    text = "\n\n".join(
        f"[Page {number}]\n{page_text}" for number, page_text in enumerate(pages, 1)
    )
    return text, page_count, {"page_texts": pages, "failures": failures}


def pypdf_text_is_readable(
    text: str,
    diagnostics: dict[str, Any],
    settings: Settings,
) -> tuple[bool, dict[str, float]]:
    page_texts = diagnostics.get("page_texts") or []
    compact = re.sub(r"\s+", "", text)
    alpha_count = sum(character.isalpha() for character in compact)
    alpha_ratio = alpha_count / max(len(compact), 1)
    readable_pages = sum(
        1
        for page in page_texts
        if len(re.sub(r"\s+", "", str(page))) >= settings.min_chars_per_page
    )
    readable_page_ratio = readable_pages / max(len(page_texts), 1)
    metrics = {
        "characters": float(len(compact)),
        "alpha_ratio": alpha_ratio,
        "readable_page_ratio": readable_page_ratio,
    }
    readable = (
        len(compact) >= settings.min_total_chars
        and alpha_ratio >= settings.min_alpha_ratio
        and readable_page_ratio >= settings.min_readable_page_ratio
    )
    return readable, metrics


def page_ranges(page_count: int, batch_size: int) -> list[str]:
    if page_count <= 0:
        return ["1-500"]
    ranges = []
    for start in range(1, page_count + 1, batch_size):
        end = min(start + batch_size - 1, page_count)
        ranges.append(f"{start}-{end}")
    return ranges


def cached_ocr_text(
    spark: SparkSession,
    settings: Settings,
    source_path: str,
    source_hash: str,
    page_range: str,
) -> str | None:
    rows = spark.sql(
        f"""
        SELECT concat_ws('\n\n',
          transform(
            try_cast(parsed:document:elements AS ARRAY<VARIANT>),
            element -> concat(
              '[',
              coalesce(try_cast(element:type AS STRING), 'text'),
              ']\n',
              coalesce(
                try_cast(element:content AS STRING),
                try_cast(element:description AS STRING),
                ''
              )
            )
          )
        ) AS full_text
        FROM {settings.parsed_cache_table}
        WHERE source_path = {sql_string(source_path)}
          AND source_hash = {sql_string(source_hash)}
          AND page_range = {sql_string(page_range)}
          AND (
            try_cast(parsed:error_status AS ARRAY<VARIANT>) IS NULL
            OR size(try_cast(parsed:error_status AS ARRAY<VARIANT>)) = 0
          )
        ORDER BY parsed_at DESC
        LIMIT 1
        """
    ).collect()
    if not rows:
        return None
    return str(rows[0]["full_text"] or "")


def parse_and_cache_ocr(
    spark: SparkSession,
    settings: Settings,
    source_path: str,
    source_hash: str,
    page_range: str,
) -> str:
    print(f"Running ai_parse_document on {source_path}, pages {page_range}...")
    spark.sql(
        f"""
        INSERT INTO {settings.parsed_cache_table}
        SELECT
          path AS source_path,
          {sql_string(source_hash)} AS source_hash,
          {sql_string(page_range)} AS page_range,
          ai_parse_document(
            content,
            MAP(
              'version', '2.0',
              'pageRange', {sql_string(page_range)},
              'descriptionElementTypes', ''
            )
          ) AS parsed,
          current_timestamp() AS parsed_at
        FROM READ_FILES({sql_string(source_path)}, format => 'binaryFile')
        """
    )
    text = cached_ocr_text(
        spark, settings, source_path, source_hash, page_range
    )
    if text is None:
        raise RuntimeError(
            f"ai_parse_document did not return usable text for {source_path}, {page_range}"
        )
    return text


def extract_pdf(
    spark: SparkSession,
    settings: Settings,
    path: Path,
    source_hash: str,
) -> tuple[str, str]:
    # pypdf can fail before page extraction for encrypted, damaged, or unusual PDFs.
    # In auto/always mode that is treated the same as unreadable text and triggers
    # the Databricks parser fallback instead of aborting the whole ingestion job.
    try:
        pypdf_text, page_count, diagnostics = extract_pdf_with_pypdf(path)
        pypdf_error: str | None = None
    except Exception as exc:
        pypdf_text, page_count = "", 0
        diagnostics = {"page_texts": [], "failures": 1}
        pypdf_error = f"{type(exc).__name__}: {exc}"
        print(f"pypdf failed for {path}; falling back when OCR is enabled: {pypdf_error}")

    readable, metrics = pypdf_text_is_readable(pypdf_text, diagnostics, settings)
    print(
        "pypdf readability: "
        + json.dumps(
            {**metrics, "readable": readable, "error": pypdf_error},
            sort_keys=True,
        )
    )

    if settings.ocr_mode == "never":
        if not pypdf_text.strip():
            detail = f" ({pypdf_error})" if pypdf_error else ""
            raise RuntimeError(f"pypdf extracted no text from {path}{detail}")
        return pypdf_text, "pypdf"
    if settings.ocr_mode == "auto" and readable:
        return pypdf_text, "pypdf"

    texts = []
    for page_range in page_ranges(page_count, settings.ocr_page_batch_size):
        text = cached_ocr_text(
            spark, settings, str(path), source_hash, page_range
        )
        if text is None:
            text = parse_and_cache_ocr(
                spark, settings, str(path), source_hash, page_range
            )
        else:
            print(f"Using cached ai_parse_document output for pages {page_range}.")
        texts.append(text)
    ocr_text = "\n\n".join(texts).strip()
    if not ocr_text:
        raise RuntimeError(f"OCR/layout parsing produced no text for {path}")
    return ocr_text, "databricks_ai_parse_document"


# ---------------------------------------------------------------------------
# FHIR conversion
# ---------------------------------------------------------------------------


def _coding_display(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    if value.get("text"):
        return str(value["text"])
    for coding in value.get("coding") or []:
        if isinstance(coding, dict):
            if coding.get("display"):
                return str(coding["display"])
            if coding.get("code"):
                return str(coding["code"])
    return ""


def _reference_id(reference: Any) -> str:
    if isinstance(reference, dict):
        text = str(reference.get("reference", ""))
    else:
        text = str(reference or "")
    return text.split("/")[-1] if text else ""


def _fhir_date(resource: dict[str, Any]) -> date | None:
    candidates = [
        resource.get("effectiveDateTime"),
        (resource.get("effectivePeriod") or {}).get("start")
        if isinstance(resource.get("effectivePeriod"), dict)
        else None,
        resource.get("issued"),
        resource.get("authoredOn"),
        resource.get("recordedDate"),
        resource.get("onsetDateTime"),
        resource.get("performedDateTime"),
        (resource.get("performedPeriod") or {}).get("start")
        if isinstance(resource.get("performedPeriod"), dict)
        else None,
        (resource.get("period") or {}).get("start")
        if isinstance(resource.get("period"), dict)
        else None,
        resource.get("date"),
        resource.get("occurrenceDateTime"),
        resource.get("birthDate"),
    ]
    for candidate in candidates:
        parsed = parse_date(candidate)
        if parsed:
            return parsed
    return None


def _fhir_event_name(resource: dict[str, Any]) -> str:
    resource_type = str(resource.get("resourceType", "FHIRResource"))
    for key in ("code", "type", "category"):
        value = resource.get(key)
        if isinstance(value, list):
            for item in value:
                display = _coding_display(item)
                if display:
                    return display
        else:
            display = _coding_display(value)
            if display:
                return display
    medication = resource.get("medicationCodeableConcept")
    display = _coding_display(medication)
    if display:
        return display
    if isinstance(resource.get("vaccineCode"), dict):
        display = _coding_display(resource["vaccineCode"])
        if display:
            return display
    return resource_type


def _flatten_json(value: Any, prefix: str = "", limit: int = 240) -> list[str]:
    lines: list[str] = []

    def visit(item: Any, path: str) -> None:
        if len(lines) >= limit:
            return
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(item, list):
            for index, child in enumerate(item[:25]):
                visit(child, f"{path}[{index}]")
        elif item not in (None, ""):
            lines.append(f"{path}: {item}")

    visit(value, prefix)
    if len(lines) >= limit:
        lines.append("... additional FHIR fields omitted ...")
    return lines


def records_from_fhir_bundle(
    path: Path,
    source_hash: str,
    default_patient_id: str,
) -> list[DocumentRecord]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    resources: list[dict[str, Any]] = []
    if isinstance(payload, dict) and payload.get("resourceType") == "Bundle":
        for entry in payload.get("entry") or []:
            resource = entry.get("resource") if isinstance(entry, dict) else None
            if isinstance(resource, dict):
                resources.append(resource)
    elif isinstance(payload, dict) and payload.get("resourceType"):
        resources = [payload]
    else:
        raise ValueError(f"JSON is not a FHIR resource or Bundle: {path}")

    patient_ids = [
        str(resource.get("id"))
        for resource in resources
        if resource.get("resourceType") == "Patient" and resource.get("id")
    ]
    bundle_patient_id = patient_ids[0] if patient_ids else default_patient_id
    records: list[DocumentRecord] = []
    for resource in resources:
        resource_type = str(resource.get("resourceType", "FHIRResource"))
        resource_id = str(resource.get("id") or uuid.uuid4().hex)
        patient_id = (
            _reference_id(resource.get("subject"))
            or _reference_id(resource.get("patient"))
            or (resource_id if resource_type == "Patient" else "")
            or bundle_patient_id
        )
        event_name = _fhir_event_name(resource)
        observed_date = _fhir_date(resource)
        value_quantity = resource.get("valueQuantity")
        numeric_value = None
        unit = ""
        if isinstance(value_quantity, dict):
            try:
                numeric_value = float(value_quantity.get("value"))
            except (TypeError, ValueError):
                numeric_value = None
            unit = str(value_quantity.get("unit") or value_quantity.get("code") or "")
        status = str(resource.get("status") or resource.get("clinicalStatus", ""))
        if isinstance(resource.get("clinicalStatus"), dict):
            status = _coding_display(resource["clinicalStatus"])
        header = [
            f"FHIR resource type: {resource_type}",
            f"Patient ID: {patient_id}",
            f"Document ID: {resource_type}-{resource_id}",
            f"Date: {observed_date.isoformat() if observed_date else 'Unknown'}",
            f"Event: {event_name}",
            f"Status: {status or 'Unknown'}",
        ]
        body = "\n".join(_flatten_json(resource))
        full_text = "\n".join(header) + "\n\nFHIR details:\n" + body
        records.append(
            DocumentRecord(
                document_id=f"{resource_type}-{resource_id}",
                patient_id=patient_id,
                document_type=resource_type,
                document_date=observed_date,
                section="FHIR resource",
                source_path=str(path),
                source_doc=path.name,
                full_text=full_text,
                extraction_method="synthea_fhir_json",
                event_name=event_name,
                numeric_value=numeric_value,
                unit=unit,
                status=status,
                content_hash=content_hash(full_text),
            )
        )
    return records


# ---------------------------------------------------------------------------
# Plain documents and chunking
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def records_from_plain_file(
    spark: SparkSession,
    settings: Settings,
    path: Path,
    source_hash: str,
) -> list[DocumentRecord]:
    metadata = {**metadata_from_filename(path), **load_sidecar(path, settings)}
    if path.suffix.lower() == ".pdf":
        text, extraction_method = extract_pdf(
            spark, settings, path, source_hash
        )
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        extraction_method = "plain_text"
    text = normalize_text(text)
    if not text:
        raise RuntimeError(f"No text was extracted from {path}")

    patient_id = str(metadata.get("patient_id") or settings.default_patient_id)
    document_type = str(metadata.get("document_type") or path.suffix.lstrip(".") or "document")
    document_date = parse_date(metadata.get("document_date"))
    document_id = str(
        metadata.get("document_id")
        or stable_id(patient_id, str(path), source_hash)[:24]
    )
    section = str(metadata.get("section") or "Document")
    event_name = str(metadata.get("event_name") or document_type)
    numeric_value = metadata.get("numeric_value")
    try:
        numeric_value = float(numeric_value) if numeric_value is not None else None
    except (TypeError, ValueError):
        numeric_value = None
    return [
        DocumentRecord(
            document_id=document_id,
            patient_id=patient_id,
            document_type=document_type,
            document_date=document_date,
            section=section,
            source_path=str(path),
            source_doc=path.name,
            full_text=text,
            extraction_method=extraction_method,
            event_name=event_name,
            numeric_value=numeric_value,
            unit=str(metadata.get("unit") or ""),
            status=str(metadata.get("status") or ""),
            content_hash=content_hash(text),
        )
    ]


# Legacy manual chunking retained for project history only. It is deliberately
# commented out and is NOT executed. The active implementation below uses the
# LangChain RecursiveCharacterTextSplitter recommended in Databricks RAG examples.
#
# def legacy_linear_chunk_text(text: str, size: int, overlap: int) -> list[str]:
#     step = size - overlap
#     chunks = []
#     for start in range(0, len(text), step):
#         chunk = text[start:start + size].strip()
#         if chunk:
#             chunks.append(chunk)
#         if start + size >= len(text):
#             break
#     return chunks


def split_record_with_langchain(
    record: DocumentRecord, settings: Settings
) -> list[ChunkRecord]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=list(settings.chunk_separators),
        length_function=len,
        add_start_index=True,
        strip_whitespace=True,
    )
    metadata = {
        "patient_id": record.patient_id,
        "document_id": record.document_id,
        "document_type": record.document_type,
        "document_date": (
            record.document_date.isoformat() if record.document_date else ""
        ),
        "section": record.section,
        "source_path": record.source_path,
        "source_doc": record.source_doc,
    }
    documents: list[Document] = splitter.create_documents(
        [record.full_text], metadatas=[metadata]
    )
    chunks = []
    for index, document in enumerate(documents):
        text = normalize_text(document.page_content)
        if not text:
            continue
        start_index = int(document.metadata.get("start_index", 0))
        chunk_id = stable_id(
            record.patient_id,
            record.document_id,
            str(index),
            str(start_index),
            content_hash(text),
        )
        retrieval_text = "\n".join(
            [
                f"Patient ID: {record.patient_id}",
                f"Document type: {record.document_type}",
                f"Document date: {record.document_date.isoformat() if record.document_date else 'Unknown'}",
                f"Event: {record.event_name}",
                f"Section: {record.section}",
                text,
            ]
        )
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                patient_id=record.patient_id,
                document_id=record.document_id,
                document_type=record.document_type,
                document_date=record.document_date,
                section=record.section,
                source_path=record.source_path,
                source_doc=record.source_doc,
                text_chunk=text,
                retrieval_text=retrieval_text,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Embedding and writes
# ---------------------------------------------------------------------------


def batched(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _is_rate_limit_error(exc: BaseException) -> bool:
    parts = []
    current: BaseException | None = exc
    while current is not None:
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__
    text = " ".join(parts).lower()
    return any(
        marker in text
        for marker in (
            "request_limit_exceeded",
            "requestlimitexceeded",
            "rate limit",
            "too many requests",
            "http 429",
            "status code 429",
            "timed out after",
        )
    )


def embed_texts(settings: Settings, texts: Sequence[str]) -> list[list[float]]:
    client = WorkspaceClient(
        config=Config(
            retry_timeout_seconds=settings.embedding_retry_timeout_seconds,
            rate_limit=1,
        )
    )
    embeddings: list[list[float]] = []
    batches = list(batched(texts, settings.embedding_batch_size))
    last_request_started: float | None = None
    for batch_number, batch in enumerate(batches, start=1):
        for attempt in range(1, settings.embedding_max_attempts + 1):
            if last_request_started is not None:
                remaining = settings.embedding_request_interval_seconds - (
                    time.monotonic() - last_request_started
                )
                if remaining > 0:
                    time.sleep(remaining)
            try:
                last_request_started = time.monotonic()
                response = client.serving_endpoints.query(
                    name=settings.embedding_endpoint_name,
                    input=list(batch),
                )
                data = list(response.data or [])
                data.sort(key=lambda item: int(getattr(item, "index", 0) or 0))
                if len(data) != len(batch):
                    raise RuntimeError(
                        f"Embedding endpoint returned {len(data)} vectors for {len(batch)} inputs"
                    )
                for item in data:
                    vector = getattr(item, "embedding", None)
                    if vector is None:
                        raise RuntimeError("Embedding response had no vector")
                    embeddings.append([float(value) for value in vector])
                print(
                    f"Embedded {len(embeddings)}/{len(texts)} chunks "
                    f"(batch {batch_number}/{len(batches)})."
                )
                break
            except Exception as exc:
                if not _is_rate_limit_error(exc) or attempt >= settings.embedding_max_attempts:
                    raise
                delay = min(
                    settings.embedding_backoff_seconds * (2 ** (attempt - 1)),
                    settings.embedding_max_backoff_seconds,
                )
                delay += random.uniform(0, min(5.0, delay * 0.1))
                print(f"Embedding rate limit; retrying in {delay:.1f}s...")
                time.sleep(delay)
    dimensions = {len(vector) for vector in embeddings}
    if len(dimensions) != 1:
        raise RuntimeError(f"Inconsistent embedding dimensions: {sorted(dimensions)}")
    return embeddings


def source_is_current(
    spark: SparkSession,
    settings: Settings,
    source_path: str,
    source_hash: str,
    signature: str,
) -> bool:
    rows = spark.sql(
        f"""
        SELECT source_hash, processing_signature
        FROM {settings.processed_documents_table}
        WHERE source_path = {sql_string(source_path)}
        ORDER BY processed_at DESC
        LIMIT 1
        """
    ).collect()
    return bool(
        rows
        and str(rows[0]["source_hash"]) == source_hash
        and str(rows[0]["processing_signature"]) == signature
    )


def write_source_records(
    spark: SparkSession,
    settings: Settings,
    source_path: str,
    source_hash: str,
    signature: str,
    documents: list[DocumentRecord],
    chunks: list[ChunkRecord],
    embeddings: list[list[float]],
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk and embedding counts differ")

    spark.sql(
        f"DELETE FROM {settings.documents_table} WHERE source_path = {sql_string(source_path)}"
    )
    spark.sql(
        f"DELETE FROM {settings.source_table} WHERE source_path = {sql_string(source_path)}"
    )
    spark.sql(
        f"DELETE FROM {settings.processed_documents_table} WHERE source_path = {sql_string(source_path)}"
    )

    document_schema = StructType(
        [
            StructField("document_id", StringType(), False),
            StructField("patient_id", StringType(), False),
            StructField("document_type", StringType(), False),
            StructField("document_date", DateType(), True),
            StructField("section", StringType(), True),
            StructField("source_path", StringType(), False),
            StructField("source_doc", StringType(), False),
            StructField("full_text", StringType(), False),
            StructField("extraction_method", StringType(), False),
            StructField("event_name", StringType(), True),
            StructField("numeric_value", DoubleType(), True),
            StructField("unit", StringType(), True),
            StructField("status", StringType(), True),
            StructField("content_hash", StringType(), False),
        ]
    )
    document_rows = [
        (
            record.document_id,
            record.patient_id,
            record.document_type,
            record.document_date,
            record.section,
            record.source_path,
            record.source_doc,
            record.full_text,
            record.extraction_method,
            record.event_name,
            record.numeric_value,
            record.unit,
            record.status,
            record.content_hash,
        )
        for record in documents
    ]
    spark.createDataFrame(document_rows, document_schema).withColumn(
        "processed_at", current_timestamp()
    ).write.mode("append").saveAsTable(settings.documents_table)

    chunk_schema = StructType(
        [
            StructField("chunk_id", StringType(), False),
            StructField("patient_id", StringType(), False),
            StructField("document_id", StringType(), False),
            StructField("document_type", StringType(), False),
            StructField("document_date", DateType(), True),
            StructField("section", StringType(), True),
            StructField("source_path", StringType(), False),
            StructField("source_doc", StringType(), False),
            StructField("text_chunk", StringType(), False),
            StructField("retrieval_text", StringType(), False),
            StructField("embedding", ArrayType(FloatType(), False), False),
        ]
    )
    chunk_rows = [
        (
            chunk.chunk_id,
            chunk.patient_id,
            chunk.document_id,
            chunk.document_type,
            chunk.document_date,
            chunk.section,
            chunk.source_path,
            chunk.source_doc,
            chunk.text_chunk,
            chunk.retrieval_text,
            [float(value) for value in vector],
        )
        for chunk, vector in zip(chunks, embeddings, strict=True)
    ]
    spark.createDataFrame(chunk_rows, chunk_schema).withColumn(
        "processed_at", current_timestamp()
    ).write.mode("append").saveAsTable(settings.source_table)

    metadata = spark.createDataFrame(
        [
            (
                source_path,
                source_hash,
                signature,
                len(documents),
                len(chunks),
            )
        ],
        "source_path STRING, source_hash STRING, processing_signature STRING, "
        "document_count INT, chunk_count INT",
    ).withColumn("processed_at", current_timestamp())
    metadata.write.mode("append").saveAsTable(settings.processed_documents_table)


# ---------------------------------------------------------------------------
# AI Search index
# ---------------------------------------------------------------------------


def ensure_and_sync_index(settings: Settings, embedding_dimension: int | None) -> None:
    client = AISearchClient(disable_notice=True)
    if not client.endpoint_exists(settings.vector_search_endpoint_name):
        if not settings.create_index_if_missing:
            raise RuntimeError(
                f"AI Search endpoint does not exist: {settings.vector_search_endpoint_name}"
            )
        client.create_endpoint_and_wait(
            name=settings.vector_search_endpoint_name,
            endpoint_type="STANDARD",
            verbose=True,
            timeout=timedelta(hours=1),
        )

    if not client.index_exists(index_name=settings.vector_search_index):
        if not settings.create_index_if_missing:
            raise RuntimeError(
                f"AI Search index does not exist: {settings.vector_search_index}"
            )
        if embedding_dimension is None:
            raise RuntimeError("Embedding dimension is required to create the index")
        index = client.create_delta_sync_index_and_wait(
            endpoint_name=settings.vector_search_endpoint_name,
            source_table_name=settings.source_table,
            index_name=settings.vector_search_index,
            pipeline_type="TRIGGERED",
            primary_key="chunk_id",
            embedding_dimension=embedding_dimension,
            embedding_vector_column="embedding",
            columns_to_sync=[
                "text_chunk",
                "retrieval_text",
                "patient_id",
                "document_id",
                "document_type",
                "document_date",
                "section",
                "source_path",
                "source_doc",
            ],
            verbose=True,
            timeout=timedelta(hours=2),
        )
    else:
        index = client.get_index(index_name=settings.vector_search_index)

    if settings.sync_index:
        index.sync()
        index.wait_until_ready(
            verbose=True,
            wait_for_updates=True,
            timeout=timedelta(hours=2),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    settings = build_settings(args)
    spark = SparkSession.builder.getOrCreate()
    ensure_tables(spark, settings)

    files = discover_files(settings)
    if not files:
        raise RuntimeError(
            f"No supported files were found under {settings.volume_path}. "
            "Upload Synthea FHIR JSON, PDFs, Markdown, or text files first."
        )
    print(f"Discovered {len(files)} source files.")

    signature = content_hash(
        json.dumps(
            {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "chunk_separators": settings.chunk_separators,
                "embedding_endpoint": settings.embedding_endpoint_name,
                "ocr_mode": settings.ocr_mode,
                "pipeline_version": 2,
            },
            sort_keys=True,
        )
    )
    embedding_dimension: int | None = None
    processed_files = 0
    skipped_files = 0

    for path in files:
        raw = path.read_bytes()
        source_hash = content_hash(raw)
        if (
            source_is_current(
                spark, settings, str(path), source_hash, signature
            )
            and not settings.force
        ):
            print(f"Unchanged; skipping {path.name}")
            skipped_files += 1
            continue

        print(f"Processing {path}")
        if path.suffix.lower() == ".json":
            try:
                documents = records_from_fhir_bundle(
                    path, source_hash, settings.default_patient_id
                )
            except ValueError:
                # Non-FHIR JSON is treated as readable text so it remains searchable.
                documents = records_from_plain_file(
                    spark, settings, path, source_hash
                )
        else:
            documents = records_from_plain_file(
                spark, settings, path, source_hash
            )

        chunks = [
            chunk
            for document in documents
            for chunk in split_record_with_langchain(document, settings)
        ]
        if not chunks:
            raise RuntimeError(f"Chunking produced no content for {path}")
        embeddings = embed_texts(
            settings, [chunk.retrieval_text for chunk in chunks]
        )
        embedding_dimension = len(embeddings[0])
        write_source_records(
            spark,
            settings,
            str(path),
            source_hash,
            signature,
            documents,
            chunks,
            embeddings,
        )
        print(
            f"Stored {len(documents)} documents and {len(chunks)} chunks for {path.name}."
        )
        processed_files += 1

    if embedding_dimension is None:
        rows = spark.sql(
            f"SELECT size(embedding) AS dimensions FROM {settings.source_table} LIMIT 1"
        ).collect()
        if rows:
            embedding_dimension = int(rows[0]["dimensions"])

    ensure_and_sync_index(settings, embedding_dimension)
    print(
        f"Preprocessing complete: {processed_files} processed, {skipped_files} reused."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Preprocessing failed: {exc}", file=sys.stderr)
        raise
