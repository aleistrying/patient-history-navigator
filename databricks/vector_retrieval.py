"""Advanced patient-isolated retrieval over Databricks AI Search.

The retriever runs each expanded query, applies a mandatory patient_id filter,
deduplicates chunks, and combines rankings with Reciprocal Rank Fusion (RRF).
It first attempts Databricks built-in hybrid search. For a self-managed embedding
index, it falls back to an ANN query plus a full-text query and fuses both lists.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from databricks.ai_search.client import AISearchClient
from databricks_langchain import DatabricksEmbeddings

from ttl_cache import TTLCache

try:
    from databricks.ai_search.reranker import DatabricksReranker
except Exception:  # Older package versions may not expose the reranker helper.
    DatabricksReranker = None  # type: ignore[assignment]


@dataclass
class SearchHit:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    rrf_score: float = 0.0


class AdvancedRetriever:
    def __init__(
        self,
        resources: dict[str, Any],
        config: dict[str, Any],
        cache_config: dict[str, Any],
    ):
        self.config = config
        self.schema = config["schema"]
        self.index_name = config["vector_search_index"]
        self.embedding = DatabricksEmbeddings(
            endpoint=resources["embedding_endpoint_name"]
        )
        self.index = AISearchClient(disable_notice=True).get_index(
            index_name=self.index_name
        )
        self.cache = TTLCache[str, list[SearchHit]](
            ttl_seconds=float(cache_config.get("retrieval_ttl_seconds", 300)),
            max_entries=int(cache_config.get("max_retrieval_entries", 512)),
        )
        # Deduplicated, order-preserving: the index rejects a query that names the
        # same column twice, and schema["document_uri"] is itself "source_path"
        # under the default config.
        self.columns = list(
            dict.fromkeys(
                [
                    self.schema["primary_key"],
                    self.schema["chunk_text"],
                    self.schema["document_uri"],
                    "patient_id",
                    "document_id",
                    "document_type",
                    "document_date",
                    "section",
                    "source_path",
                ]
            )
        )

    def retrieve(
        self,
        queries: list[str],
        patient_id: str,
        *,
        document_types: list[str] | None = None,
    ) -> list[SearchHit]:
        cache_key = self._cache_key(queries, patient_id, document_types)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        ranked_lists: list[list[SearchHit]] = []
        for query in queries:
            hits = self._search_one(query, patient_id, document_types)
            if hits:
                ranked_lists.append(hits)

        fused: dict[str, SearchHit] = {}
        rrf_constant = int(self.config.get("rrf_constant", 60))
        for ranked in ranked_lists:
            for rank, hit in enumerate(ranked, start=1):
                contribution = 1.0 / (rrf_constant + rank)
                existing = fused.get(hit.chunk_id)
                if existing is None:
                    hit.rrf_score = contribution
                    fused[hit.chunk_id] = hit
                else:
                    existing.rrf_score += contribution
                    existing.score = max(existing.score, hit.score)

        final_k = int(self.config.get("final_k", 8))
        result = sorted(
            fused.values(),
            key=lambda hit: (hit.rrf_score, hit.score),
            reverse=True,
        )[:final_k]
        self.cache.set(cache_key, result)
        return result

    def format_context(self, hits: list[SearchHit]) -> str:
        blocks = []
        for number, hit in enumerate(hits, start=1):
            metadata = hit.metadata
            blocks.append(
                "\n".join(
                    [
                        f"[Evidence {number}]",
                        f"Patient: {metadata.get('patient_id', '')}",
                        f"Document ID: {metadata.get('document_id', '')}",
                        f"Document type: {metadata.get('document_type', '')}",
                        f"Date: {metadata.get('document_date', '')}",
                        f"Section: {metadata.get('section', '')}",
                        f"Source: {metadata.get('source_path') or metadata.get('source_doc', '')}",
                        hit.text,
                    ]
                )
            )
        return "\n\n".join(blocks)

    def source_summaries(self, hits: list[SearchHit]) -> list[dict[str, Any]]:
        """One entry per source *document*, not per retrieved chunk.

        `retrieve` returns chunks, and several of them routinely come from the same
        document, so a per-chunk list showed the same PDF three times over and made
        the evidence panel look like padding. Collapsing by document_id keeps the
        list honest about how many records back the answer.

        The first chunk seen for a document is its best one -- `hits` arrives sorted
        by (rrf_score, score) descending -- so its ids and scores are the ones kept,
        and iterating in order means the deduplicated list stays ranked by relevance.
        `chunk_count` records how many retrieved chunks agreed on that document,
        which is itself a strength signal worth showing rather than discarding.

        A hit with no document_id keys on its chunk_id instead, so missing metadata
        cannot silently fuse unrelated chunks into one phantom document.
        """
        summaries: dict[str, dict[str, Any]] = {}
        for hit in hits:
            document_id = hit.metadata.get("document_id")
            key = str(document_id) if document_id else f"chunk:{hit.chunk_id}"
            existing = summaries.get(key)
            if existing is not None:
                existing["chunk_count"] += 1
                continue
            summaries[key] = {
                "chunk_id": hit.chunk_id,
                "document_id": document_id,
                "document_type": hit.metadata.get("document_type"),
                "document_date": hit.metadata.get("document_date"),
                "source_path": hit.metadata.get("source_path"),
                "score": round(hit.score, 6),
                "rrf_score": round(hit.rrf_score, 6),
                "chunk_count": 1,
            }
        return list(summaries.values())

    def _search_one(
        self,
        query: str,
        patient_id: str,
        document_types: list[str] | None,
    ) -> list[SearchHit]:
        # Standard AI Search endpoints use a filter dictionary. Storage-optimized
        # indexes use a SQL-like filter string. We try both without ever dropping
        # the mandatory patient boundary.
        standard_filters: dict[str, Any] = {"patient_id": patient_id}
        if document_types:
            standard_filters["document_type"] = document_types
        escaped_patient = patient_id.replace("'", "''")
        sql_filter = f"patient_id = '{escaped_patient}'"
        if document_types:
            values = ", ".join(
                "'" + value.replace("'", "''") + "'" for value in document_types
            )
            sql_filter += f" AND document_type IN ({values})"
        filter_candidates: list[Any] = [standard_filters, sql_filter]

        per_query_k = int(self.config.get("per_query_k", 6))
        query_vector = self.embedding.embed_query(query)

        reranker = None
        if bool(self.config.get("use_reranker", True)) and DatabricksReranker:
            reranker = DatabricksReranker(
                columns_to_rerank=[self.schema["chunk_text"], "document_type"]
            )

        # Preferred path: built-in hybrid search. Databricks combines semantic
        # and keyword ranks with RRF; a reranker can then improve final precision.
        for filters in filter_candidates:
            try:
                kwargs: dict[str, Any] = {
                    "query_text": query,
                    "query_vector": query_vector,
                    "columns": self.columns,
                    "filters": filters,
                    "num_results": per_query_k,
                    "query_type": "HYBRID",
                }
                if reranker is not None:
                    kwargs["reranker"] = reranker
                return self._parse_response(self.index.similarity_search(**kwargs))
            except Exception:
                continue

        # Compatibility fallback: ANN plus full-text, fused locally. This path
        # remains patient-filtered for both standard and storage-optimized syntax.
        last_error: Exception | None = None
        for filters in filter_candidates:
            try:
                ann = self._parse_response(
                    self.index.similarity_search(
                        query_vector=query_vector,
                        columns=self.columns,
                        filters=filters,
                        num_results=per_query_k,
                    )
                )
                try:
                    lexical = self._parse_response(
                        self.index.similarity_search(
                            query_text=query,
                            columns=self.columns,
                            filters=filters,
                            num_results=per_query_k,
                            query_type="FULL_TEXT",
                        )
                    )
                except Exception:
                    lexical = []
                return self._local_rrf([ann, lexical], per_query_k)
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return []

    def _parse_response(self, response: Any) -> list[SearchHit]:
        if hasattr(response, "as_dict"):
            response = response.as_dict()
        elif hasattr(response, "to_dict"):
            response = response.to_dict()
        if not isinstance(response, dict):
            return []

        manifest_columns = (
            response.get("manifest", {}).get("columns")
            or response.get("manifest", {}).get("schema", {}).get("columns")
            or []
        )
        names = [
            column.get("name") if isinstance(column, dict) else str(column)
            for column in manifest_columns
        ]
        rows = response.get("result", {}).get("data_array") or []
        hits: list[SearchHit] = []
        for row in rows:
            values = list(row)
            score = 0.0
            # Two response shapes exist. Some return the similarity as an extra
            # trailing value beyond the declared columns; others declare a
            # "score" column in the manifest, in which case it arrives inside
            # the zipped row and the length check below never fires. Handle both,
            # otherwise every score silently reads 0.0.
            if len(values) > len(names):
                try:
                    score = float(values[-1])
                    values = values[: len(names)]
                except (TypeError, ValueError):
                    pass
            data = dict(zip(names, values, strict=False))
            if not score:
                try:
                    score = float(data.get("score") or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
            chunk_id = str(data.get(self.schema["primary_key"], ""))
            text = str(data.get(self.schema["chunk_text"], ""))
            if not chunk_id or not text:
                continue
            metadata = {key: value for key, value in data.items() if key != self.schema["chunk_text"]}
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata,
                    score=score,
                )
            )
        return hits

    @staticmethod
    def _local_rrf(ranked_lists: list[list[SearchHit]], limit: int) -> list[SearchHit]:
        fused: dict[str, SearchHit] = {}
        for ranked in ranked_lists:
            for rank, hit in enumerate(ranked, start=1):
                contribution = 1.0 / (60 + rank)
                existing = fused.get(hit.chunk_id)
                if existing is None:
                    hit.rrf_score = contribution
                    fused[hit.chunk_id] = hit
                else:
                    existing.rrf_score += contribution
                    existing.score = max(existing.score, hit.score)
        return sorted(
            fused.values(),
            key=lambda hit: (hit.rrf_score, hit.score),
            reverse=True,
        )[:limit]

    def _cache_key(
        self,
        queries: list[str],
        patient_id: str,
        document_types: list[str] | None,
    ) -> str:
        payload = json.dumps(
            {
                "index": self.index_name,
                "queries": queries,
                "patient_id": patient_id,
                "document_types": document_types,
                "config": self.config,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
