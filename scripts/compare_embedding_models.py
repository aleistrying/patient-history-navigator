"""Compare the workspace's three embedding models on this corpus, and log the runs.

The question this answers is "would a different embedding model retrieve better
here", and it is worth answering with measurement because the names invite a wrong
guess: `databricks-qwen3-embedding-0-6b` reads like the small one next to two
models called "large", when 0.6b is 600 million parameters and the BERT-family
"large" models are roughly half that.

Method. Real chunks are pulled from the live chunk table. Queries are built from
the document titles in the corpus, which gives a gold label without hand-annotation:
the chunks belonging to a document with that title are the correct answers for a
query naming it. Each model embeds every chunk and every query; retrieval is exact
cosine over that set, with no reranker and no RRF, so the number attributed to the
embedding model is the embedding model's.

Reported per model: recall@5, MRR, and the margin between the mean similarity of
correct and incorrect chunks -- a model can rank correctly while barely separating,
which shows up as a thin margin and predicts brittleness the ranking metrics hide.

    python3 scripts/compare_embedding_models.py [--chunks 400] [--no-mlflow]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

MODELS = [
    "databricks-qwen3-embedding-0-6b",
    "databricks-gte-large-en",
    "databricks-bge-large-en",
]
WAREHOUSE = "b0affc2359ab09a3"
CHUNKS_TABLE = "workspace.default.patient_history_rag_chunks"
DOCUMENTS_TABLE = "workspace.default.patient_history_documents"
EXPERIMENT = "/Shared/dbdemos-rag-prod"
# Measured on this workspace: 25 consecutive batches of 16 ran at 22 chunks/s with
# no throttling. 16 is the safe ceiling -- 32 stalled past 180s.
BATCH = 16


def sql(statement: str) -> list[dict]:
    payload = {"warehouse_id": WAREHOUSE, "statement": statement,
               "wait_timeout": "50s", "format": "JSON_ARRAY"}
    completed = subprocess.run(
        ["databricks", "api", "post", "/api/2.0/sql/statements",
         "--json", json.dumps(payload)],
        capture_output=True, text=True, timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip()[:400])
    body = json.loads(completed.stdout)
    if (body.get("status") or {}).get("state") != "SUCCEEDED":
        raise RuntimeError(json.dumps(body.get("status"))[:400])
    schema = body["manifest"]["schema"]["columns"]
    names = [column["name"] for column in schema]
    rows = (body.get("result") or {}).get("data_array") or []
    return [dict(zip(names, row)) for row in rows]


def _post(model: str, batch: list[str], timeout: int) -> list[list[float]]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump({"input": batch}, handle)
        path = handle.name
    try:
        completed = subprocess.run(
            ["databricks", "api", "post",
             f"/serving-endpoints/{model}/invocations", "--json", "@" + path],
            capture_output=True, text=True, timeout=timeout,
        )
    finally:
        os.unlink(path)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip()[:300])
    data = json.loads(completed.stdout)["data"]
    return [item["embedding"] for item in sorted(data, key=lambda d: d.get("index", 0))]


def embed(model: str, texts: list[str]) -> tuple[list[list[float]], int]:
    """Embed everything, halving the batch whenever the endpoint stops answering.

    The three models do not share a batch ceiling: qwen3 sustains 16, while
    gte-large-en answers at 8 and then stalls past 300s at 16 rather than
    returning an error. Scoring a model on a timeout would measure the ceiling and
    call it quality, so the batch adapts and the size that worked is reported.
    """
    size = BATCH
    vectors: list[list[float]] = []
    start = 0
    while start < len(texts):
        batch = texts[start:start + size]
        try:
            vectors += _post(model, batch, timeout=90)
            start += size
        except (subprocess.TimeoutExpired, RuntimeError):
            if size == 1:
                raise
            size = max(1, size // 2)
            print(f"        {model}: backing off to batch {size}")
    return vectors, size


def normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def build_dataset(limit: int):
    """Chunks, plus one query per document title that owns at least two chunks."""
    # retrieval_text, not text_chunk: retrieval_text is what the pipeline actually
    # embeds, so comparing anything else would measure a system nobody is running.
    chunks = sql(f"""
        SELECT c.retrieval_text AS chunk_text, c.document_id,
               d.event_name, d.document_type
        FROM {CHUNKS_TABLE} c
        JOIN {DOCUMENTS_TABLE} d ON c.document_id = d.document_id
        WHERE c.retrieval_text IS NOT NULL AND length(c.retrieval_text) > 80
          AND d.event_name IS NOT NULL AND length(d.event_name) > 3
        ORDER BY c.document_id, c.chunk_id
        LIMIT {int(limit)}
    """)
    if not chunks:
        raise SystemExit("No chunks returned; is the corpus ingested?")

    by_title: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(chunks):
        by_title[str(row["event_name"]).strip()].append(index)

    queries = []
    for title, indices in sorted(by_title.items()):
        # A title owning every chunk in the set makes a trivial query; one owning a
        # single chunk makes a noisy one.
        if 2 <= len(indices) <= max(3, len(chunks) // 4):
            queries.append({"text": f"What are the recorded results for {title}?",
                            "gold": set(indices), "title": title})
    if len(queries) < 5:
        raise SystemExit(f"Only {len(queries)} usable queries; raise --chunks.")
    return chunks, queries


def score(model: str, chunks, queries, k: int = 5) -> dict:
    started = time.monotonic()
    raw_chunks, batch_used = embed(model, [str(c["chunk_text"]) for c in chunks])
    raw_queries, _ = embed(model, [q["text"] for q in queries])
    chunk_vectors = [normalise(v) for v in raw_chunks]
    query_vectors = [normalise(v) for v in raw_queries]
    elapsed = time.monotonic() - started

    hits = reciprocal = 0.0
    correct_means, other_means = [], []
    for query, query_vector in zip(queries, query_vectors):
        similarities = [(dot(query_vector, cv), i)
                        for i, cv in enumerate(chunk_vectors)]
        similarities.sort(reverse=True)
        top = [index for _, index in similarities[:k]]
        if query["gold"] & set(top):
            hits += 1
        for rank, (_, index) in enumerate(similarities, start=1):
            if index in query["gold"]:
                reciprocal += 1.0 / rank
                break
        correct = [s for s, i in similarities if i in query["gold"]]
        other = [s for s, i in similarities if i not in query["gold"]]
        correct_means.append(sum(correct) / len(correct))
        other_means.append(sum(other) / len(other))

    n = len(queries)
    margin = sum(c - o for c, o in zip(correct_means, other_means)) / n
    return {
        "recall_at_5": hits / n,
        "mrr": reciprocal / n,
        "separation_margin": margin,
        "mean_correct_similarity": sum(correct_means) / n,
        "embed_seconds": elapsed,
        "chunks_per_second": (len(chunks) + n) / elapsed,
        "batch_size_sustained": batch_used,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=400)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    chunks, queries = build_dataset(args.chunks)
    print(f"  {len(chunks)} chunks, {len(queries)} title-derived queries\n")

    tracking = None
    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri("databricks")
            mlflow.set_experiment(EXPERIMENT)
            tracking = mlflow
        except Exception as exc:
            print(f"  (MLflow unavailable, printing only: {str(exc)[:80]})\n")

    results = {}
    for model in MODELS:
        try:
            metrics = score(model, chunks, queries)
        except Exception as exc:
            print(f"  {model:<34} FAILED {str(exc)[:90]}")
            continue
        results[model] = metrics
        print(f"  {model:<34} recall@5 {metrics['recall_at_5']:.2f}  "
              f"MRR {metrics['mrr']:.3f}  margin {metrics['separation_margin']:+.4f}  "
              f"{metrics['chunks_per_second']:5.1f} chunks/s "
              f"@batch {metrics['batch_size_sustained']}")
        if tracking:
            with tracking.start_run(run_name=f"embedding-{model}"):
                tracking.log_params({"embedding_model": model,
                                     "chunks": len(chunks),
                                     "queries": len(queries),
                                     "batch_size_requested": BATCH})
                tracking.log_metrics(metrics)

    if len(results) > 1:
        best = max(results, key=lambda m: (results[m]["mrr"],
                                           results[m]["separation_margin"]))
        print(f"\n  best by MRR then margin: {best}")
        current = "databricks-qwen3-embedding-0-6b"
        if best == current:
            print("  the delivered configuration is already the best of the three.")
        else:
            gain = results[best]["mrr"] - results[current]["mrr"]
            print(f"  {best} beats the delivered model by {gain:+.3f} MRR.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
