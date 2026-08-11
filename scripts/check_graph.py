"""Drive the request graph end to end with no Databricks workspace.

Why this exists
---------------
The other suites in this directory all talk to the deployed endpoint, so finding a
node that reads a state key nobody wrote costs a deploy, a cold start and a failed
live query. That is a slow way to find a typo.

Here the four external dependencies -- the chat model, the vector index, the SQL
warehouse and the memory store -- are replaced by stubs, and the real graph, the
real nodes, the real tools and the real prompts run against them. It checks that
both paths reach an answer, that each node's state contract is satisfied, and that
patient scope cannot be reached by the model.

It does not check retrieval quality, SQL correctness, or routing judgement -- those
need the real thing, and check_behaviour.py covers them.

    python3 scripts/check_graph.py [--verbose]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

# Spans are asserted below, so they have to be readable back. Both settings are
# required before mlflow is imported: the default file store refuses reads ("in
# maintenance mode"), and with async logging on, a trace is still being written
# when the assertion runs. A throwaway SQLite file keeps this self-contained and
# keeps local runs out of the real experiment.
os.environ.setdefault(
    "MLFLOW_TRACKING_URI",
    "sqlite:///" + str(Path(tempfile.gettempdir()) / "phn_check_graph_traces.db"),
)
os.environ.setdefault("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "databricks"))

from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from mlflow.models.model_config import _set_model_config  # noqa: E402

# Bound now, before load_agent() replaces the name on the module, so the stub below
# can still reach the real implementation of the pure methods.
from vector_retrieval import AdvancedRetriever as _RealRetriever  # noqa: E402


# ---------------------------------------------------------------------------
# Stubs for the four things that need a workspace
# ---------------------------------------------------------------------------


class StubChatModel(FakeListChatModel):
    """A chat model that can also be asked to return a tool call.

    `tool_call` is set per-scenario by the test. When it is None, `bind_tools`
    returns a model that selects nothing, which is how the retrieval path is
    reached -- the same shape as the real model declining to call a tool.
    """

    tool_call: dict[str, Any] | None = None

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        calls = [self.tool_call] if self.tool_call else []

        class _Bound:
            @staticmethod
            def invoke(_messages, **_kw):
                return AIMessage(content="", tool_calls=calls)

        return _Bound()


class StubExecutor:
    """SQL warehouse stand-in. Returns nothing, which every tool must survive."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.is_configured = True
        self.queries: list[str] = []

    def query(self, statement: str) -> list[dict[str, Any]]:
        self.queries.append(statement)
        return []


def hit(text: str, *, document_id: str = "doc-1", score: float = 0.51):
    """A real SearchHit, not a stand-in.

    Deliberately the genuine dataclass: the retrieval span reads `.score`,
    `.rrf_score` and `.metadata` off each hit, and `_retrieve_with_trace` wraps
    that in a bare `except` so telemetry can never fail a request. Feed it a
    duck-typed stub instead and all six retrieval metrics vanish silently while
    every other assertion still passes.
    """
    from vector_retrieval import SearchHit

    return SearchHit(
        chunk_id=f"chunk-{document_id}",
        text=text,
        metadata={"document_id": document_id, "patient_id": "P-TEST"},
        score=score,
        rrf_score=1 / 61,
    )


class StubRetriever:
    """Vector index stand-in, with the hit list swappable per scenario."""

    hits: list[Any] = []

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def retrieve(self, queries, patient_id, document_types=None,  # noqa: ANN001
                 diagnostics=None):
        StubRetriever.last_call = {
            "queries": list(queries),
            "patient_id": patient_id,
            "document_types": document_types,
        }
        if diagnostics is not None:
            # Mirror what the real retriever records on a clean hybrid run, so the
            # span assertions below exercise the keys the deployed path fills.
            diagnostics.update({
                "cached": False,
                "probes": len(queries),
                "reranker_available": True,
                "reranked_probes": len(queries),
                "paths": {"hybrid"},
                "filter_syntax": {"dict"},
            })
        return list(StubRetriever.hits)

    def format_context(self, hits):  # noqa: ANN001
        return "\n\n".join(f"[Evidence {i}] {h}" for i, h in enumerate(hits, 1))

    def source_summaries(self, hits):  # noqa: ANN001
        # Delegate to the real implementation rather than imitating it. This method
        # is pure logic over hits with no network call, so there is nothing to stub,
        # and a stub that returned one entry per chunk would let the per-document
        # collapse regress without any check noticing.
        return _RealRetriever.source_summaries(self, hits)


class StubMemoryStore:
    """Memory store stand-in that keeps one session in a dict."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.enabled = True
        self.last_error = None
        self.saved: list[Any] = []

    def load(self, user_id: str, session_id: str):
        from session_memory import SessionMemory

        return SessionMemory(user_id=user_id, session_id=session_id)

    def save(self, memory) -> None:  # noqa: ANN001
        self.saved.append(memory)


def load_agent():
    """Import serving_agent with the external clients replaced.

    The stubs are installed on the modules serving_agent imports *from*, before it
    is imported, because it builds its clients at module scope -- the same reason
    the real thing fails fast on a missing warehouse id instead of at first query.
    """
    _set_model_config(str(ROOT / "databricks" / "navigator_config.yaml"))

    import databricks_langchain.chat_models as chat_models
    import session_memory
    import sql_execution
    import vector_retrieval

    chat_models.ChatDatabricks = lambda **kwargs: StubChatModel(responses=["stub answer"])
    vector_retrieval.AdvancedRetriever = StubRetriever
    sql_execution.DatabricksSQLExecutor = StubExecutor
    session_memory.CatalogMemoryStore = StubMemoryStore

    import serving_agent

    return serving_agent


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def request(sa, question: str, **custom):  # noqa: ANN001
    from mlflow.types.responses import ResponsesAgentRequest

    return ResponsesAgentRequest(
        input=[{"role": "user", "content": question}],
        custom_inputs={"patient_id": "P-TEST", **custom},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    sa = load_agent()
    checks: list[tuple[bool, str, str]] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        checks.append((bool(ok), label, detail))

    # ---- structure -------------------------------------------------------
    check(len(sa.PATIENT_TOOLS) == 8, "eight tools built", str(sorted(sa.PATIENT_TOOLS)))
    check(len(sa.TOOL_SPECS) == 8, "eight schemas offered")
    leaked = [
        spec["function"]["name"]
        for spec in sa.TOOL_SPECS
        if "patient_id" in (spec["function"].get("parameters") or {}).get("properties", {})
    ]
    check(not leaked, "patient_id in no model-facing schema", f"leaked: {leaked}")

    nodes = {n for n in sa.NAVIGATOR_GRAPH.get_graph().nodes if not n.startswith("__")}
    expected_nodes = {
        "load_memory", "route", "run_tool", "explain_tool",
        "expand_queries", "retrieve", "generate_answer", "save_memory",
    }
    check(nodes == expected_nodes, "graph has the eight nodes", str(sorted(nodes ^ expected_nodes)))

    # ---- retrieval path --------------------------------------------------
    sa.llm.tool_call = None
    StubRetriever.hits = [hit("sodium 142 mmol/L on 2026-06-25")]
    answer, meta = sa._execute(request(sa, "what do the labs say?"))
    check(bool(answer), "retrieval path answers", repr(answer[:60]))
    check(meta["tool"] is None, "retrieval path reports no tool", str(meta["tool"]))
    check(bool(meta.get("retrieval_queries")), "retrieval path records its probes",
          str(meta.get("retrieval_queries"))[:80])
    check(meta["patient_id"] == "P-TEST", "patient scope honoured", meta["patient_id"])
    check(StubRetriever.last_call["patient_id"] == "P-TEST",
          "retrieval filtered by patient", str(StubRetriever.last_call["patient_id"]))
    check(len(meta["sources"]) == 1, "sources returned", str(len(meta["sources"])))

    # Retrieval returns chunks and several routinely come from one document, so the
    # evidence list is per document. Two chunks of one document must be one source
    # carrying a count of two, not two rows naming the same PDF.
    StubRetriever.hits = [
        hit("first passage", document_id="doc-A"),
        hit("second passage", document_id="doc-A", score=0.44),
        hit("other document", document_id="doc-B", score=0.41),
    ]
    _answer, meta = sa._execute(request(sa, "what do the labs say?"))
    sources = meta["sources"]
    check(len(sources) == 2, "three chunks from two documents give two sources",
          f"got {len(sources)}")
    counts = {s.get("document_id"): s.get("chunk_count") for s in sources}
    check(counts.get("doc-A") == 2 and counts.get("doc-B") == 1,
          "chunk_count records how many chunks agreed", str(counts))
    check([s.get("document_id") for s in sources] == ["doc-A", "doc-B"],
          "deduplicated sources stay ranked by relevance",
          str([s.get("document_id") for s in sources]))

    # ---- no evidence: the model must not be asked to invent one ----------
    StubRetriever.hits = []
    answer, meta = sa._execute(request(sa, "anything about allergies?"))
    check("could not find matching evidence" in answer,
          "no evidence refuses without calling the model", repr(answer[:70]))

    # ---- tool path, chosen by the model ---------------------------------
    sa.llm.tool_call = {"name": "count_patient_documents", "args": {}, "id": "call_1"}
    answer, meta = sa._execute(request(sa, "how many records?"))
    check(meta["tool"] == "count_documents", "model-chosen tool ran", str(meta["tool"]))
    check(bool(answer), "tool path answers", repr(answer[:60]))

    # ---- tool path, pinned by the caller --------------------------------
    sa.llm.tool_call = None
    answer, meta = sa._execute(request(sa, "how many records?", tool="count_documents"))
    check(meta["tool"] == "count_documents", "pinned tool ran without the model",
          str(meta["tool"]))

    # ---- a pinned tool must not be silently ignored ----------------------
    try:
        sa._execute(request(sa, "x", tool="not_a_tool"))
        check(False, "unknown pinned tool rejected", "no error raised")
    except ValueError:
        check(True, "unknown pinned tool rejected")

    # ---- the model cannot redirect patient scope ------------------------
    # A tool call carrying patient_id must not change whose records are read.
    sa.llm.tool_call = {
        "name": "count_patient_documents",
        "args": {"patient_id": "SOMEONE-ELSE"},
        "id": "call_2",
    }
    answer, meta = sa._execute(request(sa, "how many records?"))
    check(meta["patient_id"] == "P-TEST",
          "model cannot redirect patient scope", meta["patient_id"])

    # ---- MLflow spans ----------------------------------------------------
    # The deck and the report both name these four spans and the retrieval
    # attributes, so an edit that stops emitting one invalidates documentation as
    # well as observability. Asserted here because the alternative is reading the
    # Tracing UI by eye after every deploy.
    import mlflow

    def spans_for(req) -> dict[str, str]:  # noqa: ANN001
        sa._execute(req)
        trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
        return {span.name: str(span.span_type) for span in trace.data.spans}

    sa.llm.tool_call = None
    StubRetriever.hits = [hit("sodium 142 mmol/L"), hit("sodium 138 mmol/L", document_id="doc-2")]
    retrieval_spans = spans_for(request(sa, "what do the labs say?"))
    check("patient_history_request" in retrieval_spans, "CHAIN span emitted")
    check(retrieval_spans.get("patient_scoped_ai_search") == "RETRIEVER",
          "RETRIEVER span emitted", str(retrieval_spans.get("patient_scoped_ai_search")))
    check("tool_selection" in retrieval_spans, "LLM tool-selection span emitted",
          str(sorted(retrieval_spans)))

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    retrieval_span = next(
        (s for s in trace.data.spans if s.name == "patient_scoped_ai_search"), None
    )
    wanted_attributes = {
        "retrieval.query_count", "retrieval.hit_count", "retrieval.distinct_documents",
        "retrieval.top_score", "retrieval.mean_score", "retrieval.top_rrf_score",
    }
    present = set((retrieval_span.attributes or {}) if retrieval_span else {})
    check(wanted_attributes <= present, "six retrieval metrics on the span",
          f"missing: {sorted(wanted_attributes - present)}")

    # Which search path served is not visible in the answer: an unreranked fallback
    # carries the same citations and format as a reranked hybrid result, just worse
    # evidence. The span has to say so, or that downgrade is undetectable.
    provenance = {
        "retrieval.path", "retrieval.filter_syntax", "retrieval.cached",
        "retrieval.probes_run", "retrieval.reranked_probes",
        "retrieval.reranker_available", "retrieval.hybrid_failures",
    }
    check(provenance <= present, "search path recorded on the span",
          f"missing: {sorted(provenance - present)}")
    attributes = (retrieval_span.attributes or {}) if retrieval_span else {}
    check(attributes.get("retrieval.path") == "hybrid",
          "the path taken is named", str(attributes.get("retrieval.path")))

    root = next((s for s in trace.data.spans if s.name == "patient_history_request"), None)
    root_attributes = set((root.attributes or {}) if root else {})
    check({"request.patient_id", "request.answer_path"} <= root_attributes,
          "root span carries patient and answer path", str(sorted(root_attributes)))

    sa.llm.tool_call = {"name": "count_patient_documents", "args": {}, "id": "call_3"}
    tool_spans = spans_for(request(sa, "how many records?"))
    check(tool_spans.get("focused_patient_record_tool") == "TOOL",
          "TOOL span emitted on the tool path",
          str(tool_spans.get("focused_patient_record_tool")))
    check("patient_scoped_ai_search" not in tool_spans,
          "tool path emits no retriever span (why groundedness is scored per group)")

    # ---- report ----------------------------------------------------------
    passed = sum(1 for ok, _, _ in checks if ok)
    failed = len(checks) - passed
    print()
    for ok, label, detail in checks:
        mark = "ok  " if ok else "FAIL"
        suffix = f"   {detail}" if (detail and (args.verbose or not ok)) else ""
        print(f"  {mark} {label}{suffix}")
    print(f"\n  {passed} passed, {failed} failed, {len(checks)} checks")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
