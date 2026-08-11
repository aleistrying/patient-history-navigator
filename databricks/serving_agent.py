"""Patient History Navigator serving application.

The model is intentionally *tool-assisted* rather than an unconstrained autonomous
agent. One request is a LangGraph traversal, declared in `orchestration.py`:

    load_memory -> route -> { run_tool -> explain_tool
                            | expand_queries -> retrieve -> generate_answer }
                         -> save_memory

`route` is the only branch, and it is the model's decision: it selects at most one
of the eight tools in `patient_tools.py`, or none, which is the signal to answer
from retrieved record text instead. Both paths converge on `save_memory`.

This module supplies the node bodies and the runtime clients; the topology lives in
orchestration.py so the shape of a request can be read without also reading prompt
text, SQL, or retrieval internals.

What the layers are, and why they are separate:

* `record_tools.py`   - SQL over Unity Catalog. Every number in an answer is
                        computed here, never generated.
* `patient_tools.py`  - those functions as LangChain StructuredTools: the schemas
                        the model chooses from, plus argument discipline. Binds
                        `patient_id` server-side, so the model chooses the
                        operation and the application chooses the patient.
* `orchestration.py`  - the graph: nodes, edges, and the one branch.
* this module         - node bodies, prompts, clients, MLflow spans, and the
                        ResponsesAgent contract Model Serving calls.

Three spans are emitted per request: `patient_history_request` (CHAIN),
`tool_selection` (LLM), and then whichever path answered --
`focused_patient_record_tool` (TOOL) or `patient_scoped_ai_search` (RETRIEVER).
Which of the last two is absent is itself the signal for how the question was
answered. `scripts/check_graph.py` asserts all four names across both paths
offline, along with the six `retrieval.*` metrics.

The application organizes synthetic records. It does not diagnose, prescribe, or
interpret clinical significance.
"""

from __future__ import annotations

import json
from typing import Any, Generator
from uuid import uuid4

import mlflow
from databricks_langchain.chat_models import ChatDatabricks
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from mlflow.entities import SpanType
from mlflow.models import ModelConfig, set_model
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)

from session_memory import CatalogMemoryStore, SessionMemory
from query_strategies import STRATEGIES, QueryStrategyEngine
from vector_retrieval import AdvancedRetriever, SearchHit
from sql_execution import DatabricksSQLExecutor
from record_tools import PatientRecordTools, ToolResult
from patient_tools import (
    TOOL_NAME_TO_KEY,
    build_patient_tools,
    model_facing_specs,
)
from orchestration import NavigatorNodes, NavigatorState, build_navigator_graph

# MLflow records supported LangChain prompt, model, and parser spans automatically.
# The custom modules also add useful fields to custom_outputs for debugging.
mlflow.langchain.autolog()


# ---------------------------------------------------------------------------
# Configuration and shared runtime clients
# ---------------------------------------------------------------------------

model_config = ModelConfig()
project_config = model_config.get("project_config") or {}
resources = model_config.get("databricks_resources") or {}
retriever_config = model_config.get("retriever_config") or {}
query_config = model_config.get("query_strategy_config") or {}
memory_config = model_config.get("memory_config") or {}
cache_config = model_config.get("cache_config") or {}
tools_config = model_config.get("tools_config") or {}
llm_config = model_config.get("llm_config") or {}

required_resources = [
    "vector_search_endpoint_name",
    "embedding_endpoint_name",
    "llm_endpoint_name",
]
missing_resources = [key for key in required_resources if not resources.get(key)]
if missing_resources:
    raise ValueError(
        "Missing required databricks_resources values: " + ", ".join(missing_resources)
    )
if not retriever_config.get("vector_search_index"):
    raise ValueError("retriever_config.vector_search_index is required")

llm = ChatDatabricks(
    endpoint=resources["llm_endpoint_name"],
    extra_params=llm_config.get(
        "llm_parameters",
        {"temperature": 0.1, "max_tokens": 900},
    ),
)
query_engine = QueryStrategyEngine(llm, query_config, cache_config)
retriever = AdvancedRetriever(resources, retriever_config, cache_config)

sql_executor = DatabricksSQLExecutor(
    warehouse_id=str(resources.get("sql_warehouse_id", "")),
    timeout_seconds=int(tools_config.get("sql_timeout_seconds", 50)),
)
record_tools = PatientRecordTools(
    executor=sql_executor,
    documents_table=str(
        tools_config.get(
            "documents_table",
            "workspace.default.patient_history_documents",
        )
    ),
    config=tools_config,
)
memory_store = CatalogMemoryStore(
    full_name=str(
        resources.get(
            "memory_store_name",
            "workspace.default.patient_history_memory",
        )
    ),
    enabled=bool(memory_config.get("enabled", True)),
    fail_open=bool(memory_config.get("fail_open", True)),
    timeout_seconds=int(memory_config.get("timeout_seconds", 20)),
)

# Tell MLflow which AI Search columns represent retrieved content and source IDs.
vector_schema = retriever_config.get("schema", {})
mlflow.models.set_retriever_schema(
    primary_key=vector_schema.get("primary_key", "chunk_id"),
    text_column=vector_schema.get("chunk_text", "text_chunk"),
    doc_uri=vector_schema.get("document_uri", "source_path"),
)

answer_prompt = PromptTemplate(
    template=llm_config.get(
        "llm_prompt_template",
        """You are Patient History Navigator, a record-navigation assistant for synthetic patient data.

Safety and grounding rules:
- Use only the evidence supplied below.
- Never diagnose, prescribe, recommend treatment, or claim clinical significance.
- Distinguish recorded facts from missing information.
- Cite supporting chunks inline as [Evidence 1], [Evidence 2], and so on.
- Never combine facts from another patient.
- When evidence is insufficient or conflicting, say exactly what is missing or conflicting.

Persistent session context:
{memory_context}

Recent conversation:
{chat_history}

Retrieved record evidence:
{context}

User question:
{question}

Answer:""",
    ),
    input_variables=[
        "memory_context",
        "chat_history",
        "context",
        "question",
    ],
)
answer_chain = answer_prompt | llm | StrOutputParser()

tool_explanation_prompt = PromptTemplate.from_template(
    """You are Patient History Navigator. Report the record result below as the answer.

Rules:
- Treat the result as authoritative and answer the question directly from it.
- Report the findings themselves: name each record, value, change and date the
  result contains, and carry the numbers over exactly as given.
- Never summarise several findings into a phrase like "various values differ".
- Never describe the tool, name it, mention that a tool ran, or explain the
  format, notation or layout of its output.
- Preserve document IDs and dates.
- Do not diagnose, prescribe, recommend treatment, or interpret clinical significance.
- Clearly state when requested inputs or records were missing.
- Be concise and structured.

User question:
{question}

Record operation:
{tool_name}

Result:
{tool_output}

Answer:"""
)
tool_explanation_chain = tool_explanation_prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Request parsing helpers
# ---------------------------------------------------------------------------


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "to_dict"):
        dumped = value.to_dict()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _messages(request: ResponsesAgentRequest) -> list[dict[str, Any]]:
    return to_chat_completions_input(
        [item.model_dump() for item in request.input]
    )


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content or "")


def _latest_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role", "")).lower() == "user":
            value = _message_text(message.get("content"))
            if value.strip():
                return value.strip()
    raise ValueError("At least one non-empty user message is required")


def _recent_chat(messages: list[dict[str, Any]], limit: int = 10) -> str:
    lines: list[str] = []
    for message in messages[-limit:]:
        role = str(message.get("role", "unknown"))
        content = _message_text(message.get("content"))
        if content.strip():
            lines.append(f"{role}: {content.strip()}")
    return "\n".join(lines)


def _stored_or_request_history(
    messages: list[dict[str, Any]], memory: SessionMemory
) -> str:
    """The conversation so far, from the request or, failing that, from memory.

    The request is authoritative when it carries prior turns: a client that sends
    its own transcript knows what the user has actually seen. But a caller can send
    only the current question and still be mid-conversation -- the Playground does
    this, and so does the web UI after a page reload -- and until now that produced
    "this conversation just started" while the memory store held the turns all
    along. Memory was storing recent_messages that nothing ever read.

    Falling back to them makes the store load-bearing rather than decorative, and
    keeps the precedence honest: stored turns fill a gap, they never override what
    the client sent.
    """
    limit = int(memory_config.get("request_history_limit", 10))
    supplied = [
        message for message in messages
        if _message_text(message.get("content")).strip()
    ]
    # One message is the current question with no history behind it.
    if len(supplied) > 1 or not memory.recent_messages:
        return _recent_chat(messages, limit)
    combined = list(memory.recent_messages) + supplied
    return _recent_chat(combined, limit)


def _memory_context(memory: SessionMemory) -> str:
    parts = []
    if memory.active_patient_id:
        parts.append(f"Active synthetic patient: {memory.active_patient_id}")
    if memory.summary:
        parts.append(f"Session summary: {memory.summary}")
    if memory.last_tool:
        parts.append(f"Most recent deterministic tool: {memory.last_tool}")
    return "\n".join(parts) or "No stored session context."


def _context_ids(request: ResponsesAgentRequest) -> tuple[str, str]:
    context = _as_dict(getattr(request, "context", None))
    custom = _as_dict(getattr(request, "custom_inputs", None))
    user_id = str(
        context.get("user_id")
        or custom.get("user_id")
        or memory_config.get("default_user_id", "demo-user")
    )
    session_id = str(
        context.get("conversation_id")
        or custom.get("session_id")
        or custom.get("conversation_id")
        or memory_config.get("default_session_id", "demo-session")
    )
    return user_id, session_id


def _patient_id(custom: dict[str, Any], memory: SessionMemory) -> str:
    return str(
        custom.get("patient_id")
        or memory.active_patient_id
        or project_config.get("default_patient_id", "demo-patient")
    )


def _list_value(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        result = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        result = [str(item).strip() for item in value if str(item).strip()]
    else:
        result = [str(value).strip()]
    return result or None


# ---------------------------------------------------------------------------
# Model-selected tool routing and deterministic execution
# ---------------------------------------------------------------------------


# The eight tools the model may call, and the schemas it chooses from, both built
# from patient_tools.py where one object per tool declares the callable and the
# contract together.
#
# patient_id is injected rather than declared, so the model chooses the operation
# and the application chooses the patient.
#
# The descriptions in those schemas are the routing rules, so
# scripts/check_tool_schemas.py pins what is generated here -- offline, in under a
# second, instead of as a routing regression found by a live test.
PATIENT_TOOLS = build_patient_tools(record_tools, tools_config)
TOOL_SPECS: list[dict[str, Any]] = model_facing_specs(PATIENT_TOOLS)

TOOL_SELECTION_SYSTEM_PROMPT = (
    "You answer questions about one synthetic patient's medical records.\n"
    "Call a tool only when the question needs an exact count, a date-ordered "
    "timeline, a numeric trend, a comparison of two named documents, or medication "
    "changes.\n"
    "Answering WITHOUT a tool is the default and the most common case. Any question "
    "about what the records say, show, mean or contain -- results, findings, "
    "symptoms, history, summaries, or anything needing interpretation of the text -- "
    "must be answered with NO tool call, from retrieved record text.\n"
    "Only call a tool when the question is one of those specific operations, or when "
    "it names a particular document or document type to read or list.\n"
    "Never invent dates, document ids, measurement names or record types. Omit any "
    "argument the user did not state: an omitted filter means 'all of them', which "
    "is nearly always what a general question wants. Narrowing to something the user "
    "did not ask for can silently return zero results.\n"
    "Never ask for a patient id; the system supplies it."
)


@mlflow.trace(name="tool_selection", span_type=SpanType.LLM)
def _select_tool_with_llm(question: str, chat_history: str) -> tuple[str | None, dict[str, Any]]:
    """Let the model choose at most one tool and extract its arguments.

    Returns the internal tool key and the model's arguments, or (None, {}) when the
    question should be answered from retrieved record text instead.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": TOOL_SELECTION_SYSTEM_PROMPT}
    ]
    if chat_history:
        messages.append(
            {"role": "system", "content": f"Recent conversation:\n{chat_history}"}
        )
    messages.append({"role": "user", "content": question})
    try:
        response = llm.bind_tools(TOOL_SPECS).invoke(messages)
    except Exception:
        # Selection must never break the request; fall back to retrieval.
        return None, {}
    calls = getattr(response, "tool_calls", None) or []
    if not calls:
        return None, {}
    first = calls[0]
    name = str(first.get("name") or "")
    key = TOOL_NAME_TO_KEY.get(name)
    if not key:
        return None, {}
    arguments = first.get("args")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
    return key, dict(arguments or {})


def _route_tool(
    question: str,
    custom: dict[str, Any],
    chat_history: str = "",
) -> tuple[str | None, dict[str, Any]]:
    """Resolve which focused function to run, and with what arguments.

    An explicit custom_inputs.tool always wins, so evaluation rows and scripted
    callers stay deterministic. Otherwise the model decides.
    """

    if not bool(tools_config.get("enabled", True)):
        return None, {}

    explicit = str(custom.get("tool") or "").strip().lower()
    # Every model-facing name is accepted, plus the internal keys and short forms a
    # caller is likely to type. Spreading TOOL_NAME_TO_KEY keeps the eight canonical
    # names in step with the tools themselves.
    aliases = {
        **TOOL_NAME_TO_KEY,
        "timeline": "timeline",
        "build_timeline": "timeline",
        "compare": "compare_documents",
        "diff": "compare_documents",
        "compare_documents": "compare_documents",
        "lab_trend": "lab_trend",
        "medication_changes": "medication_changes",
        "count": "count_documents",
        "count_documents": "count_documents",
        "patient": "set_active_patient",
        "set_patient": "set_active_patient",
        "list": "list_documents",
        "list_documents": "list_documents",
        "read": "read_documents",
        "read_documents": "read_documents",
    }
    # Pin the retrieval path explicitly, so an evaluation row meant to exercise
    # retrieval cannot be routed to a tool at runtime, produce no retriever span,
    # and make RetrievalGroundedness unscorable for that row.
    if explicit in {"none", "no_tool", "retrieval", "search"}:
        return None, {}
    if explicit:
        if explicit not in aliases:
            raise ValueError(
                "Unknown tool. Use one of: "
                + ", ".join(sorted(set(aliases.values())))
                + ", or 'none' to pin the retrieval path."
            )
        # Scripted callers pass the arguments themselves in custom_inputs.
        return aliases[explicit], dict(custom)

    if not bool(tools_config.get("enable_intent_routing", True)):
        return None, {}

    return _select_tool_with_llm(question, chat_history)


@mlflow.trace(name="focused_patient_record_tool", span_type=SpanType.TOOL)
def _run_tool(
    tool_name: str,
    patient_id: str,
    arguments: dict[str, Any],
) -> ToolResult:
    """Invoke one StructuredTool with the patient bound by the application.

    `arguments` come from the model's tool call, or from custom_inputs when a caller
    pinned the tool. Nothing here parses the question text: argument extraction is
    the model's job, date normalisation and absent-argument refusals belong to each
    tool's wrapper in patient_tools.py, and patient_id is bound below so the model
    cannot change which patient is in scope.
    """
    tool = PATIENT_TOOLS.get(tool_name)
    if tool is None:
        raise ValueError(f"Unsupported routed tool: {tool_name}")

    # A pinned caller passes its whole custom_inputs dict, which carries keys that
    # are not tool arguments -- tool, session_id, user_id. Pydantic drops unknown
    # keys, so they are harmless; patient_id is assigned last and therefore always
    # wins over anything the caller or the model put there.
    payload = dict(arguments or {})
    fields = getattr(getattr(tool, "args_schema", None), "model_fields", {}) or {}
    if "patient_id" in fields:
        payload["patient_id"] = patient_id
    else:
        # set_active_patient resolves a patient rather than reading one's records,
        # so it takes no patient_id and must not be handed the current scope.
        payload.pop("patient_id", None)
    return tool.invoke(payload)


# ---------------------------------------------------------------------------
# Graph nodes. Each returns only the state keys it writes; the topology that
# connects them is in orchestration.py.
# ---------------------------------------------------------------------------


def _node_load_memory(state: NavigatorState) -> dict[str, Any]:
    memory = memory_store.load(
        user_id=state["user_id"], session_id=state["session_id"]
    )
    return {
        "memory": memory,
        "patient_id": _patient_id(state["custom"], memory),
        "history": _stored_or_request_history(state["messages"], memory),
        "memory_context": _memory_context(memory),
    }


def _node_route(state: NavigatorState) -> dict[str, Any]:
    tool_key, tool_args = _route_tool(
        state["question"], state["custom"], state["history"]
    )
    return {"tool_key": tool_key, "tool_args": tool_args}


def _node_run_tool(state: NavigatorState) -> dict[str, Any]:
    tool_key = state["tool_key"]
    result = _run_tool(tool_key, state["patient_id"], state["tool_args"])
    updates: dict[str, Any] = {"tool_result": result, "tool_data": result.data}

    # Switching patient is handled here rather than inside the tool because it is
    # the one operation that changes request scope instead of reading records. It
    # lets a caller that cannot send custom_inputs -- the Databricks AI Playground
    # -- pick a patient by asking, while the UI keeps setting it explicitly. An
    # explicit custom_inputs.patient_id still wins on every later turn, because
    # _patient_id consults it before memory.
    if tool_key == "set_active_patient":
        resolved = str((result.data or {}).get("patient_id") or "")
        if resolved:
            state["memory"].active_patient_id = resolved
            updates["patient_id"] = resolved
    return updates


def _node_explain_tool(state: NavigatorState) -> dict[str, Any]:
    result = state["tool_result"]
    answer = tool_explanation_chain.invoke(
        {
            "question": state["question"],
            "tool_name": result.name,
            "tool_output": result.output,
        }
    )
    return {
        "answer": answer,
        "sources": [
            {"document_id": document_id, "source": "deterministic_tool"}
            for document_id in result.source_document_ids
        ],
    }


def _node_expand_queries(state: NavigatorState) -> dict[str, Any]:
    bundle = query_engine.build(
        question=state["question"],
        chat_history=state["history"],
        strategies_override=_list_value(state["custom"].get("query_strategies")),
    )
    return {"query_bundle": bundle}


def _node_retrieve(state: NavigatorState) -> dict[str, Any]:
    hits = _retrieve_with_trace(
        queries=state["query_bundle"].retrieval_queries,
        patient_id=state["patient_id"],
        document_types=_list_value(state["custom"].get("document_types")),
    )
    return {"hits": hits, "context": retriever.format_context(hits)}


def _node_generate_answer(state: NavigatorState) -> dict[str, Any]:
    hits = state.get("hits") or []
    context = state.get("context") or ""
    if not context:
        # No evidence means no answer to ground, so the model is not called at all.
        answer = (
            f"I could not find matching evidence for patient {state['patient_id']}. "
            "Check the patient ID, date range, document type, and whether the "
            "AI Search index has been synchronized."
        )
    else:
        answer = answer_chain.invoke(
            {
                "memory_context": state["memory_context"],
                "chat_history": state["history"],
                "context": context,
                "question": state["question"],
            }
        )
    return {"answer": answer, "sources": retriever.source_summaries(hits)}


def _node_save_memory(state: NavigatorState) -> dict[str, Any]:
    saved = _update_memory(
        state["memory"],
        patient_id=state["patient_id"],
        question=state["question"],
        answer=state["answer"],
        tool_name=state.get("tool_key"),
    )
    return {"memory_saved": saved}


NAVIGATOR_GRAPH = build_navigator_graph(
    NavigatorNodes(
        load_memory=_node_load_memory,
        route=_node_route,
        run_tool=_node_run_tool,
        explain_tool=_node_explain_tool,
        expand_queries=_node_expand_queries,
        retrieve=_node_retrieve,
        generate_answer=_node_generate_answer,
        save_memory=_node_save_memory,
    )
)


# ---------------------------------------------------------------------------
# Traced retrieval wrapper
# ---------------------------------------------------------------------------


@mlflow.trace(name="patient_scoped_ai_search", span_type=SpanType.RETRIEVER)
def _retrieve_with_trace(
    *,
    queries: list[str],
    patient_id: str,
    document_types: list[str] | None,
) -> list[SearchHit]:
    """Expose retrieval as a dedicated MLflow span while preserving patient filters."""

    # Filled by the retriever with which search path actually served each probe.
    # Both the hybrid path and the reranker are optional at runtime and degrade to
    # an answer that looks completely normal -- same citations, same format, worse
    # evidence -- so the trace has to say which one ran.
    diagnostics: dict[str, Any] = {}
    hits = retriever.retrieve(
        queries=queries,
        patient_id=patient_id,
        document_types=document_types,
        diagnostics=diagnostics,
    )
    # Retrieval metrics on the span itself, so the Tracing UI shows how many
    # probes ran and how well they scored, not just that retrieval happened.
    # Telemetry must never break a request, hence the guard.
    try:
        span = mlflow.get_current_active_span()
        if span is not None:
            scores = [h.score for h in hits if isinstance(h.score, (int, float))]
            fused = [h.rrf_score for h in hits if isinstance(h.rrf_score, (int, float))]
            documents = {
                str(h.metadata.get("document_id") or "") for h in hits if h.metadata
            }
            span.set_attributes(
                {
                    "retrieval.query_count": len(queries),
                    "retrieval.hit_count": len(hits),
                    "retrieval.patient_id": patient_id,
                    "retrieval.document_type_filter": ",".join(document_types or []) or "none",
                    "retrieval.top_score": round(max(scores), 6) if scores else 0.0,
                    "retrieval.mean_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
                    "retrieval.top_rrf_score": round(max(fused), 6) if fused else 0.0,
                    "retrieval.distinct_documents": len(documents - {""}),
                    # Which path served, and whether it was reranked. Sets are
                    # joined because span attributes take scalars, and a request
                    # runs several probes that can take different paths.
                    "retrieval.path": ",".join(
                        sorted(diagnostics.get("paths") or [])
                    ) or ("cached" if diagnostics.get("cached") else "none"),
                    "retrieval.filter_syntax": ",".join(
                        sorted(diagnostics.get("filter_syntax") or [])
                    ) or "none",
                    "retrieval.cached": bool(diagnostics.get("cached")),
                    "retrieval.probes_run": int(diagnostics.get("probes", 0)),
                    "retrieval.reranked_probes": int(
                        diagnostics.get("reranked_probes", 0)
                    ),
                    "retrieval.reranker_available": bool(
                        diagnostics.get("reranker_available")
                    ),
                    "retrieval.hybrid_failures": int(
                        diagnostics.get("hybrid_failures", 0)
                    ),
                    "retrieval.fulltext_failures": int(
                        diagnostics.get("fulltext_failures", 0)
                    ),
                    "retrieval.warnings": ",".join(
                        sorted(diagnostics.get("warnings") or [])
                    ) or "none",
                    # Only present when something downgraded, so an empty string
                    # here means the preferred path served every probe.
                    "retrieval.first_error": str(
                        diagnostics.get("first_hybrid_error")
                        or diagnostics.get("first_fallback_error")
                        or ""
                    ),
                }
            )
    except Exception:
        pass
    return hits


# ---------------------------------------------------------------------------
# Memory update and complete request execution
# ---------------------------------------------------------------------------


def _update_memory(
    memory: SessionMemory,
    *,
    patient_id: str,
    question: str,
    answer: str,
    tool_name: str | None,
) -> bool:
    memory.active_patient_id = patient_id
    memory.last_tool = tool_name
    memory.recent_messages.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )
    keep = int(memory_config.get("recent_message_limit", 10))
    memory.recent_messages = memory.recent_messages[-max(2, keep) :]
    # A compact deterministic summary prevents unbounded memory growth and does not
    # require another LLM call. It stores workflow state, not clinical conclusions.
    memory.summary = (
        f"Active patient {patient_id}; {len(memory.recent_messages)} recent messages "
        f"retained; last operation {tool_name or 'retrieval answer'}."
    )
    memory_store.save(memory)
    return memory_store.enabled and memory_store.last_error is None


@mlflow.trace(name="patient_history_request", span_type=SpanType.CHAIN)
def _execute(request: ResponsesAgentRequest) -> tuple[str, dict[str, Any]]:
    """Run one request through the graph and shape the response metadata.

    Everything between loading memory and saving it is the graph's business; this
    function only adapts the ResponsesAgent request into initial state and the final
    state into custom_outputs.
    """
    messages = _messages(request)
    custom = _as_dict(getattr(request, "custom_inputs", None))
    user_id, session_id = _context_ids(request)

    final: NavigatorState = NAVIGATOR_GRAPH.invoke(
        {
            "question": _latest_user_message(messages),
            "messages": messages,
            "custom": custom,
            "user_id": user_id,
            "session_id": session_id,
        }
    )

    patient_id = str(final.get("patient_id") or "")
    tool_name = final.get("tool_key")
    answer = final.get("answer") or ""

    # Tag the trace so the Tracing UI can be filtered by patient or by which path
    # answered. Wrapped because telemetry must never fail a request.
    try:
        span = mlflow.get_current_active_span()
        if span is not None:
            span.set_attributes(
                {
                    "request.patient_id": patient_id,
                    "request.answer_path": tool_name or "retrieval",
                    "request.tool": tool_name or "none",
                    "request.session_id": session_id,
                }
            )
    except Exception:
        pass

    metadata: dict[str, Any] = {
        "application": "patient-history-navigator",
        "patient_id": patient_id,
        "user_id": user_id,
        "session_id": session_id,
        "tool": tool_name,
        "memory_enabled": memory_store.enabled,
        "memory_saved": bool(final.get("memory_saved")),
        "memory_error": memory_store.last_error,
        "sources": final.get("sources") or [],
        # Structured result of the tool that ran, so clients chart the computed
        # numbers instead of parsing them back out of the model's prose.
        "tool_data": final.get("tool_data"),
        "cache_policy": {
            "query_plan_ttl_seconds": cache_config.get("query_plan_ttl_seconds", 600),
            "retrieval_ttl_seconds": cache_config.get("retrieval_ttl_seconds", 300),
            "final_answers_cached": False,
        },
    }

    query_bundle = final.get("query_bundle")
    if query_bundle is not None:
        metadata["standalone_question"] = query_bundle.standalone_question
        metadata["retrieval_queries"] = query_bundle.retrieval_queries
        metadata["query_strategy_outputs"] = query_bundle.strategy_outputs
        metadata["active_query_strategies"] = [
            name for name in query_bundle.strategy_outputs if name in STRATEGIES
        ]
        metadata["query_strategy_errors"] = {
            name: value
            for name, value in query_bundle.strategy_outputs.items()
            if name.endswith("_error")
        }
    return str(answer), metadata


class PatientHistoryResponsesAgent(ResponsesAgent):
    """MLflow ResponsesAgent wrapper around deterministic tools and LangChain RAG."""

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        answer, metadata = _execute(request)
        return ResponsesAgentResponse(
            output=[
                self.create_text_output_item(
                    text=answer,
                    id=f"msg_{uuid4().hex}",
                )
            ],
            custom_outputs=metadata,
        )

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        answer, _metadata = _execute(request)
        item_id = f"msg_{uuid4().hex}"
        chunk_size = 160
        for start in range(0, len(answer), chunk_size):
            yield ResponsesAgentStreamEvent(
                **self.create_text_delta(
                    delta=answer[start : start + chunk_size],
                    item_id=item_id,
                )
            )
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=answer, id=item_id),
        )


set_model(PatientHistoryResponsesAgent())
