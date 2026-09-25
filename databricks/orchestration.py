"""The request graph: what happens, in what order, and where it branches.

This module holds only topology. Every node body is a callable supplied by the
caller (see `NavigatorNodes`), so reading this file tells you the shape of a
request without also reading prompt text, SQL, or retrieval internals.

    START
      v
    load_memory ......... session state, then resolve patient scope
      v
    route ............... the model chooses one tool, or none
      |
      +-- tool ------> run_tool ------> explain_tool ---+
      |                                                 |
      +-- retrieval --> expand_queries                  |
                          v                             |
                        retrieve                        |
                          v                             |
                        generate_answer ----------------+
                                                        v
                                                   save_memory
                                                        v
                                                       END

The branch is data, so an unroutable tool cannot fall through into the wrong path.
Each node reads the state keys it declares and writes the ones it owns, so the two
paths share no mutable locals. And the topology is inspectable:
`compiled.get_graph().draw_mermaid()` renders it, and `scripts/check_graph.py`
drives both paths of the compiled graph end to end.

Retrieval is the common path and `route` is a full LLM call, so the two run in
sequence. Starting retrieval speculatively alongside `route` would halve the
latency of the common case, at the cost of an extra LLM call on every tool answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypedDict


from langgraph.graph import END, START, StateGraph


class NavigatorState(TypedDict, total=False):
    """Everything one request carries.

    `total=False` because nodes return partial updates and the two paths populate
    disjoint halves: a tool answer has no `hits`, a retrieval answer no
    `tool_result`. At runtime the state is a dict, so reading a key the other path
    owns raises `KeyError` rather than yielding a plausible-looking empty value.

    Every key a node reads must be declared here. LangGraph drops input keys the
    schema does not mention, and does so without warning.
    """

    # ---- request, set before the graph runs
    question: str
    # The whole message list, not just the question: load_memory needs it to tell
    # whether the caller supplied its own transcript or the stored one fills the gap.
    messages: list[dict[str, Any]]
    patient_id: str
    custom: dict[str, Any]
    user_id: str
    session_id: str

    # ---- written by load_memory
    memory: Any  # SessionMemory; opaque here on purpose
    history: str
    memory_context: str

    # ---- written by route
    tool_key: str | None
    tool_args: dict[str, Any]

    # ---- written by the tool path
    tool_result: Any  # ToolResult
    tool_data: dict[str, Any] | None

    # ---- written by the retrieval path
    query_bundle: Any  # QueryBundle
    hits: list[Any]
    context: str

    # ---- written by whichever path answered
    answer: str
    sources: list[dict[str, Any]]
    memory_saved: bool


@dataclass(frozen=True)
class NavigatorNodes:
    """The node bodies, injected so this module stays pure topology.

    Each callable takes the state and returns only the keys it writes. That is the
    LangGraph update convention, and it is what stops a node clobbering a sibling's
    output.
    """

    load_memory: Callable[[NavigatorState], dict[str, Any]]
    route: Callable[[NavigatorState], dict[str, Any]]
    run_tool: Callable[[NavigatorState], dict[str, Any]]
    explain_tool: Callable[[NavigatorState], dict[str, Any]]
    expand_queries: Callable[[NavigatorState], dict[str, Any]]
    retrieve: Callable[[NavigatorState], dict[str, Any]]
    generate_answer: Callable[[NavigatorState], dict[str, Any]]
    save_memory: Callable[[NavigatorState], dict[str, Any]]


def chose_tool(state: NavigatorState) -> str:
    """The one branch in the graph: did the model select a tool, or not?

    A named function so the edge carries a legible label in the rendered diagram,
    and so it can be tested without standing up the graph.
    """
    return "tool" if state.get("tool_key") else "retrieval"


def build_navigator_graph(nodes: NavigatorNodes):
    """Wire the nodes into a compiled graph.

    Call `.get_graph().draw_mermaid()` on the result for the diagram used in the
    technical report.
    """
    graph = StateGraph(NavigatorState)

    graph.add_node("load_memory", nodes.load_memory)
    graph.add_node("route", nodes.route)
    graph.add_node("run_tool", nodes.run_tool)
    graph.add_node("explain_tool", nodes.explain_tool)
    graph.add_node("expand_queries", nodes.expand_queries)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("generate_answer", nodes.generate_answer)
    graph.add_node("save_memory", nodes.save_memory)

    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "route")
    graph.add_conditional_edges(
        "route",
        chose_tool,
        {"tool": "run_tool", "retrieval": "expand_queries"},
    )

    # Tool path: run the deterministic function, then let the model explain it.
    graph.add_edge("run_tool", "explain_tool")
    graph.add_edge("explain_tool", "save_memory")

    # Retrieval path: expand into probes, fuse the results, ground the answer.
    graph.add_edge("expand_queries", "retrieve")
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", "save_memory")

    # Both paths converge: every answered turn is persisted the same way.
    graph.add_edge("save_memory", END)

    return graph.compile()
