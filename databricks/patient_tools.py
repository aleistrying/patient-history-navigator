"""The eight patient-record tools, as LangChain StructuredTools.

One object per tool declares both the schema the model chooses from and the
callable that runs, so a schema cannot advertise an argument the executor ignores.

`patient_id` is never model-selectable: the model chooses the *operation*, the
application chooses the *patient*. `Annotated[str, InjectedToolArg]` enforces that
in both directions -- LangChain excludes injected arguments from the schema sent to
the model, so the model cannot see or set the field, and the field has no default,
so `tool.invoke()` raises `ValidationError` if the application fails to bind it.

Optional string arguments default to `""`, not `None`, to keep the generated schema
a plain `{"type": "string"}`. `Optional[str]` renders as `anyOf: [string, null]`,
a more complex contract to hand a model for no gain. Every consumer here treats
empty as absent.

Argument descriptions are the routing rules -- they are how the model decides which
tool applies and what to extract. `scripts/check_tool_schemas.py` pins them, so
edit them with that test to hand.
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from langchain_core.tools import InjectedToolArg, StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field

from record_tools import PatientRecordTools, ToolResult

# Bound server-side on every call. Repeated as a constant so the eight schemas
# below cannot disagree about it.
_INJECTED_PATIENT = Annotated[str, InjectedToolArg]
_PATIENT_FIELD = Field(
    description="Bound by the application from the request. Never model-supplied."
)


def normalise_date(value: Any, *, end_of_year: bool = False) -> str | None:
    """Accept what the model supplies and make it a date SQL can parse.

    The model legitimately answers "between 2014 and 2026" with bare years, which
    DATE('2014') will not accept.
    """
    # The model occasionally emits a value with stray quoting from malformed JSON,
    # e.g. "2023-01-01', ". Trim that rather than discarding an otherwise valid
    # date, which would silently drop the filter the question asked for.
    text = str(value or "").strip().strip("'\"`,; \t")
    if not text:
        return None
    if re.fullmatch(r"\d{4}", text):
        return f"{text}-12-31" if end_of_year else f"{text}-01-01"
    if re.fullmatch(r"\d{4}-\d{2}", text):
        return f"{text}-28" if end_of_year else f"{text}-01"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    return None


def _clean(value: Any) -> str | None:
    """Empty, whitespace-only and missing all mean 'argument not supplied'."""
    text = str(value or "").strip()
    return text or None


# ---------------------------------------------------------------------------
# Argument schemas. Descriptions are the routing rules -- see module docstring.
# ---------------------------------------------------------------------------


class TimelineArgs(BaseModel):
    patient_id: _INJECTED_PATIENT = _PATIENT_FIELD
    start_date: str = Field(
        default="",
        description="ISO date. Omit unless the question names a start date or year.",
    )
    end_date: str = Field(
        default="",
        description="ISO date. Omit unless the question names an end date or year.",
    )
    document_types: str = Field(
        default="",
        description="Optional comma-separated document types to restrict to.",
    )


class CompareArgs(BaseModel):
    patient_id: _INJECTED_PATIENT = _PATIENT_FIELD
    date_a: str = Field(
        default="",
        description=(
            "The earlier date the user named, as YYYY-MM-DD. Use this "
            "whenever the user refers to documents by when they were "
            "recorded."
        ),
    )
    date_b: str = Field(
        default="",
        description="The later date the user named, as YYYY-MM-DD.",
    )
    document_type: str = Field(
        default="",
        description=(
            "The kind of document to compare, e.g. encounter_note, "
            "laboratory_report, diagnostic_imaging_report. Use this "
            "instead of guessing ids, and alongside dates to narrow "
            "which document on each date is meant."
        ),
    )
    document_id_a: str = Field(
        default="",
        description="Only if a real id was given to you earlier.",
    )
    document_id_b: str = Field(
        default="",
        description="Only if a real id was given to you earlier.",
    )


class LabTrendArgs(BaseModel):
    patient_id: _INJECTED_PATIENT = _PATIENT_FIELD
    test_name: str = Field(
        default="",
        description=(
            "One measurement name such as Hemoglobin, Body Weight, "
            "Heart rate. Omit for an overview of all measurements."
        ),
    )
    start_date: str = Field(
        default="", description="Omit unless the question names a period."
    )
    end_date: str = Field(
        default="", description="Omit unless the question names a period."
    )


class MedicationChangesArgs(BaseModel):
    """The only tool whose schema marks arguments required -- advisory, not enforced.

    "What changed" is meaningless without two points to compare and there is no
    defensible default pair, so the model is told both dates are required.

    Execution still tolerates their absence. A field with no default makes `invoke`
    raise `ValidationError`, which reaches the user as a failed request; the tool
    instead returns a sentence naming what is missing, which is an answer. So the
    fields carry defaults, `medication_changes` handles the empty case, and
    `model_facing_specs` re-asserts the required list onto the emitted schema.
    """

    patient_id: _INJECTED_PATIENT = _PATIENT_FIELD
    earlier_date: str = Field(
        default="", description="The earlier of the two dates to compare."
    )
    later_date: str = Field(
        default="", description="The later of the two dates to compare."
    )

    # A `__get_pydantic_json_schema__` override here would not run: LangChain builds
    # a separate model for the model-facing schema, with injected arguments stripped,
    # and that derived model carries none of this class's hooks.


class SetActivePatientArgs(BaseModel):
    patient: str = Field(
        default="",
        description=(
            "Name, id, id prefix or ON-SYN display id. Omit to list "
            "the available patients."
        ),
    )


class ListDocumentsArgs(BaseModel):
    patient_id: _INJECTED_PATIENT = _PATIENT_FIELD
    document_type: str = Field(
        default="",
        description="Optional single type, e.g. diagnostic_imaging_report.",
    )
    start_date: str = Field(
        default="", description="Omit unless the question names a period."
    )
    end_date: str = Field(
        default="", description="Omit unless the question names a period."
    )


class ReadDocumentsArgs(BaseModel):
    patient_id: _INJECTED_PATIENT = _PATIENT_FIELD
    document_type: str = Field(
        default="",
        description=(
            "e.g. encounter_note, specialist_note, discharge_summary, "
            "diagnostic_imaging_report."
        ),
    )
    document_ids: str = Field(
        default="",
        description="Comma-separated real ids. Never invent these.",
    )
    start_date: str = Field(
        default="", description="Omit unless the question names a period."
    )
    end_date: str = Field(
        default="", description="Omit unless the question names a period."
    )


class CountDocumentsArgs(BaseModel):
    patient_id: _INJECTED_PATIENT = _PATIENT_FIELD
    document_type: str = Field(
        default="",
        description=(
            "ONE record type, only when the user names it. Omit it for "
            "any general count, including 'how many records', 'how many "
            "of each' or 'per category' -- omitting returns the full "
            "breakdown by type. Never guess a type the user did not say."
        ),
    )
    start_date: str = Field(
        default="",
        description="Omit unless the user names a period. Omitting counts all time.",
    )
    end_date: str = Field(
        default="",
        description="Omit unless the user names a period. Omitting counts all time.",
    )


# Tool-level descriptions, kept beside the schemas they belong to.
TOOL_DESCRIPTIONS: dict[str, str] = {
    "build_patient_timeline": (
        "List the patient's recorded events in date order. Use for "
        "'timeline', 'chronology', or 'what happened when'."
    ),
    "compare_patient_documents": (
        "Show the textual differences between two of the patient's documents. "
        "Document ids are opaque hashes: never invent them. If the user names "
        "two dates, pass them as date_a and date_b — that is the normal case. "
        "Otherwise omit the ids and pass document_type, and the two most recent "
        "documents of that type will be compared. Only pass ids that appeared "
        "verbatim earlier in the conversation."
    ),
    "analyze_lab_trend": (
        "Compute how a recorded numeric measurement changed over time (a "
        "lab analyte or a vital sign). Use when the question asks about a "
        "trend, change, or progression of a measurement. Omit test_name to "
        "summarise every recorded measurement."
    ),
    "detect_medication_changes": (
        "Compare the medications recorded at two dates and list what was "
        "added or removed. Use only when the question asks what medications "
        "changed, started, or stopped."
    ),
    "set_active_patient": (
        "Choose which patient the conversation is about, and remember it for "
        "later turns. Use when the user names a patient, asks to switch "
        "patient, or asks which patients are available. Accepts a patient id, "
        "an id prefix, an ON-SYN display id, or a person's name. Call it with "
        "no argument to list the patients on record."
    ),
    "list_patient_documents": (
        "List every record the patient has: date, type, title and id. Use "
        "whenever the question asks what documents or records exist, or which "
        "documents are of some kind. Retrieval sees only a few documents and "
        "will under-report, so prefer this for any 'what documents' question."
    ),
    "read_patient_documents": (
        "Return the verbatim text of up to five documents of ONE named kind. "
        "Use this only when the user names a specific document or document "
        "type to read, such as 'read the discharge summary', 'what do the "
        "specialist notes say', or 'show me the encounter notes'. "
        "Do NOT use it for open questions about what the records contain, "
        "show, or mean -- including summaries, findings, results and any "
        "clinical-content question. Those are answered from retrieved "
        "evidence with no tool call."
    ),
    "count_patient_documents": (
        "Return an exact COUNT of records, broken down by type. Use ONLY for "
        "questions asking how many, how much, or the number of records. Never "
        "use it to describe, list, or summarise what the records contain: a "
        "question like 'what laboratory results are recorded' wants the "
        "contents, not a count."
    ),
}

# Model-facing tool name -> internal key used by pinned callers and traces.
TOOL_NAME_TO_KEY: dict[str, str] = {
    "build_patient_timeline": "timeline",
    "compare_patient_documents": "compare_documents",
    "analyze_lab_trend": "lab_trend",
    "detect_medication_changes": "medication_changes",
    "count_patient_documents": "count_documents",
    "set_active_patient": "set_active_patient",
    "list_patient_documents": "list_documents",
    "read_patient_documents": "read_documents",
}
KEY_TO_TOOL_NAME: dict[str, str] = {
    key: name for name, key in TOOL_NAME_TO_KEY.items()
}

# Arguments advertised to the model as required, though execution tolerates their
# absence -- see MedicationChangesArgs for why the two differ. Kept here because
# Pydantic derives required-ness from the absence of a default, and a field with no
# default makes `invoke` raise instead of letting the tool explain itself.
REQUIRED_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "medication_changes": ("earlier_date", "later_date"),
}


def model_facing_specs(tools: dict[str, StructuredTool]) -> list[dict[str, Any]]:
    """The OpenAI-style schemas to hand to `bind_tools`.

    Generated from the tools themselves, so the contract offered and the code run
    stay one object -- with the single exception of `REQUIRED_ARGUMENTS`.
    `scripts/check_tool_schemas.py` pins this function's output, so a routing
    contract change is caught offline instead of as a live regression.
    """
    specs: list[dict[str, Any]] = []
    for key, tool in tools.items():
        spec = convert_to_openai_tool(tool)
        required = REQUIRED_ARGUMENTS.get(key)
        if required:
            spec["function"].setdefault("parameters", {})["required"] = list(required)
        specs.append(spec)
    return specs


def build_patient_tools(
    record_tools: PatientRecordTools,
    tools_config: dict[str, Any],
) -> dict[str, StructuredTool]:
    """Build the eight tools, closed over the record layer and its config.

    Keyed by *internal key* rather than model-facing name, because the graph, the
    trace attributes and pinned evaluation rows all address tools that way.
    `KEY_TO_TOOL_NAME` maps back when the model-facing name is needed.

    Each wrapper owns the argument discipline for its own tool: date normalisation,
    and the refusals for arguments that have no safe default. Keeping them here
    means every call site inherits them.
    """

    def timeline(
        patient_id: _INJECTED_PATIENT,
        start_date: str = "",
        end_date: str = "",
        document_types: str = "",
    ) -> ToolResult:
        return record_tools.build_timeline(
            patient_id=patient_id,
            start_date=normalise_date(start_date),
            end_date=normalise_date(end_date, end_of_year=True),
            document_types=_clean(document_types),
            limit=int(tools_config.get("timeline_default_rows", 200)),
        )

    def compare_documents(
        patient_id: _INJECTED_PATIENT,
        date_a: str = "",
        date_b: str = "",
        document_type: str = "",
        document_id_a: str = "",
        document_id_b: str = "",
    ) -> ToolResult:
        resolved_a = normalise_date(date_a)
        resolved_b = normalise_date(date_b)
        kind = _clean(document_type)
        ids = [_clean(document_id_a), _clean(document_id_b)]

        # A comparison needs to know *which two*. Every other tool has a defensible
        # default for an absent argument -- all documents, all measurements -- but
        # "compare two things" has none: with nothing supplied the tool would pick a
        # pair the user never asked about and report a similarity for it, which
        # reads as an answer. So say what is missing instead of guessing.
        if bool(resolved_a) != bool(resolved_b):
            given, missing = ("first", "second") if resolved_a else ("second", "first")
            return ToolResult(
                name="compare_patient_documents",
                output=(
                    f"Only the {given} date was given "
                    f"({resolved_a or resolved_b}). Comparing needs two: name the "
                    f"{missing} date as well, and the document type if the patient "
                    "has more than one document on those days."
                ),
                source_document_ids=[],
            )
        if not resolved_a and not all(ids) and not kind:
            return ToolResult(
                name="compare_patient_documents",
                output=(
                    "Comparing needs to know which two documents. Either name two "
                    "dates, or name a document type — encounter note, laboratory "
                    "report, vital signs report — and the two most recent of that "
                    "type will be compared. Ask for the document list to see what "
                    "is on record."
                ),
                source_document_ids=[],
            )
        return record_tools.compare_documents(
            patient_id=patient_id,
            document_id_a=ids[0],
            document_id_b=ids[1],
            document_type=kind,
            date_a=resolved_a,
            date_b=resolved_b,
            context_lines=3,
        )

    def lab_trend(
        patient_id: _INJECTED_PATIENT,
        test_name: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> ToolResult:
        # An absent test_name is valid: the tool then reports every measurement.
        return record_tools.analyze_lab_trend(
            patient_id=patient_id,
            test_name=str(test_name or "").strip(),
            start_date=normalise_date(start_date),
            end_date=normalise_date(end_date, end_of_year=True),
        )

    def medication_changes(
        patient_id: _INJECTED_PATIENT,
        earlier_date: str = "",
        later_date: str = "",
    ) -> ToolResult:
        earlier = normalise_date(earlier_date)
        later = normalise_date(later_date, end_of_year=True)
        # Required in the schema, but the model can still supply an unparseable
        # value, and a dropped date would silently compare the wrong span.
        if not earlier or not later:
            return ToolResult(
                name="detect_medication_changes",
                output=(
                    "Comparing medications needs two dates to compare between. "
                    "Name the two dates or years in your question."
                ),
                source_document_ids=[],
            )
        return record_tools.detect_medication_changes(
            patient_id=patient_id,
            earlier_date=earlier,
            later_date=later,
        )

    def set_active_patient(patient: str = "") -> ToolResult:
        # Resolution only. Applying the result to the session is the graph's job,
        # because changing patient scope changes request state, not record data.
        requested = str(patient or "").strip()
        resolved, message = record_tools.find_patient(requested)
        return ToolResult(
            name="set_active_patient",
            output=message,
            source_document_ids=[],
            data={
                "kind": "patient_scope",
                "patient_id": resolved,
                "resolved": bool(resolved),
            },
        )

    def list_documents(
        patient_id: _INJECTED_PATIENT,
        document_type: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> ToolResult:
        return record_tools.list_documents(
            patient_id=patient_id,
            document_type=_clean(document_type),
            start_date=normalise_date(start_date),
            end_date=normalise_date(end_date, end_of_year=True),
        )

    def read_documents(
        patient_id: _INJECTED_PATIENT,
        document_type: str = "",
        document_ids: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> ToolResult:
        kind = _clean(document_type)
        ids = _clean(document_ids)
        start = normalise_date(start_date)
        end = normalise_date(end_date, end_of_year=True)
        # Reading needs to know *what* to read. The model routes open content
        # questions here despite the description telling it not to, and with no
        # arguments this would return the three most recent documents as a confident
        # answer about whatever was asked -- including record types the patient does
        # not have. Listing the available types instead also answers the question,
        # because the type asked about is visibly absent from the list.
        if not kind and not ids and not start and not end:
            return record_tools.available_document_types(
                patient_id,
                lead=(
                    "Reading a document needs to know which one. Name a document "
                    "type or a document id. This patient has:"
                ),
            )
        return record_tools.read_documents(
            patient_id=patient_id,
            document_type=kind,
            document_ids=ids,
            start_date=start,
            end_date=end,
        )

    def count_documents(
        patient_id: _INJECTED_PATIENT,
        document_type: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> ToolResult:
        return record_tools.count_documents(
            patient_id=patient_id,
            document_type=_clean(document_type),
            start_date=normalise_date(start_date),
            end_date=normalise_date(end_date, end_of_year=True),
        )

    specs: list[tuple[str, Any, type[BaseModel]]] = [
        ("timeline", timeline, TimelineArgs),
        ("compare_documents", compare_documents, CompareArgs),
        ("lab_trend", lab_trend, LabTrendArgs),
        ("medication_changes", medication_changes, MedicationChangesArgs),
        ("set_active_patient", set_active_patient, SetActivePatientArgs),
        ("list_documents", list_documents, ListDocumentsArgs),
        ("read_documents", read_documents, ReadDocumentsArgs),
        ("count_documents", count_documents, CountDocumentsArgs),
    ]
    return {
        key: StructuredTool.from_function(
            func=func,
            name=KEY_TO_TOOL_NAME[key],
            description=TOOL_DESCRIPTIONS[KEY_TO_TOOL_NAME[key]],
            args_schema=args_schema,
        )
        for key, func, args_schema in specs
    }
