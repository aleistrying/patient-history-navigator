"""Deterministic record operations over Unity Catalog data.

These are normal Python functions, not an open-ended autonomous agent loop. Each
performs one specific, inspectable operation, and the LLM only explains the result
-- every number in an answer is computed here, never generated.

This module is the record layer and knows nothing about models or tool calling.
`patient_tools.py` wraps these functions as the LangChain StructuredTools the model
chooses from, and owns argument discipline: date normalisation, and refusing an
operation whose arguments have no safe default. The split matters because the two
have different reasons to change -- a new SQL column here, a rephrased description
there -- and because it keeps a query testable without a model.
"""

from __future__ import annotations

import difflib
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


from sql_execution import DatabricksSQLExecutor, sql_literal, validate_fqn


@dataclass
class ToolResult:
    name: str
    output: str
    source_document_ids: list[str]
    # Machine-readable form of the same result, surfaced in custom_outputs so a
    # client can chart or tabulate it. `output` is prose for the LLM to explain
    # and the model routinely rewrites its formatting, so anything that needs the
    # numbers must read them from here rather than parsing the answer text.
    data: dict[str, Any] | None = None


class PatientRecordTools:
    def __init__(
        self,
        executor: DatabricksSQLExecutor,
        documents_table: str,
        config: dict[str, Any],
    ):
        self.executor = executor
        self.documents_table = validate_fqn(documents_table)
        self.config = config
        # Numeric trends read from a per-analyte table when one is configured.
        # documents_table holds one row per document, so a lab panel PDF has a
        # single numeric_value column that cannot represent its many analytes --
        # only the FHIR ingestion path ever fills it. Falls back to
        # documents_table so existing deployments keep working unchanged.
        observations_table = str(config.get("observations_table", "") or "").strip()
        self.observations_table = (
            validate_fqn(observations_table) if observations_table else None
        )

    def build_timeline(
        self,
        patient_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        document_types: str | None = None,
        limit: int = 200,
    ) -> ToolResult:
        where = [f"patient_id = {sql_literal(patient_id)}"]
        if start_date:
            where.append(f"document_date >= DATE({sql_literal(start_date)})")
        if end_date:
            where.append(f"document_date <= DATE({sql_literal(end_date)})")
        # Resolve any requested types against the ones that exist. An unrecognised
        # value ("medication", "medicine") would otherwise filter every row out and
        # report an empty timeline for a patient who has records.
        unresolved: list[str] = []
        if document_types:
            requested = [item.strip() for item in document_types.split(",") if item.strip()]
            resolved: list[str] = []
            for item in requested:
                match = self._resolve_document_type(item)
                if match:
                    resolved.append(match)
                else:
                    unresolved.append(item)
            if resolved:
                where.append(
                    "document_type IN ("
                    + ", ".join(sql_literal(item) for item in sorted(set(resolved)))
                    + ")"
                )
        safe_limit = min(max(int(limit), 1), int(self.config.get("timeline_max_rows", 500)))
        rows = self.executor.query(
            f"""
            SELECT document_id, document_date, document_type, event_name,
                   status, section, source_path,
                   substring(full_text, 1, 1200) AS excerpt
            FROM {self.documents_table}
            WHERE {' AND '.join(where)}
            ORDER BY document_date ASC NULLS LAST, document_id ASC
            LIMIT {safe_limit}
            """
        )
        if not rows:
            message = "No timeline events matched the patient and date filters."
            if unresolved:
                message += (
                    " These requested record types are not used in this corpus and were "
                    "ignored: " + ", ".join(sorted(set(unresolved)))
                    + ". Available types: "
                    + ", ".join(sorted(self._TYPE_SYNONYMS))
                    + "."
                )
            return ToolResult(
                name="build_patient_timeline",
                output=message,
                source_document_ids=[],
            )
        lines = ["Chronological patient timeline:"]
        if unresolved:
            lines.insert(
                0,
                "Ignored unrecognised record types: " + ", ".join(sorted(set(unresolved))) + ".",
            )
        document_ids = []
        for row in rows:
            document_id = str(row.get("document_id", ""))
            document_ids.append(document_id)
            label = row.get("event_name") or row.get("document_type") or "Record"
            lines.append(
                f"- {row.get('document_date') or 'Unknown date'} | {label} "
                f"[{document_id}] | status={row.get('status') or 'n/a'}\n"
                f"  {str(row.get('excerpt') or '').strip()}"
            )
        return ToolResult(
            name="build_patient_timeline",
            output="\n".join(lines),
            source_document_ids=document_ids,
        )

    def _latest_pair_of_type(
        self, patient_id: str, document_type: str | None
    ) -> tuple[str, str] | None:
        """The two most recent comparable documents, optionally of one type.

        "Most recent two" is not sufficient on its own. A laboratory report can be a
        metabolic panel or a urinalysis, and the two most recent were once one of
        each -- a diff of two unrelated panels that reports a meaningless similarity
        percentage. Two documents sharing an event_name are preferred, and only if
        no title has two documents does this fall back to the plain recency pair.
        """
        where = [f"patient_id = {sql_literal(patient_id)}"]
        resolved = self._resolve_document_type(document_type) if document_type else None
        if resolved:
            where.append(f"document_type = {sql_literal(resolved)}")
        rows = self.executor.query(
            f"""
            SELECT document_id, event_name
            FROM {self.documents_table}
            WHERE {' AND '.join(where)}
            ORDER BY document_date DESC NULLS LAST, document_id DESC
            LIMIT 40
            """
        )
        by_title: dict[str, list[str]] = {}
        for row in rows:
            document_id = str(row.get("document_id") or "")
            if not document_id:
                continue
            by_title.setdefault(str(row.get("event_name") or ""), []).append(document_id)
        for title, ids in by_title.items():
            if title and len(ids) >= 2:
                # Oldest-first, so a diff reads as a change forward in time.
                return (ids[1], ids[0])
        ids = [str(row.get("document_id")) for row in rows if row.get("document_id")]
        return (ids[1], ids[0]) if len(ids) >= 2 else None

    def _pair_by_dates(
        self, patient_id: str, date_a: str, date_b: str, document_type: str | None
    ) -> tuple[tuple[str, str] | None, str]:
        """Resolve two dates to two documents, matching titles across them if possible.

        Asked to compare the panels of two visits, the caller names dates, not ids --
        ids are content hashes nobody can know. A date can carry several documents,
        so where both dates share a title (two metabolic panels, say) that pair is
        chosen; comparing a metabolic panel against a urinalysis would produce a
        number that looks like an answer and means nothing.
        """
        resolved = self._resolve_document_type(document_type) if document_type else None
        candidates: dict[str, list[dict[str, Any]]] = {}
        for date in (date_a, date_b):
            where = [
                f"patient_id = {sql_literal(patient_id)}",
                f"to_date(document_date) = to_date({sql_literal(date)})",
            ]
            if resolved:
                where.append(f"document_type = {sql_literal(resolved)}")
            candidates[date] = self.executor.query(
                f"""
                SELECT document_id, event_name, document_type
                FROM {self.documents_table}
                WHERE {' AND '.join(where)}
                ORDER BY document_id ASC
                """
            )

        missing = [date for date, rows in candidates.items() if not rows]
        if missing:
            scope = f" of type {resolved!r}" if resolved else ""
            return None, (
                f"No document{scope} is recorded on {' or '.join(missing)} for this "
                "patient, so those dates could not be compared."
            )

        left_rows, right_rows = candidates[date_a], candidates[date_b]
        right_by_title = {str(r.get("event_name") or ""): r for r in right_rows}
        for row in left_rows:
            title = str(row.get("event_name") or "")
            if title and title in right_by_title:
                return (
                    str(row.get("document_id")),
                    str(right_by_title[title].get("document_id")),
                ), ""

        note = ""
        if len(left_rows) > 1 or len(right_rows) > 1:
            note = (
                f"No document title appears on both dates, so the first document of "
                f"each was compared: {left_rows[0].get('event_name')} against "
                f"{right_rows[0].get('event_name')}. Naming a document type would "
                "narrow this.\n"
            )
        return (
            str(left_rows[0].get("document_id")),
            str(right_rows[0].get("document_id")),
        ), note

    def compare_documents(
        self,
        patient_id: str,
        document_id_a: str | None = None,
        document_id_b: str | None = None,
        context_lines: int = 3,
        document_type: str | None = None,
        date_a: str | None = None,
        date_b: str | None = None,
    ) -> ToolResult:
        """Diff two documents, resolving which two when ids are not supplied.

        A caller asking for "the last two encounter notes" cannot know the ids --
        they are content hashes. Two other ways in exist for that reason: naming two
        dates, which is how people actually refer to visits, or naming a
        document_type and taking the two most recent of it.
        """
        requested = [str(i).strip() for i in (document_id_a, document_id_b) if str(i or "").strip()]
        note = ""
        by_id: dict[str, Any] = {}

        # Dates win over ids. A model that was given no real ids tends to invent
        # them, and an invented id must not silently outrank a date the user typed.
        pair_dates = [str(d).strip() for d in (date_a, date_b) if str(d or "").strip()]
        if len(pair_dates) == 2:
            pair, message = self._pair_by_dates(
                patient_id, pair_dates[0], pair_dates[1], document_type
            )
            if not pair:
                return ToolResult(
                    name="compare_patient_documents", output=message,
                    source_document_ids=[],
                )
            requested, note = [], message
            document_id_a, document_id_b = pair
            rows = self.executor.query(
                f"""
                SELECT document_id, document_date, document_type, event_name,
                       source_path, full_text
                FROM {self.documents_table}
                WHERE patient_id = {sql_literal(patient_id)}
                  AND document_id IN ({sql_literal(document_id_a)}, {sql_literal(document_id_b)})
                """
            )
            by_id = {str(row.get("document_id")): row for row in rows}

        elif len(requested) == 2:
            rows = self.executor.query(
                f"""
                SELECT document_id, document_date, document_type, event_name,
                       source_path, full_text
                FROM {self.documents_table}
                WHERE patient_id = {sql_literal(patient_id)}
                  AND document_id IN ({', '.join(sql_literal(i) for i in requested)})
                """
            )
            by_id = {str(row.get("document_id")): row for row in rows}

        if len(by_id) != 2:
            # Either no ids were given, or they do not exist for this patient.
            pair = self._latest_pair_of_type(patient_id, document_type)
            if not pair:
                available = self.executor.query(
                    f"""
                    SELECT document_type, count(*) AS records
                    FROM {self.documents_table}
                    WHERE patient_id = {sql_literal(patient_id)}
                    GROUP BY document_type
                    HAVING count(*) >= 2
                    ORDER BY records DESC
                    """
                )
                types = [
                    f"{row.get('document_type')} ({row.get('records')})" for row in available
                ]
                message = (
                    "Two documents of the same kind are needed to compare. "
                    + (
                        "Record types with at least two documents: " + "; ".join(types) + "."
                        if types
                        else "This patient has no record type with two or more documents."
                    )
                )
                return ToolResult(
                    name="compare_patient_documents", output=message, source_document_ids=[]
                )
            if requested:
                note = (
                    "The supplied document ids were not found for this patient, so the two "
                    "most recent matching documents were compared instead.\n"
                )
            document_id_a, document_id_b = pair
            rows = self.executor.query(
                f"""
                SELECT document_id, document_date, document_type, event_name,
                       source_path, full_text
                FROM {self.documents_table}
                WHERE patient_id = {sql_literal(patient_id)}
                  AND document_id IN ({sql_literal(document_id_a)}, {sql_literal(document_id_b)})
                """
            )
            by_id = {str(row.get("document_id")): row for row in rows}
            if len(by_id) != 2:
                return ToolResult(
                    name="compare_patient_documents",
                    output="Could not load two comparable documents for this patient.",
                    source_document_ids=list(by_id),
                )
        document_id_a = str(document_id_a)
        document_id_b = str(document_id_b)
        left = str(by_id[document_id_a].get("full_text") or "").splitlines()
        right = str(by_id[document_id_b].get("full_text") or "").splitlines()
        ratio = difflib.SequenceMatcher(None, "\n".join(left), "\n".join(right)).ratio()
        diff = list(
            difflib.unified_diff(
                left,
                right,
                fromfile=document_id_a,
                tofile=document_id_b,
                lineterm="",
                n=max(0, min(int(context_lines), 10)),
            )
        )
        max_lines = int(self.config.get("diff_max_lines", 250))
        clipped = diff[:max_lines]
        if len(diff) > max_lines:
            clipped.append(f"... diff clipped after {max_lines} lines ...")
        left_row = by_id[document_id_a]
        right_row = by_id[document_id_b]
        def describe(row: dict[str, Any], document_id: str) -> str:
            # Title first, then date, then id. An id alone is unreadable, and the
            # reader needs to know *which* panel was compared, not just that two
            # laboratory reports were.
            title = str(row.get("event_name") or "").strip()
            kind = str(row.get("document_type") or "document").strip()
            label = f"{title} ({kind})" if title else kind
            return f"{label} dated {row.get('document_date')} [{document_id}]"

        output = (
            note
            + f"Comparing {describe(left_row, document_id_a)} "
            f"with {describe(right_row, document_id_b)}.\n"
            f"Document similarity: {ratio:.1%}\n"
            f"Removed lines begin with '-'; added lines begin with '+'.\n\n"
            + "\n".join(clipped or ["No textual differences detected."])
        )
        return ToolResult(
            name="compare_patient_documents",
            output=output,
            source_document_ids=[document_id_a, document_id_b],
        )

    # A medication carrying one of these is not in effect, whatever its date.
    _TERMINAL_STATUSES = frozenset({"stopped", "completed", "cancelled"})

    # Phrases that ask for "the trend" generally rather than naming one analyte.
    _TREND_OVERVIEW_TERMS = frozenset(
        {
            "lab", "labs", "lab result", "lab results", "laboratory",
            "laboratory result", "laboratory results", "result", "results",
            "test", "tests", "value", "values", "measurement", "measurements",
            "vital", "vitals", "vital sign", "vital signs", "all", "everything",
            "any", "trend", "trends", "data",
        }
    )

    @staticmethod
    def _summarise(points: list[tuple[str, float, str, str]]) -> tuple[float, float, float, float]:
        first, last = points[0][1], points[-1][1]
        delta = last - first
        percent = (delta / first * 100.0) if first else math.nan
        return first, last, delta, percent

    def _match_series(self, wanted: str, names: list[str]) -> list[str]:
        """Validate a measurement name against the names actually on record.

        The caller supplies a measurement name, not a sentence -- the model
        extracts it. This only has to bridge naming differences between what a
        person calls a measurement and the LOINC-style label stored with it, e.g.
        "Hemoglobin" for "Hemoglobin [Mass/volume] in Blood".
        """
        needle = " ".join(str(wanted or "").lower().split())
        if not needle or needle in self._TREND_OVERVIEW_TERMS:
            return []
        exact = [n for n in names if needle == n.lower()]
        if exact:
            return exact
        contains = [n for n in names if needle in n.lower() or n.lower() in needle]
        if contains:
            return contains
        # Shared significant words, so "eye pressure" finds both
        # "Right eye Intraocular pressure" and its left-eye counterpart.
        tokens = {t for t in re.split(r"[^a-z0-9]+", needle) if len(t) > 3}
        if not tokens:
            return []
        scored = []
        for name in names:
            name_tokens = {t for t in re.split(r"[^a-z0-9]+", name.lower()) if len(t) > 3}
            overlap = len(tokens & name_tokens)
            if overlap:
                scored.append((overlap, name))
        if not scored:
            return []
        best = max(score for score, _ in scored)
        return [name for score, name in scored if score == best]

    def analyze_lab_trend(
        self,
        patient_id: str,
        test_name: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> ToolResult:
        """Numeric trend for one measurement, or an overview across all of them.

        Asked for a specific analyte, this reports that series. Asked something
        general ("what is the trend of the lab results"), it reports every series
        with at least two points rather than failing on an unmatched name.
        """
        where = [
            f"patient_id = {sql_literal(patient_id)}",
            "numeric_value IS NOT NULL",
        ]
        if start_date:
            where.append(f"document_date >= DATE({sql_literal(start_date)})")
        if end_date:
            where.append(f"document_date <= DATE({sql_literal(end_date)})")
        # One query for everything on record; matching happens in Python so a
        # near-miss on a long LOINC-style name can still resolve.
        rows = self.executor.query(
            f"""
            SELECT document_id, document_date, event_name, numeric_value, unit
            FROM {self.observations_table or self.documents_table}
            WHERE {' AND '.join(where)}
            ORDER BY event_name ASC, document_date ASC NULLS LAST
            LIMIT 2000
            """
        )

        series: dict[str, list[tuple[str, float, str, str]]] = defaultdict(list)
        for row in rows:
            try:
                name = str(row.get("event_name") or "").strip()
                if not name:
                    continue
                series[name].append(
                    (
                        str(row.get("document_date") or ""),
                        float(row["numeric_value"]),
                        str(row.get("unit") or ""),
                        str(row.get("document_id") or ""),
                    )
                )
            except (TypeError, ValueError, KeyError):
                continue

        if not series:
            return ToolResult(
                name="analyze_lab_trend",
                output=(
                    "No numeric measurements are recorded for this patient in the "
                    "configured observations table."
                ),
                source_document_ids=[],
            )

        matched = self._match_series(str(test_name or ""), list(series))

        # ---- one measurement: full series with every point ----
        if len(matched) == 1:
            name = matched[0]
            points = series[name]
            if len(points) < 2:
                observed = points[0]
                return ToolResult(
                    name="analyze_lab_trend",
                    output=(
                        f"{name} has a single recorded value, so no trend can be "
                        f"computed: {observed[1]:g} {observed[2]} on "
                        f"{observed[0] or 'an unknown date'}."
                    ),
                    source_document_ids=[observed[3]],
                )
            first, last, delta, percent = self._summarise(points)
            unit = next((u for _, _, u, _ in points if u), "")
            direction = "increased" if delta > 0 else "decreased" if delta < 0 else "was unchanged"
            lines = [
                f"Numeric trend for {name}: {direction} from {first:g} to {last:g} {unit}.".replace(
                    "  ", " "
                ),
                f"Absolute change: {delta:+g} {unit}".strip(),
                f"Percent change: {percent:+.1f}%"
                if math.isfinite(percent)
                else "Percent change: undefined",
            ]
            # First-to-last alone can mislead: a value can end higher than it
            # started while having peaked well above both.
            values = [value for _, value, _, _ in points]
            low, high = min(values), max(values)
            if high != low and (high > max(first, last) or low < min(first, last)):
                lines.append(
                    f"Range across all {len(points)} values: {low:g} to {high:g} {unit}, "
                    f"so the series did not move in one direction only.".replace("  ", " ")
                )
            lines.append("Observed values:")
            for observed_date, value, point_unit, document_id in points:
                lines.append(
                    f"- {observed_date or 'Unknown date'}: {value:g} {point_unit} [{document_id}]"
                )
            lines.append(
                "This tool reports recorded numeric changes only; it does not "
                "interpret clinical significance."
            )
            return ToolResult(
                name="analyze_lab_trend",
                output="\n".join(lines),
                source_document_ids=[point[3] for point in points],
                data={
                    "kind": "trend",
                    "measurement": name,
                    "unit": unit,
                    "first": first,
                    "last": last,
                    "delta": delta,
                    "percent": percent if math.isfinite(percent) else None,
                    "low": low,
                    "high": high,
                    "points": [
                        {
                            "date": observed_date,
                            "value": value,
                            "unit": point_unit,
                            "document_id": document_id,
                        }
                        for observed_date, value, point_unit, document_id in points
                    ],
                },
            )

        # ---- several measurements, or a general request: one line each ----
        selected = matched or [
            name for name, points in series.items() if len(points) >= 2
        ]
        selected = sorted(selected, key=lambda name: (-len(series[name]), name))
        if not selected:
            single = sorted(series)
            return ToolResult(
                name="analyze_lab_trend",
                output=(
                    "Every recorded measurement for this patient has only one value, "
                    "so no trend can be computed. Measurements on record: "
                    + "; ".join(single[:40])
                ),
                source_document_ids=[series[name][0][3] for name in single[:20]],
            )

        if matched:
            heading = f"{len(selected)} recorded measurements match {test_name!r}:"
        else:
            heading = (
                f"Trends across all {len(selected)} recorded measurements with at "
                "least two values:"
            )
        lines = [heading]
        document_ids: list[str] = []
        for name in selected[:25]:
            points = series[name]
            first, last, delta, percent = self._summarise(points)
            unit = next((u for _, _, u, _ in points if u), "")
            change = "no change" if delta == 0 else f"{delta:+g}"
            percent_text = f", {percent:+.1f}%" if math.isfinite(percent) and delta else ""
            lines.append(
                f"* {name}: {first:g} -> {last:g} {unit} ({change}{percent_text}) "
                f"over {len(points)} values, {points[0][0]} to {points[-1][0]}".replace(
                    "  ", " "
                )
            )
            document_ids += [point[3] for point in points]
        if len(selected) > 25:
            lines.append(f"* ... and {len(selected) - 25} more measurements.")
        lines.append(
            "Ask about any single measurement by name for its full list of values."
        )
        lines.append(
            "This tool reports recorded numeric changes only; it does not interpret "
            "clinical significance."
        )
        summaries = []
        for name in selected[:25]:
            points = series[name]
            first, last, delta, percent = self._summarise(points)
            values = [value for _, value, _, _ in points]
            summaries.append(
                {
                    "measurement": name,
                    "unit": next((u for _, _, u, _ in points if u), ""),
                    "first": first,
                    "last": last,
                    "delta": delta,
                    "percent": percent if math.isfinite(percent) else None,
                    "low": min(values),
                    "high": max(values),
                    "count": len(points),
                    "from": points[0][0],
                    "to": points[-1][0],
                }
            )
        return ToolResult(
            name="analyze_lab_trend",
            output="\n".join(lines),
            source_document_ids=document_ids[:60],
            data={
                "kind": "trend_overview",
                "matched": bool(matched),
                "total": len(selected),
                "series": summaries,
            },
        )

    # Words a question might use for each document_type on record.
    _TYPE_SYNONYMS = {
        "laboratory_report": ["lab", "labs", "laboratory", "blood test", "panel"],
        "vital_signs_report": ["vital", "vitals", "observation"],
        "diagnostic_imaging_report": ["imaging", "scan", "x-ray", "xray", "radiology", "mri", "ct"],
        "encounter_note": ["encounter", "visit", "appointment"],
        "specialist_note": ["specialist", "consult", "consultation", "referral"],
        "prescription_record": ["prescription", "prescribed"],
        "medication_history": ["medication", "medications", "drug", "drugs", "meds"],
        "immunization_history": ["immunisation", "immunization", "vaccine", "vaccination"],
        "allergy_report": ["allergy", "allergies", "allergic"],
        "problem_list": ["problem", "problems", "condition", "conditions", "diagnosis list"],
        "care_plan": ["care plan", "plan of care"],
        "discharge_summary": ["discharge", "admission"],
        "patient_summary": ["summary", "overview"],
        "longitudinal_timeline": ["longitudinal"],
    }

    def find_patient(self, query: str) -> tuple[str | None, str]:
        """Resolve a free-text patient reference to a patient_id on record.

        Lets a caller that cannot set custom_inputs -- the Databricks AI
        Playground, for instance -- choose a patient by talking. Accepts a full
        id, an id prefix, the ON-SYN display id, or a name appearing in the
        record text. Returns (patient_id, human-readable message).
        """
        needle = " ".join(str(query or "").split()).strip()
        # The recorded name is pulled from the report header so the roster can be
        # read by a person: an id alone tells nobody which patient is which.
        rows = self.executor.query(
            f"""
            SELECT patient_id, count(*) AS records,
                   min(document_date) AS earliest, max(document_date) AS latest,
                   max(regexp_extract(full_text,
                       'Patient\\\\n([A-Za-z][A-Za-z0-9]*[ ][A-Za-z][A-Za-z0-9]*)\\\\nSynthetic Patient',
                       1)) AS patient_name
            FROM {self.documents_table}
            GROUP BY patient_id
            ORDER BY records DESC
            """
        )
        known = [
            {
                "patient_id": str(r.get("patient_id") or ""),
                "records": int(r.get("records") or 0),
                "earliest": str(r.get("earliest") or ""),
                "latest": str(r.get("latest") or ""),
                "name": str(r.get("patient_name") or "").strip(),
            }
            for r in rows
            if str(r.get("patient_id") or "")
        ]

        def label(entry: dict[str, Any]) -> str:
            name = entry["name"] or "name not recorded"
            return f"{name} - {entry['patient_id']} ({self._display_id(entry['patient_id'])})"

        def roster() -> str:
            lines = ["Patients on record:"]
            for entry in known:
                lines.append(
                    f"- {label(entry)}: {entry['records']} records, "
                    f"{entry['earliest']} to {entry['latest']}"
                )
            lines.append("Say a name or id to switch to that patient.")
            return "\n".join(lines)

        def chosen(entry: dict[str, Any]) -> tuple[str, str]:
            return entry["patient_id"], (
                f"Active patient set to {label(entry)}. "
                f"{entry['records']} records, {entry['earliest']} to {entry['latest']}."
            )

        if not needle:
            return None, roster()

        lowered = needle.lower()
        compact = lowered.replace("-", "").replace(" ", "")
        for entry in known:  # id, exact or prefix
            pid = entry["patient_id"].lower()
            if lowered == pid or pid.startswith(lowered):
                return chosen(entry)
        for entry in known:  # ON-SYN display id
            if compact and compact in self._display_id(entry["patient_id"]).lower().replace("-", ""):
                return chosen(entry)
        for entry in known:  # recorded name, substring either way
            name = entry["name"].lower()
            if name and (lowered in name or name.startswith(lowered)):
                return chosen(entry)
        # Names carry generator suffixes ("Kayleigh718"), and people mistype them,
        # so compare against letters-only forms with a similarity cutoff.
        letters = re.sub(r"[^a-z ]", "", lowered).strip()
        candidates: dict[str, dict[str, Any]] = {}
        for entry in known:
            for token in re.sub(r"[^a-z ]", "", entry["name"].lower()).split():
                if token:
                    candidates.setdefault(token, entry)
        for token in letters.split() or [letters]:
            close = difflib.get_close_matches(token, list(candidates), n=1, cutoff=0.75)
            if close:
                return chosen(candidates[close[0]])
        # Last resort: the name may appear in the record text but not the header.
        matches = self.executor.query(
            f"""
            SELECT DISTINCT patient_id
            FROM {self.documents_table}
            WHERE lower(full_text) LIKE lower({sql_literal('%' + needle + '%')})
            LIMIT 5
            """
        )
        by_id = {entry["patient_id"]: entry for entry in known}
        found = [
            by_id[str(m.get("patient_id"))]
            for m in matches
            if str(m.get("patient_id") or "") in by_id
        ]
        if len(found) == 1:
            return chosen(found[0])
        if len(found) > 1:
            return None, (
                f"{needle!r} matches more than one patient: "
                + ", ".join(label(e) for e in found)
                + ". Name one of them.\n" + roster()
            )
        return None, f"No patient matches {needle!r}.\n" + roster()

    @staticmethod
    def _display_id(patient_id: str) -> str:
        """ON-SYN-XXXXXXXXXX, the identifier printed on the generated reports."""
        compact = str(patient_id or "").replace("-", "").upper()
        return f"ON-SYN-{compact[:10]}" if compact else ""

    def _resolve_document_type(self, value: str) -> str | None:
        """Map a requested record type onto a document_type that exists.

        Returns None for anything unrecognised. Callers must treat that as "this
        record type is not on file" and say so -- **not** as licence to drop the
        filter. Dropping it is what made "how many genome sequencing reports?"
        answer 44: the filter vanished, the query counted every document, and the
        total was reported under the name of a record type the patient never had.
        """
        needle = " ".join(str(value or "").lower().split())
        if not needle:
            return None
        if needle in self._TYPE_SYNONYMS:
            return needle
        for stored, synonyms in self._TYPE_SYNONYMS.items():
            if needle == stored or needle in synonyms:
                return stored
        for stored, synonyms in self._TYPE_SYNONYMS.items():
            if any(synonym in needle for synonym in synonyms):
                return stored
        return None

    def available_document_types(self, patient_id: str, lead: str) -> ToolResult:
        """What this patient actually has, as an answer to an under-specified ask.

        Used where a tool cannot proceed without knowing which records are meant.
        Listing the types is not a consolation prize: it usually answers the original
        question, because a record type the user asked about is visibly not in the
        list. That is the same reasoning behind the zero-result fallback in
        count_documents.
        """
        rows = self.executor.query(
            f"""
            SELECT document_type, count(*) AS records
            FROM {self.documents_table}
            WHERE patient_id = {sql_literal(patient_id)}
            GROUP BY document_type
            ORDER BY records DESC, document_type ASC
            """
        )
        entries = [
            (str(row.get("document_type") or "unknown"), int(row.get("records") or 0))
            for row in rows
            if row.get("document_type")
        ]
        if not entries:
            return ToolResult(
                name="read_patient_documents",
                output="No documents are on record for this patient.",
                source_document_ids=[],
            )
        listing = "; ".join(f"{name} ({count})" for name, count in entries)
        return ToolResult(
            name="read_patient_documents",
            output=f"{lead} {listing}.",
            source_document_ids=[],
            data={
                # The same shape count_documents emits, so the UI draws its census
                # slip -- "Records on file, 44 total" -- which is what this listing
                # actually is. The key must be by_type: countsSlip() reads by_type,
                # so a differently named key renders nothing at all and the reader
                # loses the table that answers their question.
                "kind": "counts",
                "total": sum(count for _, count in entries),
                "by_type": [{"document_type": name, "records": count}
                            for name, count in entries],
            },
        )

    def count_documents(
        self,
        patient_id: str,
        document_type: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> ToolResult:
        """Count records for a patient, exactly, from the whole table.

        Retrieval cannot answer "how many": it only ever sees the handful of
        chunks it retrieved, so it under-counts. This aggregates every row.
        """
        where = [f"patient_id = {sql_literal(patient_id)}"]
        if start_date:
            where.append(f"document_date >= DATE({sql_literal(start_date)})")
        if end_date:
            where.append(f"document_date <= DATE({sql_literal(end_date)})")

        wanted = self._resolve_document_type(document_type) if document_type else None
        if document_type and not wanted:
            # Without this the filter is dropped and the query counts everything,
            # which reported 44 as the number of "genome sequencing" reports.
            return self._unknown_type_result(
                patient_id, "count_patient_documents", str(document_type).strip())
        if wanted:
            where.append(f"document_type = {sql_literal(wanted)}")

        rows = self.executor.query(
            f"""
            SELECT document_type, count(*) AS records,
                   min(document_date) AS earliest, max(document_date) AS latest
            FROM {self.documents_table}
            WHERE {' AND '.join(where)}
            GROUP BY document_type
            ORDER BY records DESC, document_type ASC
            """
        )
        counts: list[tuple[str, int, str, str]] = []
        total = 0
        for row in rows:
            try:
                records = int(row["records"])
            except (TypeError, ValueError, KeyError):
                continue
            counts.append(
                (
                    str(row.get("document_type") or "unknown"),
                    records,
                    str(row.get("earliest") or ""),
                    str(row.get("latest") or ""),
                )
            )
            total += records

        if not counts:
            # A zero is nearly always a filter that matched nothing rather than an
            # empty chart, so answer the question that was actually asked by
            # showing what the patient does have.
            if wanted or start_date or end_date:
                fallback = self.executor.query(
                    f"""
                    SELECT document_type, count(*) AS records
                    FROM {self.documents_table}
                    WHERE patient_id = {sql_literal(patient_id)}
                    GROUP BY document_type
                    ORDER BY records DESC
                    """
                )
                available = [
                    f"{str(r.get('document_type')).replace('_', ' ')}: {r.get('records')}"
                    for r in fallback
                ]
                scope = f" of type {wanted!r}" if wanted else ""
                window = " in that date range" if (start_date or end_date) else ""
                if available:
                    total_all = sum(int(r.get("records") or 0) for r in fallback)
                    return ToolResult(
                        name="count_patient_documents",
                        output=(
                            f"This patient has no records{scope}{window}. "
                            f"Across all types they have {total_all} records: "
                            + "; ".join(available)
                            + ".\nCounts are exact totals from the record table, not a sample."
                        ),
                        source_document_ids=[],
                        data={"kind": "counts", "total": total_all, "document_type": None,
                              "earliest": "", "latest": "",
                              "by_type": [
                                  {"document_type": str(r.get("document_type")),
                                   "records": int(r.get("records") or 0),
                                   "earliest": "", "latest": ""}
                                  for r in fallback
                              ]},
                    )
            return ToolResult(
                name="count_patient_documents",
                output="This patient has no records at all.",
                source_document_ids=[],
            )

        window = ""
        earliest = min(c[2] for c in counts if c[2]) if any(c[2] for c in counts) else ""
        latest = max(c[3] for c in counts if c[3]) if any(c[3] for c in counts) else ""
        if earliest and latest:
            window = f", dated {earliest} to {latest}"

        if wanted:
            label = wanted.replace("_", " ")
            lines = [f"This patient has exactly {total} {label} record(s){window}."]
        else:
            lines = [f"This patient has exactly {total} records{window}, by type:"]
            for stored, records, first, last in counts:
                span = f" ({first} to {last})" if first and last and first != last else (
                    f" ({first})" if first else ""
                )
                lines.append(f"* {stored.replace('_', ' ')}: {records}{span}")
        lines.append("Counts are exact totals from the record table, not a sample.")
        return ToolResult(
            name="count_patient_documents",
            output="\n".join(lines),
            source_document_ids=[],
            data={
                "kind": "counts",
                "total": total,
                "document_type": wanted,
                "earliest": earliest,
                "latest": latest,
                "by_type": [
                    {
                        "document_type": stored,
                        "records": records,
                        "earliest": first,
                        "latest": last,
                    }
                    for stored, records, first, last in counts
                ],
            },
        )

    def _document_filters(
        self,
        patient_id: str,
        document_type: str | None,
        start_date: str | None,
        end_date: str | None,
    ) -> tuple[list[str], str | None, str | None]:
        """Build the WHERE clause, and report a requested type that does not exist.

        The third return value is the requested type when it could not be resolved.
        It used to be dropped: an unrecognised type simply added no filter, on the
        reasoning that a dropped filter beats silently matching no rows. That is
        backwards. "How many genome sequencing reports?" resolved to nothing, ran
        unfiltered, and answered **44** -- every document the patient has, presented
        as a count of a record type they do not have. A false zero is a wrong answer;
        a false total is a wrong answer that looks researched.

        Callers must handle the third value rather than ignore it.
        """
        where = [f"patient_id = {sql_literal(patient_id)}"]
        resolved = self._resolve_document_type(document_type) if document_type else None
        unresolved = None
        if document_type and not resolved:
            unresolved = str(document_type).strip()
        if resolved:
            where.append(f"document_type = {sql_literal(resolved)}")
        if start_date:
            where.append(f"document_date >= DATE({sql_literal(start_date)})")
        if end_date:
            where.append(f"document_date <= DATE({sql_literal(end_date)})")
        return where, resolved, unresolved

    def _unknown_type_result(self, patient_id: str, name: str,
                             requested: str) -> ToolResult:
        """Say the requested record type is not on file, and list what is."""
        result = self.available_document_types(
            patient_id,
            lead=(
                f"No record type matching {requested!r} exists for this patient, so "
                "nothing was counted or read under that name. On file:"
            ),
        )
        return ToolResult(name=name, output=result.output,
                          source_document_ids=[], data=result.data)

    def list_documents(
        self,
        patient_id: str,
        document_type: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 200,
    ) -> ToolResult:
        """Complete inventory of a patient's records: id, date, type, title.

        Retrieval can only describe the handful of chunks it retrieved, so asked
        "what documents does this patient have" it under-reports. This lists every
        matching row.
        """
        where, resolved, unresolved = self._document_filters(
            patient_id, document_type, start_date, end_date)
        if unresolved:
            return self._unknown_type_result(
                patient_id, "list_patient_documents", unresolved)
        safe_limit = min(max(int(limit), 1), 400)
        rows = self.executor.query(
            f"""
            SELECT document_id, document_date, document_type, event_name
            FROM {self.documents_table}
            WHERE {' AND '.join(where)}
            ORDER BY document_date ASC NULLS LAST, document_type ASC, document_id ASC
            LIMIT {safe_limit}
            """
        )
        if not rows:
            scope = f" of type {resolved!r}" if resolved else ""
            return ToolResult(
                name="list_patient_documents",
                output=f"This patient has no records{scope} in the requested range.",
                source_document_ids=[],
            )
        entries = [
            {
                "document_id": str(row.get("document_id") or ""),
                "document_date": str(row.get("document_date") or ""),
                "document_type": str(row.get("document_type") or ""),
                "title": str(row.get("event_name") or ""),
            }
            for row in rows
        ]
        scope = f" {resolved.replace('_', ' ')}" if resolved else ""
        lines = [f"This patient has {len(entries)}{scope} record(s), listed oldest first:"]
        for entry in entries:
            lines.append(
                f"- {entry['document_date'] or 'Unknown date'} | "
                f"{entry['document_type'].replace('_', ' ')} | {entry['title']} "
                f"[{entry['document_id']}]"
            )
        lines.append("This is the complete list from the record table, not a sample.")
        return ToolResult(
            name="list_patient_documents",
            output="\n".join(lines),
            source_document_ids=[entry["document_id"] for entry in entries],
            data={
                "kind": "document_list",
                "total": len(entries),
                "document_type": resolved,
                "documents": entries,
            },
        )

    def read_documents(
        self,
        patient_id: str,
        document_type: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        document_ids: str | None = None,
        limit: int = 3,
    ) -> ToolResult:
        """Return the recorded text of a few documents, newest first.

        Needed for questions about what a note actually says. Retrieval returns
        fragments chosen for similarity; this returns whole documents.
        """
        where, _resolved, unresolved = self._document_filters(
            patient_id, document_type, start_date, end_date)
        if unresolved:
            return self._unknown_type_result(
                patient_id, "read_patient_documents", unresolved)
        wanted_ids = [item.strip() for item in str(document_ids or "").split(",") if item.strip()]
        if wanted_ids:
            where.append(
                "document_id IN (" + ", ".join(sql_literal(i) for i in wanted_ids) + ")"
            )
        safe_limit = min(max(int(limit), 1), 5)
        rows = self.executor.query(
            f"""
            SELECT document_id, document_date, document_type, event_name, full_text
            FROM {self.documents_table}
            WHERE {' AND '.join(where)}
            ORDER BY document_date DESC NULLS LAST, document_id DESC
            LIMIT {safe_limit}
            """
        )
        if not rows:
            return ToolResult(
                name="read_patient_documents",
                output=(
                    "No matching documents were found to read. Ask for the document "
                    "list first to see what is on record."
                ),
                source_document_ids=[],
            )
        per_document = 3000
        lines: list[str] = []
        entries = []
        for row in rows:
            text = str(row.get("full_text") or "").strip()
            clipped = text[:per_document]
            entries.append(
                {
                    "document_id": str(row.get("document_id") or ""),
                    "document_date": str(row.get("document_date") or ""),
                    "document_type": str(row.get("document_type") or ""),
                    "title": str(row.get("event_name") or ""),
                    "characters": len(text),
                }
            )
            lines.append(
                f"=== {row.get('document_type')} | {row.get('document_date')} | "
                f"{row.get('event_name')} [{row.get('document_id')}] ==="
            )
            lines.append(clipped if clipped else "(This document has no recorded text.)")
            if len(text) > per_document:
                lines.append(f"... text clipped after {per_document} characters ...")
            lines.append("")
        lines.append(
            f"Returned the recorded text of {len(entries)} document(s) verbatim. "
            "Summarise only what appears above."
        )
        return ToolResult(
            name="read_patient_documents",
            output="\n".join(lines),
            source_document_ids=[entry["document_id"] for entry in entries],
            data={"kind": "documents", "total": len(entries), "documents": entries},
        )

    def detect_medication_changes(
        self,
        patient_id: str,
        earlier_date: str,
        later_date: str,
    ) -> ToolResult:
        rows = self.executor.query(
            f"""
            SELECT document_id, document_date, event_name, status
            FROM {self.documents_table}
            WHERE patient_id = {sql_literal(patient_id)}
              AND lower(document_type) IN ('medicationrequest', 'medication', 'medicationstatement')
              AND document_date <= DATE({sql_literal(later_date)})
            ORDER BY document_date ASC NULLS LAST
            LIMIT 2000
            """
        )
        before: set[str] = set()
        after: set[str] = set()
        sources: list[str] = []
        for row in rows:
            medication = str(row.get("event_name") or "").strip()
            observed = str(row.get("document_date") or "")
            status = str(row.get("status") or "").lower()
            if not medication or not observed:
                continue
            sources.append(str(row.get("document_id") or ""))
            if observed <= earlier_date and status not in self._TERMINAL_STATUSES:
                before.add(medication)
            # No date test here: the query already bounds document_date by
            # later_date, so every surviving row is on or before it.
            if status not in self._TERMINAL_STATUSES:
                after.add(medication)
            if status in self._TERMINAL_STATUSES:
                after.discard(medication)
        added = sorted(after - before)
        removed = sorted(before - after)
        unchanged = sorted(before & after)
        output = "\n".join(
            [
                f"Medication set comparison: {earlier_date} -> {later_date}",
                "Added: " + (", ".join(added) if added else "None detected"),
                "Removed: " + (", ".join(removed) if removed else "None detected"),
                "Unchanged: " + (", ".join(unchanged) if unchanged else "None detected"),
                "This result depends on the status and dates present in the source records.",
            ]
        )
        return ToolResult(
            name="detect_medication_changes",
            output=output,
            source_document_ids=list(dict.fromkeys(sources)),
        )
