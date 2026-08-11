"""Exercise every tool against the deployed endpoint, with and without arguments.

Two cases per tool, because they fail differently:

  supplied   the tool is given what it needs and must return a real result
  bare       the tool is given nothing and must either fall back to a defensible
             default (all documents, all measurements) or say what it needs

The second case is the one worth testing. A tool that quietly invents a default it
should have asked about returns something that reads like an answer -- a comparison
of two documents nobody asked for, complete with a similarity percentage.

Each tool is pinned through custom_inputs, so this measures the tools themselves
rather than the model's choice of tool.

    python3 scripts/check_tools.py [--patient <id-or-prefix>] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time

ENDPOINT = "dbdemos-rag-prod"
PATIENT = "73b97ede-ebda-6de2-36c8-c7c6cc9ad9ff"  # Buddy254 Boyle917, 44 documents

# expectation: a regex the answer must match, or a callable over the tool payload.
CASES = [
    # ---------------------------------------------------------------- supplied
    ("count_documents", "supplied",
     {"document_type": "laboratory report"},
     r"\b16\b", "counts the 16 laboratory reports"),
    ("count_documents", "bare",
     {},
     r"\b44\b", "no filter means every document, not an error"),

    ("list_documents", "supplied",
     {"document_type": "laboratory report"},
     r"laboratory report", "lists only laboratory reports"),
    ("list_documents", "bare",
     {},
     r"\b44\b|document", "no filter means the complete list"),

    ("timeline", "supplied",
     {"start_date": "2022-01-01", "end_date": "2022-12-31"},
     r"2022", "honours the date window"),
    ("timeline", "bare",
     {},
     r"201\d.*20[12]\d", "no window means the whole record"),

    ("lab_trend", "supplied",
     {"test_name": "Body Weight"},
     r"(?i)body weight|kg", "reports the named measurement"),
    ("lab_trend", "bare",
     {},
     r"(?i)measurement|analyte|trend|recorded", "no name means the overview"),

    ("read_documents", "supplied",
     {"document_type": "encounter note"},
     r"(?i)encounter", "reads the named type"),
    ("read_documents", "bare",
     {},
     r"(?i)(type or ID|specific document|which document|document type).{0,400}"
     r"(laboratory_report|laboratory report|encounter)",
     "asks which document, listing what exists"),

    ("compare_documents", "supplied",
     {"date_a": "2026-01-26", "date_b": "2026-06-25"},
     r"(?i)2026-01-26.*2026-06-25|similarity", "compares the two dates asked for"),
    ("compare_documents", "one date",
     {"date_a": "2026-01-26"},
     r"(?i)two|second date", "asks for the missing date instead of guessing"),
    ("compare_documents", "bare",
     {},
     r"(?i)which two|name two dates|document type",
     "asks which two documents instead of picking a pair"),

    ("medication_changes", "supplied",
     {"earlier_date": "2014-01-01", "later_date": "2026-12-31"},
     r"(?i)medication|started|stopped|no change", "reports the medication delta"),
    ("medication_changes", "bare",
     {},
     r"(?i)two dates", "asks for the two dates"),

    ("set_active_patient", "supplied",
     {"patient": "Kayleigh"},
     r"(?i)kayleigh", "resolves the name"),
    ("set_active_patient", "bare",
     {},
     r"(?i)patients on record|say a name", "offers the roster"),
]


# A realistic question per tool. "run the <tool> tool" is not something a user would
# ask, and the model answers it by describing the tool rather than reporting the
# records -- which fails an assertion for a reason unrelated to the tool. The tool is
# pinned through custom_inputs either way, so the wording only shapes the prose.
QUESTIONS = {
    "count_documents": "how many records of that kind are on file?",
    "list_documents": "what documents does this patient have?",
    "timeline": "build a chronological timeline of this patient's recorded events",
    "lab_trend": "what is the recorded trend?",
    "read_documents": "what do the recorded notes actually say?",
    "compare_documents": "what is the difference between those two documents?",
    "medication_changes": "which medications were started or stopped?",
    "set_active_patient": "switch to that patient",
}


def invoke(tool: str, arguments: dict, patient: str, timeout: int = 300) -> dict:
    payload = {
        "input": [{"role": "user",
                   "content": QUESTIONS.get(tool, f"run the {tool} tool")}],
        "context": {"user_id": "tool-check", "conversation_id": f"check-{tool}"},
        "custom_inputs": {"patient_id": patient, "tool": tool, **arguments},
    }
    completed = subprocess.run(
        ["databricks", "api", "post",
         f"/serving-endpoints/{ENDPOINT}/invocations", "--json", json.dumps(payload)],
        capture_output=True, text=True, timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "").strip()[:300])
    return json.loads(completed.stdout)


def answer_text(response: dict) -> str:
    parts: list[str] = []
    for item in response.get("output") or []:
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts += [b.get("text", "") for b in content
                      if isinstance(b, dict) and b.get("text")]
    return "\n".join(p for p in parts if p).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default=PATIENT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"  endpoint {ENDPOINT}   patient {args.patient[:13]}…\n")
    passed = failed = 0
    for tool, case, arguments, pattern, why in CASES:
        started = time.monotonic()
        try:
            response = invoke(tool, arguments, args.patient)
            text = answer_text(response)
            ran = (response.get("custom_outputs") or {}).get("tool") or "-"
            ok = bool(re.search(pattern, text, re.S))
        except Exception as exc:
            text, ran, ok = f"ERROR {exc}", "-", False
        seconds = time.monotonic() - started
        passed, failed = (passed + ok, failed + (not ok))
        print(f"  {'ok  ' if ok else 'FAIL'} {tool:<20} {case:<10} "
              f"{seconds:5.1f}s  ran={ran:<20} {why}")
        if args.verbose or not ok:
            print(f"        {' '.join(text.split())[:230]}")
    print(f"\n  {passed} passed, {failed} failed, {len(CASES)} cases")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
