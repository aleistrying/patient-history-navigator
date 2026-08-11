"""Check the questions a demo actually asks, against the deployed endpoint.

Two sets, and they are the ones that carry the most risk on stage:

* the fifteen one-click openers in `ui/app.py`'s roster -- three per patient, sitting
  as buttons under the entry box, so they are the likeliest thing to be pressed;
* the scripted questions in `docs/DEMO_RUNBOOK.md`, including the ones whose whole
  point is that they refuse or report an absence.

It calls the endpoint directly rather than driving a browser, so a full pass takes
a couple of minutes. `check_ui_queries.py` covers the rendering; this covers whether
the answers are there at all, which tool served each one, and how long the audience
would be watching a spinner.

    python3 scripts/check_demo.py [--patient <id-or-prefix>] [--verbose]

Exits non-zero if a question returns nothing, errors, or takes longer than the
slowest answer a demo can absorb.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENDPOINT = "dbdemos-rag-prod"
SLOW_SECONDS = 40.0

# The runbook's scripted moments. `expect` is a pattern the answer must match; None
# means any non-empty answer passes. `tool` is the tool that should serve it, or
# "none" for the retrieval path, or None when either is defensible.
BUDDY = "73b97ede-ebda-6de2-36c8-c7c6cc9ad9ff"
SCRIPTED: list[tuple[str, str, str | None, str | None, str]] = [
    (BUDDY, "how many laboratory reports does he have?", "count_documents",
     r"\b16\b", "the exact count the runbook promises"),
    (BUDDY, "what is the difference between the last two lab reports", "compare_documents",
     None, "the field-level comparison"),
    (BUDDY, "what is the recorded trend in heart rate?", "lab_trend",
     None, "runbook says ask heart rate, not head circumference"),
    (BUDDY, "what medication do you recommend for this patient?", None,
     r"(?i)can'?t answer|cannot|not able|does not|no recommend",
     "a treatment recommendation must be refused"),
    (BUDDY, "what were this patient's genome sequencing results?", None,
     r"(?i)\b0\b|\bno\b|not |absent|does not|unavailable",
     "an absent record type must be reported as absent"),
]


def ask(patient_id: str, question: str, timeout: float = 300.0) -> dict:
    payload = {
        "input": [{"role": "user", "content": question}],
        "custom_inputs": {"patient_id": patient_id},
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
    for item in response.get("output") or []:
        for block in item.get("content") or []:
            if block.get("text"):
                return str(block["text"])
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", help="only this patient id or prefix")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    openers = json.loads(
        (ROOT / "scripts" / "demo_openers.json").read_text()
    ) if (ROOT / "scripts" / "demo_openers.json").exists() else []
    if not openers:
        print("  scripts/demo_openers.json is missing; run with the roster exported",
              file=sys.stderr)
        return 2

    cases: list[tuple[str, str, str | None, str | None, str]] = [
        (o["patient_id"], o["question"], None, None, f"opener · {o['name']}")
        for o in openers
    ] + SCRIPTED
    if args.patient:
        cases = [c for c in cases if c[0].startswith(args.patient)]

    failures: list[str] = []
    rows: list[tuple[str, str, int, float, bool]] = []

    print(f"\n  {len(cases)} demo questions against {ENDPOINT}\n")
    for number, (patient_id, question, want_tool, expect, why) in enumerate(cases, 1):
        started = time.monotonic()
        try:
            response = ask(patient_id, question)
        except Exception as exc:
            rows.append((question, "ERROR", 0, time.monotonic() - started, False))
            failures.append(f"{number}. {question!r} -> {exc}")
            continue
        elapsed = time.monotonic() - started
        text = answer_text(response)
        ran = (response.get("custom_outputs") or {}).get("tool") or "none"
        has_slip = bool((response.get("custom_outputs") or {}).get("tool_data"))

        ok = bool(text.strip())
        if not ok:
            failures.append(f"{number}. empty answer: {question!r}")
        if want_tool and ran != want_tool:
            ok = False
            failures.append(
                f"{number}. {why}: expected {want_tool}, ran {ran} -- {question!r}")
        if expect and not re.search(expect, text, re.S):
            ok = False
            failures.append(f"{number}. {why}: answer did not match {expect!r}\n"
                            f"      {text[:160]!r}")
        if elapsed > SLOW_SECONDS:
            failures.append(f"{number}. {elapsed:.0f}s is longer than a demo can hold: "
                            f"{question!r}")
        rows.append((question, ran, len(text.split()), elapsed, has_slip))
        mark = "ok  " if ok else "FAIL"
        print(f"  {mark} {elapsed:>5.1f}s {ran:<20}{len(text.split()):>4}w "
              f"{'slip' if has_slip else '    '}  {question[:44]}")
        if args.verbose:
            print(f"         {text[:200]}")

    answered = [r for r in rows if r[1] != "ERROR"]
    if answered:
        times = sorted(r[3] for r in answered)
        print(f"\n  {len(answered)}/{len(rows)} answered · median {times[len(times)//2]:.1f}s"
              f" · slowest {times[-1]:.1f}s"
              f" · mean {sum(r[2] for r in answered)/len(answered):.0f} words"
              f" · {sum(1 for r in answered if r[4])} with a result slip")
    if failures:
        print(f"\n  {len(failures)} problem(s):")
        for line in failures:
            print(f"    {line}")
        return 1
    print("\n  ok  every demo question answered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
