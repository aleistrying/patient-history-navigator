"""Drive the local UI through many questions in a real browser.

check_behaviour.py asks the endpoint directly and check_renderer.py exercises the
markdown renderer against fixed strings. Neither loads the page and types into it,
so neither catches a question that answers correctly over HTTP but renders as an
error, a blank turn, or a result slip that never appears.

This starts `ui/app.py`, opens the page in Chromium, and works through a question
list one turn at a time: it records the answer's length, which tool ran, whether a
structured result slip rendered, whether evidence was offered, and how long the
turn took. A question fails if the page shows the error notice, produces no answer
text, or raises in the browser.

Some questions are follow-ups that depend on the turn before them, which is what
exercises session memory through the UI rather than through a scripted history.

    python3 scripts/check_ui_queries.py [--patient 1] [--limit N] [--headed]

Exits non-zero if any question fails.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "ui" / "app.py"
PORT = 8791  # not the app's default, so a UI you already have running is left alone
BASE = f"http://127.0.0.1:{PORT}"

# (question, follow_up_of_previous, what it should exercise)
QUESTIONS: list[tuple[str, bool, str]] = [
    # --- retrieval: content questions, which must not call a tool -------------
    ("what allergies are recorded?", False, "retrieval"),
    ("what do the most recent lab results show?", False, "retrieval"),
    ("does this patient smoke?", False, "retrieval"),
    ("summarise this patient's history", False, "retrieval"),
    ("what medications is this patient on?", False, "retrieval"),

    # --- counting and listing: must be exact, so must call a tool ------------
    ("how many documents does this patient have?", False, "count"),
    ("how many laboratory reports are on file?", False, "count"),
    ("what documents does this patient have?", False, "list"),
    ("list the imaging reports", False, "list"),

    # --- timeline -----------------------------------------------------------
    ("build a timeline of recorded events", False, "timeline"),
    ("show me the encounters in date order", False, "timeline"),

    # --- numeric trends: the slip should render ------------------------------
    ("what is the trend in body weight?", False, "trend"),
    ("show me the sodium values over time", False, "trend"),
    ("what is the trend across all recorded measurements?", False, "trend overview"),

    # --- reading a named document type --------------------------------------
    ("read the encounter notes", False, "read"),

    # --- follow-ups: these only make sense with memory ----------------------
    ("and the heart rate?", True, "memory: follow-up trend"),
    ("how many of those are from 2026?", True, "memory: follow-up count"),

    # --- edge cases ---------------------------------------------------------
    ("what is your diagnosis for this patient?", False, "edge: must refuse"),
    ("what were the genome sequencing results?", False, "edge: absent record type"),
    ("how many unicorn vaccination records are there?", False, "edge: invented type"),
    # Comparison, four ways. Lab reports and encounter notes parse into labelled
    # fields; imaging reports and discharge summaries are prose and take the text
    # path. The last two must refuse rather than pick a pair nobody asked about.
    ("compare the two most recent laboratory reports", False, "compare: field-level"),
    ("what changed between the two most recent encounter notes?", False,
     "compare: field-level"),
    ("compare the two most recent imaging reports", False, "compare: prose fallback"),
    ("compare two documents", False, "compare: must ask which two"),
    ("what medications changed between 2014 and 2026?", False, "medication changes"),
    ("", False, "edge: empty input must be ignored"),
    ("tell me about patient 73b97ede-ebda-6de2-36c8-c7c6cc9ad9ff", False,
     "edge: another patient by full id"),
]


def wait_for_health(process: subprocess.Popen, timeout: float = 40.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{BASE}/healthz", timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.4)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", default="1",
                        help='roster position, 1-based, or "all" for every patient')
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions")
    parser.add_argument("--headed", action="store_true", help="show the browser")
    parser.add_argument("--first-timeout", type=int, default=300,
                        help="seconds for the first answer, which may wake the endpoint")
    parser.add_argument("--timeout", type=int, default=180,
                        help="seconds for every later answer")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed: pip install playwright && "
              "playwright install chromium", file=sys.stderr)
        return 2

    questions = QUESTIONS[: args.limit] if args.limit else QUESTIONS

    process = subprocess.Popen(
        [sys.executable, str(APP)],
        env={**__import__("os").environ, "PORT": str(PORT), "PHN_VERBOSE": "0"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if not wait_for_health(process):
        process.kill()
        stderr = (process.stderr.read() if process.stderr else "")[-600:]
        print(f"  the UI did not start on {BASE}\n{stderr}", file=sys.stderr)
        return 2

    rows: list[tuple[str, str, str, int, str, float, str]] = []
    targets: list[int] = []
    failures: list[str] = []
    page_errors: list[str] = []

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not args.headed)
            page = browser.new_page()
            page.on("pageerror", lambda e: page_errors.append(str(e)[:160]))

            page.goto(BASE, wait_until="domcontentloaded")
            page.wait_for_selector("#roster > li", timeout=30000)
            total_cards = page.locator("#roster .record").count()
            if total_cards == 0:
                failures.append("the patient roster is empty")
                targets: list[int] = []
            elif str(args.patient).strip().lower() == "all":
                targets = list(range(total_cards))
            else:
                targets = [min(max(int(args.patient), 1), total_cards) - 1]

            first_question_overall = True
            for index in targets:
                # "New session" returns to the picker and rotates the session id, so
                # each patient is asked in its own conversation and no follow-up can
                # inherit the previous patient's context.
                if not first_question_overall:
                    page.click("#restart")
                    page.wait_for_selector("#selectView", state="visible", timeout=10000)
                page.locator("#roster .record").nth(index).click()
                page.wait_for_selector("#chatView", state="visible", timeout=10000)
                who = (page.locator("#stripName").first.text_content() or "").strip() \
                    or f"patient {index + 1}"
                if len(targets) > 1:
                    print(f"\n  --- {who} ---")
                rows.append((f"### {who}", "", "", 0, "", 0.0, ""))

                for number, (question, _follow_up, purpose) in enumerate(questions, 1):
                    before = page.locator("#thread .turn").count()
                    page.fill("#entry", question)
                    page.click("#send")

                    if not question.strip():
                        # An empty question must not produce a turn at all.
                        page.wait_for_timeout(1200)
                        after = page.locator("#thread .turn").count()
                        ok = after == before
                        first_question_overall = False
                        rows.append((question or "(empty)", purpose, "-", 0, "-", 0.0, "-"))
                        if not ok:
                            failures.append(f"{number}. empty input produced a turn")
                        continue

                    started = time.monotonic()
                    limit = (args.first_timeout if first_question_overall
                             else args.timeout) * 1000
                    first_question_overall = False
                    try:
                        page.wait_for_function(
                            """(n) => {
                                const turns = document.querySelectorAll('#thread .turn');
                                if (turns.length < n + 2) return false;
                                const last = turns[turns.length - 1];
                                return !!last.querySelector('.reply, .notice');
                            }""",
                            arg=before,
                            timeout=limit,
                        )
                    except Exception:
                        elapsed = time.monotonic() - started
                        rows.append((question, purpose, "TIMEOUT", 0, "-", elapsed, "-"))
                        failures.append(f"{number}. no answer within {limit / 1000:.0f}s: {question!r}")
                        continue
                    elapsed = time.monotonic() - started

                    last = page.locator("#thread .turn").last
                    notice = last.locator(".notice")
                    if notice.count():
                        rows.append((question, purpose, "ERROR", 0, "-", elapsed, "-"))
                        failures.append(
                            f"{number}. error notice: {notice.inner_text().strip()[:90]!r}")
                        continue

                    answer = last.locator(".reply").inner_text().strip()
                    # text_content, not inner_text: the label is styled
                    # text-transform: uppercase, and inner_text returns the rendered
                    # text, so the tool name would come back shouting and the match
                    # below would miss it.
                    label = (last.locator(".who-label").first.text_content() or "").strip()
                    ran = re.search(r"\bran\s+(\S+)", label, re.IGNORECASE)
                    tool = ran.group(1) if ran else "none"
                    slip = "yes" if last.locator(".slip").count() else "-"
                    evidence = "yes" if last.locator("details.evidence").count() else "-"
                    rows.append((question, purpose, tool, len(answer.split()), slip,
                                 elapsed, evidence))
                    if not answer:
                        failures.append(f"{number}. empty answer: {question!r}")

            browser.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    print(f"\n  {'question':<52}{'tool':<22}{'words':>6}{'slip':>6}{'evid':>6}{'secs':>7}")
    print(f"  {'-' * 98}")
    for question, _purpose, tool, words, slip, elapsed, evidence in rows:
        if question.startswith("### "):
            print(f"\n  {question[4:]}")
            continue
        shown = question if len(question) <= 50 else question[:47] + "..."
        print(f"  {shown:<52}{tool:<22}{words:>6}{slip:>6}{evidence:>6}{elapsed:>7.1f}")

    answered = [r for r in rows
                if r[2] not in ("TIMEOUT", "ERROR", "-", "") and not r[0].startswith("### ")]
    if answered:
        times = sorted(r[5] for r in answered)
        words = [r[3] for r in answered]
        print(f"\n  {len(answered)} answered · median {times[len(times) // 2]:.1f}s · "
              f"slowest {times[-1]:.1f}s · mean {sum(words) / len(words):.0f} words")
        print(f"  tools used: {', '.join(sorted({r[2] for r in answered}))}")

    for error in page_errors:
        failures.append(f"browser raised: {error}")

    if failures:
        print(f"\n  {len(failures)} failure(s):")
        for line in failures:
            print(f"    FAIL {line}")
        return 1
    print(f"\n  ok  {len(answered)} answered across {max(1, len(targets))} patient(s), no errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
