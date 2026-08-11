"""Smoke-test the deployed page in a real browser. Run after every `vercel --prod`.

sync_from_ui.py checks the generated HTML, and the API routes can be curled, and
both of those passed while the deployed page was completely unusable: a JavaScript
identifier the gate needed was never declared, so the page threw on load and nothing
could be clicked. No amount of string-checking finds that. Loading the page and
clicking does.

    python3 vercel/verify_deployment.py [--url URL] [--password demo]

Exits non-zero if any step fails, so it can gate a release.
"""

from __future__ import annotations

import argparse
import sys

DEFAULT_URL = "https://patient-history-navigator.vercel.app"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--password", default="demo")
    parser.add_argument("--ask", action="store_true",
                        help="also send a real question (slow: wakes the endpoint)")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed: pip install playwright && "
              "playwright install chromium", file=sys.stderr)
        return 2

    failures: list[str] = []
    console: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.on("pageerror", lambda e: console.append(f"pageerror: {str(e)[:180]}"))
        page.on("console", lambda m: console.append(f"console.{m.type}: {m.text[:180]}")
                if m.type == "error" else None)

        page.goto(args.url, wait_until="domcontentloaded")

        # 1. The gate renders.
        if page.locator("#gatePass").count() == 0:
            failures.append("the password field is not on the page")
        else:
            page.fill("#gatePass", args.password)
            page.click("#gateBtn")
            page.wait_for_timeout(9000)

            # 2. The gate opens. A JavaScript error on load shows up here first.
            if not page.locator("#gateView").is_hidden():
                message = page.locator("#gateMsg").inner_text().strip()
                failures.append(f"the gate did not open (message: {message!r})")

            # 3. The roster renders and is clickable. This is the step that catches
            #    a page that loaded but whose script died.
            cards = page.locator("#roster > li").count()
            if cards == 0:
                failures.append("the patient roster is empty")
            else:
                try:
                    page.locator("#roster .record").first.click(timeout=6000)
                    page.wait_for_timeout(1500)
                    if not page.locator("#chatView").is_visible():
                        failures.append("clicking a patient did not open the record")
                except Exception as exc:
                    failures.append(f"a patient card is not clickable: "
                                    f"{str(exc).splitlines()[0][:110]}")

            # 4. The session survives a reload, which is what makes memory work.
            before = page.evaluate("sessionStorage.getItem('phn-session')")
            page.reload(wait_until="domcontentloaded")
            page.wait_for_timeout(7000)
            after = page.evaluate("sessionStorage.getItem('phn-session')")
            if not after or after != before:
                failures.append(f"the session id did not survive a reload "
                                f"({before} -> {after})")
            if page.locator("#roster > li").count() == 0:
                failures.append("the roster is empty after a reload")

            if args.ask:
                page.locator("#roster .record").first.click(timeout=8000)
                page.fill("#entry", "how many lab records does he have?")
                page.click("#send")
                try:
                    page.wait_for_selector("text=/ran count_documents/i", timeout=240000)
                except Exception:
                    failures.append("a real question did not produce a tool answer")

        browser.close()

    fatal = [line for line in console if line.startswith("pageerror")]
    for line in fatal:
        failures.append(line)

    if failures:
        print(f"  {len(failures)} problem(s):")
        for line in failures:
            print(f"    FAIL {line}")
        return 1
    print(f"  ok  {args.url} loads, unlocks, lists patients, opens a record, "
          "and keeps its session across a reload")
    if console:
        print(f"  ({len(console)} non-fatal console messages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
