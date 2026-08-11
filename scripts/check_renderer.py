"""Check the UI's hand-rolled markdown renderer in a real browser engine.

ui/index.html carries its own renderMarkdown/inlineMd pair rather than a library,
because the page's premise is that it is one dependency-free file. That trade means
the renderer needs its own tests.

Citations are the fragile part: the pattern has to cover multi-reference brackets
like `[Evidence 1, Evidence 2]` and `[Evidence 1, 2, 3]`, not just `[Evidence 1]`,
and every chip has to keep the "Ev" prefix so a lone number does not read as a
stray digit next to prose.

Escaping is the case that matters most: every fragment must be HTML-escaped before
any tag is introduced, so model output cannot inject markup.

    python3 scripts/check_renderer.py
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGE = ROOT / "ui" / "index.html"

# (input, must appear in output, must NOT appear in output)
CASES: list[tuple[str, list[str], list[str]]] = [
    ("flagged as high (H) [Evidence 1].",
     ['<em class="cite">Ev 1</em>'], ["[Evidence"]),
    # Past three, chips collapse into a count: a model citing every chunk it was
    # given is not really citing, and eight chips in a row read as noise.
    ("values rose [Evidence 1, 2, 3, 4, 5].",
     ['<em class="cite">Ev 3</em>', '+2'], ["Ev 4", "Ev 5", "[Evidence"]),
    # Three still render in full, so the cap does not fire early.
    ("stable across [Evidence 1, 2, 3].",
     ['<em class="cite">Ev 3</em>'], ["+", "[Evidence"]),
    ("a 4.3% increase [Evidence 1, Evidence 2].",
     ['<em class="cite">Ev 1</em>', '<em class="cite">Ev 2</em>'], ["[Evidence"]),
    ("values rose [Evidence 1, 2, 3].",
     ['Ev 1', 'Ev 2', 'Ev 3'], ["[Evidence"]),
    ("see [evidence 4] lowercase", ['Ev 4'], ["[evidence"]),
    ("a bracket that is not a citation [see note] stays",
     ["[see note]"], ['class="cite"']),
    ("no citation here at all", ["no citation here at all"], ['class="cite"']),
    ("**bold** and *italic* and `code`",
     ["<strong>bold</strong>", "<em>italic</em>", "<code>code</code>"], []),
    ("# A heading", ["<h3", "A heading"], []),
    ("- one\n- two", ["<ul>", "<li>one</li>", "<li>two</li>"], []),
    ("1. first\n2. second", ["<ol>", "first"], []),
    # Escaping: the payloads must survive as text, never as markup.
    ("<script>alert(1)</script>", ["&lt;script&gt;"], ["<script>"]),
    ('<img src=x onerror="alert(1)">', ["&lt;img"], ["<img"]),
    ("[Evidence 1]<script>bad()</script>",
     ['<em class="cite">Ev 1</em>', "&lt;script&gt;"], ["<script>"]),
    ("a & b < c > d", ["&amp;", "&lt;", "&gt;"], []),
]


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed: pip install playwright && "
              "playwright install chromium", file=sys.stderr)
        return 2

    failures = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        # A file:// origin, because the script touches sessionStorage on load and
        # that throws on the opaque origin set_content gives you.
        page.goto(PAGE.resolve().as_uri(), wait_until="domcontentloaded")
        page.wait_for_timeout(600)

        for source, expected, forbidden in CASES:
            rendered = page.evaluate("(s) => renderMarkdown(s)", source)
            missing = [token for token in expected if token not in rendered]
            leaked = [token for token in forbidden if token in rendered]
            ok = not missing and not leaked
            failures += not ok
            label = " ".join(source.split())[:52]
            print(f"  {'ok  ' if ok else 'FAIL'} {label}")
            if not ok:
                if missing:
                    print(f"         missing: {missing}")
                if leaked:
                    print(f"         LEAKED:  {leaked}")
                print(f"         got: {re.sub(r'\\s+', ' ', rendered)[:180]}")
        browser.close()

    print(f"\n  {len(CASES) - failures} passed, {failures} failed, "
          f"{len(CASES)} cases")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
