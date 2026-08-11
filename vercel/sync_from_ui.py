"""Regenerate vercel/index.html from the canonical ui/index.html.

The two pages are the same application; the Vercel copy only adds a password gate,
because its URL is public. Keeping that difference in a script rather than in two
hand-edited files means the local UI stays canonical and the deployed copy cannot
silently drift from it.

    python3 vercel/sync_from_ui.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "ui" / "index.html"
TARGET = ROOT / "vercel" / "index.html"

GATE_MARKUP = """<!-- PASSWORD GATE (Vercel deployment only) -->
<main class="select-view" id="gateView">
  <header class="masthead">
    <h1>Patient History Navigator</h1>
    <p>This demonstration is password protected. Enter the password you were given
       to open a synthetic patient record.</p>
  </header>
  <div style="max-width:380px">
    <p class="field" style="margin:0 0 8px">Password</p>
    <div class="entry">
      <input id="gatePass" type="password" autocomplete="current-password"
             style="flex:1;font-family:inherit;font-size:15px;color:var(--ink);
                    background:var(--surface);border:1px solid var(--rule);
                    border-radius:var(--radius);padding:11px 13px">
      <button class="send" id="gateBtn">Open</button>
    </div>
    <p class="note" id="gateMsg" style="margin-top:10px;color:var(--flag-high)"></p>
  </div>
  <p class="stamp field">Synthetic demonstration data — not for clinical use</p>
</main>

"""

GATE_SCRIPT = """
/* ---------- password gate (Vercel deployment) ---------- */
async function unlock(){
  const value = $('gatePass').value.trim();
  if(!value){ $('gateMsg').textContent = 'Enter the password.'; return; }
  $('gateBtn').disabled = true;
  $('gateMsg').textContent = 'Checking…';
  try{
    // validate_only checks the password and returns immediately. Sending a real
    // question here made the gate inherit the endpoint's cold start, so opening the
    // page could sit on "Checking..." for a minute before letting anyone in.
    const res = await fetch('/api/ask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ validate_only: true, password: value }),
    });
    if(res.status === 401){ $('gateMsg').textContent = 'Wrong password.'; return; }
    sitePassword = value;
    sessionStorage.setItem('phn-pass', value);
    $('gateView').hidden = true; $('selectView').hidden = false;
    boot();
  }catch{
    $('gateMsg').textContent = 'Could not reach the server. Try again.';
  }finally{
    $('gateBtn').disabled = false;
  }
}
$('gateBtn').addEventListener('click', unlock);
$('gatePass').addEventListener('keydown', (e) => { if(e.key === 'Enter') unlock(); });
if(sitePassword){ $('gateView').hidden = true; $('selectView').hidden = false; boot(); }
"""


class AnchorMissing(RuntimeError):
    """An anchor this script edits is no longer present in ui/index.html."""


def must_replace(text: str, anchor: str, replacement: str, what: str) -> str:
    """Replace exactly one anchor, or fail loudly.

    str.replace on a missing anchor silently returns the text unchanged, which is
    how a broken page got deployed: the declaration of `sitePassword` was anchored
    to the `let sessionId = ...` line, that line was edited for session
    persistence, the anchor stopped matching, and the gate threw
    "sitePassword is not defined" on load. The structural checks below still passed,
    because they tested that the password was *used* and never that it was
    *declared*. Every edit is now verified at the point it is made.
    """
    if anchor not in text:
        raise AnchorMissing(
            f"{what}: anchor not found in ui/index.html.\n  looked for: {anchor[:90]}"
        )
    return text.replace(anchor, replacement, 1)


def main() -> int:
    src = SOURCE.read_text()

    if "gateView" in src:
        print("ui/index.html already contains the gate; nothing to sync.", file=sys.stderr)
        return 1

    out = must_replace(src, "<!-- STEP 1 -->", GATE_MARKUP + "<!-- STEP 1 -->",
                       "gate markup insertion")
    out = must_replace(
        out,
        '<main class="select-view" id="selectView">',
        '<main class="select-view" id="selectView" hidden>',
        "hide the picker until the gate opens",
    )
    # Anchored on the 'use strict' preamble rather than any particular variable
    # line, because variable lines get edited and this declaration must not be
    # coupled to one.
    out = must_replace(
        out,
        "const $ = (id) => document.getElementById(id);",
        "const $ = (id) => document.getElementById(id);\n"
        "let sitePassword = sessionStorage.getItem('phn-pass') || '';",
        "declare sitePassword",
    )
    # Every request carries the password; the function checks it before any work.
    out = must_replace(
        out,
        "        session_id: sessionId, user_id: 'ui-user',",
        "        session_id: sessionId, user_id: 'ui-user',\n        password: sitePassword,",
        "send the password with each question",
    )
    # The warm probe is a GET, so it cannot carry the JSON body /api/ask uses. It
    # still spends Model Serving compute, so it takes the password as a query
    # parameter instead.
    out = must_replace(
        out,
        "await (await fetch('/api/warm')).json()",
        "await (await fetch('/api/warm?password=' + "
        "encodeURIComponent(sitePassword))).json()",
        "authenticate the warm probe",
    )
    # boot() is called by the gate instead of on load.
    out = must_replace(out, "\nboot();", GATE_SCRIPT, "defer boot() to the gate")

    checks = {
        "gate markup": "gateView" in out,
        "selection hidden": 'id="selectView" hidden' in out,
        # Declared *and* used. Checking only the use is what let a page deploy
        # where sitePassword was referenced but never declared, so the gate threw
        # "sitePassword is not defined" and nothing on the page could be clicked.
        "password declared": "let sitePassword =" in out,
        "password sent": "password: sitePassword" in out,
        "gate handler": "async function unlock" in out,
        "boot deferred": out.count("boot();") >= 2,
        "wake bar carried over": "warmBar" in out,
        # Without this the probe hits /api/warm unauthenticated and 401s, and the
        # bar polls forever without ever reporting ready.
        "warm probe authenticated": "/api/warm?password=" in out,
        # Nothing may reference an identifier the gate script never defines.
        "no undeclared gate identifier":
            out.count("sitePassword") >= 3 and "let sitePassword =" in out,
    }
    for name, ok in checks.items():
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if not all(checks.values()):
        print("Sync aborted: the source page changed shape.", file=sys.stderr)
        return 1

    TARGET.write_text(out)
    print(f"\nWrote {TARGET.relative_to(ROOT)} ({len(out):,} chars)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnchorMissing as error:
        print(f"Sync aborted. {error}", file=sys.stderr)
        print("ui/index.html was edited in a way this script does not account for. "
              "Update the anchor above rather than hand-editing vercel/index.html.",
              file=sys.stderr)
        raise SystemExit(1) from None
