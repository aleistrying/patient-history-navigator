"""Vercel serverless function: start the endpoint scaling up, and say whether it is up.

Mirrors the /api/warm route that ui/app.py serves locally. Model Serving scales to
zero after roughly ten minutes idle and takes about a minute to wake, so the first
question after a quiet period is slow and every later one is not. A short-timeout
probe is enough to trigger the scale-up — the wake continues server-side even after
this call gives up — so the page can poll until one probe succeeds and turn a
mysterious delay into a visible, finished step.

Returns {"status": "ready"|"warming", "seconds": n}.

The password is required, as a query parameter, because a probe spends Model
Serving compute and this URL is public. GET cannot carry the JSON body /api/ask
uses, so vercel/sync_from_ui.py appends ?password= to the page's fetch.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler

ENDPOINT = os.environ.get("PHN_ENDPOINT", "dbdemos-rag-prod")
# Long enough to succeed against a warm endpoint (4-16s), short enough that a cold
# one reports "warming" rather than holding the connection open for the full wake.
PROBE_SECONDS = int(os.environ.get("PHN_WARM_PROBE_SECONDS", "45"))

PROBE = {
    "input": [{"role": "user", "content": "ping"}],
    "context": {"user_id": "warmup", "conversation_id": "warmup"},
    "custom_inputs": {"patient_id": "demo-patient"},
}


class handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        expected = os.environ.get("PHN_PASSWORD") or ""
        if expected and (query.get("password") or [""])[0] != expected:
            self._send(401, {"error": "Wrong password."})
            return

        host = (os.environ.get("DATABRICKS_HOST") or "").rstrip("/")
        token = os.environ.get("DATABRICKS_TOKEN") or ""
        if not host or not token:
            self._send(500, {"status": "warming",
                             "error": "Missing DATABRICKS_HOST or DATABRICKS_TOKEN."})
            return

        request = urllib.request.Request(
            f"{host}/serving-endpoints/{ENDPOINT}/invocations",
            data=json.dumps(PROBE).encode("utf-8"),
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=PROBE_SECONDS):
                pass
        except Exception:
            # Includes the timeout, which is the normal cold-start outcome: the
            # probe has still done its job of starting the scale-up.
            self._send(200, {"status": "warming",
                             "seconds": round(time.monotonic() - started)})
            return
        self._send(200, {"status": "ready",
                         "seconds": round(time.monotonic() - started)})
