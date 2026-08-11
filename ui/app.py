"""Local web UI for the Patient History Navigator serving endpoint.

Runs two ways from one codebase:

  Local        python3 ui/app.py            -> http://127.0.0.1:8765
               Authenticates with your Databricks CLI credentials.

  Databricks   deployed as a Databricks App
  Apps         Authenticates with the app's own service principal, which Apps
               injects as DATABRICKS_CLIENT_ID / DATABRICKS_CLIENT_SECRET.

Either way the browser never receives a token: the page talks only to this
process, and this process signs the request to Model Serving.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENDPOINT = os.environ.get("PHN_ENDPOINT", "dbdemos-rag-prod")
PORT = int(os.environ.get("PORT", "8765"))
# Long enough to succeed when the endpoint is already warm, short enough that a
# cold endpoint returns "warming" rather than blocking the page. A warm answer
# takes 4-16s and a cold one over 55s, so 45s separates the two cleanly.
WARM_PROBE_SECONDS = int(os.environ.get("PHN_WARM_PROBE_SECONDS", "45"))

# The five ingested patients. Ingestion covers the patient_7* subset only, so this
# list is fixed rather than discovered -- it avoids waking the SQL warehouse just
# to draw the picker. Counts come from patient_history_documents and
# patient_history_lab_observations.
PATIENTS = [
    {
        "id": "73b97ede-ebda-6de2-36c8-c7c6cc9ad9ff",
        "specimen": "ON-SYN-73B97EDEEB",
        "name": "Buddy254 Boyle917",
        "sex": "M",
        "born": "1985-08-31",
        "documents": 44,
        "span": "2014 – 2026",
        "measurements": 77,
        "note": "Widest record span. BMI rises 27.5 to 29.9.",
        "openers": [
            {"label": "Laboratory results", "text": "What laboratory results are recorded for this patient?"},
            {"label": "Record timeline", "text": "Build a chronological timeline of this patient's recorded events."},
            {"label": "Body weight trend", "text": "What is the recorded trend in body weight?"},
        ],
    },
    {
        "id": "7c6f0701-02e7-6c85-1f82-c0f615d02bb4",
        "specimen": "ON-SYN-7C6F070102",
        "name": "Joleen561 Witting912",
        "sex": "F",
        "born": "1978-08-19",
        "documents": 52,
        "span": "2016 – 2026",
        "measurements": 81,
        "note": "Most measurements on file, 36 distinct analytes.",
        "openers": [
            {"label": "Laboratory results", "text": "What laboratory results are recorded for this patient?"},
            {"label": "Record timeline", "text": "Build a chronological timeline of this patient's recorded events."},
            {"label": "Heart rate trend", "text": "What is the recorded trend in heart rate?"},
        ],
    },
    {
        "id": "7dd90132-76f1-9559-8d81-e732c8633880",
        "specimen": "ON-SYN-7DD9013276",
        "name": "Kayleigh718 Schimmel440",
        "sex": "F",
        "born": "2025-08-21",
        "documents": 29,
        "span": "2025 – 2026",
        "measurements": 73,
        "note": "Infant record. Growth series across six visits.",
        "openers": [
            {"label": "Body weight trend", "text": "What is the recorded trend in body weight?"},
            {"label": "Head circumference", "text": "What is the recorded trend in head occipital-frontal circumference?"},
            {"label": "Record timeline", "text": "Build a chronological timeline of this patient's recorded events."},
        ],
    },
    {
        "id": "7c50e28d-07cc-a4cf-eb58-b2a15bc552a3",
        "specimen": "ON-SYN-7C50E28D07",
        "name": "Margot418 Strosin214",
        "sex": "F",
        "born": "1960-03-14",
        "documents": 46,
        "span": "2010 – 2014",
        "measurements": 69,
        "note": "Earliest records. Six-point pain score series.",
        "openers": [
            {"label": "Laboratory results", "text": "What laboratory results are recorded for this patient?"},
            {"label": "Pain score trend", "text": "What is the recorded trend in pain severity score?"},
            {"label": "Record timeline", "text": "Build a chronological timeline of this patient's recorded events."},
        ],
    },
    {
        "id": "7ad140ab-bfae-c3ae-20a3-2244b1c4d0e2",
        "specimen": "ON-SYN-7AD140ABBF",
        "name": "Guadalupe206 Valencia279",
        "sex": "F",
        "born": "1954-04-26",
        "documents": 54,
        "span": "2021 – 2024",
        "measurements": 37,
        "note": "Eye care record. Intraocular pressure series.",
        "openers": [
            {"label": "Laboratory results", "text": "What laboratory results are recorded for this patient?"},
            {"label": "Eye pressure trend", "text": "What is the recorded trend in right eye intraocular pressure?"},
            {"label": "Record timeline", "text": "Build a chronological timeline of this patient's recorded events."},
        ],
    },
]


def _invoke_via_sdk(payload: dict, timeout: int = 300) -> dict | None:
    """Preferred path. Works locally and inside Databricks Apps."""
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.core import Config
    except Exception:
        return None
    # The endpoint scales to zero and the agent makes its own LLM calls, so a cold
    # request can run well past the default client timeout. The warm probe passes a
    # shorter one so it can report "warming" instead of blocking the page.
    #
    # The timeout goes through Config rather than a WorkspaceClient keyword.
    # WorkspaceClient does not accept http_timeout_seconds -- passing it raises
    # TypeError, and because every call here is wrapped, that surfaces in the page
    # as a failed question rather than as a stack trace.
    client = WorkspaceClient(config=Config(http_timeout_seconds=timeout))
    return client.api_client.do(
        "POST", f"/serving-endpoints/{ENDPOINT}/invocations", body=payload
    )


def _invoke_via_cli(payload: dict, timeout: int = 300) -> dict:
    """Fallback for a local machine that has the CLI but not the Python SDK."""
    completed = subprocess.run(
        [
            "databricks",
            "api",
            "post",
            f"/serving-endpoints/{ENDPOINT}/invocations",
            "--json",
            json.dumps(payload),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "").strip()[:500])
    return json.loads(completed.stdout)


def _is_wakeup_timeout(error: Exception) -> bool:
    text = str(error).lower()
    return "timed out" in text or "timeout" in text or "deadline" in text


def invoke(payload: dict, attempts: int = 2, timeout: int = 300) -> dict:
    """Call the endpoint, retrying once past a cold-start timeout.

    The CLI gives up after 60s of inactivity and its timeout is not settable by
    environment variable, so a scaled-to-zero endpoint reliably fails the first
    call. That first call is what wakes it, so one retry succeeds.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            result = _invoke_via_sdk(payload, timeout=timeout)
            if result is None:
                result = _invoke_via_cli(payload, timeout=timeout)
            return result
        except Exception as exc:
            last = exc
            if attempt + 1 >= attempts or not _is_wakeup_timeout(exc):
                raise
            print("Endpoint was cold; retrying once.")
    raise last if last else RuntimeError("The endpoint could not be reached.")


def answer_text(response: dict) -> str:
    output = response.get("output")
    if isinstance(output, str):
        return output
    parts: list[str] = []
    for item in output or []:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts += [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("text")
            ]
    return "\n".join(part for part in parts if part).strip()


class Handler(BaseHTTPRequestHandler):
    server_version = "PatientHistoryNavigatorUI"

    def log_message(self, fmt: str, *args) -> None:  # quieter console
        if os.environ.get("PHN_VERBOSE"):
            super().log_message(fmt, *args)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _warm(self) -> None:
        """Start the endpoint scaling up and report whether it is ready yet.

        Model Serving scales to zero after roughly ten minutes idle and takes about
        a minute to wake. A short-timeout probe is enough to trigger the scale-up:
        the wake continues server-side even when this call gives up, so the page can
        poll until a probe succeeds.
        """
        payload = {
            "input": [{"role": "user", "content": "ping"}],
            "context": {"user_id": "warmup", "conversation_id": "warmup"},
            "custom_inputs": {"patient_id": "demo-patient"},
        }
        started = time.monotonic()
        try:
            invoke(payload, attempts=1, timeout=WARM_PROBE_SECONDS)
        except Exception:
            self._send_json(200, {"status": "warming",
                                  "seconds": round(time.monotonic() - started)})
            return
        self._send_json(200, {"status": "ready",
                              "seconds": round(time.monotonic() - started)})

    def do_GET(self) -> None:
        route = self.path.split("?", 1)[0]
        if route == "/api/warm":
            self._warm()
            return
        if route in ("/", "/index.html"):
            page = (ROOT / "index.html").read_bytes()
            self._send(200, page, "text/html; charset=utf-8")
        elif route == "/api/patients":
            self._send_json(200, {"patients": PATIENTS, "endpoint": ENDPOINT})
        elif route == "/healthz":
            self._send_json(200, {"ok": True})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/ask":
            self._send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            request = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._send_json(400, {"error": "Could not read the request."})
            return

        question = str(request.get("question") or "").strip()
        patient_id = str(request.get("patient_id") or "").strip()
        if not question or not patient_id:
            self._send_json(400, {"error": "A patient and a question are both required."})
            return

        custom: dict[str, object] = {"patient_id": patient_id}
        # The agent routes tools itself; these only pass through when the caller
        # deliberately pins one (the UI does not).
        if request.get("tool"):
            custom["tool"] = request["tool"]
        for key, value in (request.get("custom") or {}).items():
            custom[key] = value

        payload = {
            "input": request.get("history") or [{"role": "user", "content": question}],
            "context": {
                "user_id": str(request.get("user_id") or "ui-user"),
                "conversation_id": str(request.get("session_id") or "ui-session"),
            },
            "custom_inputs": custom,
        }

        try:
            response = invoke(payload)
        except subprocess.TimeoutExpired:
            self._send_json(
                504,
                {"error": "The endpoint did not respond in time. It may be waking up — try again."},
            )
            return
        except Exception as exc:  # surface the real reason, trimmed
            self._send_json(502, {"error": str(exc)[:400]})
            return

        custom_outputs = response.get("custom_outputs") or {}
        self._send_json(
            200,
            {
                "answer": answer_text(response),
                "patient_id": custom_outputs.get("patient_id"),
                "tool": custom_outputs.get("tool"),
                # Structured tool result; the page charts this rather than
                # re-parsing numbers out of the prose answer.
                "tool_data": custom_outputs.get("tool_data"),
                "sources": custom_outputs.get("sources") or [],
                "memory_saved": custom_outputs.get("memory_saved"),
                "memory_error": custom_outputs.get("memory_error"),
            },
        )


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Patient History Navigator UI  ->  http://127.0.0.1:{PORT}")
    print(f"Serving endpoint: {ENDPOINT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
