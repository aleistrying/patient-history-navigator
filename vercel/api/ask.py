"""Vercel serverless function: sign and forward one question to Model Serving.

The browser never receives Databricks credentials. It posts here, this function
adds the bearer token from an encrypted Vercel environment variable, and returns
only the answer and its metadata.

Required environment variables (Vercel project settings, not the repo):
    DATABRICKS_HOST    https://dbc-....cloud.databricks.com
    DATABRICKS_TOKEN   a personal access token with CAN QUERY on the endpoint
    PHN_PASSWORD       shared password callers must supply
    PHN_ENDPOINT       optional, defaults to dbdemos-rag-prod

Duration: a cold Model Serving endpoint takes about a minute to wake, and a
tool-backed answer runs 15-35s warm, so vercel.json sets maxDuration to 300s.
"""

import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler

ENDPOINT = os.environ.get("PHN_ENDPOINT", "dbdemos-rag-prod")
TIMEOUT_SECONDS = 280


def answer_text(response: dict) -> str:
    output = response.get("output")
    if isinstance(output, str):
        return output
    parts = []
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


def call_databricks(payload: dict) -> dict:
    host = (os.environ.get("DATABRICKS_HOST") or "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN") or ""
    if not host or not token:
        raise RuntimeError("The server is missing DATABRICKS_HOST or DATABRICKS_TOKEN.")
    request = urllib.request.Request(
        f"{host}/serving-endpoints/{ENDPOINT}/invocations",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


class handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            request = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._send(400, {"error": "Could not read the request."})
            return

        expected = os.environ.get("PHN_PASSWORD") or ""
        if expected and str(request.get("password") or "") != expected:
            self._send(401, {"error": "Wrong password."})
            return

        # The gate only needs to know whether the password is right. It used to prove
        # that by sending a real question, which meant the gate inherited the
        # endpoint's cold start: type the password against a scaled-to-zero endpoint
        # and the button sat on "Checking..." for about a minute before opening. The
        # password has already been checked above, so answer now and touch nothing.
        if request.get("validate_only"):
            self._send(200, {"ok": True})
            return

        question = str(request.get("question") or "").strip()
        patient_id = str(request.get("patient_id") or "").strip()
        if not question:
            self._send(400, {"error": "A question is required."})
            return

        custom = {}
        if patient_id:
            custom["patient_id"] = patient_id
        for key, value in (request.get("custom") or {}).items():
            custom[key] = value

        payload = {
            "input": request.get("history") or [{"role": "user", "content": question}],
            "context": {
                "user_id": str(request.get("user_id") or "web-user"),
                "conversation_id": str(request.get("session_id") or "web-session"),
            },
        }
        if custom:
            payload["custom_inputs"] = custom

        try:
            response = call_databricks(payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:300]
            self._send(502, {"error": f"Databricks returned {exc.code}. {detail}"})
            return
        except Exception as exc:
            self._send(
                504,
                {
                    "error": (
                        f"{exc}. The endpoint scales to zero, so the first question "
                        "after a pause can take about a minute. Ask again."
                    )
                },
            )
            return

        custom_outputs = response.get("custom_outputs") or {}
        self._send(
            200,
            {
                "answer": answer_text(response),
                "patient_id": custom_outputs.get("patient_id"),
                "tool": custom_outputs.get("tool"),
                "tool_data": custom_outputs.get("tool_data"),
                "sources": custom_outputs.get("sources") or [],
                "memory_error": custom_outputs.get("memory_error"),
            },
        )
