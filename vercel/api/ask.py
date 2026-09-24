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
tool-backed answer runs 5-12s warm while a retrieval answer runs 9-28s, so
vercel.json sets maxDuration to 300s. That 300 is genuinely available here and is
not silently clamped: the project runs on fluid compute, and the Vercel API
reports functionDefaultTimeout=300 for this (Hobby) project with the deployment's
resolved functionTimeout also 300. Verify before changing it -- the numbers below
are derived from maxDuration and have to move with it.
"""

import io
import json
import os
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler

ENDPOINT = os.environ.get("PHN_ENDPOINT", "dbdemos-rag-prod")

# Must match vercel.json's maxDuration for "api/ask.py". Everything below is a
# budget carved out of it, because the client timeout has to expire *before* the
# platform kills the function: if the platform wins, the socket dies with no body
# and the browser can only report a bare "Failed to fetch", with no way to tell
# the user what happened. If the platform loses, the caller gets real JSON.
MAX_DURATION_SECONDS = 300
# Held back for the function's own cold start, reading the request, and writing
# the response, so the last attempt still has room to answer before the cap.
RESERVED_SECONDS = 20
# Mirrors ui/app.py's invoke(attempts=2): the endpoint scales to zero and the
# first call after a pause is what wakes it, so that call times out and the
# retry is the one that succeeds. Two attempts must fit the budget together --
# a retry the platform kills mid-flight is worse than no retry at all.
ATTEMPTS = 2
TIMEOUT_SECONDS = (MAX_DURATION_SECONDS - RESERVED_SECONDS) // ATTEMPTS
# A spurious 404 comes back in about a second, so these retries cost ~15s at most
# and come out of RESERVED_SECONDS rather than the timeout budget.
ROUTING_RETRIES = 4
ROUTING_RETRY_SECONDS = 3


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


def endpoint_exists() -> bool:
    """Whether the serving endpoint is defined at all (it may still be starting)."""
    host = (os.environ.get("DATABRICKS_HOST") or "").rstrip("/")
    request = urllib.request.Request(
        f"{host}/api/2.0/serving-endpoints/{ENDPOINT}",
        headers={"Authorization": f"Bearer {os.environ.get('DATABRICKS_TOKEN') or ''}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15):
            return True
    except urllib.error.HTTPError as exc:
        return exc.code != 404
    except Exception:
        return True


def is_timeout(error: Exception) -> bool:
    """Whether the call gave up waiting, as opposed to being refused an answer.

    Mirrors ui/app.py's _is_wakeup_timeout. urllib raises the socket timeout
    wrapped in URLError, so the type check alone misses it and the message check
    alone is fragile; do both. HTTPError is deliberately excluded by the caller:
    a 4xx or 5xx is a real reply and retrying it only spends the clock twice.
    """
    if isinstance(error, TimeoutError):
        return True
    reason = getattr(error, "reason", None)
    if isinstance(reason, TimeoutError):
        return True
    text = f"{error} {reason}".lower()
    return "timed out" in text or "timeout" in text or "deadline" in text


def call_databricks_past_routing_404(payload: dict) -> dict:
    """Call the endpoint, retrying a spurious 404 ENDPOINT_NOT_FOUND.

    A READY endpoint intermittently answers 404 ENDPOINT_NOT_FOUND within about a
    second (measured: 1 call in 6, and on consecutive probes right after a
    deploy), then answers normally on the next call. The request never reaches
    the agent, so a retry cannot repeat any work. A genuinely missing endpoint
    still fails after ROUTING_RETRIES and the caller reports it.
    """
    for attempt in range(ROUTING_RETRIES + 1):
        try:
            return call_databricks(payload)
        except urllib.error.HTTPError as exc:
            body = exc.read()
            if (exc.code != 404 or b"ENDPOINT_NOT_FOUND" not in body
                    or attempt >= ROUTING_RETRIES):
                raise urllib.error.HTTPError(
                    exc.url, exc.code, exc.msg, exc.hdrs, io.BytesIO(body)
                ) from None
            time.sleep(ROUTING_RETRY_SECONDS)
    raise RuntimeError("unreachable")


def call_databricks_with_retry(payload: dict) -> dict:
    """Call the endpoint, retrying once past a cold-start timeout.

    Model Serving scales to zero after roughly ten minutes idle. The call that
    finds it cold is the call that wakes it, so it times out and the next one
    lands on a running endpoint. ui/app.py has always done this; this path did
    not, which is why a caller here had to press send again by hand.
    """
    last: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            return call_databricks_past_routing_404(payload)
        except urllib.error.HTTPError:
            raise
        except Exception as exc:
            last = exc
            if attempt + 1 >= ATTEMPTS or not is_timeout(exc):
                raise
    raise last if last else RuntimeError("The endpoint could not be reached.")


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

        # The password is already checked above, so answer without calling Model
        # Serving. That keeps the gate off the endpoint's cold-start path, which
        # would otherwise hold the button on "Checking..." for about a minute.
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
            response = call_databricks_with_retry(payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:300]
            if exc.code == 404 and "ENDPOINT_NOT_FOUND" in detail:
                # A just-deployed endpoint also answers 404 for a few minutes, so
                # only call it missing when the endpoint itself is not there.
                if endpoint_exists():
                    self._send(503, {"error": (
                        f"{ENDPOINT} was just deployed and is not accepting "
                        "requests yet. Ask again in a minute or two."
                    )})
                else:
                    # Gone, not asleep. /api/warm rebuilds it.
                    self._send(502, {"error": (
                        f"The serving endpoint {ENDPOINT} does not exist. Open the "
                        "patient list and press 'Wake it now' to rebuild it; that "
                        "takes 20-30 minutes."
                    )})
                return
            self._send(502, {"error": f"Databricks returned {exc.code}. {detail}"})
            return
        except Exception as exc:
            # Only a timeout is evidence of a slow or sleeping endpoint. Attaching
            # the scale-to-zero explanation to every failure made it advice for a
            # cause we had not established: the endpoint is often demonstrably warm
            # when a question fails, and telling the user to wait a minute for a
            # wake-up that already happened sends them looking in the wrong place.
            if is_timeout(exc):
                self._send(
                    504,
                    {
                        "error": (
                            f"The endpoint did not answer within "
                            f"{TIMEOUT_SECONDS}s, across {ATTEMPTS} attempts. "
                            "If it had been idle for a while it may still be "
                            "waking up; ask again."
                        )
                    },
                )
            else:
                self._send(502, {"error": f"The endpoint could not be reached. {exc}"})
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
