"""Vercel serverless function: start the endpoint scaling up, and say whether it is up.

Mirrors the /api/warm route that ui/app.py serves locally. Model Serving scales to
zero after roughly ten minutes idle and takes about a minute to wake, so the first
question after a quiet period is slow and every later one is not. A short-timeout
probe is enough to trigger the scale-up — the wake continues server-side even after
this call gives up — so the page can poll until one probe succeeds and turn a
mysterious delay into a visible, finished step.

Asleep is not the only way the endpoint can be missing. After about six weeks idle
the workspace lost both the serving endpoint and the AI Search endpoint with no
delete in the audit log, and the page reported a bare 404 ENDPOINT_NOT_FOUND. So a
probe that gets that 404 rebuilds what is gone instead, one step per call, in the
order each depends on the last:

    1. AI Search endpoint   (the index lives on it)
    2. AI Search index      (the served model declares it as a resource)
    3. serving endpoint     (the @champion version of the registered model)

Every step only starts work that Databricks finishes server-side, so each call
returns well inside maxDuration and the page keeps polling until "ready". The
embeddings are already stored in the source table, so the index is a sync, not a
re-embed. The whole rebuild takes roughly 20-30 minutes.

Returns {"status": "ready"|"warming"|"restoring", "seconds": n, "detail": text}.

The password is required, as a query parameter, because a probe spends Model
Serving compute and a restore creates resources, and this URL is public. GET cannot
carry the JSON body /api/ask uses, so vercel/sync_from_ui.py appends ?password= to
the page's fetch.
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

# What a restore rebuilds. Defaults are databricks/navigator_config.yaml's values;
# change them there first, then here.
VS_ENDPOINT = os.environ.get("PHN_VS_ENDPOINT", "rag-vs-endpoint")
VS_INDEX = os.environ.get("PHN_VS_INDEX", "workspace.default.patient_history_chunks_index")
SOURCE_TABLE = os.environ.get("PHN_SOURCE_TABLE", "workspace.default.patient_history_rag_chunks")
MODEL_NAME = os.environ.get("PHN_MODEL_NAME", "workspace.default.dbdemos_rag_demo")
# qwen3-embedding-0-6b, the model the stored vectors came from.
EMBEDDING_DIMENSION = 1024
# Same list as ingest_records.ensure_and_sync_index.
COLUMNS_TO_SYNC = [
    "text_chunk", "retrieval_text", "patient_id", "document_id", "document_type",
    "document_date", "section", "source_path", "source_doc",
]
EXPERIMENT_ID = "1387270176798220"
MEMORY_SECRET = "{{secrets/patient-history-navigator/memory-store-token}}"

PROBE = {
    "input": [{"role": "user", "content": "ping"}],
    "context": {"user_id": "warmup", "conversation_id": "warmup"},
    "custom_inputs": {"patient_id": "demo-patient"},
}


class Workspace:
    """Minimal signed JSON client. Returns (status, body); never raises on 4xx/5xx."""

    def __init__(self, host: str, token: str) -> None:
        self.host = host
        self.token = token

    def call(self, method: str, path: str, body: dict | None = None,
             timeout: float = 20) -> tuple[int, dict]:
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            headers={"Authorization": f"Bearer {self.token}",
                     "Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                return response.status, json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                return exc.code, json.loads(raw) if raw else {}
            except ValueError:
                return exc.code, {"message": raw.decode("utf-8", "ignore")[:300]}


def serving_config(version: str) -> dict:
    """The endpoint config release_pipeline.endpoint_config builds. Keep the two equal."""
    entity = f"{MODEL_NAME.split('.')[-1]}-{version}"[:63]
    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    return {
        "served_entities": [{
            "name": entity,
            "entity_name": MODEL_NAME,
            "entity_version": version,
            "workload_type": "CPU",
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
            "environment_vars": {
                "ENABLE_MLFLOW_TRACING": "true",
                "MLFLOW_EXPERIMENT_ID": EXPERIMENT_ID,
                "DATABRICKS_TOKEN": MEMORY_SECRET,
                "DATABRICKS_HOST": host,
            },
        }],
        "traffic_config": {
            "routes": [{"served_model_name": entity, "traffic_percentage": 100}]
        },
    }


def restore_step(ws: Workspace) -> str | None:
    """Advance the rebuild by one step and describe where it stands.

    Returns None when nothing is missing and the endpoint reports READY.

    Called on every poll, so each branch must be safe to repeat: it creates a
    resource only after a GET has shown it missing, and otherwise just reports.
    """
    # 1. AI Search endpoint.
    status, body = ws.call("GET", f"/api/2.0/vector-search/endpoints/{VS_ENDPOINT}")
    if status == 404:
        status, body = ws.call("POST", "/api/2.0/vector-search/endpoints",
                               {"name": VS_ENDPOINT, "endpoint_type": "STANDARD"})
        if status >= 400:
            raise RuntimeError(f"Could not create {VS_ENDPOINT}: {body.get('message', body)}")
        return f"Step 1 of 3: creating the search endpoint {VS_ENDPOINT} (5-10 min)."
    if status >= 400:
        raise RuntimeError(f"Could not read {VS_ENDPOINT}: {body.get('message', body)}")
    state = (body.get("endpoint_status") or {}).get("state", "UNKNOWN")
    if state != "ONLINE":
        return f"Step 1 of 3: search endpoint {VS_ENDPOINT} is {state.lower()}."

    # 2. AI Search index. An index whose endpoint was deleted also reads as 404,
    # but its Unity Catalog entry can survive, so a create that collides with it
    # clears the stale entry first and is retried on the next poll.
    status, body = ws.call("GET", f"/api/2.0/vector-search/indexes/{VS_INDEX}")
    if status == 404:
        spec = {
            "name": VS_INDEX,
            "endpoint_name": VS_ENDPOINT,
            "primary_key": "chunk_id",
            "index_type": "DELTA_SYNC",
            "delta_sync_index_spec": {
                "source_table": SOURCE_TABLE,
                "pipeline_type": "TRIGGERED",
                "embedding_vector_columns": [
                    {"name": "embedding", "embedding_dimension": EMBEDDING_DIMENSION}
                ],
                "columns_to_sync": COLUMNS_TO_SYNC,
            },
        }
        status, body = ws.call("POST", "/api/2.0/vector-search/indexes", spec)
        if status >= 400:
            if "exist" in json.dumps(body).lower():
                ws.call("DELETE", f"/api/2.0/vector-search/indexes/{VS_INDEX}")
                return "Step 2 of 3: cleared a stale index entry; recreating it next."
            raise RuntimeError(f"Could not create {VS_INDEX}: {body.get('message', body)}")
        return "Step 2 of 3: creating the search index from the stored embeddings."
    if status >= 400:
        raise RuntimeError(f"Could not read {VS_INDEX}: {body.get('message', body)}")
    if not (body.get("status") or {}).get("ready"):
        message = (body.get("status") or {}).get("message", "syncing")
        return f"Step 2 of 3: search index is not ready yet ({message[:120]})."

    # 3. Serving endpoint, on whichever version currently holds @champion.
    status, body = ws.call("GET", f"/api/2.0/serving-endpoints/{ENDPOINT}")
    if status == 404:
        status, alias = ws.call(
            "GET", f"/api/2.1/unity-catalog/models/{MODEL_NAME}/aliases/champion")
        if status >= 400 or not alias.get("version"):
            raise RuntimeError(f"No @champion version of {MODEL_NAME}: {alias.get('message', alias)}")
        version = str(alias["version"])
        status, body = ws.call("POST", "/api/2.0/serving-endpoints",
                               {"name": ENDPOINT, "config": serving_config(version)})
        if status >= 400:
            raise RuntimeError(f"Could not create {ENDPOINT}: {body.get('message', body)}")
        return f"Step 3 of 3: creating {ENDPOINT} on model version {version} (10-20 min)."
    if status >= 400:
        raise RuntimeError(f"Could not read {ENDPOINT}: {body.get('message', body)}")
    state = body.get("state") or {}
    if state.get("config_update") == "UPDATE_FAILED":
        raise RuntimeError(f"{ENDPOINT} failed to deploy; see its Events tab in Databricks.")
    if state.get("ready") == "READY":
        # Everything exists. A freshly created endpoint reports READY a few
        # minutes before its invocations URL stops answering 404, so this is a
        # wake, not a rebuild -- reporting "restoring" here held the page in
        # "Rebuilding" with the button disabled while nothing was being rebuilt.
        return None
    return f"Step 3 of 3: {ENDPOINT} is {str(state.get('ready', 'starting')).lower().replace('_', ' ')}."


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

        started = time.monotonic()
        ws = Workspace(host, token)
        try:
            ws_status, _ = ws.call("POST", f"/serving-endpoints/{ENDPOINT}/invocations",
                                   PROBE, timeout=PROBE_SECONDS)
        except Exception:
            # The timeout, which is the normal cold-start outcome: the probe has
            # still done its job of starting the scale-up.
            self._send(200, {"status": "warming",
                             "seconds": round(time.monotonic() - started)})
            return

        if ws_status < 400:
            self._send(200, {"status": "ready",
                             "seconds": round(time.monotonic() - started)})
            return
        if ws_status != 404:
            # Present but refusing (starting up, 503, a permission error): not a
            # case a rebuild fixes, so keep reporting it as waking.
            self._send(200, {"status": "warming",
                             "seconds": round(time.monotonic() - started)})
            return

        try:
            detail = restore_step(ws)
        except Exception as exc:
            self._send(200, {"status": "restoring", "error": str(exc)[:300],
                             "seconds": round(time.monotonic() - started)})
            return
        if detail is None:
            self._send(200, {"status": "warming",
                             "detail": "Deployed; waiting for it to accept requests.",
                             "seconds": round(time.monotonic() - started)})
            return
        self._send(200, {"status": "restoring", "detail": detail,
                         "seconds": round(time.monotonic() - started)})
