"""Build, evaluate, register, deploy, and smoke-test Patient History Navigator.

This script is designed for a Databricks Asset Bundle / Lakeflow Job task. A model
version is deployed only after local loading, offline evaluation, and the configured
quality gate succeed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow import MlflowClient
from mlflow.deployments import get_deploy_client
from mlflow.genai.scorers import RelevanceToQuery, RetrievalGroundedness, Safety, scorer
from mlflow.models.resources import (
    DatabricksSQLWarehouse,
    DatabricksServingEndpoint,
    DatabricksTable,
    DatabricksVectorSearchIndex,
)


def _runtime_directory() -> Path:
    module_file = globals().get("__file__")
    if module_file:
        return Path(str(module_file)).resolve().parent
    if sys.argv and sys.argv[0] and sys.argv[0] not in {"-c", "-m", ""}:
        candidate = Path(sys.argv[0])
        if candidate.exists():
            return candidate.resolve().parent
    return Path.cwd()


ROOT = _runtime_directory()
DEFAULT_AGENT_FILE = ROOT / "serving_agent.py"
DEFAULT_CONFIG_FILE = ROOT / "navigator_config.yaml"
DEFAULT_EVAL_FILE = ROOT / "evaluation_set.json"
CODE_MODULES = [
    "ttl_cache.py",
    "session_memory.py",
    "query_strategies.py",
    "vector_retrieval.py",
    "sql_execution.py",
    "record_tools.py",
    "patient_tools.py",
    "orchestration.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--agent-file", type=Path, default=DEFAULT_AGENT_FILE)
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--workload-size", default="Small")
    parser.add_argument("--scale-to-zero", action="store_true")
    parser.add_argument("--deployment-timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return value


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def git_sha() -> str:
    for variable in ("GITHUB_SHA", "CI_COMMIT_SHA"):
        if os.getenv(variable):
            return str(os.getenv(variable))
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def extract_response_text(response: Any) -> str:
    if hasattr(response, "to_dict"):
        response = response.to_dict()
    if hasattr(response, "model_dump"):
        response = response.model_dump()
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        output = response.get("output") or response.get("predictions") or []
        if isinstance(output, dict):
            output = output.get("output") or []
        for item in reversed(output if isinstance(output, list) else []):
            if not isinstance(item, dict):
                continue
            content = item.get("content") or []
            if isinstance(content, str):
                return content
            for block in content if isinstance(content, list) else []:
                if isinstance(block, dict) and block.get("text"):
                    return str(block["text"])
            if item.get("text"):
                return str(item["text"])
    raise ValueError(f"Could not extract response text from {response!r}")


def model_input(
    question: str,
    *,
    patient_id: str = "demo-patient",
    tool: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    custom_inputs: dict[str, Any] = {"patient_id": patient_id}
    if tool:
        custom_inputs["tool"] = tool
    if extra:
        custom_inputs.update(extra)
    return {
        "input": [{"role": "user", "content": question}],
        "context": {
            "user_id": "release-evaluator",
            "conversation_id": f"evaluation-{patient_id}",
        },
        "custom_inputs": custom_inputs,
    }


def build_resources(config: dict[str, Any]) -> list[Any]:
    resources_config = config.get("databricks_resources", {})
    retriever = config.get("retriever_config", {})
    tools = config.get("tools_config", {})
    values: list[Any] = [
        DatabricksVectorSearchIndex(
            index_name=str(retriever["vector_search_index"])
        ),
        DatabricksServingEndpoint(
            endpoint_name=str(resources_config["embedding_endpoint_name"])
        ),
        DatabricksServingEndpoint(
            endpoint_name=str(resources_config["llm_endpoint_name"])
        ),
        DatabricksTable(
            table_name=str(
                tools.get(
                    "documents_table",
                    "workspace.default.patient_history_documents",
                )
            )
        ),
    ]
    # analyze_lab_trend reads this table, so it needs its own grant.
    observations_table = str(tools.get("observations_table", "") or "").strip()
    if observations_table:
        values.append(DatabricksTable(table_name=observations_table))
    warehouse_id = str(resources_config.get("sql_warehouse_id", ""))
    if warehouse_id and "REPLACE" not in warehouse_id.upper():
        values.append(DatabricksSQLWarehouse(warehouse_id=warehouse_id))
    return values


def log_agent(args: argparse.Namespace, config: dict[str, Any]):
    code_paths = [str(ROOT / name) for name in CODE_MODULES]
    with mlflow.start_run(run_name=f"build-{git_sha()[:8]}") as run:
        mlflow.log_params(
            {
                "git_sha": git_sha(),
                "model_name": args.model_name,
                "endpoint_name": args.endpoint_name,
                "query_strategies": ",".join(
                    config.get("query_strategy_config", {}).get(
                        "active_strategies", []
                    )
                ),
            }
        )
        logged = mlflow.pyfunc.log_model(
            python_model=str(args.agent_file),
            name="agent",
            model_config=str(args.config_file),
            input_example=config.get("input_example"),
            resources=build_resources(config),
            code_paths=code_paths,
            pip_requirements=[
                "mlflow[databricks]>=3.1,<4",
                "databricks-sdk>=0.50,<1",
                "databricks-langchain>=0.5,<1",
                "databricks-ai-search>=0.60,<1",
                "langchain-core>=0.3,<1",
                "langchain-text-splitters>=0.3,<1",
                "langgraph>=1.0,<2",
                "pydantic>=2,<3",
                "requests>=2.31,<3",
                "pyyaml>=6,<7",
            ],
        )
        mlflow.log_artifact(str(args.config_file), artifact_path="release_inputs")
        mlflow.log_artifact(str(args.eval_file), artifact_path="release_inputs")
        return logged, run.info.run_id


def local_smoke_test(model_uri: str) -> None:
    model = mlflow.pyfunc.load_model(model_uri)
    response = model.predict(
        model_input(
            "What information is available in the indexed record?",
            patient_id="demo-patient",
        )
    )
    text = extract_response_text(response)
    if len(text.strip()) < 10:
        raise RuntimeError("Local smoke test returned an empty response")
    print(f"Local smoke test passed: {text[:180]!r}")


@scorer
def keyword_coverage(
    outputs: str,
    expectations: dict[str, Any] | None = None,
) -> float:
    expected = (expectations or {}).get("expected_keywords", [])
    if not expected:
        return 1.0
    normalized = outputs.casefold()
    return sum(keyword.casefold() in normalized for keyword in expected) / len(expected)


class _ScorerFailureRecorder(logging.Handler):
    """Capture the harness warning that reports failed scorer invocations.

    MLflow keeps the per-scorer failure counts in a local and only surfaces them
    through this warning; EvaluationResult exposes metrics, artifacts and run_id
    only. Without this, partially-failed scoring is indistinguishable from a
    clean pass, because the recorded metric is the mean of whatever succeeded.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "scorer invocations failed" in message.casefold():
            self.messages.append(message)


def evaluate_agent(
    model_uri: str,
    eval_file: Path,
) -> tuple[str, dict[str, float]]:
    model = mlflow.pyfunc.load_model(model_uri)
    evaluation_data = load_json(eval_file)

    def predict_fn(
        question: str,
        patient_id: str = "demo-patient",
        tool: str = "",
        tool_inputs_json: str = "{}",
    ) -> str:
        extra = json.loads(tool_inputs_json or "{}")
        response = model.predict(
            model_input(
                question,
                patient_id=patient_id,
                tool=tool or None,
                extra=extra,
            )
        )
        return extract_response_text(response)

    # RetrievalGroundedness needs a retriever span, which only the RAG path emits.
    # Rows that force a deterministic tool produce no such span, so scoring them
    # would fail every time. Those failures are invisible in the run's metrics --
    # MLflow averages the invocations that succeeded -- which previously let the
    # quality gate pass on a single sample. Score each group with only the
    # scorers that can apply to it, and treat any remaining failure as fatal.
    # "none" pins the retrieval path. An empty tool no longer implies retrieval:
    # tool selection is a model decision now, so an unpinned row could be routed
    # to a tool at runtime and produce no retriever span to score.
    def uses_retrieval(row: dict[str, Any]) -> bool:
        pinned = str(row.get("inputs", {}).get("tool") or "").strip().lower()
        return pinned in {"", "none", "no_tool", "retrieval", "search"}

    retrieval_rows = [row for row in evaluation_data if uses_retrieval(row)]
    tool_rows = [row for row in evaluation_data if not uses_retrieval(row)]
    print(
        f"Evaluating {len(retrieval_rows)} retrieval row(s) and {len(tool_rows)} tool row(s)."
    )

    def run_group(rows: list[dict[str, Any]], scorers: list[Any], label: str):
        """Evaluate one row group, failing if any scorer invocation errored."""
        failures = _ScorerFailureRecorder()
        harness_logger = logging.getLogger("mlflow.genai.evaluation.harness")
        harness_logger.addHandler(failures)
        try:
            result = mlflow.genai.evaluate(
                data=rows, predict_fn=predict_fn, scorers=scorers
            )
        finally:
            harness_logger.removeHandler(failures)
        if failures.messages:
            raise RuntimeError(
                f"Scorer invocations failed while evaluating {label} rows, so the "
                "reported means cover only part of the eval set: "
                + " | ".join(failures.messages)
            )
        run = mlflow.search_runs(filter_string=f"run_id = '{result.run_id}'")
        return result.run_id, {
            column.removeprefix("metrics."): float(run.iloc[0][column])
            for column in run.columns
            if column.startswith("metrics.")
            and column.endswith("/mean")
            and run.iloc[0][column] is not None
        }

    metrics: dict[str, float] = {}
    primary_run_id = None

    if tool_rows:
        # No RetrievalGroundedness here: these rows never produce a retriever span.
        primary_run_id, tool_metrics = run_group(
            tool_rows, [keyword_coverage, RelevanceToQuery(), Safety()], "tool"
        )
        metrics.update(tool_metrics)

    if retrieval_rows:
        retrieval_run_id, retrieval_metrics = run_group(
            retrieval_rows,
            [keyword_coverage, RelevanceToQuery(), RetrievalGroundedness(), Safety()],
            "retrieval",
        )
        # Average the shared scorers across both groups, weighted by row count, so
        # a metric always reflects every row it was actually able to score.
        for name, value in retrieval_metrics.items():
            if name in metrics and tool_rows:
                metrics[name] = (
                    value * len(retrieval_rows) + metrics[name] * len(tool_rows)
                ) / (len(retrieval_rows) + len(tool_rows))
            else:
                metrics[name] = value
        primary_run_id = primary_run_id or retrieval_run_id

    # If the eval set has no retrieval rows, retrieval_groundedness is simply absent
    # and find_metric() fails the gate with an explicit "was not produced" message.
    print("Evaluation metrics:", json.dumps(metrics, indent=2, sort_keys=True))
    return primary_run_id, metrics


def find_metric(metrics: dict[str, float], expected_name: str) -> float:
    wanted = expected_name.casefold().replace("-", "_")
    for name, value in metrics.items():
        normalized = name.casefold().replace("-", "_")
        if normalized == wanted or normalized.startswith(f"{wanted}/"):
            return value
    raise RuntimeError(
        f"Metric {expected_name!r} was not produced. Available: {sorted(metrics)}"
    )


def enforce_quality_gate(metrics: dict[str, float], config: dict[str, Any]) -> None:
    thresholds = config.get(
        "quality_gate",
        {
            "keyword_coverage": 0.50,
            "relevance_to_query": 0.65,
            "retrieval_groundedness": 0.65,
            "safety": 1.00,
        },
    )
    failures: list[str] = []
    for name, minimum in thresholds.items():
        value = find_metric(metrics, str(name))
        if value < float(minimum):
            failures.append(f"{name}={value:.3f} < {float(minimum):.3f}")
    if failures:
        raise RuntimeError("Quality gate failed: " + "; ".join(failures))
    print("Quality gate passed.")


def register_model(
    model_uri: str,
    model_name: str,
    build_run_id: str,
    evaluation_run_id: str,
) -> str:
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = str(registered.version)
    client = MlflowClient()
    tags = {
        "git_sha": git_sha(),
        "build_run_id": build_run_id,
        "evaluation_run_id": evaluation_run_id,
        "release_status": "validated",
        "application": "patient-history-navigator",
    }
    for key, value in tags.items():
        client.set_model_version_tag(model_name, version, key, value)
    client.set_registered_model_alias(model_name, "candidate", version)
    print(f"Registered {model_name} version {version}.")
    return version


def served_entity_name(model_name: str, version: str) -> str:
    short = model_name.split(".")[-1]
    safe = "".join(char if char.isalnum() or char in "-_" else "-" for char in short)
    return f"{safe}-{version}"[:63]


def endpoint_config(
    model_name: str,
    version: str,
    workload_size: str,
    scale_to_zero: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    entity = served_entity_name(model_name, version)
    serving = config.get("serving_config", {})
    environment_vars = {"ENABLE_MLFLOW_TRACING": "true"}
    experiment_id = str(serving.get("mlflow_experiment_id", "")).strip()
    if experiment_id:
        environment_vars["MLFLOW_EXPERIMENT_ID"] = experiment_id

    # Manual authentication for the Unity Catalog memory store, which supports
    # neither automatic passthrough nor OBO on Model Serving. The token is read
    # from a Databricks secret at endpoint start; the value never appears here.
    # Setting DATABRICKS_TOKEN turns off automatic passthrough for every other
    # declared resource, so the token's identity must be able to reach them all.
    secret_scope = str(serving.get("memory_auth_secret_scope", "")).strip()
    secret_key = str(serving.get("memory_auth_secret_key", "")).strip()
    workspace_host = str(serving.get("workspace_host", "")).strip()
    if secret_scope and secret_key:
        environment_vars["DATABRICKS_TOKEN"] = (
            "{{secrets/" + secret_scope + "/" + secret_key + "}}"
        )
        if workspace_host:
            environment_vars["DATABRICKS_HOST"] = workspace_host
    return {
        "served_entities": [
            {
                "name": entity,
                "entity_name": model_name,
                "entity_version": version,
                "workload_type": "CPU",
                "workload_size": workload_size,
                "scale_to_zero_enabled": scale_to_zero,
                "environment_vars": environment_vars,
            }
        ],
        "traffic_config": {
            "routes": [
                {"served_model_name": entity, "traffic_percentage": 100}
            ]
        },
    }


def endpoint_exists(client, endpoint_name: str) -> bool:
    try:
        client.get_endpoint(endpoint=endpoint_name)
        return True
    except Exception as exc:
        if "RESOURCE_DOES_NOT_EXIST" in str(exc) or "404" in str(exc):
            return False
        raise


def deploy_model(
    args: argparse.Namespace,
    version: str,
    config: dict[str, Any],
) -> None:
    client = get_deploy_client("databricks")
    deployment = endpoint_config(
        args.model_name,
        version,
        args.workload_size,
        args.scale_to_zero,
        config,
    )
    if endpoint_exists(client, args.endpoint_name):
        client.update_endpoint_config(endpoint=args.endpoint_name, config=deployment)
        print(f"Updating {args.endpoint_name} to model version {version}.")
    else:
        client.create_endpoint(name=args.endpoint_name, config=deployment)
        print(f"Creating {args.endpoint_name} with model version {version}.")

    deadline = time.monotonic() + args.deployment_timeout_seconds
    while time.monotonic() < deadline:
        endpoint = client.get_endpoint(endpoint=args.endpoint_name)
        state = endpoint.get("state", {})
        if state.get("ready") == "READY" and state.get("config_update") == "NOT_UPDATING":
            active = {
                str(item.get("entity_version"))
                for item in endpoint.get("config", {}).get("served_entities", [])
            }
            if version in active:
                print(f"Endpoint {args.endpoint_name} is serving version {version}.")
                return
        if state.get("ready") == "NOT_READY" and state.get("config_update") == "UPDATE_FAILED":
            raise RuntimeError(f"Endpoint update failed: {endpoint}")
        time.sleep(15)
    raise TimeoutError(f"Endpoint {args.endpoint_name} did not become ready")


def endpoint_smoke_test(endpoint_name: str) -> None:
    client = get_deploy_client("databricks")
    response = client.predict(
        endpoint=endpoint_name,
        inputs=model_input(
            "What information can you retrieve for this patient?",
            patient_id="demo-patient",
        ),
    )
    text = extract_response_text(response)
    if len(text.strip()) < 10:
        raise RuntimeError("Production smoke test returned an invalid response")
    print(f"Endpoint smoke test passed: {text[:180]!r}")


def promote(model_name: str, version: str) -> None:
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "champion", version)
    client.set_model_version_tag(model_name, version, "release_status", "production")
    print(f"Promoted {model_name} version {version} to @champion.")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config_file)
    evaluation_data = load_json(args.eval_file)
    print(f"Model: {args.model_name}")
    print(f"Endpoint: {args.endpoint_name}")
    print("Evaluation questions:")
    for row in evaluation_data:
        print(f"- {row.get('inputs', {}).get('question', '<missing>')}")

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(args.experiment_name)

    logged, build_run_id = log_agent(args, config)
    local_smoke_test(logged.model_uri)
    evaluation_run_id, metrics = evaluate_agent(logged.model_uri, args.eval_file)
    enforce_quality_gate(metrics, config)
    version = register_model(
        logged.model_uri,
        args.model_name,
        build_run_id,
        evaluation_run_id,
    )
    deploy_model(args, version, config)
    endpoint_smoke_test(args.endpoint_name)
    promote(args.model_name, version)


if __name__ == "__main__":
    main()
