#!/usr/bin/env bash
# Query the deployed Patient History Navigator endpoint.
#
#   ./ask.sh <patient> "<question>" [tool] [key=value ...]
#
# <patient> accepts a full id, a short prefix (73b9), or a roster number (1-5).
# The Databricks AI Playground cannot send custom_inputs, so it always falls back
# to demo-patient; this script sets patient_id explicitly, which is the only way
# to reach a real Synthea patient.
#
#   ./ask.sh 73b9 "What laboratory results are recorded?"
#   ./ask.sh 73b9 "Build a timeline" timeline
#   ./ask.sh 73b9 "Weight trend?" lab_trend test_name="Body Weight"
#   ./ask.sh --list
set -uo pipefail

ENDPOINT="${ENDPOINT:-dbdemos-rag-prod}"
SESSION="${SESSION:-cli-session}"

PATIENTS=(
  "73b97ede-ebda-6de2-36c8-c7c6cc9ad9ff|Buddy254 Boyle917|M 1985|44 docs 2014-2026"
  "7ad140ab-bfae-c3ae-20a3-2244b1c4d0e2|Guadalupe206 Valencia279|F 1954|54 docs 2021-2024"
  "7c50e28d-07cc-a4cf-eb58-b2a15bc552a3|Margot418 Strosin214|F 1960|46 docs 2010-2014"
  "7c6f0701-02e7-6c85-1f82-c0f615d02bb4|Joleen561 Witting912|F 1978|52 docs 2016-2026"
  "7dd90132-76f1-9559-8d81-e732c8633880|Kayleigh718 Schimmel440|F 2025|29 docs infant growth"
)

if [[ "${1:-}" == "--list" || $# -lt 2 ]]; then
  printf '%-3s %-38s %-26s %-8s %s\n' "#" "patient_id" "name" "sex/born" "records"
  i=1
  for row in "${PATIENTS[@]}"; do
    IFS='|' read -r id name sex info <<<"$row"
    printf '%-3s %-38s %-26s %-8s %s\n' "$i" "$id" "$name" "$sex" "$info"
    i=$((i+1))
  done
  echo
  echo 'Usage: ./ask.sh <patient> "<question>" [timeline|lab_trend|compare_documents|medication_changes] [key=value ...]'
  exit 0
fi

# Resolve the patient argument to a full id.
want="$1"; shift
PATIENT=""
if [[ "$want" =~ ^[1-5]$ ]]; then
  IFS='|' read -r PATIENT _ <<<"${PATIENTS[$((want-1))]}"
else
  for row in "${PATIENTS[@]}"; do
    IFS='|' read -r id _ <<<"$row"
    [[ "$id" == "$want"* ]] && PATIENT="$id" && break
  done
fi
if [[ -z "$PATIENT" ]]; then
  echo "No patient matches '$want'. Run ./ask.sh --list" >&2
  exit 1
fi

QUESTION="$1"; shift
TOOL=""
if [[ $# -gt 0 && "$1" != *=* ]]; then TOOL="$1"; shift; fi

# Remaining key=value pairs become custom_inputs (test_name, start_date, ...).
PAYLOAD=$(PATIENT="$PATIENT" QUESTION="$QUESTION" TOOL="$TOOL" SESSION="$SESSION" \
  python3 -c '
import json, os, sys
ci = {"patient_id": os.environ["PATIENT"]}
if os.environ.get("TOOL"): ci["tool"] = os.environ["TOOL"]
for arg in sys.argv[1:]:
    k, _, v = arg.partition("=")
    ci[k] = v
print(json.dumps({
    "input": [{"role": "user", "content": os.environ["QUESTION"]}],
    "context": {"user_id": "cli-user", "conversation_id": os.environ["SESSION"]},
    "custom_inputs": ci,
}))' "$@")

TMP=$(mktemp)
printf '%s' "$PAYLOAD" > "$TMP"
databricks api post "/serving-endpoints/$ENDPOINT/invocations" --json "@$TMP" \
  | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit("Endpoint returned no JSON (is it mid-deployment?)")
co = d.get("custom_outputs") or {}
out = d.get("output")
if isinstance(out, str):
    text = out
else:
    parts = []
    for item in out or []:
        if isinstance(item, dict):
            c = item.get("content")
            if isinstance(c, str): parts.append(c)
            elif isinstance(c, list):
                parts += [x.get("text", "") for x in c if isinstance(x, dict)]
    text = "\n".join(parts)
print(text or "(no answer text)")
print("\n--- diagnostics ---")
print("patient_id :", co.get("patient_id"))
print("tool       :", co.get("tool") or "(retrieval)")
print("sources    :", len(co.get("sources") or []))
if co.get("memory_error"): print("memory_err :", str(co["memory_error"])[:120])
for s in (co.get("sources") or [])[:3]:
    print("  -", str(s.get("source_path", "")).split("/")[-1][:88])
'
rm -f "$TMP"
