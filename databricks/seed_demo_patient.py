"""Write a tiny, clearly synthetic demo corpus to the configured Unity Catalog Volume.

The real project should ingest Synthea exports and selected synthetic narrative
reports. This seed exists so the deployment smoke tests and classroom demo have one
known patient even before the larger corpus is uploaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _runtime_directory() -> Path:
    module_file = globals().get("__file__")
    return Path(str(module_file)).resolve().parent if module_file else Path.cwd()


ROOT = _runtime_directory()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-file", type=Path, default=ROOT / "navigator_config.yaml"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_if_needed(path: Path, contents: str, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        print(f"Keeping existing demo file: {path}")
        return
    path.write_text(contents, encoding="utf-8")
    print(f"Wrote synthetic demo file: {path}")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config_file)
    volume = Path(
        config.get("preprocessing_config", {}).get(
            "volume_path", "/Volumes/workspace/default/rag/patient_history"
        )
    )
    demo_dir = volume / "demo"

    fhir_bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "demo-patient",
                    "birthDate": "1988-08-12",
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-2025-01",
                    "status": "finished",
                    "subject": {"reference": "Patient/demo-patient"},
                    "period": {"start": "2025-01-15T10:00:00Z"},
                    "type": [
                        {
                            "coding": [
                                {
                                    "code": "AMB",
                                    "display": "Routine outpatient visit",
                                }
                            ]
                        }
                    ],
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "a1c-2025-01",
                    "status": "final",
                    "subject": {"reference": "Patient/demo-patient"},
                    "effectiveDateTime": "2025-01-15",
                    "code": {
                        "coding": [
                            {
                                "code": "4548-4",
                                "display": "Hemoglobin A1c",
                            }
                        ]
                    },
                    "valueQuantity": {"value": 5.8, "unit": "%"},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "a1c-2025-06",
                    "status": "final",
                    "subject": {"reference": "Patient/demo-patient"},
                    "effectiveDateTime": "2025-06-20",
                    "code": {
                        "coding": [
                            {
                                "code": "4548-4",
                                "display": "Hemoglobin A1c",
                            }
                        ]
                    },
                    "valueQuantity": {"value": 5.6, "unit": "%"},
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-a-active",
                    "status": "active",
                    "subject": {"reference": "Patient/demo-patient"},
                    "authoredOn": "2025-01-15",
                    "medicationCodeableConcept": {"text": "Medication A"},
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-b-active",
                    "status": "active",
                    "subject": {"reference": "Patient/demo-patient"},
                    "authoredOn": "2025-06-20",
                    "medicationCodeableConcept": {"text": "Medication B"},
                }
            },
        ],
    }
    write_if_needed(
        demo_dir / "demo_patient_fhir_bundle.json",
        json.dumps(fhir_bundle, indent=2),
        args.overwrite,
    )

    report_one = """# Synthetic Imaging Report - January 2025

Patient ID: demo-patient
Study date: 2025-01-20
Document ID: imaging-2025-01

## Findings
A small finding is documented in the lower-left region. The report records a maximum
measurement of 8 mm. No comparison study was available.

## Impression
Baseline examination documented for future comparison. This synthetic report is for
software demonstration only.
"""
    report_two = """# Synthetic Imaging Report - June 2025

Patient ID: demo-patient
Study date: 2025-06-24
Document ID: imaging-2025-06

## Findings
The previously documented lower-left finding is again described. The maximum
measurement is recorded as 6 mm. The report compares this study with 2025-01-20.

## Impression
The recorded measurement is smaller than on the prior synthetic examination. This
report does not provide treatment advice.
"""
    first_path = demo_dir / "demo-patient__2025-01-20__imaging_report__imaging-2025-01.md"
    second_path = demo_dir / "demo-patient__2025-06-24__imaging_report__imaging-2025-06.md"
    write_if_needed(first_path, report_one, args.overwrite)
    write_if_needed(second_path, report_two, args.overwrite)

    # Sidecars make the metadata deterministic and demonstrate how arbitrary
    # narrative reports are attached to the same synthetic patient.
    for path, metadata in [
        (
            first_path,
            {
                "patient_id": "demo-patient",
                "document_id": "imaging-2025-01",
                "document_type": "imaging_report",
                "document_date": "2025-01-20",
                "section": "Imaging report",
                "event_name": "Synthetic imaging examination",
                "status": "final",
            },
        ),
        (
            second_path,
            {
                "patient_id": "demo-patient",
                "document_id": "imaging-2025-06",
                "document_type": "imaging_report",
                "document_date": "2025-06-24",
                "section": "Imaging report",
                "event_name": "Synthetic imaging follow-up",
                "status": "final",
            },
        ),
    ]:
        write_if_needed(
            Path(str(path) + ".metadata.json"),
            json.dumps(metadata, indent=2),
            args.overwrite,
        )

    print("Demo corpus ready. Upload the larger Synthea corpus to the same Volume.")


if __name__ == "__main__":
    main()
