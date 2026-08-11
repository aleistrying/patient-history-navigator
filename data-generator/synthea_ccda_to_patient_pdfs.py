#!/usr/bin/env python3
"""Convert Synthea C-CDA XML files into patient-isolated, RAG-ready PDF folders.

The generated corpus is intended for synthetic-data demonstrations only.
Every PDF is watermarked and every document receives a JSON metadata sidecar.

Examples
--------
python synthea_ccda_to_patient_pdfs.py \
  --input-dir ./100files \
  --output-dir ./generated_patient_records

python synthea_ccda_to_patient_pdfs.py \
  --input-dir ./100files \
  --output-dir ./generated_patient_records \
  --augment-missing \
  --max-labs 12 --max-encounters 10 --max-prescriptions 10

python synthea_ccda_to_patient_pdfs.py \
  --run-synthea-dir ../synthea \
  --synthea-population 100 \
  --output-dir ./generated_patient_records \
  --augment-missing
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import html
import json
import random
import re
import shutil
import subprocess
import sys
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from lxml import etree
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

CDA_NS = "urn:hl7-org:v3"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
NS = {"cda": CDA_NS, "xsi": XSI_NS}

APP_VERSION = "2.0.0"
BRAND = "Ontario Synthetic Health Network"
SYNTHETIC_NOTICE = "SYNTHETIC DEMONSTRATION DATA - NOT FOR CLINICAL USE"

ONTARIO_LOCATIONS = [
    ("Ottawa", "K1P 1J1"),
    ("Toronto", "M5G 1X8"),
    ("Hamilton", "L8P 1A1"),
    ("Kingston", "K7L 1A1"),
    ("London", "N6A 1A1"),
    ("Waterloo", "N2L 3G1"),
    ("Windsor", "N9A 1A1"),
    ("Sudbury", "P3E 3A1"),
    ("Barrie", "L4M 1A1"),
    ("Thunder Bay", "P7A 1A1"),
]

SYNTHETIC_FACILITIES = [
    "Capital Region Community Hospital",
    "Lakeshore Diagnostic Centre",
    "Northern Ontario Family Health Clinic",
    "Grand River Ambulatory Centre",
    "Maple Health Primary Care Network",
    "Ontario Central Medical Centre",
    "Thames Valley Community Hospital",
    "Bayview Synthetic Care Centre",
]

IMAGING_KEYWORDS = (
    "x-ray",
    "xray",
    "radiograph",
    "computed tomography",
    " ct ",
    "magnetic resonance",
    " mri",
    "ultrasound",
    "mammogr",
    "imaging",
    "scan",
)

INPATIENT_KEYWORDS = (
    "hospital admission",
    "emergency room admission",
    "intensive care",
    "surgical department",
    "obstetric emergency",
    "hospice",
)

CONSULT_KEYWORDS = ("consultation", "specialist", "follow-up")

SECTION_FOLDER = {
    "patient_summary": "summaries",
    "longitudinal_timeline": "summaries",
    "problem_list": "summaries",
    "allergy_report": "summaries",
    "medication_history": "medications",
    "prescription_record": "medications",
    "laboratory_report": "labs",
    "vital_signs_report": "labs",
    "encounter_note": "encounters",
    "discharge_summary": "encounters",
    "specialist_note": "encounters",
    "diagnostic_imaging_report": "imaging",
    "immunization_history": "care",
    "care_plan": "care",
}

# These are intentionally labelled as synthetic demo intervals. They are not
# clinical reference intervals and must never be used for patient care.
DEMO_LAB_PANELS: dict[str, dict[str, tuple[float, float, str, int]]] = {
    "Complete Blood Count": {
        "Hemoglobin": (120.0, 170.0, "g/L", 0),
        "White blood cell count": (4.0, 11.0, "x10^9/L", 1),
        "Platelet count": (150.0, 400.0, "x10^9/L", 0),
        "Hematocrit": (0.36, 0.50, "L/L", 2),
    },
    "Comprehensive Metabolic Panel": {
        "Sodium": (135.0, 145.0, "mmol/L", 0),
        "Potassium": (3.5, 5.1, "mmol/L", 1),
        "Creatinine": (45.0, 110.0, "umol/L", 0),
        "Glucose": (3.9, 7.8, "mmol/L", 1),
        "ALT": (7.0, 56.0, "U/L", 0),
    },
    "Urinalysis": {
        "Specific gravity": (1.005, 1.030, "ratio", 3),
        "pH": (5.0, 8.0, "pH", 1),
    },
}

QUALITATIVE_URINALYSIS = [
    ("Protein", "Negative"),
    ("Glucose", "Negative"),
    ("Blood", "Negative"),
    ("Leukocyte esterase", "Negative"),
    ("Nitrite", "Negative"),
]


@dataclass
class Event:
    section: str
    start: Optional[datetime]
    stop: Optional[datetime]
    description: str
    code: str = ""
    value: str = ""
    source_id: str = ""
    status: str = ""

    @property
    def event_date(self) -> Optional[datetime]:
        return self.start or self.stop


@dataclass
class LabObservation:
    name: str
    code: str
    value: str
    unit: str = ""
    status: str = "completed"
    reference: str = ""
    flag: str = ""


@dataclass
class LabPanel:
    panel_id: str
    panel_name: str
    panel_code: str
    occurred_at: Optional[datetime]
    status: str
    observations: list[LabObservation] = field(default_factory=list)
    provenance: str = "source_ccda"


@dataclass
class PatientRecord:
    patient_id: str
    display_patient_id: str
    name: str
    given_names: str
    family_name: str
    birth_date: Optional[date]
    gender: str
    race: str
    ethnicity: str
    source_address: str
    display_address: str
    source_organization: str
    display_facility: str
    source_file: Path
    source_sha256: str
    effective_date: Optional[datetime]
    events_by_section: dict[str, list[Event]] = field(default_factory=dict)
    lab_panels: list[LabPanel] = field(default_factory=list)

    def all_events(self) -> list[Event]:
        events: list[Event] = []
        for items in self.events_by_section.values():
            events.extend(items)
        return events


@dataclass
class GeneratedDocument:
    document_id: str
    patient_id: str
    display_patient_id: str
    document_type: str
    document_date: str
    section: str
    title: str
    event_name: str
    status: str
    source_path: str
    source_record_path: str
    output_path: str
    provenance: str
    source_event_id: str
    comparison_key: str
    synthetic: bool
    full_text: str
    tags: list[str]


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slugify(value: str, max_length: int = 80) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return (value or "document")[:max_length]


def collapse_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def xpath_string(node: etree._Element | etree._ElementTree, expression: str) -> str:
    return collapse_text(str(node.xpath(f"string({expression})", namespaces=NS)))


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None

    # ISO-8601 narrative table values.
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        pass

    # C-CDA compact timestamps can contain precision from year to seconds and
    # may append a timezone offset. We keep the first 14 digits.
    digits = re.match(r"^(\d{4,14})", value)
    if not digits:
        return None
    raw = digits.group(1)
    formats = {
        4: "%Y",
        6: "%Y%m",
        8: "%Y%m%d",
        10: "%Y%m%d%H",
        12: "%Y%m%d%H%M",
        14: "%Y%m%d%H%M%S",
    }
    fmt = formats.get(len(raw))
    if not fmt:
        return None
    try:
        return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def iso_date(value: Optional[datetime | date]) -> str:
    if value is None:
        return "undated"
    if isinstance(value, datetime):
        return value.date().isoformat()
    return value.isoformat()


def human_date(value: Optional[datetime | date]) -> str:
    if value is None:
        return "Not recorded"
    if isinstance(value, datetime):
        value = value.date()
    return value.strftime("%d %b %Y")


def date_sort_key(value: Optional[datetime]) -> datetime:
    return value or datetime.min.replace(tzinfo=timezone.utc)


def status_for_event(event: Event) -> str:
    if event.status:
        return event.status
    if event.stop:
        return "completed"
    return "active"


def deterministic_rng(patient_id: str, namespace: str) -> random.Random:
    digest = hashlib.sha256(f"{patient_id}:{namespace}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def stable_document_id(
    patient_id: str,
    document_type: str,
    document_date: str,
    source_event_id: str,
    extra: str = "",
) -> str:
    key = f"{patient_id}|{document_type}|{document_date}|{source_event_id}|{extra}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"patient-history-navigator:{key}"))


def relative_to_output(path: Path, output_root: Path) -> str:
    try:
        return path.relative_to(output_root).as_posix()
    except ValueError:
        return path.as_posix()


def event_code_only(code: str) -> str:
    match = re.search(r"([A-Za-z0-9.-]+)\s*$", code or "")
    return match.group(1) if match else code


def make_display_location(patient_id: str) -> tuple[str, str]:
    rng = deterministic_rng(patient_id, "ontario-location")
    city, postal = ONTARIO_LOCATIONS[rng.randrange(len(ONTARIO_LOCATIONS))]
    street_number = rng.randint(10, 999)
    street_name = rng.choice(
        [
            "Maple Avenue",
            "Rideau Street",
            "Lakeshore Drive",
            "King Street",
            "Cedar Road",
            "Wellington Avenue",
            "Bay Street",
            "Laurier Boulevard",
        ]
    )
    address = f"{street_number} {street_name}, {city}, Ontario {postal}"
    facility = SYNTHETIC_FACILITIES[rng.randrange(len(SYNTHETIC_FACILITIES))]
    return address, facility


# ---------------------------------------------------------------------------
# C-CDA parsing
# ---------------------------------------------------------------------------


def parse_table_rows(section: etree._Element, section_title: str) -> list[Event]:
    headers = [
        collapse_text("".join(cell.itertext()))
        for cell in section.xpath(".//cda:thead/cda:tr/cda:th", namespaces=NS)
    ]
    events: list[Event] = []
    for row_index, row in enumerate(
        section.xpath(".//cda:tbody/cda:tr", namespaces=NS), start=1
    ):
        cells = [
            collapse_text("".join(cell.itertext()))
            for cell in row.xpath("./cda:td", namespaces=NS)
        ]
        mapped = {headers[i]: cells[i] for i in range(min(len(headers), len(cells)))}
        start = parse_datetime(mapped.get("Start") or (cells[0] if cells else ""))
        stop = parse_datetime(mapped.get("Stop") or (cells[1] if len(cells) > 1 else ""))
        description = mapped.get("Description") or (cells[2] if len(cells) > 2 else "")
        code = mapped.get("Code") or (cells[3] if len(cells) > 3 else "")
        value = mapped.get("Value") or (cells[4] if len(cells) > 4 else "")
        if not any([start, stop, description, code, value]):
            continue
        source_id = hashlib.sha1(
            f"{section_title}|{row_index}|{start}|{stop}|{description}|{code}|{value}".encode(
                "utf-8"
            )
        ).hexdigest()[:20]
        events.append(
            Event(
                section=section_title,
                start=start,
                stop=stop,
                description=description,
                code=code,
                value=value,
                source_id=source_id,
            )
        )
    return events


def parse_lab_value(observation: etree._Element) -> tuple[str, str]:
    value_node = observation.find(f"{{{CDA_NS}}}value")
    if value_node is None:
        return "", ""
    value = value_node.get("value") or value_node.get("displayName") or collapse_text(
        "".join(value_node.itertext())
    )
    if not value:
        value = value_node.get("code", "")
    return collapse_text(value), collapse_text(value_node.get("unit", ""))


def parse_lab_panels(tree: etree._ElementTree) -> list[LabPanel]:
    panels: list[LabPanel] = []
    organizers = tree.xpath(
        "//cda:section[cda:code/@code='30954-2']/cda:entry/cda:organizer",
        namespaces=NS,
    )
    for index, organizer in enumerate(organizers, start=1):
        panel_id = xpath_string(organizer, "cda:id/@root") or f"panel-{index}"
        panel_name = xpath_string(organizer, "cda:code/@displayName") or "Diagnostic panel"
        panel_code = xpath_string(organizer, "cda:code/@code")
        occurred_at = parse_datetime(
            xpath_string(organizer, "cda:effectiveTime/cda:low/@value")
            or xpath_string(organizer, "cda:effectiveTime/@value")
        )
        status = xpath_string(organizer, "cda:statusCode/@code") or "completed"
        observations: list[LabObservation] = []
        for obs in organizer.xpath("./cda:component/cda:observation", namespaces=NS):
            name = xpath_string(obs, "cda:code/@displayName") or "Observation"
            code = xpath_string(obs, "cda:code/@code")
            value, unit = parse_lab_value(obs)
            observations.append(
                LabObservation(
                    name=name,
                    code=code,
                    value=value,
                    unit=unit,
                    status=xpath_string(obs, "cda:statusCode/@code") or status,
                )
            )
        panels.append(
            LabPanel(
                panel_id=panel_id,
                panel_name=panel_name,
                panel_code=panel_code,
                occurred_at=occurred_at,
                status=status,
                observations=observations,
            )
        )
    return panels


def parse_ccda(path: Path, localize_ontario: bool = True) -> PatientRecord:
    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False)
    tree = etree.parse(str(path), parser)

    patient_id = xpath_string(
        tree, "//cda:recordTarget/cda:patientRole/cda:id/@extension"
    ) or xpath_string(tree, "/cda:ClinicalDocument/cda:id/@extension")
    if not patient_id:
        patient_id = str(uuid.uuid5(uuid.NAMESPACE_URL, sha256_file(path)))

    given_nodes = tree.xpath(
        "//cda:recordTarget/cda:patientRole/cda:patient/cda:name/cda:given",
        namespaces=NS,
    )
    given_names = " ".join(collapse_text("".join(node.itertext())) for node in given_nodes)
    family_name = xpath_string(
        tree,
        "//cda:recordTarget/cda:patientRole/cda:patient/cda:name/cda:family",
    )
    name = collapse_text(f"{given_names} {family_name}") or "Synthetic Patient"
    birth_dt = parse_datetime(
        xpath_string(
            tree,
            "//cda:recordTarget/cda:patientRole/cda:patient/cda:birthTime/@value",
        )
    )
    gender = xpath_string(
        tree,
        "//cda:recordTarget/cda:patientRole/cda:patient/cda:administrativeGenderCode/@code",
    )
    race = xpath_string(
        tree,
        "//cda:recordTarget/cda:patientRole/cda:patient/cda:raceCode/@displayName",
    )
    ethnicity = xpath_string(
        tree,
        "//cda:recordTarget/cda:patientRole/cda:patient/cda:ethnicGroupCode/@displayName",
    )

    address_parts = tree.xpath(
        "//cda:recordTarget/cda:patientRole/cda:addr/*/text()", namespaces=NS
    )
    source_address = collapse_text(", ".join(str(part) for part in address_parts))
    source_organization = xpath_string(
        tree, "//cda:author//cda:representedOrganization/cda:name"
    )
    effective_date = parse_datetime(xpath_string(tree, "/cda:ClinicalDocument/cda:effectiveTime/@value"))

    display_address, display_facility = make_display_location(patient_id)
    if not localize_ontario:
        display_address = source_address or "Address not recorded"
        display_facility = source_organization or "Synthetic Health Organization"

    events_by_section: dict[str, list[Event]] = {}
    for section in tree.xpath(
        "//cda:structuredBody/cda:component/cda:section", namespaces=NS
    ):
        title = xpath_string(section, "cda:title") or "Untitled Section"
        rows = parse_table_rows(section, title)
        if rows:
            events_by_section[title] = rows
        elif section.get("nullFlavor"):
            text = xpath_string(section, "cda:text") or "No information recorded."
            events_by_section[title] = [
                Event(
                    section=title,
                    start=None,
                    stop=None,
                    description=text,
                    source_id=hashlib.sha1(f"{title}|{text}".encode()).hexdigest()[:20],
                    status="not-recorded",
                )
            ]

    display_patient_id = f"ON-SYN-{patient_id.replace('-', '')[:10].upper()}"
    return PatientRecord(
        patient_id=patient_id,
        display_patient_id=display_patient_id,
        name=name,
        given_names=given_names,
        family_name=family_name,
        birth_date=birth_dt.date() if birth_dt else None,
        gender=gender,
        race=race,
        ethnicity=ethnicity,
        source_address=source_address,
        display_address=display_address,
        source_organization=source_organization,
        display_facility=display_facility,
        source_file=path,
        source_sha256=sha256_file(path),
        effective_date=effective_date,
        events_by_section=events_by_section,
        lab_panels=parse_lab_panels(tree),
    )


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------


def register_fonts() -> tuple[str, str]:
    candidates = [
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ),
        (
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ),
    ]
    for regular, bold in candidates:
        if Path(regular).exists() and Path(bold).exists():
            try:
                pdfmetrics.registerFont(TTFont("SyntheticSans", regular))
                pdfmetrics.registerFont(TTFont("SyntheticSans-Bold", bold))
                return "SyntheticSans", "SyntheticSans-Bold"
            except Exception:
                continue
    return "Helvetica", "Helvetica-Bold"


FONT_REGULAR, FONT_BOLD = register_fonts()

PALETTE = {
    "navy": colors.HexColor("#17324D"),
    "blue": colors.HexColor("#2E6F95"),
    "teal": colors.HexColor("#3A8D8C"),
    "light_blue": colors.HexColor("#EAF3F8"),
    "light_teal": colors.HexColor("#EAF6F4"),
    "light_gray": colors.HexColor("#F3F5F7"),
    "mid_gray": colors.HexColor("#D9E0E6"),
    "dark_gray": colors.HexColor("#48545E"),
    "warning": colors.HexColor("#8A4B08"),
    "warning_bg": colors.HexColor("#FFF4E5"),
}


class PDFRenderer:
    def __init__(self) -> None:
        styles = getSampleStyleSheet()
        self.styles = {
            "title": ParagraphStyle(
                "Title",
                parent=styles["Title"],
                fontName=FONT_BOLD,
                fontSize=18,
                leading=22,
                textColor=PALETTE["navy"],
                spaceAfter=5,
            ),
            "subtitle": ParagraphStyle(
                "Subtitle",
                parent=styles["Normal"],
                fontName=FONT_REGULAR,
                fontSize=9.5,
                leading=13,
                textColor=PALETTE["dark_gray"],
                spaceAfter=10,
            ),
            "heading": ParagraphStyle(
                "Heading",
                parent=styles["Heading2"],
                fontName=FONT_BOLD,
                fontSize=11.5,
                leading=14,
                textColor=PALETTE["navy"],
                spaceBefore=8,
                spaceAfter=5,
            ),
            "body": ParagraphStyle(
                "Body",
                parent=styles["BodyText"],
                fontName=FONT_REGULAR,
                fontSize=9,
                leading=13,
                textColor=colors.HexColor("#25313B"),
                spaceAfter=5,
            ),
            "small": ParagraphStyle(
                "Small",
                parent=styles["BodyText"],
                fontName=FONT_REGULAR,
                fontSize=7.6,
                leading=10.5,
                textColor=PALETTE["dark_gray"],
            ),
            "small_bold": ParagraphStyle(
                "SmallBold",
                parent=styles["BodyText"],
                fontName=FONT_BOLD,
                fontSize=7.6,
                leading=10.5,
                textColor=PALETTE["navy"],
            ),
            "table_header": ParagraphStyle(
                "TableHeader",
                parent=styles["BodyText"],
                fontName=FONT_BOLD,
                fontSize=7.6,
                leading=10.0,
                textColor=colors.white,
            ),
            "callout": ParagraphStyle(
                "Callout",
                parent=styles["BodyText"],
                fontName=FONT_REGULAR,
                fontSize=8,
                leading=11,
                textColor=PALETTE["warning"],
                borderColor=colors.HexColor("#F0C36D"),
                borderWidth=0.6,
                borderPadding=7,
                backColor=PALETTE["warning_bg"],
                spaceBefore=5,
                spaceAfter=8,
            ),
            "center": ParagraphStyle(
                "Center",
                parent=styles["BodyText"],
                fontName=FONT_REGULAR,
                fontSize=8,
                alignment=TA_CENTER,
                leading=11,
            ),
        }

    def p(self, value: Any, style: str = "body") -> Paragraph:
        text = html.escape(collapse_text(str(value))) if value is not None else ""
        return Paragraph(text or "&nbsp;", self.styles[style])

    def table(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[Any]],
        widths: Optional[Sequence[float]] = None,
        font_size: float = 7.6,
        alignments: Optional[dict[int, str]] = None,
    ) -> Table:
        data: list[list[Any]] = [
            [self.p(header, "table_header") for header in headers]
        ]
        for row in rows:
            data.append([self.p(cell, "small") for cell in row])
        table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        commands: list[tuple[Any, ...]] = [
            ("BACKGROUND", (0, 0), (-1, 0), PALETTE["navy"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
            ("FONTNAME", (0, 1), (-1, -1), FONT_REGULAR),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("GRID", (0, 0), (-1, -1), 0.35, PALETTE["mid_gray"]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        for row_index in range(1, len(data)):
            if row_index % 2 == 0:
                commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PALETTE["light_gray"]))
        for index, alignment in (alignments or {}).items():
            commands.append(("ALIGN", (index, 1), (index, -1), alignment))
        table.setStyle(TableStyle(commands))
        return table

    def key_value_table(self, pairs: Sequence[tuple[str, Any]], width: float) -> Table:
        rows = []
        for key, value in pairs:
            rows.append([self.p(key, "small_bold"), self.p(value, "small")])
        table = Table(rows, colWidths=[width * 0.28, width * 0.72], hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), PALETTE["light_blue"]),
                    ("GRID", (0, 0), (-1, -1), 0.35, PALETTE["mid_gray"]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return table

    def patient_banner(self, patient: PatientRecord, record_date: str, width: float) -> Table:
        gender_map = {"M": "Male", "F": "Female", "UN": "Unknown"}
        cells = [
            [
                self.p("Patient", "small_bold"),
                self.p(patient.name, "small"),
                self.p("Synthetic Patient ID", "small_bold"),
                self.p(patient.display_patient_id, "small"),
            ],
            [
                self.p("Date of birth", "small_bold"),
                self.p(human_date(patient.birth_date), "small"),
                self.p("Gender", "small_bold"),
                self.p(gender_map.get(patient.gender, patient.gender or "Not recorded"), "small"),
            ],
            [
                self.p("Facility", "small_bold"),
                self.p(patient.display_facility, "small"),
                self.p("Document date", "small_bold"),
                self.p(record_date, "small"),
            ],
        ]
        table = Table(cells, colWidths=[width * 0.16, width * 0.34, width * 0.18, width * 0.32])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), PALETTE["light_blue"]),
                    ("GRID", (0, 0), (-1, -1), 0.35, PALETTE["mid_gray"]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return table

    def write(
        self,
        output_path: Path,
        patient: PatientRecord,
        document_id: str,
        title: str,
        subtitle: str,
        document_date: str,
        blocks: Sequence[dict[str, Any]],
        provenance: str,
    ) -> str:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=LETTER,
            rightMargin=0.55 * inch,
            leftMargin=0.55 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.6 * inch,
            title=title,
            author=BRAND,
            subject=SYNTHETIC_NOTICE,
        )
        story: list[Any] = [
            self.p(title, "title"),
            self.p(subtitle, "subtitle"),
            self.patient_banner(patient, document_date, doc.width),
            Spacer(1, 7),
            Paragraph(SYNTHETIC_NOTICE, self.styles["callout"]),
        ]
        full_text_parts = [title, subtitle, SYNTHETIC_NOTICE]

        for block in blocks:
            kind = block["kind"]
            if kind == "heading":
                story.append(self.p(block["text"], "heading"))
                full_text_parts.append(block["text"])
            elif kind == "paragraph":
                story.append(self.p(block["text"], block.get("style", "body")))
                full_text_parts.append(collapse_text(str(block["text"])))
            elif kind == "callout":
                story.append(Paragraph(html.escape(collapse_text(str(block["text"]))), self.styles["callout"]))
                full_text_parts.append(collapse_text(str(block["text"])))
            elif kind == "spacer":
                story.append(Spacer(1, block.get("height", 6)))
            elif kind == "pagebreak":
                story.append(PageBreak())
            elif kind == "kv":
                story.append(self.key_value_table(block["pairs"], doc.width))
                for key, value in block["pairs"]:
                    full_text_parts.append(f"{key}: {collapse_text(str(value))}")
            elif kind == "table":
                widths = block.get("widths")
                if widths and max(widths) <= 1.0:
                    widths = [doc.width * item for item in widths]
                story.append(
                    self.table(
                        block["headers"],
                        block["rows"],
                        widths=widths,
                        font_size=block.get("font_size", 7.6),
                        alignments=block.get("alignments"),
                    )
                )
                full_text_parts.append(" | ".join(block["headers"]))
                full_text_parts.extend(
                    " | ".join(collapse_text(str(cell)) for cell in row)
                    for row in block["rows"]
                )
            else:
                raise ValueError(f"Unsupported block kind: {kind}")
            story.append(Spacer(1, 3))

        def draw_page(canvas: Canvas, pdf_doc: SimpleDocTemplate) -> None:
            canvas.saveState()
            canvas.setTitle(title)
            canvas.setAuthor(BRAND)
            canvas.setSubject(SYNTHETIC_NOTICE)
            canvas.setKeywords(
                f"synthetic,patient_id={patient.patient_id},document_id={document_id},provenance={provenance}"
            )

            # Header line and brand.
            canvas.setStrokeColor(PALETTE["teal"])
            canvas.setLineWidth(1.2)
            canvas.line(0.55 * inch, LETTER[1] - 0.43 * inch, LETTER[0] - 0.55 * inch, LETTER[1] - 0.43 * inch)
            canvas.setFont(FONT_BOLD, 8)
            canvas.setFillColor(PALETTE["navy"])
            canvas.drawString(0.55 * inch, LETTER[1] - 0.34 * inch, BRAND)
            canvas.setFont(FONT_REGULAR, 6.8)
            canvas.setFillColor(PALETTE["dark_gray"])
            canvas.drawRightString(
                LETTER[0] - 0.55 * inch,
                LETTER[1] - 0.34 * inch,
                f"Patient: {patient.display_patient_id}",
            )

            # Watermark.
            canvas.setFillColor(colors.Color(0.75, 0.78, 0.80, alpha=0.18))
            canvas.setFont(FONT_BOLD, 32)
            canvas.translate(LETTER[0] / 2, LETTER[1] / 2)
            canvas.rotate(35)
            canvas.drawCentredString(0, 0, "SYNTHETIC DEMO")
            canvas.rotate(-35)
            canvas.translate(-LETTER[0] / 2, -LETTER[1] / 2)

            # Footer.
            canvas.setStrokeColor(PALETTE["mid_gray"])
            canvas.setLineWidth(0.5)
            canvas.line(0.55 * inch, 0.42 * inch, LETTER[0] - 0.55 * inch, 0.42 * inch)
            canvas.setFont(FONT_REGULAR, 6.5)
            canvas.setFillColor(PALETTE["dark_gray"])
            canvas.drawString(0.55 * inch, 0.27 * inch, f"Doc: {document_id[:8]}...{document_id[-6:]}")
            canvas.drawCentredString(LETTER[0] / 2, 0.27 * inch, SYNTHETIC_NOTICE)
            canvas.drawRightString(
                LETTER[0] - 0.55 * inch,
                0.27 * inch,
                f"Page {pdf_doc.page}",
            )
            canvas.restoreState()

        doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
        return "\n".join(part for part in full_text_parts if part)


# ---------------------------------------------------------------------------
# Document generation helpers
# ---------------------------------------------------------------------------


def select_recent(events: Sequence[Event], limit: int) -> list[Event]:
    ordered = sorted(events, key=lambda item: date_sort_key(item.event_date), reverse=True)
    if limit <= 0:
        return ordered
    return ordered[:limit]


def document_output_path(
    patient_dir: Path,
    patient: PatientRecord,
    document_type: str,
    document_date: str,
    document_id: str,
) -> Path:
    folder = SECTION_FOLDER.get(document_type, "other")
    filename = f"{patient.patient_id}__{document_date}__{document_type}__{document_id}.pdf"
    return patient_dir / "documents" / folder / filename


def write_sidecar(pdf_path: Path, metadata: GeneratedDocument) -> None:
    sidecar_path = pdf_path.with_suffix(pdf_path.suffix + ".metadata.json")
    sidecar_path.write_text(
        json.dumps(asdict(metadata), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def make_doc(
    renderer: PDFRenderer,
    output_root: Path,
    patient_dir: Path,
    patient: PatientRecord,
    document_type: str,
    document_date: str,
    section: str,
    title: str,
    subtitle: str,
    blocks: Sequence[dict[str, Any]],
    event_name: str = "",
    status: str = "completed",
    source_event_id: str = "",
    comparison_key: str = "",
    provenance: str = "source_ccda",
    tags: Optional[list[str]] = None,
    extra_id: str = "",
) -> GeneratedDocument:
    document_id = stable_document_id(
        patient.patient_id,
        document_type,
        document_date,
        source_event_id,
        extra=extra_id,
    )
    pdf_path = document_output_path(
        patient_dir, patient, document_type, document_date, document_id
    )
    full_text = renderer.write(
        pdf_path,
        patient,
        document_id,
        title,
        subtitle,
        human_date(parse_datetime(document_date)) if document_date != "undated" else "Undated",
        blocks,
        provenance,
    )
    metadata = GeneratedDocument(
        document_id=document_id,
        patient_id=patient.patient_id,
        display_patient_id=patient.display_patient_id,
        document_type=document_type,
        document_date=document_date,
        section=section,
        title=title,
        event_name=event_name or title,
        status=status,
        source_path=relative_to_output(pdf_path, output_root),
        source_record_path=relative_to_output(
            patient_dir / "source" / patient.source_file.name, output_root
        ),
        output_path=relative_to_output(pdf_path, output_root),
        provenance=provenance,
        source_event_id=source_event_id,
        comparison_key=comparison_key,
        synthetic=True,
        full_text=full_text,
        tags=tags or [],
    )
    write_sidecar(pdf_path, metadata)
    return metadata


def events_for(patient: PatientRecord, *names: str) -> list[Event]:
    result: list[Event] = []
    lowered = [name.lower() for name in names]
    for section_name, events in patient.events_by_section.items():
        if section_name.lower() in lowered:
            result.extend(events)
    return result


def active_medications(patient: PatientRecord) -> list[Event]:
    meds = events_for(patient, "Medications")
    return [event for event in meds if not event.stop]


def latest_labs(patient: PatientRecord, limit: int = 8) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for panel in sorted(
        patient.lab_panels,
        key=lambda item: date_sort_key(item.occurred_at),
        reverse=True,
    ):
        for obs in panel.observations:
            rows.append((human_date(panel.occurred_at), obs.name, f"{obs.value} {obs.unit}".strip()))
            if len(rows) >= limit:
                return rows
    return rows


def nearby_events(patient: PatientRecord, when: Optional[datetime], days: int = 1) -> list[Event]:
    if when is None:
        return []
    result = []
    for event in patient.all_events():
        if event.event_date and abs((event.event_date.date() - when.date()).days) <= days:
            result.append(event)
    return sorted(result, key=lambda item: (item.section, item.description))


def build_base_documents(
    renderer: PDFRenderer,
    output_root: Path,
    patient_dir: Path,
    patient: PatientRecord,
) -> list[GeneratedDocument]:
    generated: list[GeneratedDocument] = []
    doc_date = iso_date(patient.effective_date)

    # Patient summary.
    problem_rows = [
        [human_date(e.start), e.description, status_for_event(e), event_code_only(e.code)]
        for e in select_recent(events_for(patient, "Problems"), 12)
    ]
    medication_rows = [
        [human_date(e.start), e.description, status_for_event(e), event_code_only(e.code)]
        for e in select_recent(events_for(patient, "Medications"), 12)
    ]
    allergy_events = events_for(patient, "Allergies", "Allergies and Adverse Reactions")
    allergy_rows = [
        [human_date(e.start), e.description, status_for_event(e), event_code_only(e.code)]
        for e in allergy_events[:10]
    ]
    encounter_rows = [
        [human_date(e.start), e.description, human_date(e.stop), event_code_only(e.code)]
        for e in select_recent(events_for(patient, "Encounters"), 10)
    ]
    lab_rows = [list(row) for row in latest_labs(patient, 10)]
    blocks: list[dict[str, Any]] = [
        {"kind": "heading", "text": "Patient identity and source"},
        {
            "kind": "kv",
            "pairs": [
                ("Internal patient_id", patient.patient_id),
                ("Synthetic display ID", patient.display_patient_id),
                ("Localized address", patient.display_address),
                ("Source C-CDA file", patient.source_file.name),
                ("Source SHA-256", patient.source_sha256),
                ("Presentation jurisdiction", "Ontario, Canada - synthetic localization"),
            ],
        },
        {"kind": "heading", "text": "Recent recorded problems"},
        {
            "kind": "table",
            "headers": ["Start", "Description", "Status", "Code"],
            "rows": problem_rows or [["-", "No problems recorded", "-", "-"]],
            "widths": [0.14, 0.55, 0.13, 0.18],
        },
        {"kind": "heading", "text": "Medication history snapshot"},
        {
            "kind": "table",
            "headers": ["Start", "Medication", "Status", "Code"],
            "rows": medication_rows or [["-", "No medications recorded", "-", "-"]],
            "widths": [0.14, 0.55, 0.13, 0.18],
        },
        {"kind": "heading", "text": "Allergies and adverse reactions"},
        {
            "kind": "table",
            "headers": ["Start", "Description", "Status", "Code"],
            "rows": allergy_rows or [["-", "No current allergies recorded", "-", "-"]],
            "widths": [0.14, 0.55, 0.13, 0.18],
        },
        {"kind": "heading", "text": "Latest laboratory observations"},
        {
            "kind": "table",
            "headers": ["Date", "Observation", "Result"],
            "rows": lab_rows or [["-", "No structured laboratory observations found", "-"]],
            "widths": [0.17, 0.58, 0.25],
        },
        {"kind": "heading", "text": "Recent encounters"},
        {
            "kind": "table",
            "headers": ["Start", "Encounter", "End", "Code"],
            "rows": encounter_rows or [["-", "No encounters recorded", "-", "-"]],
            "widths": [0.14, 0.50, 0.14, 0.22],
        },
        {
            "kind": "callout",
            "text": "This summary is a deterministic presentation of the supplied synthetic C-CDA record. It does not diagnose, prescribe, recommend treatment, or establish clinical significance.",
        },
    ]
    generated.append(
        make_doc(
            renderer,
            output_root,
            patient_dir,
            patient,
            "patient_summary",
            doc_date,
            "Summary",
            "Synthetic Longitudinal Patient Summary",
            "A source-grounded overview for retrieval and demonstration workflows",
            blocks,
            event_name="Patient summary",
            source_event_id=patient.source_sha256[:20],
            comparison_key="patient-summary",
            tags=["summary", "demographics", "longitudinal"],
        )
    )

    # Longitudinal timeline.
    timeline_rows: list[list[str]] = []
    timeline_events = sorted(
        [event for event in patient.all_events() if event.event_date],
        key=lambda event: date_sort_key(event.event_date),
    )
    for event in timeline_events:
        timeline_rows.append(
            [
                human_date(event.start),
                event.section,
                event.description,
                event.value,
                status_for_event(event),
            ]
        )
    generated.append(
        make_doc(
            renderer,
            output_root,
            patient_dir,
            patient,
            "longitudinal_timeline",
            doc_date,
            "Timeline",
            "Longitudinal Record Timeline",
            "Chronological events extracted from the supplied synthetic C-CDA",
            [
                {
                    "kind": "table",
                    "headers": ["Date", "Section", "Event", "Value", "Status"],
                    "rows": timeline_rows or [["-", "-", "No dated events found", "-", "-"]],
                    "widths": [0.12, 0.16, 0.46, 0.14, 0.12],
                    "font_size": 7.2,
                }
            ],
            event_name="Longitudinal patient timeline",
            source_event_id=patient.source_sha256[:20],
            comparison_key="timeline",
            tags=["timeline", "all-events"],
        )
    )

    # Problem list.
    all_problems = sorted(events_for(patient, "Problems"), key=lambda e: date_sort_key(e.start))
    generated.append(
        make_doc(
            renderer,
            output_root,
            patient_dir,
            patient,
            "problem_list",
            doc_date,
            "Problems",
            "Problem and Condition History",
            "Recorded problem-list entries, without diagnostic interpretation",
            [
                {
                    "kind": "table",
                    "headers": ["Start", "Stop", "Description", "Code", "Status"],
                    "rows": [
                        [
                            human_date(e.start),
                            human_date(e.stop) if e.stop else "Open",
                            e.description,
                            event_code_only(e.code),
                            status_for_event(e),
                        ]
                        for e in all_problems
                    ]
                    or [["-", "-", "No problem-list entries", "-", "-"]],
                    "widths": [0.12, 0.12, 0.48, 0.16, 0.12],
                }
            ],
            event_name="Problem list",
            source_event_id=patient.source_sha256[:20],
            comparison_key="problem-list",
            tags=["conditions", "problems"],
        )
    )

    # Medication history.
    all_meds = sorted(events_for(patient, "Medications"), key=lambda e: date_sort_key(e.start))
    generated.append(
        make_doc(
            renderer,
            output_root,
            patient_dir,
            patient,
            "medication_history",
            doc_date,
            "Medications",
            "Medication Start and Stop History",
            "Medication entries preserved from the synthetic source record",
            [
                {
                    "kind": "table",
                    "headers": ["Start", "Stop", "Medication", "Code", "Status"],
                    "rows": [
                        [
                            human_date(e.start),
                            human_date(e.stop) if e.stop else "Ongoing in source",
                            e.description,
                            event_code_only(e.code),
                            status_for_event(e),
                        ]
                        for e in all_meds
                    ]
                    or [["-", "-", "No medications recorded", "-", "-"]],
                    "widths": [0.12, 0.14, 0.48, 0.14, 0.12],
                },
                {
                    "kind": "callout",
                    "text": "Medication directions, dispensing quantities, and refill instructions are not inferred when they are absent from the C-CDA source.",
                },
            ],
            event_name="Medication history",
            source_event_id=patient.source_sha256[:20],
            comparison_key="medications",
            tags=["medications", "starts", "stops"],
        )
    )

    # Allergy report.
    generated.append(
        make_doc(
            renderer,
            output_root,
            patient_dir,
            patient,
            "allergy_report",
            doc_date,
            "Allergies",
            "Allergy and Adverse Reaction Record",
            "Synthetic source entries; absence of entries does not imply clinical clearance",
            [
                {
                    "kind": "table",
                    "headers": ["Start", "Stop", "Description", "Code", "Status"],
                    "rows": [
                        [
                            human_date(e.start),
                            human_date(e.stop) if e.stop else "Open / not supplied",
                            e.description,
                            event_code_only(e.code),
                            status_for_event(e),
                        ]
                        for e in allergy_events
                    ]
                    or [["-", "-", "No allergy entries found", "-", "-"]],
                    "widths": [0.12, 0.15, 0.45, 0.16, 0.12],
                }
            ],
            event_name="Allergy record",
            source_event_id=patient.source_sha256[:20],
            comparison_key="allergies",
            tags=["allergies", "adverse-reactions"],
        )
    )

    # Immunizations.
    immunizations = sorted(events_for(patient, "Immunizations"), key=lambda e: date_sort_key(e.start))
    if immunizations:
        generated.append(
            make_doc(
                renderer,
                output_root,
                patient_dir,
                patient,
                "immunization_history",
                doc_date,
                "Immunizations",
                "Immunization History",
                "Recorded synthetic immunization events",
                [
                    {
                        "kind": "table",
                        "headers": ["Date", "Immunization", "Code", "Status"],
                        "rows": [
                            [human_date(e.start), e.description, event_code_only(e.code), status_for_event(e)]
                            for e in immunizations
                        ],
                        "widths": [0.16, 0.55, 0.17, 0.12],
                    }
                ],
                event_name="Immunization history",
                source_event_id=patient.source_sha256[:20],
                comparison_key="immunizations",
                tags=["immunizations"],
            )
        )

    # Care plan.
    care_events = sorted(events_for(patient, "Plan of Care"), key=lambda e: date_sort_key(e.start))
    if care_events:
        generated.append(
            make_doc(
                renderer,
                output_root,
                patient_dir,
                patient,
                "care_plan",
                doc_date,
                "Care plan",
                "Recorded Plans and Follow-up Items",
                "Source-grounded plan entries for follow-up retrieval demonstrations",
                [
                    {
                        "kind": "table",
                        "headers": ["Start", "Stop", "Plan", "Code", "Status"],
                        "rows": [
                            [
                                human_date(e.start),
                                human_date(e.stop) if e.stop else "Open",
                                e.description,
                                event_code_only(e.code),
                                status_for_event(e),
                            ]
                            for e in care_events
                        ],
                        "widths": [0.13, 0.13, 0.47, 0.15, 0.12],
                    }
                ],
                event_name="Care plan",
                source_event_id=patient.source_sha256[:20],
                comparison_key="care-plan",
                tags=["care-plan", "follow-up"],
            )
        )

    return generated


def build_lab_documents(
    renderer: PDFRenderer,
    output_root: Path,
    patient_dir: Path,
    patient: PatientRecord,
    max_labs: int,
) -> list[GeneratedDocument]:
    generated: list[GeneratedDocument] = []
    panels = sorted(patient.lab_panels, key=lambda p: date_sort_key(p.occurred_at), reverse=True)
    if max_labs > 0:
        panels = panels[:max_labs]
    for panel in panels:
        document_date = iso_date(panel.occurred_at)
        is_augmented = panel.provenance == "synthetic_augmentation"
        subtitle = (
            "Fictional structured observations created only for RAG comparison and trend demonstrations"
            if is_augmented
            else "Structured observations extracted from the synthetic C-CDA diagnostic organizer"
        )
        reference_note = (
            "Synthetic demo intervals are shown in the results table"
            if is_augmented
            else "Not supplied by source C-CDA"
        )
        interpretation_note = (
            "No clinical interpretation is generated. Values and intervals in this augmented report are fictional demonstration data."
            if is_augmented
            else "No clinical interpretation is generated. Units and values are reproduced from synthetic source data."
        )
        rows = [
            [
                observation.name,
                observation.code,
                observation.value,
                observation.unit,
                observation.reference or "Not supplied",
                observation.flag,
            ]
            for observation in panel.observations
        ]
        generated.append(
            make_doc(
                renderer,
                output_root,
                patient_dir,
                patient,
                "laboratory_report",
                document_date,
                "Diagnostic Results",
                panel.panel_name,
                subtitle,
                [
                    {
                        "kind": "kv",
                        "pairs": [
                            ("Panel code", panel.panel_code or "Not supplied"),
                            ("Panel status", panel.status),
                            ("Source panel ID", panel.panel_id),
                            ("Collection / result date", human_date(panel.occurred_at)),
                            ("Reference intervals", reference_note),
                        ],
                    },
                    {
                        "kind": "heading",
                        "text": "Results",
                    },
                    {
                        "kind": "table",
                        "headers": ["Test", "Code", "Result", "Unit", "Reference", "Flag"],
                        "rows": rows or [["No component observations", "-", "-", "-", "-", "-"]],
                        "widths": [0.31, 0.13, 0.13, 0.12, 0.22, 0.09],
                        "font_size": 7.1,
                    },
                    {
                        "kind": "callout",
                        "text": interpretation_note,
                    },
                ],
                event_name=panel.panel_name,
                status=panel.status,
                source_event_id=panel.panel_id,
                comparison_key=panel.panel_code or slugify(panel.panel_name),
                provenance=panel.provenance,
                tags=["laboratory", "diagnostic-result", panel.panel_code],
            )
        )
    return generated


def build_prescription_documents(
    renderer: PDFRenderer,
    output_root: Path,
    patient_dir: Path,
    patient: PatientRecord,
    max_prescriptions: int,
) -> list[GeneratedDocument]:
    generated: list[GeneratedDocument] = []
    medications = select_recent(events_for(patient, "Medications"), max_prescriptions)
    for event in medications:
        document_date = iso_date(event.start)
        status = status_for_event(event)
        generated.append(
            make_doc(
                renderer,
                output_root,
                patient_dir,
                patient,
                "prescription_record",
                document_date,
                "Medications",
                "Synthetic Medication Order Record",
                "Timeline-focused representation of a medication entry",
                [
                    {
                        "kind": "kv",
                        "pairs": [
                            ("Medication", event.description),
                            ("Order start", human_date(event.start)),
                            ("Order stop", human_date(event.stop) if event.stop else "No stop date supplied"),
                            ("Status", status),
                            ("Code", event.code or "Not supplied"),
                            ("Synthetic prescriber", f"Clinician {patient.display_patient_id[-4:]}"),
                            ("Directions", "Exact directions were not included in the supplied C-CDA export"),
                            ("Dispensing details", "Not supplied - no quantity or refill instructions inferred"),
                        ],
                    },
                    {
                        "kind": "callout",
                        "text": "This is not a prescription and must never be used to obtain or administer medication. It exists only to demonstrate medication start/stop retrieval.",
                    },
                ],
                event_name=event.description,
                status=status,
                source_event_id=event.source_id,
                comparison_key=event_code_only(event.code) or slugify(event.description),
                tags=["medication", "prescription-record", status],
            )
        )
    return generated


def build_encounter_documents(
    renderer: PDFRenderer,
    output_root: Path,
    patient_dir: Path,
    patient: PatientRecord,
    max_encounters: int,
) -> list[GeneratedDocument]:
    generated: list[GeneratedDocument] = []
    encounters = select_recent(events_for(patient, "Encounters"), max_encounters)
    for encounter in encounters:
        document_date = iso_date(encounter.start)
        related = [e for e in nearby_events(patient, encounter.start, days=1) if e.source_id != encounter.source_id]
        related_rows = [
            [e.section, human_date(e.start), e.description, e.value, status_for_event(e)]
            for e in related[:30]
        ]
        generated.append(
            make_doc(
                renderer,
                output_root,
                patient_dir,
                patient,
                "encounter_note",
                document_date,
                "Encounters",
                "Encounter Record",
                "Deterministic note assembled from same-day synthetic record events",
                [
                    {
                        "kind": "kv",
                        "pairs": [
                            ("Encounter", encounter.description),
                            ("Start", human_date(encounter.start)),
                            ("End", human_date(encounter.stop) if encounter.stop else "Not supplied"),
                            ("Encounter code", encounter.code or "Not supplied"),
                            ("Facility", patient.display_facility),
                            ("Status", status_for_event(encounter)),
                        ],
                    },
                    {"kind": "heading", "text": "Same-day record activity"},
                    {
                        "kind": "table",
                        "headers": ["Section", "Date", "Event", "Value", "Status"],
                        "rows": related_rows or [["-", "-", "No additional same-day events found", "-", "-"]],
                        "widths": [0.16, 0.13, 0.46, 0.14, 0.11],
                    },
                    {
                        "kind": "callout",
                        "text": "The narrative is not a clinician-authored note. It is a deterministic grouping of synthetic entries occurring near the encounter date.",
                    },
                ],
                event_name=encounter.description,
                status=status_for_event(encounter),
                source_event_id=encounter.source_id,
                comparison_key=event_code_only(encounter.code) or slugify(encounter.description),
                tags=["encounter", "timeline"],
            )
        )

        lowered = f" {encounter.description.lower()} "
        if any(keyword in lowered for keyword in INPATIENT_KEYWORDS):
            generated.append(
                make_doc(
                    renderer,
                    output_root,
                    patient_dir,
                    patient,
                    "discharge_summary",
                    document_date,
                    "Encounters",
                    "Synthetic Discharge / Transition Summary",
                    "Record-derived transition document for comparison and follow-up demonstrations",
                    [
                        {
                            "kind": "kv",
                            "pairs": [
                                ("Encounter", encounter.description),
                                ("Admission / start", human_date(encounter.start)),
                                ("Discharge / end", human_date(encounter.stop) if encounter.stop else "Not supplied"),
                                ("Disposition", "Not supplied by source C-CDA"),
                                ("Source encounter code", encounter.code or "Not supplied"),
                            ],
                        },
                        {"kind": "heading", "text": "Recorded events during transition window"},
                        {
                            "kind": "table",
                            "headers": ["Section", "Date", "Recorded event", "Value"],
                            "rows": [[e.section, human_date(e.start), e.description, e.value] for e in related[:35]]
                            or [["-", "-", "No related entries found", "-"]],
                            "widths": [0.18, 0.14, 0.54, 0.14],
                        },
                        {
                            "kind": "callout",
                            "text": "No discharge instructions or clinical recommendations are invented when absent from the source record.",
                        },
                    ],
                    event_name=f"Discharge summary: {encounter.description}",
                    status="completed",
                    source_event_id=encounter.source_id,
                    comparison_key="discharge-summary",
                    tags=["discharge", "transition", "encounter"],
                    extra_id="discharge",
                )
            )

        if any(keyword in lowered for keyword in CONSULT_KEYWORDS):
            generated.append(
                make_doc(
                    renderer,
                    output_root,
                    patient_dir,
                    patient,
                    "specialist_note",
                    document_date,
                    "Encounters",
                    "Synthetic Consultation Note",
                    "Source-grounded consultation context without diagnosis or recommendations",
                    [
                        {
                            "kind": "kv",
                            "pairs": [
                                ("Consultation event", encounter.description),
                                ("Date", human_date(encounter.start)),
                                ("Source code", encounter.code or "Not supplied"),
                                ("Consulting service", "Not specified in source C-CDA"),
                            ],
                        },
                        {"kind": "heading", "text": "Related synthetic record entries"},
                        {
                            "kind": "table",
                            "headers": ["Section", "Date", "Entry", "Status"],
                            "rows": [[e.section, human_date(e.start), e.description, status_for_event(e)] for e in related[:25]]
                            or [["-", "-", "No related entries found", "-"]],
                            "widths": [0.18, 0.15, 0.55, 0.12],
                        },
                    ],
                    event_name=f"Consultation: {encounter.description}",
                    status="completed",
                    source_event_id=encounter.source_id,
                    comparison_key="specialist-note",
                    tags=["consultation", "specialist-note"],
                    extra_id="consultation",
                )
            )
    return generated


def build_vital_documents(
    renderer: PDFRenderer,
    output_root: Path,
    patient_dir: Path,
    patient: PatientRecord,
    max_vitals: int,
) -> list[GeneratedDocument]:
    generated: list[GeneratedDocument] = []
    grouped: dict[str, list[Event]] = defaultdict(list)
    for event in events_for(patient, "Vital Signs"):
        grouped[iso_date(event.start)].append(event)
    dates = sorted(grouped.keys(), reverse=True)
    if max_vitals > 0:
        dates = dates[:max_vitals]
    for document_date in dates:
        events = grouped[document_date]
        source_id = hashlib.sha1(
            "|".join(sorted(e.source_id for e in events)).encode("utf-8")
        ).hexdigest()[:20]
        generated.append(
            make_doc(
                renderer,
                output_root,
                patient_dir,
                patient,
                "vital_signs_report",
                document_date,
                "Vital Signs",
                "Vital Signs and Measurements",
                "Measurements grouped by recorded date",
                [
                    {
                        "kind": "table",
                        "headers": ["Measurement", "Value", "Code", "Status"],
                        "rows": [[e.description, e.value, event_code_only(e.code), status_for_event(e)] for e in events],
                        "widths": [0.47, 0.20, 0.19, 0.14],
                    }
                ],
                event_name="Vital signs",
                source_event_id=source_id,
                comparison_key="vital-signs",
                tags=["vitals", "measurements"],
            )
        )
    return generated


def build_imaging_documents(
    renderer: PDFRenderer,
    output_root: Path,
    patient_dir: Path,
    patient: PatientRecord,
    max_imaging: int,
) -> list[GeneratedDocument]:
    generated: list[GeneratedDocument] = []
    procedures = events_for(patient, "Surgeries")
    candidates = [
        event
        for event in procedures
        if any(keyword in f" {event.description.lower()} " for keyword in IMAGING_KEYWORDS)
    ]
    candidates = select_recent(candidates, max_imaging)
    for event in candidates:
        document_date = iso_date(event.start)
        generated.append(
            make_doc(
                renderer,
                output_root,
                patient_dir,
                patient,
                "diagnostic_imaging_report",
                document_date,
                "Diagnostic Imaging",
                "Diagnostic Imaging Procedure Record",
                "Imaging occurrence extracted from the synthetic C-CDA procedure history",
                [
                    {
                        "kind": "kv",
                        "pairs": [
                            ("Procedure", event.description),
                            ("Procedure date", human_date(event.start)),
                            ("Completion date", human_date(event.stop) if event.stop else "Not supplied"),
                            ("Procedure code", event.code or "Not supplied"),
                            ("Imaging facility", patient.display_facility),
                            ("Image series", "Not included in the source C-CDA export"),
                        ],
                    },
                    {"kind": "heading", "text": "Findings"},
                    {
                        "kind": "paragraph",
                        "text": "The source record confirms that the imaging-related procedure occurred. A detailed radiologist findings narrative was not present in the supplied C-CDA, so no findings are inferred.",
                    },
                    {"kind": "heading", "text": "Impression"},
                    {
                        "kind": "paragraph",
                        "text": "Not available in source record.",
                    },
                    {
                        "kind": "callout",
                        "text": "This document is intentionally honest about missing image findings. Use --augment-missing only when you need clearly labelled fictional comparison cases.",
                    },
                ],
                event_name=event.description,
                status=status_for_event(event),
                source_event_id=event.source_id,
                comparison_key=event_code_only(event.code) or slugify(event.description),
                tags=["imaging", "procedure", "source-grounded"],
            )
        )
    return generated


# ---------------------------------------------------------------------------
# Optional clearly labelled demo augmentation
# ---------------------------------------------------------------------------


def patient_anchor_date(patient: PatientRecord) -> datetime:
    dates = [event.event_date for event in patient.all_events() if event.event_date]
    dates.extend(panel.occurred_at for panel in patient.lab_panels if panel.occurred_at)
    if patient.effective_date:
        dates.append(patient.effective_date)
    return max(dates) if dates else datetime.now(timezone.utc)


def generate_demo_lab_panel(
    patient: PatientRecord,
    panel_name: str,
    occurred_at: datetime,
    sequence: int,
) -> LabPanel:
    rng = deterministic_rng(patient.patient_id, f"augmentation:{panel_name}:{sequence}")
    observations: list[LabObservation] = []
    for name, (low, high, unit, decimals) in DEMO_LAB_PANELS[panel_name].items():
        span = high - low
        value = low + span * rng.uniform(0.15, 0.85)
        # Roughly one observation in the second report is outside the fictional
        # demo interval, giving the comparison tool something to find.
        flag = ""
        if sequence == 2 and rng.random() < 0.22:
            if rng.random() < 0.5:
                value = low - span * rng.uniform(0.04, 0.15)
                flag = "L"
            else:
                value = high + span * rng.uniform(0.04, 0.15)
                flag = "H"
        formatted = f"{value:.{decimals}f}"
        observations.append(
            LabObservation(
                name=name,
                code=f"DEMO-{slugify(name).upper()[:16]}",
                value=formatted,
                unit=unit,
                reference=f"{low:.{decimals}f}-{high:.{decimals}f} (synthetic demo interval)",
                flag=flag,
            )
        )
    if panel_name == "Urinalysis":
        for name, result in QUALITATIVE_URINALYSIS:
            observations.append(
                LabObservation(
                    name=name,
                    code=f"DEMO-{slugify(name).upper()[:16]}",
                    value=result,
                    unit="",
                    reference="Negative (synthetic demo interval)",
                    flag="",
                )
            )
    panel_code = f"DEMO-{slugify(panel_name).upper()}"
    return LabPanel(
        panel_id=str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{patient.patient_id}|augmentation|{panel_name}|{occurred_at.date()}|{sequence}",
            )
        ),
        panel_name=f"{panel_name} - Synthetic Augmentation",
        panel_code=panel_code,
        occurred_at=occurred_at,
        status="completed",
        observations=observations,
        provenance="synthetic_augmentation",
    )


def build_augmented_documents(
    renderer: PDFRenderer,
    output_root: Path,
    patient_dir: Path,
    patient: PatientRecord,
    existing_docs: Sequence[GeneratedDocument],
) -> list[GeneratedDocument]:
    generated: list[GeneratedDocument] = []
    anchor = patient_anchor_date(patient)
    rng = deterministic_rng(patient.patient_id, "augmentation-documents")

    # Add two versions of each missing demo lab family to support trend and
    # document-comparison demonstrations.
    existing_panel_names = " ".join(panel.panel_name.lower() for panel in patient.lab_panels)
    augmented_panels: list[LabPanel] = []
    checks = {
        "Complete Blood Count": ("cbc", "complete blood"),
        "Comprehensive Metabolic Panel": ("metabolic", "chemistry"),
        "Urinalysis": ("urinalysis", "urine"),
    }
    for panel_name, keywords in checks.items():
        if not any(keyword in existing_panel_names for keyword in keywords):
            augmented_panels.append(generate_demo_lab_panel(patient, panel_name, anchor - timedelta(days=180), 1))
            augmented_panels.append(generate_demo_lab_panel(patient, panel_name, anchor - timedelta(days=30), 2))

    if augmented_panels:
        augmented_patient = copy.copy(patient)
        augmented_patient.events_by_section = patient.events_by_section
        augmented_patient.lab_panels = augmented_panels
        generated.extend(
            build_lab_documents(
                renderer,
                output_root,
                patient_dir,
                augmented_patient,
                max_labs=0,
            )
        )

    existing_types = {doc.document_type for doc in existing_docs}

    # Add a fictional imaging report only when the source has no imaging report.
    if "diagnostic_imaging_report" not in existing_types:
        occurred = anchor - timedelta(days=rng.randint(40, 120))
        scenario = rng.choice(
            [
                (
                    "Chest radiograph - two views",
                    "Cardiomediastinal silhouette is within the synthetic expected range. No focal air-space opacity or pleural fluid is represented in this fictional demonstration.",
                    "No acute synthetic chest finding.",
                ),
                (
                    "Right knee radiograph - three views",
                    "Alignment is maintained in this fictional report. No displaced fracture is represented. Mild synthetic joint-space narrowing is noted for comparison demonstrations.",
                    "Mild synthetic degenerative change; no acute synthetic osseous finding.",
                ),
            ]
        )
        generated.append(
            make_doc(
                renderer,
                output_root,
                patient_dir,
                patient,
                "diagnostic_imaging_report",
                iso_date(occurred),
                "Diagnostic Imaging",
                f"{scenario[0]} - Synthetic Augmentation",
                "Fictional narrative created only to exercise imaging retrieval and document comparison",
                [
                    {
                        "kind": "kv",
                        "pairs": [
                            ("Study", scenario[0]),
                            ("Study date", human_date(occurred)),
                            ("Accession", f"SYN-{patient.display_patient_id[-6:]}-{occurred.strftime('%Y%m%d')}"),
                            ("Modality", "Synthetic radiography"),
                            ("Image availability", "Illustrative report only; no diagnostic image included"),
                        ],
                    },
                    {"kind": "heading", "text": "Findings"},
                    {"kind": "paragraph", "text": scenario[1]},
                    {"kind": "heading", "text": "Impression"},
                    {"kind": "paragraph", "text": scenario[2]},
                    {
                        "kind": "callout",
                        "text": "Every finding in this report is fictional augmentation and is not derived from the source patient record.",
                    },
                ],
                event_name=scenario[0],
                status="completed",
                source_event_id=f"augmentation-imaging-{iso_date(occurred)}",
                comparison_key=slugify(scenario[0]),
                provenance="synthetic_augmentation",
                tags=["imaging", "augmentation", "fictional"],
            )
        )

    # Add a fictional specialist follow-up note if the record has none.
    if "specialist_note" not in existing_types:
        occurred = anchor - timedelta(days=rng.randint(15, 70))
        generated.append(
            make_doc(
                renderer,
                output_root,
                patient_dir,
                patient,
                "specialist_note",
                iso_date(occurred),
                "Consultation",
                "Synthetic Specialist Follow-up Note - Augmented",
                "Fictional follow-up document for open-action retrieval demonstrations",
                [
                    {
                        "kind": "kv",
                        "pairs": [
                            ("Service", rng.choice(["Internal Medicine", "Cardiology", "Orthopaedics", "Endocrinology"])),
                            ("Visit date", human_date(occurred)),
                            ("Reason", "Review of longitudinal synthetic record and previously recorded results"),
                            ("Status", "Completed synthetic visit"),
                        ],
                    },
                    {"kind": "heading", "text": "Record review"},
                    {
                        "kind": "paragraph",
                        "text": "The synthetic specialist reviewed available medication history, recent laboratory reports, and the most recent encounter summary. No diagnosis or treatment recommendation is generated by this demo pipeline.",
                    },
                    {"kind": "heading", "text": "Documented follow-up action"},
                    {
                        "kind": "paragraph",
                        "text": "Repeat the synthetic laboratory panel and arrange a record-review follow-up in approximately eight weeks. This action exists only to test follow-up retrieval.",
                    },
                    {
                        "kind": "callout",
                        "text": "Fictional augmentation: the source C-CDA did not contain this consultation note.",
                    },
                ],
                event_name="Synthetic specialist follow-up",
                status="completed",
                source_event_id=f"augmentation-specialist-{iso_date(occurred)}",
                comparison_key="specialist-follow-up",
                provenance="synthetic_augmentation",
                tags=["specialist-note", "follow-up", "augmentation"],
            )
        )

    # Add a second summary-like transition document when no discharge document
    # exists, giving compare_documents a realistic narrative pair.
    if "discharge_summary" not in existing_types:
        occurred = anchor - timedelta(days=rng.randint(5, 40))
        generated.append(
            make_doc(
                renderer,
                output_root,
                patient_dir,
                patient,
                "discharge_summary",
                iso_date(occurred),
                "Transition of Care",
                "Synthetic Short-Stay Discharge Summary - Augmented",
                "Fictional narrative for timeline, follow-up, and document-difference testing",
                [
                    {
                        "kind": "kv",
                        "pairs": [
                            ("Encounter type", "Synthetic short observation stay"),
                            ("Admission", human_date(occurred - timedelta(days=1))),
                            ("Discharge", human_date(occurred)),
                            ("Disposition", "Synthetic discharge to home"),
                            ("Primary purpose", "Record-review and observation scenario"),
                        ],
                    },
                    {"kind": "heading", "text": "Summary"},
                    {
                        "kind": "paragraph",
                        "text": "The synthetic patient completed a brief observation encounter. The demonstration record contains no clinical decision support and no treatment recommendation.",
                    },
                    {"kind": "heading", "text": "Outstanding follow-up"},
                    {
                        "kind": "paragraph",
                        "text": "A synthetic follow-up record review is marked as pending. The purpose is to provide an unresolved action for RAG evaluation.",
                    },
                    {
                        "kind": "callout",
                        "text": "Fictional augmentation: this transition summary is not present in the supplied C-CDA.",
                    },
                ],
                event_name="Synthetic discharge summary",
                status="completed",
                source_event_id=f"augmentation-discharge-{iso_date(occurred)}",
                comparison_key="discharge-summary",
                provenance="synthetic_augmentation",
                tags=["discharge", "follow-up", "augmentation"],
            )
        )

    return generated


# ---------------------------------------------------------------------------
# Corpus generation and optional Synthea execution
# ---------------------------------------------------------------------------


def run_synthea(
    synthea_dir: Path,
    population: int,
    seed: int,
    work_dir: Path,
) -> Path:
    runner = synthea_dir / ("run_synthea.bat" if sys.platform.startswith("win") else "run_synthea")
    if not runner.exists():
        raise FileNotFoundError(f"Synthea runner not found: {runner}")
    base_dir = work_dir / "synthea_output"
    base_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(runner),
        "-p",
        str(population),
        "-s",
        str(seed),
        "--exporter.ccda.export=true",
        "--exporter.fhir.export=true",
        f"--exporter.baseDirectory={base_dir}",
        "Massachusetts",
    ]
    stderr("Running Synthea: " + " ".join(command))
    subprocess.run(command, cwd=synthea_dir, check=True)
    ccda_dir = base_dir / "ccda"
    if not ccda_dir.exists():
        raise RuntimeError(f"Synthea completed but no C-CDA folder was found at {ccda_dir}")
    return ccda_dir


def write_patient_manifest(
    patient_dir: Path,
    patient: PatientRecord,
    documents: Sequence[GeneratedDocument],
    output_root: Path,
) -> None:
    manifest = {
        "schema_version": APP_VERSION,
        "synthetic": True,
        "patient_id": patient.patient_id,
        "display_patient_id": patient.display_patient_id,
        "patient_name": patient.name,
        "birth_date": patient.birth_date.isoformat() if patient.birth_date else None,
        "gender": patient.gender,
        "race": patient.race,
        "ethnicity": patient.ethnicity,
        "display_address": patient.display_address,
        "display_facility": patient.display_facility,
        "localization": "Ontario, Canada - presentation layer only",
        "source": {
            "file_name": patient.source_file.name,
            "source_path": str(patient.source_file),
            "sha256": patient.source_sha256,
            "source_organization": patient.source_organization,
            "source_address": patient.source_address,
        },
        "privacy_boundary": {
            "required_filter": "patient_id",
            "instruction": "Apply patient_id before retrieval; never search all patients and filter after ranking.",
        },
        "documents": [
            {
                "document_id": doc.document_id,
                "document_type": doc.document_type,
                "document_date": doc.document_date,
                "output_path": doc.output_path,
                "provenance": doc.provenance,
                "comparison_key": doc.comparison_key,
            }
            for doc in documents
        ],
    }
    (patient_dir / "patient_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def generate_patient(
    renderer: PDFRenderer,
    output_root: Path,
    xml_path: Path,
    args: argparse.Namespace,
) -> tuple[PatientRecord, list[GeneratedDocument]]:
    patient = parse_ccda(xml_path, localize_ontario=not args.keep_source_location)
    patient_dir = output_root / f"patient_{patient.patient_id}"
    source_dir = patient_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(xml_path, source_dir / xml_path.name)

    documents: list[GeneratedDocument] = []
    documents.extend(build_base_documents(renderer, output_root, patient_dir, patient))
    documents.extend(
        build_lab_documents(
            renderer, output_root, patient_dir, patient, max_labs=args.max_labs
        )
    )
    documents.extend(
        build_prescription_documents(
            renderer,
            output_root,
            patient_dir,
            patient,
            max_prescriptions=args.max_prescriptions,
        )
    )
    documents.extend(
        build_encounter_documents(
            renderer,
            output_root,
            patient_dir,
            patient,
            max_encounters=args.max_encounters,
        )
    )
    documents.extend(
        build_vital_documents(
            renderer,
            output_root,
            patient_dir,
            patient,
            max_vitals=args.max_vitals,
        )
    )
    documents.extend(
        build_imaging_documents(
            renderer,
            output_root,
            patient_dir,
            patient,
            max_imaging=args.max_imaging,
        )
    )

    if args.augment_missing:
        documents.extend(
            build_augmented_documents(
                renderer, output_root, patient_dir, patient, documents
            )
        )

    write_patient_manifest(patient_dir, patient, documents, output_root)
    return patient, documents


def generate_corpus(args: argparse.Namespace) -> None:
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.run_synthea_dir:
        input_dir = run_synthea(
            Path(args.run_synthea_dir).expanduser().resolve(),
            args.synthea_population,
            args.synthea_seed,
            output_root / "_generation_work",
        )
    elif args.input_dir:
        input_dir = Path(args.input_dir).expanduser().resolve()
    else:
        raise ValueError("Provide --input-dir or --run-synthea-dir")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    xml_files = sorted(input_dir.glob(args.pattern))
    if not xml_files:
        raise FileNotFoundError(
            f"No XML files matched {args.pattern!r} in {input_dir}"
        )

    renderer = PDFRenderer()
    all_documents: list[GeneratedDocument] = []
    patients: list[PatientRecord] = []
    failures: list[dict[str, str]] = []

    for index, xml_path in enumerate(xml_files, start=1):
        stderr(f"[{index}/{len(xml_files)}] Processing {xml_path.name}")
        try:
            patient, documents = generate_patient(
                renderer, output_root, xml_path, args
            )
            patients.append(patient)
            all_documents.extend(documents)
            stderr(
                f"  -> {patient.display_patient_id}: {len(documents)} documents"
            )
        except Exception as exc:
            failures.append({"file": str(xml_path), "error": str(exc)})
            stderr(f"  !! Failed: {exc}")
            if args.fail_fast:
                raise

    # Corpus-level JSONL is directly useful for a Delta full-document table.
    with (output_root / "documents.jsonl").open("w", encoding="utf-8") as handle:
        for document in all_documents:
            handle.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")

    with (output_root / "patients.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "patient_id",
                "display_patient_id",
                "patient_name",
                "birth_date",
                "gender",
                "document_count",
                "source_file",
                "source_sha256",
            ],
        )
        writer.writeheader()
        counts = defaultdict(int)
        for document in all_documents:
            counts[document.patient_id] += 1
        for patient in patients:
            writer.writerow(
                {
                    "patient_id": patient.patient_id,
                    "display_patient_id": patient.display_patient_id,
                    "patient_name": patient.name,
                    "birth_date": patient.birth_date.isoformat() if patient.birth_date else "",
                    "gender": patient.gender,
                    "document_count": counts[patient.patient_id],
                    "source_file": patient.source_file.name,
                    "source_sha256": patient.source_sha256,
                }
            )

    comparison_groups: dict[tuple[str, str], list[GeneratedDocument]] = defaultdict(list)
    for document in all_documents:
        if document.comparison_key:
            comparison_groups[(document.patient_id, document.comparison_key)].append(document)
    comparison_pairs: list[dict[str, Any]] = []
    for (patient_id, comparison_key), docs in comparison_groups.items():
        dated = sorted(
            [doc for doc in docs if doc.document_date != "undated"],
            key=lambda doc: doc.document_date,
        )
        if len(dated) >= 2:
            comparison_pairs.append(
                {
                    "patient_id": patient_id,
                    "comparison_key": comparison_key,
                    "document_a_id": dated[-2].document_id,
                    "document_b_id": dated[-1].document_id,
                    "document_a_path": dated[-2].output_path,
                    "document_b_path": dated[-1].output_path,
                }
            )
    (output_root / "comparison_pairs.json").write_text(
        json.dumps(comparison_pairs, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    corpus_manifest = {
        "schema_version": APP_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "synthetic": True,
        "input_directory": str(input_dir),
        "output_directory": str(output_root),
        "patient_count": len(patients),
        "document_count": len(all_documents),
        "failure_count": len(failures),
        "failures": failures,
        "augmentation_enabled": bool(args.augment_missing),
        "localization": "Source locations retained"
        if args.keep_source_location
        else "Ontario, Canada - synthetic presentation localization",
        "privacy_rule": "Every retrieval and deterministic tool query must filter by patient_id before searching or ranking.",
        "files": {
            "documents_jsonl": "documents.jsonl",
            "patients_csv": "patients.csv",
            "comparison_pairs": "comparison_pairs.json",
        },
    }
    (output_root / "corpus_manifest.json").write_text(
        json.dumps(corpus_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    stderr(
        f"Complete: {len(patients)} patients, {len(all_documents)} PDFs, {len(failures)} failures."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Synthea C-CDA XML into patient-isolated, RAG-ready synthetic PDFs."
    )
    parser.add_argument("--input-dir", help="Directory containing Synthea C-CDA XML files")
    parser.add_argument(
        "--output-dir",
        default="./generated_patient_records",
        help="Destination root containing one folder per patient",
    )
    parser.add_argument("--pattern", default="*.xml", help="Input glob pattern")
    parser.add_argument(
        "--augment-missing",
        action="store_true",
        help="Create clearly labelled fictional labs, imaging, specialist, and discharge examples when source categories are missing",
    )
    parser.add_argument(
        "--keep-source-location",
        action="store_true",
        help="Keep source US addresses and organizations instead of Ontario presentation localization",
    )
    parser.add_argument("--max-labs", type=int, default=12, help="Maximum source lab PDFs per patient; 0 means all")
    parser.add_argument("--max-encounters", type=int, default=10, help="Maximum encounter PDFs per patient; 0 means all")
    parser.add_argument("--max-prescriptions", type=int, default=10, help="Maximum medication-order PDFs per patient; 0 means all")
    parser.add_argument("--max-vitals", type=int, default=6, help="Maximum vital-sign date PDFs per patient; 0 means all")
    parser.add_argument("--max-imaging", type=int, default=6, help="Maximum imaging PDFs per patient; 0 means all")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first invalid XML or generation error")

    parser.add_argument(
        "--run-synthea-dir",
        help="Optional path to a local Synthea checkout; generates source C-CDA before conversion",
    )
    parser.add_argument("--synthea-population", type=int, default=100)
    parser.add_argument("--synthea-seed", type=int, default=20260730)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        generate_corpus(args)
        return 0
    except KeyboardInterrupt:
        stderr("Interrupted")
        return 130
    except Exception as exc:
        stderr(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
