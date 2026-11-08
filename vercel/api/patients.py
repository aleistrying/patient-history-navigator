"""Vercel serverless function: the patient roster.

Static because ingestion covers a fixed five-patient subset; serving it from here
avoids waking the SQL warehouse just to draw the picker.
"""

import json
import os
from http.server import BaseHTTPRequestHandler

from auth_gate import access_error

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

class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        denied = access_error(self.headers.get("X-PHN-Password"))
        response = {"error": denied[1]} if denied else {
            "patients": PATIENTS,
            "endpoint": os.environ.get("PHN_ENDPOINT", "dbdemos-rag-prod"),
        }
        body = json.dumps(response).encode("utf-8")
        self.send_response(denied[0] if denied else 200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
