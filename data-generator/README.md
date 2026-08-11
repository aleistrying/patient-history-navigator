# Synthea C-CDA to patient PDF corpus

This tool converts a directory of Synthea C-CDA XML files into a folder-per-patient PDF corpus designed for the **Patient History Navigator** RAG project.

Every generated PDF is visibly marked as synthetic, carries a stable `patient_id` and `document_id`, and has a `.metadata.json` sidecar. The corpus also includes `documents.jsonl`, `patients.csv`, and `comparison_pairs.json` for deterministic timeline, document-comparison, lab-trend, and medication-change workflows.

## Output layout

```text
generated_patient_records/
  corpus_manifest.json
  patients.csv
  documents.jsonl
  comparison_pairs.json
  patient_<patient_uuid>/
    patient_manifest.json
    source/
      original_ccda.xml
    documents/
      summaries/
      labs/
      medications/
      encounters/
      imaging/
      care/
```

The PDF filename contract is:

```text
patient_id__YYYY-MM-DD__document_type__document_id.pdf
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Convert your existing `./100files` directory

```bash
python synthea_ccda_to_patient_pdfs.py \
  --input-dir ./100files \
  --output-dir ./generated_patient_records
```

For a richer demo corpus, add only clearly labelled fictional categories that are missing from a patient's source record:

```bash
python synthea_ccda_to_patient_pdfs.py \
  --input-dir ./100files \
  --output-dir ./generated_patient_records \
  --augment-missing
```

The augmentation mode creates deterministic fictional comparison cases such as two CBC/CMP/urinalysis reports, one imaging report, one specialist follow-up note, and one short-stay discharge summary. Their metadata uses:

```json
"provenance": "synthetic_augmentation"
```

Source-grounded PDFs use:

```json
"provenance": "source_ccda"
```

## Control corpus size

The defaults keep the most recent documents of high-volume types:

```bash
--max-labs 12
--max-encounters 10
--max-prescriptions 10
--max-vitals 6
--max-imaging 6
```

Set a value to `0` to generate every source event of that type:

```bash
python synthea_ccda_to_patient_pdfs.py \
  --input-dir ./100files \
  --output-dir ./generated_patient_records \
  --max-labs 0 --max-encounters 0 --max-prescriptions 0
```

## Generate new source patients with a local Synthea checkout

Synthea must already be cloned and runnable. The script does not download or modify Synthea.

```bash
python synthea_ccda_to_patient_pdfs.py \
  --run-synthea-dir ../synthea \
  --synthea-population 100 \
  --synthea-seed 20260730 \
  --output-dir ./generated_patient_records \
  --augment-missing
```

This invokes the local runner with C-CDA and FHIR export enabled, then converts the generated C-CDA folder.

## Ontario localization

By default, the display layer uses deterministic fictional Ontario addresses and organizations while preserving the original XML under `source/`. This avoids presenting Massachusetts source organizations as Ontario institutions.

Use `--keep-source-location` to preserve source addresses and organizations in the PDF presentation.

The internal UUID from Synthea remains the authoritative `patient_id`. A readable ID such as `ON-SYN-AB07C69812` is display-only.

## Planned: scanned documents that only OCR can read

**Not implemented yet.** Every PDF this tool currently emits is a `reportlab` document with
a real text layer, so `pypdf` reads all of them and the navigator's OCR fallback never
runs — 225 of 225 ingested PDFs passed the readability gate and the parse cache holds zero
rows. The corpus cannot fail that gate, which means it cannot test it.

The planned change adds a post-render pass over a seeded random subset: rasterise the pages
and rebuild the PDF from the images, so the file has no text layer at all and only OCR can
read it. The text version is kept as ground truth for scoring OCR output against.

- `--scan-ratio` selects the subset; the existing `--synthea-seed` keeps it reproducible.
- Quality varies per document — DPI ~100–300, JPEG quality ~40–95, ±1.5° skew, light blur
  and noise, occasional grayscale/bilevel and a scanner-style border — aiming for a spread
  that **straddles** the navigator's readability thresholds (≥200 chars, ≥30 per page,
  ≥60 % readable pages, ≥20 % alphabetic) rather than one that always fails.
- The sidecar, `documents.jsonl`, and `corpus_manifest.json` each record
  `rendering: "text" | "scanned_image"`, `scan_dpi`, `scan_quality`, `ocr_required`, so
  evaluation can be split by rendering mode.
- Needs a rasteriser: `PyMuPDF`, or `pdf2image` plus poppler. `requirements.txt` currently
  has only `lxml` and `reportlab`.
- The synthetic-data banner must remain legible after degradation. An unmarked degraded copy
  is a provenance regression, not a test asset.

Rationale and the full acceptance criteria are in `docs/IMPLEMENTATION_GUIDE.md` section 30.

## Required RAG privacy boundary

Filter on `patient_id` **before** vector search, full-text search, SQL tools, reranking, or result fusion. Do not retrieve globally and filter after ranking.

Example metadata filter concept:

```python
filters = {"patient_id": authorized_patient_id}
```

Your full-document table can load `documents.jsonl`; your chunk table should copy the same `patient_id`, `document_id`, `document_type`, `document_date`, `section`, `source_path`, and `provenance` into every chunk.

## Important safety and provenance rules

- All records are synthetic demonstration data.
- The tool does not diagnose, prescribe, recommend treatment, or claim clinical significance.
- Medication directions are never inferred from drug names.
- Source-only imaging reports state when findings were not present in C-CDA.
- Fictional augmentation is visually marked and machine-labelled separately.
- The fictional lab intervals are for UI demonstration only and are not clinical reference intervals.
