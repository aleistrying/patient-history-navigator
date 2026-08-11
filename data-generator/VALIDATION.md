# Validation results

The generator was tested against the seven supplied Synthea C-CDA files.

- Patients processed: 7
- PDFs generated: 176
- Metadata sidecars: 176
- Comparison pairs generated: 23
- Parsing/generation failures: 0
- PDFs with zero pages: 0
- PDFs with insufficient extractable text: 0
- Patient-folder / metadata ID mismatches: 0
- Missing copied source-record references: 0

Representative patient summary, laboratory, encounter, and imaging PDFs were rendered to PNG and visually checked for wrapping, clipping, table headers, watermarking, and footer layout.

Test command:

```bash
python synthea_ccda_to_patient_pdfs.py \
  --input-dir /mnt/data \
  --pattern '*.xml' \
  --output-dir ./sample_output \
  --augment-missing \
  --max-labs 4 \
  --max-encounters 3 \
  --max-prescriptions 3 \
  --max-vitals 2 \
  --max-imaging 2
```
