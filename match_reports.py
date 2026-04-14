#!/usr/bin/env python3
"""
match_reports.py
================
Pipeline position: STEP 3 of 4
Depends on: download_data.py (./brats_data/ must exist)
Run after: run_inference.py   Run before: evaluate.py

Fetches brats23_metadata-report-deepseek-r1:1.5b.json from the KurtLabUW
BTReport GitHub repo, which contains reports for BraTS-GLI patients generated
by DeepSeek R1 1.5B from quantitative imaging metadata (VASARI features, tumor
volumes, midline shift). Uses the 'Findings' field per patient as the reference
report for evaluation.

These reports are grounded in real quantitative measurements extracted from the
actual BraTS scans, making them a more meaningful reference than synthetically
generated text.

JSON structure:
  - Top-level keys: BraTS-GLI patient IDs (e.g. "BraTS-GLI-00000-000")
  - Per-entry fields:
      "Findings"   ← narrative radiology report (used as reference)
      "Metadata"   ← structured quantitative features (VASARI, volumes, etc.)

Output: ./reference_reports/<patient_id>.txt for all matched patients.

Requirements: requests
"""

import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METADATA_REPORTS_URL = (
    "https://raw.githubusercontent.com/KurtLabUW/BTReport/main/"
    "btreport/llm_report_generation/example_reports/"
    "brats23_metadata-report-deepseek-r1%3A1.5b.json"
)
BRATS_DIR = Path("./brats_data")
REF_DIR   = Path("./reference_reports")


# ---------------------------------------------------------------------------
# Step 1 – Fetch the metadata reports JSON
# ---------------------------------------------------------------------------

def fetch_json(url: str) -> dict:
    print(f"Fetching {url} …")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Step 2 – Discover local BraTS patient IDs
# ---------------------------------------------------------------------------

def get_local_patient_ids() -> list[str]:
    if not BRATS_DIR.exists():
        sys.exit(f"ERROR: {BRATS_DIR} does not exist. Run download_data.py first.")
    ids = sorted(
        d.name for d in BRATS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("BraTS-")
    )
    if not ids:
        sys.exit(f"ERROR: No BraTS patient directories found in {BRATS_DIR}.")
    print(f"Local BraTS patients ({len(ids)}): {ids}")
    return ids


# ---------------------------------------------------------------------------
# Step 3 – Match and save reference reports
# ---------------------------------------------------------------------------

def save_reference_reports(data: dict, patient_ids: list[str]) -> None:
    """
    For each local patient ID, look up the 'Findings' field in the metadata
    reports JSON and save it as the reference report.
    """
    REF_DIR.mkdir(parents=True, exist_ok=True)
    matched, unmatched = 0, 0

    for pid in patient_ids:
        ref_path = REF_DIR / f"{pid}.txt"

        if ref_path.exists():
            print(f"  {pid}  → already exists, skipping.")
            matched += 1
            continue

        if pid not in data:
            print(f"  {pid}  → NOT FOUND in metadata reports JSON.")
            unmatched += 1
            continue

        findings = data[pid].get("Findings", "").strip()
        if not findings:
            print(f"  {pid}  → found but 'Findings' field is empty.")
            unmatched += 1
            continue

        header = (
            f"# REFERENCE REPORT\n"
            f"# Patient ID : {pid}\n"
            f"# Source     : BTReport / DeepSeek R1 1.5B (metadata-based)\n"
            f"# URL        : {METADATA_REPORTS_URL}\n"
            f"{'#'*72}\n\n"
        )
        ref_path.write_text(header + findings, encoding="utf-8")
        print(f"  {pid}  → saved to {ref_path}")
        matched += 1

    print(f"\nDone. {matched}/{len(patient_ids)} reference reports saved to {REF_DIR}/")
    if unmatched:
        print(f"WARNING: {unmatched} patient(s) had no reference report.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data        = fetch_json(METADATA_REPORTS_URL)
    patient_ids = get_local_patient_ids()
    save_reference_reports(data, patient_ids)

    saved = sorted(REF_DIR.glob("*.txt")) if REF_DIR.exists() else []
    for f in saved:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
