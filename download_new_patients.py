#!/usr/bin/env python3
"""
download_new_patients.py
========================
Finds and downloads new BraTS-GLI patient scans that haven't been
inferred yet, then organises them into the brats_data/ structure
that run_inference.py expects.

How it works
────────────
1. Reads btreport_brats23.json (already cached) to get the full list of
   1,251 valid patient IDs.
2. Compares against what's already in brats_data/ and reports/ to find
   patients that haven't been processed.
3. Selects N candidates (default 7) starting from the first available ID
   after the last one you've already done.
4. Downloads the BraTS 2023 glioma dataset from Kaggle and copies the
   relevant patient folders into brats_data/.

Prerequisites
─────────────
1. Kaggle credentials:
     Go to https://www.kaggle.com/settings/account → Create New Token
     Save kaggle.json to ~/.kaggle/kaggle.json
     chmod 600 ~/.kaggle/kaggle.json
2. pip install kaggle>=1.6.0  (already in requirements.txt)

Dataset note
────────────
BraTS 2023 glioma data is on Synapse (syn51514105) and mirrored on Kaggle
by community users. Set KAGGLE_DATASET below to the correct identifier.

Known Kaggle mirrors to try (check availability at https://kaggle.com/datasets):
  "awsaf49/brats2023-local-data"
  "dschettler8845/brats-2021-task1"   (BraTS 2021, slightly different naming)

If no Kaggle mirror is available:
  1. Download from Synapse: https://www.synapse.org/#!Synapse:syn51514105
  2. Extract patient folders somewhere (e.g. ~/Downloads/brats/)
  3. Run:  python download_new_patients.py --skip_kaggle --src ~/Downloads/brats/

Expected file layout after this script:
  brats_data/
    BraTS-GLI-00008-000/
      BraTS-GLI-00008-000-t1c.nii
      BraTS-GLI-00008-000-t1n.nii
      BraTS-GLI-00008-000-t2f.nii
      BraTS-GLI-00008-000-t2w.nii
    BraTS-GLI-00009-000/
      ...

Usage:
  python download_new_patients.py                     # default: 7 new patients
  python download_new_patients.py --n 3               # download 3
  python download_new_patients.py --first_only        # skip -001 timepoints
  python download_new_patients.py --skip_kaggle --src /path/to/extracted/data
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "brats_data"
BTREPORT = ROOT / "btreport_brats23.json"

# ── Configure this to match the actual Kaggle dataset ─────────────────────────
KAGGLE_DATASET = "awsaf49/brats2023-local-data"  # adjust if needed
MODALITIES     = ["t1c", "t1n", "t2f", "t2w"]
# ─────────────────────────────────────────────────────────────────────────────


def already_in_data_dir() -> set:
    if not DATA_DIR.exists():
        return set()
    return {d.name for d in DATA_DIR.iterdir()
            if d.is_dir() and d.name.startswith("BraTS-GLI-")}


def already_inferred() -> set:
    """Patient IDs that already have a report (base or finetuned)."""
    done = set()
    for folder in [ROOT / "reports", ROOT / "reports_finetuned"]:
        if folder.exists():
            done |= {p.stem for p in folder.glob("*.txt")}
    return done


def patients_in_btreport(first_only: bool) -> list:
    """Return sorted list of all patient IDs from cached btreport JSON."""
    if not BTREPORT.exists():
        print("⚠ btreport_brats23.json not found. Run match_reports.py first.")
        sys.exit(1)
    with open(BTREPORT) as f:
        data = json.load(f)
    keys = sorted(k for k in data if k.startswith("BraTS-GLI-"))
    if first_only:
        # Keep only the -000 timepoint for each patient number
        keys = [k for k in keys if k.endswith("-000")]
    return keys


def select_new_patients(all_patients: list, exclude: set, n: int) -> list:
    candidates = [p for p in all_patients if p not in exclude]
    if not candidates:
        print("✓ No new patients available — all patients in btreport already processed.")
        sys.exit(0)
    return candidates[:n]


def check_kaggle_auth():
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_file = kaggle_json.exists()
    has_env  = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if not has_file and not has_env:
        print("✗ Kaggle credentials not found.")
        print("  1. Visit https://www.kaggle.com/settings/account")
        print("  2. Click 'Create New Token' → downloads kaggle.json")
        print("  3. mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json")
        print("  4. chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)


def kaggle_download(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {KAGGLE_DATASET} → {dest}/")
    print("(This can take several minutes for a large dataset.)")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "-p", str(dest), "--unzip"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"\n✗ Kaggle download failed:\n{result.stderr.strip()}")
        print("\nTroubleshooting:")
        print(f"  • Try manually: kaggle datasets download -d {KAGGLE_DATASET} -p {dest} --unzip")
        print(f"  • Check the dataset exists: https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
        print("  • Alternative: download from Synapse https://www.synapse.org/#!Synapse:syn51514105")
        print("    then re-run with:  python download_new_patients.py --skip_kaggle --src /path/to/data")
        sys.exit(1)
    print("[✓] Download complete.")


def find_patient_folder(search_root: Path, patient_id: str) -> Path | None:
    """Recursively find a directory named exactly patient_id."""
    for match in search_root.rglob(patient_id):
        if match.is_dir():
            return match
    return None


def copy_patient(src: Path, patient_id: str) -> bool:
    """Copy one patient folder into brats_data/, verifying all 4 modalities."""
    dst = DATA_DIR / patient_id
    if dst.exists():
        print(f"  {patient_id}: already in brats_data/, skipping.")
        return True

    # Locate each modality file (accepts .nii or .nii.gz)
    nii_files = {}
    for mod in MODALITIES:
        for suffix in [".nii", ".nii.gz"]:
            candidate = src / f"{patient_id}-{mod}{suffix}"
            if candidate.exists():
                nii_files[mod] = candidate
                break
        if mod not in nii_files:
            # Fallback: search anywhere under src
            matches = list(src.rglob(f"*{mod}*"))
            nii_matches = [m for m in matches if m.suffix in (".nii", ".gz")]
            if nii_matches:
                nii_files[mod] = nii_matches[0]
            else:
                print(f"  ⚠ {patient_id}: cannot find {mod} file under {src}")
                return False

    dst.mkdir(parents=True, exist_ok=True)
    for mod, src_file in nii_files.items():
        # Normalise to .nii (non-compressed) name to match run_inference.py expectations
        dst_name = f"{patient_id}-{mod}.nii"
        shutil.copy2(src_file, dst / dst_name)
        print(f"    ✓ {dst_name}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download new BraTS patients for inference")
    parser.add_argument("--n",          type=int, default=7,
                        help="Number of new patients to download (default: 7)")
    parser.add_argument("--first_only", action="store_true",
                        help="Only download -000 timepoint patients (skip -001, -002, etc.)")
    parser.add_argument("--skip_kaggle", action="store_true",
                        help="Skip Kaggle download; use --src to point at already-extracted data")
    parser.add_argument("--src",        type=str, default="/tmp/brats_download",
                        help="Directory where Kaggle data is/will be extracted")
    args = parser.parse_args()

    src_dir = Path(args.src)

    print("━" * 64)
    print("BraTS New Patient Downloader")
    print("━" * 64)

    # What we already have
    in_data  = already_in_data_dir()
    inferred = already_inferred()
    exclude  = in_data | inferred
    print(f"\nAlready in brats_data/  : {len(in_data)}")
    print(f"Already inferred         : {len(inferred)}")

    # Available patient IDs from btreport
    all_patients = patients_in_btreport(first_only=args.first_only)
    print(f"Patients in btreport.json: {len(all_patients)}"
          + (" (first timepoints only)" if args.first_only else ""))

    # Select new candidates
    new_patients = select_new_patients(all_patients, exclude, args.n)
    print(f"\nSelected {len(new_patients)} new patient(s):")
    for p in new_patients:
        print(f"  {p}")

    # Download
    if not args.skip_kaggle:
        check_kaggle_auth()
        kaggle_download(src_dir)
    else:
        if not src_dir.exists():
            print(f"\n✗ --src path does not exist: {src_dir}")
            sys.exit(1)
        print(f"\nUsing pre-extracted data from: {src_dir}")

    # Copy selected patients into brats_data/
    DATA_DIR.mkdir(exist_ok=True)
    print(f"\nCopying patient folders → {DATA_DIR}/")
    success, failed = [], []

    for pid in new_patients:
        src = find_patient_folder(src_dir, pid)
        if src is None:
            print(f"  ✗ {pid}: not found anywhere under {src_dir}")
            failed.append(pid)
            continue
        print(f"  {pid}  (found at {src.relative_to(src_dir)})")
        if copy_patient(src, pid):
            success.append(pid)
        else:
            failed.append(pid)

    # Summary
    print(f"\n{'─' * 64}")
    print(f"Successfully copied: {len(success)}/{len(new_patients)}")
    if failed:
        print(f"Failed:             {failed}")

    if success:
        print("\nNext steps:")
        print("  1.  python run_inference.py           # generate reports")
        print("  2.  python match_reports.py           # fetch reference reports")
        print("  3.  python evaluate.py                # compute NLP metrics")
        print("  4.  python error_analysis.py          # detect error patterns")
        print("  5.  python analyze_reports.py         # visualise comparisons")

    if failed:
        print("\nFor patients that failed, try downloading manually from:")
        print("  https://www.synapse.org/#!Synapse:syn51514105")
        print("Then re-run with --skip_kaggle --src /path/to/extracted/")


if __name__ == "__main__":
    main()
