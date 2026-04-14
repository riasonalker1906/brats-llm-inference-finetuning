#!/usr/bin/env python3
"""
error_analysis.py
=================
Scalable error pattern detection for radiology report generation.
Designed to grow with the patient cohort — run it again after adding
new patients and the aggregate statistics update automatically.

Error taxonomy (10 categories)
───────────────────────────────
  OMIT_MASS_EFFECT       Reference mentions mass effect/midline shift;
                         model omits it entirely.
  OMIT_QUANTIFICATION    Reference gives mm / cm / % measurements;
                         model gives none.
  OMIT_NECROSIS          Necrosis in reference; absent from model.
  OMIT_HEMORRHAGE        Hemorrhage / susceptibility in reference; absent.
  OMIT_VENTRICLE_EFFECT  Ventricle compression / entrapment in reference; absent.
  OMIT_INVASION          Ependymal or white-matter invasion in reference; absent.
  HALLUCINATED_NEGATIVE  Model states "no mass effect / no necrosis" where
                         reference implies the finding IS present.
  TEMPLATE_COLLAPSE      Model report is ≥ 70% text-similar to another patient
                         report from the same model (copy-paste behaviour).
  SPECIFICITY_LOSS       Model uses identical location phrase ("left frontal lobe")
                         across ALL patients — never differentiates.
  GENERIC_IMPRESSION     IMPRESSION section is ≥ 85% identical to another patient's
                         impression (boilerplate conclusion).

Usage:
  python error_analysis.py                      # analyse both models
  python error_analysis.py --model base         # base model only
  python error_analysis.py --model finetuned    # finetuned model only

Outputs → ./error_analysis/
  per_patient_errors_<model>.csv    binary error matrix (patients × error types)
  aggregate_errors_<model>.csv      error frequency table
  error_heatmap_<model>.png         visual error matrix
  error_frequency_<model>.png       bar chart of most common errors
  comparison_delta.png              base vs finetuned error count comparison
                                    (only when --model both)
"""

import argparse
import difflib
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
OUT  = ROOT / "error_analysis"
OUT.mkdir(exist_ok=True)

# ── Error taxonomy ────────────────────────────────────────────────────────────
# Each entry is either:
#   "pattern" style  →  ref_requires + model_absent patterns
#   "special" style  →  custom detection logic below
#
# For pattern errors: error fires when ref matches ref_requires AND model
# does NOT match model_absent (i.e. the model missed something the reference had).

ERRORS = {
    "OMIT_MASS_EFFECT": {
        "desc":         "Mass effect / midline shift in reference but missing from model",
        "ref_requires": [r"mass\s+effect|herniation|effacement"
                          r"|\d+\.?\d*\s*mm\s*(shift|deviation)|midline\s+shift"],
        "model_absent": [r"mass\s+effect|herniation|effacement|midline\s+shift"],
    },
    "OMIT_QUANTIFICATION": {
        "desc":         "Reference gives numeric measurements (mm/cm/%); model gives none",
        "ref_requires": [r"\d+\.?\d*\s*(mm|cm|%|cm[³3])"],
        "model_absent": [r"\d+\.?\d*\s*(mm|cm|%)"],
    },
    "OMIT_NECROSIS": {
        "desc":         "Necrosis in reference; model never mentions it",
        "ref_requires": [r"necros"],
        "model_absent": [r"necros"],
    },
    "OMIT_HEMORRHAGE": {
        "desc":         "Hemorrhage / susceptibility in reference; absent from model",
        "ref_requires": [r"hemorrhag|haemorrhag|susceptib|intratumoral\s+bleed"],
        "model_absent": [r"hemorrhag|haemorrhag|susceptib|bleed"],
    },
    "OMIT_VENTRICLE_EFFECT": {
        "desc":         "Ventricle compression / entrapment in reference; absent from model",
        "ref_requires": [r"entrapment|enlarg.*ventricle|compress.*ventricle|hydrocephalus"],
        "model_absent": [r"entrapment|enlarg.*ventricle|compress.*ventricle|hydrocephalus"],
    },
    "OMIT_INVASION": {
        "desc":         "Ependymal or white-matter invasion in reference; absent from model",
        "ref_requires": [r"ependymal|white\s+matter\s+invasion|deep\s+white"],
        "model_absent": [r"ependymal|white\s+matter\s+invasion"],
    },
    "HALLUCINATED_NEGATIVE": {
        "desc":    "Model asserts 'no mass effect / no necrosis' that reference contradicts",
        "special": "hallucinated_negative",
    },
    "TEMPLATE_COLLAPSE": {
        "desc":      "Report is ≥70% text-similar to another patient's report (same model)",
        "special":   "template_collapse",
        "threshold": 0.70,
    },
    "SPECIFICITY_LOSS": {
        "desc":    "Model uses identical location ('left frontal lobe') for every patient",
        "special": "specificity_loss",
    },
    "GENERIC_IMPRESSION": {
        "desc":      "IMPRESSION section is ≥85% identical to another patient's impression",
        "special":   "generic_impression",
        "threshold": 0.85,
    },
}

# Phrases where the model incorrectly asserts absence of a finding
NEGATION_PATTERNS = [
    r"no\s+(clear\s+)?(mass\s+effect|midline\s+shift|herniation)",
    r"no\s+(significant\s+)?(edema|necrosis|hemorrhag)",
    r"no\s+(apparent|obvious)\s+(mass\s+effect|shift)",
    r"without\s+(significant\s+)?mass\s+effect",
    r"absence\s+of\s+(mass\s+effect|shift)",
    r"no\s+clear\s+mass\s+effect\s+or\s+midline\s+shift",
]


# ── Utilities ─────────────────────────────────────────────────────────────────

def _strip(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.startswith("#")).strip()

def _any(text: str, patterns: list) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)

def _short(pid: str) -> str:
    return pid.replace("BraTS-GLI-", "").replace("-000", "")

def _extract_impression(text: str) -> str:
    m = re.search(r"IMPRESSION[:\s]*(.*?)$", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


# ── Report loading ────────────────────────────────────────────────────────────

def load_reports(model: str) -> dict:
    folder = ROOT / ("reports" if model == "base" else "reports_finetuned")
    ref_dir = ROOT / "reference_reports"
    patients = sorted(p.stem for p in ref_dir.glob("*.txt"))
    result = {}
    for pid in patients:
        mp = folder  / f"{pid}.txt"
        rp = ref_dir / f"{pid}.txt"
        if mp.exists() and rp.exists():
            result[pid] = {
                "model": _strip(mp.read_text()),
                "ref":   _strip(rp.read_text()),
            }
    return result


# ── Error detection ───────────────────────────────────────────────────────────

def detect_errors(reports: dict) -> pd.DataFrame:
    """Return a binary DataFrame: patients × error_types."""
    patients        = sorted(reports.keys())
    all_model_texts = {pid: reports[pid]["model"] for pid in patients}
    rows = []

    for pid in patients:
        mt  = reports[pid]["model"]
        rt  = reports[pid]["ref"]
        row = {"patient_id": pid}

        for err_name, err_def in ERRORS.items():
            special = err_def.get("special")

            # ── Standard pattern check ─────────────────────────────────────
            if special is None:
                ref_has   = _any(rt, err_def["ref_requires"])
                model_has = _any(mt, err_def["model_absent"])
                row[err_name] = int(ref_has and not model_has)

            # ── Hallucinated negative ──────────────────────────────────────
            elif special == "hallucinated_negative":
                ref_has_mass_effect = _any(rt, [
                    r"mass\s+effect|midline\s+shift|\d+\.?\d*\s*mm\s*shift|herniation"
                ])
                model_negates = _any(mt, NEGATION_PATTERNS)
                row[err_name] = int(ref_has_mass_effect and model_negates)

            # ── Template collapse ──────────────────────────────────────────
            elif special == "template_collapse":
                thr        = err_def.get("threshold", 0.70)
                collapsed  = False
                for other_pid, other_text in all_model_texts.items():
                    if other_pid == pid:
                        continue
                    if difflib.SequenceMatcher(None, mt, other_text).ratio() >= thr:
                        collapsed = True
                        break
                row[err_name] = int(collapsed)

            # ── Specificity loss ───────────────────────────────────────────
            # Fires if EVERY patient report from this model contains "left frontal lobe"
            # (i.e. the model never differentiates tumour location).
            elif special == "specificity_loss":
                lfl_count  = sum(
                    1 for t in all_model_texts.values()
                    if re.search(r"left\s+frontal\s+lobe", t, re.IGNORECASE)
                )
                row[err_name] = int(lfl_count == len(patients) and len(patients) > 1)

            # ── Generic impression ─────────────────────────────────────────
            elif special == "generic_impression":
                thr    = err_def.get("threshold", 0.85)
                my_imp = _extract_impression(mt)
                if not my_imp:
                    row[err_name] = 0
                    continue
                generic = False
                for other_pid, other_text in all_model_texts.items():
                    if other_pid == pid:
                        continue
                    other_imp = _extract_impression(other_text)
                    if difflib.SequenceMatcher(None, my_imp, other_imp).ratio() >= thr:
                        generic = True
                        break
                row[err_name] = int(generic)

        rows.append(row)

    return pd.DataFrame(rows).set_index("patient_id")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_error_heatmap(error_df: pd.DataFrame, model_name: str):
    short_patients = [_short(p) for p in error_df.index]
    error_names    = list(error_df.columns)
    mat            = error_df.values.T.astype(float)

    fig_w = max(8, len(short_patients) * 1.4)
    fig_h = max(5, len(error_names) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("err", ["#F2F2F2", "#C00000"])
    ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(short_patients)))
    ax.set_xticklabels(short_patients, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(error_names)))
    ax.set_yticklabels(error_names, fontsize=8)
    ax.set_title(
        f"Error Matrix — {model_name}  ({len(short_patients)} patients)\n"
        f"Red = error present  ·  = no error",
        fontweight="bold",
    )
    for i in range(len(error_names)):
        for j in range(len(short_patients)):
            ax.text(j, i, "✗" if mat[i, j] else "·",
                    ha="center", va="center", fontsize=11,
                    color="#C00000" if mat[i, j] else "#BFBFBF")

    plt.tight_layout()
    plt.savefig(OUT / f"error_heatmap_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] error_heatmap_{model_name}.png")


def plot_error_frequency(error_df: pd.DataFrame, model_name: str):
    freq       = error_df.sum().sort_values(ascending=True)
    n_patients = len(error_df)

    fig, ax = plt.subplots(figsize=(10, max(4, len(freq) * 0.52)))
    colors = [
        "#C00000" if v == n_patients else
        "#ED7D31" if v > n_patients / 2 else
        "#4472C4"
        for v in freq.values
    ]
    bars = ax.barh(freq.index, freq.values, color=colors, alpha=0.85)
    ax.set_xlabel("Number of Patients Affected")
    ax.set_title(
        f"Error Frequency — {model_name}  (n = {n_patients} patients)\n"
        f"Red = affects all patients  ·  Orange = majority  ·  Blue = minority",
        fontweight="bold",
    )
    ax.axvline(n_patients, color="#7F7F7F", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlim(0, n_patients + 2)

    descs = {k: v["desc"] for k, v in ERRORS.items()}
    for bar, err_name, v in zip(bars, freq.index, freq.values):
        pct = v / n_patients * 100
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{v}/{n_patients}  ({pct:.0f}%)", va="center", fontsize=8)

    # Annotate each row with the description
    y_labels = []
    for e in freq.index:
        desc = descs.get(e, "")
        truncated = desc[:48] + "…" if len(desc) > 48 else desc
        y_labels.append(f"{e}\n  {truncated}")
    ax.set_yticklabels(y_labels, fontsize=7.5)

    plt.tight_layout()
    plt.savefig(OUT / f"error_frequency_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] error_frequency_{model_name}.png")


def plot_comparison_delta(base_errors: pd.DataFrame, ft_errors: pd.DataFrame):
    """Show how many fewer (or more) patients are affected by each error after finetuning."""
    # Align on common patients
    common = base_errors.index.intersection(ft_errors.index)
    delta  = ft_errors.loc[common].sum() - base_errors.loc[common].sum()
    delta  = delta.sort_values()

    fig, ax = plt.subplots(figsize=(9, max(4, len(delta) * 0.52)))
    colors  = ["#70AD47" if v < 0 else "#C00000" if v > 0 else "#7F7F7F" for v in delta]
    ax.barh(delta.index, delta.values, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ Patients Affected  (Finetuned − Base)  —  negative = finetuning helped")
    ax.set_title("Finetuning Effect on Error Rates\nGreen = reduced errors  ·  Red = introduced new errors",
                 fontweight="bold")
    for i, (err_name, v) in enumerate(delta.items()):
        if v != 0:
            ax.text(v + (0.05 if v > 0 else -0.05), i, f"{v:+d}",
                    va="center", ha="left" if v > 0 else "right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "comparison_delta.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] comparison_delta.png")


# ── Aggregate console report ──────────────────────────────────────────────────

def aggregate_analysis(error_df: pd.DataFrame, model_name: str):
    n = len(error_df)
    print(f"\n{'═' * 72}")
    print(f"AGGREGATE ERROR ANALYSIS — {model_name}  ({n} patients)")
    print(f"{'═' * 72}")
    print(f"\n  {'Error Type':<28} {'Affected':>10} {'Rate':>8}   Description")
    print("  " + "─" * 70)

    for col in error_df.columns:
        count = int(error_df[col].sum())
        pct   = count / n * 100
        desc  = ERRORS[col]["desc"][:40]
        flag  = "🔴" if pct == 100 else "🟡" if pct > 50 else "🟢"
        print(f"  {flag} {col:<26} {count:>5}/{n:<4} {pct:>7.0f}%   {desc}")

    worst = error_df.sum(axis=1).sort_values(ascending=False)
    print(f"\n  Most error-prone patients:")
    for pid, count in worst.items():
        print(f"    {_short(pid):<18} {int(count)}/{len(ERRORS)} error types")

    # Save CSVs
    error_df.to_csv(OUT / f"per_patient_errors_{model_name}.csv")
    print(f"\n  [✓] per_patient_errors_{model_name}.csv")

    freq_df = pd.DataFrame([{
        "error":             col,
        "description":       ERRORS[col]["desc"],
        "affected_patients": int(error_df[col].sum()),
        "rate":              float(error_df[col].mean()),
    } for col in error_df.columns])
    freq_df.to_csv(OUT / f"aggregate_errors_{model_name}.csv", index=False)
    print(f"  [✓] aggregate_errors_{model_name}.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scalable radiology report error analysis for BraTS"
    )
    parser.add_argument("--model", choices=["base", "finetuned", "both"],
                        default="both", help="Which model's reports to analyse")
    args = parser.parse_args()

    models = ["base", "finetuned"] if args.model == "both" else [args.model]
    error_dfs = {}

    for model_name in models:
        print(f"\n{'━' * 72}")
        print(f"Analysing: {model_name}")
        print(f"{'━' * 72}")
        reports = load_reports(model_name)
        if not reports:
            print(f"  ⚠ No reports found. Run run_inference.py first.")
            continue
        print(f"  Patients found: {len(reports)}")

        error_df = detect_errors(reports)
        error_dfs[model_name] = error_df

        aggregate_analysis(error_df, model_name)
        plot_error_heatmap(error_df, model_name)
        plot_error_frequency(error_df, model_name)

    # Cross-model comparison
    if args.model == "both" and "base" in error_dfs and "finetuned" in error_dfs:
        plot_comparison_delta(error_dfs["base"], error_dfs["finetuned"])
        print(f"\n{'═' * 72}")
        print("FINETUNING EFFECT ON ERROR COUNTS  (negative = finetuning helped)")
        print("═" * 72)
        common = error_dfs["base"].index.intersection(error_dfs["finetuned"].index)
        delta  = error_dfs["finetuned"].loc[common].sum() - error_dfs["base"].loc[common].sum()
        for err, diff in delta.items():
            arrow = "↓" if diff < 0 else ("↑" if diff > 0 else "→")
            print(f"  {arrow}  {err:<30}  {diff:+d}")

    print(f"\n✓ All outputs → {OUT}/")


if __name__ == "__main__":
    main()
