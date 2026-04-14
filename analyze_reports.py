#!/usr/bin/env python3
"""
analyze_reports.py
==================
Comprehensive comparison of reference, base-model, and finetuned-model
radiology reports for all BraTS patients with available reports.

What this script does
─────────────────────
1. Loads all three versions of each patient report (reference / base / finetuned).
2. Extracts 18 medical concepts via regex (midline shift, necrosis, ventricle
   effects, BraTS segments, etc.) and checks which model captures which.
3. Detects "template collapse" — whether the model generates nearly identical
   text for different patients (a key failure mode).
4. Produces 5 publication-quality plots + a summary CSV.

Outputs (all written to ./analysis_output/):
  metric_comparison.png   — grouped bar: ROUGE-L / METEOR / BLEU-4 / BERTScore
  finetuning_delta.png    — Δ per metric from base → finetuned (+ = improvement)
  concept_heatmap.png     — which medical concepts each model captures per patient
  concept_recall.png      — % of reference concepts recalled, per patient
  template_collapse.png   — pairwise report similarity within each model
  analysis_summary.csv    — per-patient metrics + concept recall scores

Usage:
  python analyze_reports.py
"""

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
from matplotlib.colors import LinearSegmentedColormap

# ── Directory layout ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DIRS = {
    "reference": ROOT / "reference_reports",
    "base":      ROOT / "reports",
    "finetuned": ROOT / "reports_finetuned",
}
EVAL_FILES = {
    "base":      ROOT / "evaluation"           / "results.csv",
    "finetuned": ROOT / "evaluation_finetuned" / "results.csv",
}
OUT = ROOT / "analysis_output"
OUT.mkdir(exist_ok=True)

# ── Medical concept dictionary ────────────────────────────────────────────────
# Each value is a list of regex patterns (OR-combined, case-insensitive).
# A concept is "present" (1) if ANY pattern matches in the report text.
#
# Design principle: patterns are conservative — they require the actual clinical
# term, not loose synonyms, so false positives are rare.
CONCEPTS = {
    "midline_shift_mentioned":  [r"midline\s+shift", r"midline\s+deviation"],
    "midline_shift_quantified": [r"\d+\.?\d*\s*mm\s*(shift|deviation)",
                                  r"shift.*\d+\.?\d*\s*mm"],
    "mass_effect":              [r"mass\s+effect", r"herniation", r"effacement"],
    "edema_mentioned":          [r"\bedema\b", r"\boedema\b", r"vasogenic"],
    "edema_quantified":         [r"\d+\s*%.*edema", r"edema.*\d+\.?\d*\s*cm",
                                  r"edema.*volume"],
    "ring_enhancement":         [r"ring\s+enhanc", r"rim\s+enhanc",
                                  r"peripheral\s+enhanc"],
    "necrosis":                 [r"necros", r"necrotic"],
    "hemorrhage":               [r"hemorrhag", r"haemorrhag", r"susceptib",
                                  r"intratumoral\s+bleed"],
    "ventricles_mentioned":     [r"ventricle", r"ventricular"],
    "ventricle_effect":         [r"enlarg.*ventricle", r"compress.*ventricle",
                                  r"entrapment", r"hydrocephalus"],
    "tumor_dimensions":         [r"\d+\.?\d*\s*[×xX]\s*\d+\.?\d*",
                                  r"\d+\.?\d*\s*cm\b"],
    "specific_lobe":            [r"frontal\s+lobe", r"temporal\s+lobe",
                                  r"parietal\s+lobe", r"occipital\s+lobe",
                                  r"insula", r"cerebellum", r"basal\s+ganglia"],
    "ependymal_invasion":       [r"ependymal\s+invasion", r"subependymal",
                                  r"ependymal"],
    "white_matter_invasion":    [r"white\s+matter", r"deep\s+white"],
    "brats_segments":           [r"\bNCR\b", r"\bET\b", r"\bED\b",
                                  r"necrotic\s+core", r"enhancing\s+tumor",
                                  r"peritumoral\s+edema"],
    "diffusion_restriction":    [r"restricted\s+diffusion",
                                  r"diffusion\s+restrict", r"\bDWI\b", r"\bADC\b"],
    "vascularity":              [r"vascularity", r"vasculariz"],
    "multifocal":               [r"multifocal", r"multiple\s+(lesion|tumor)",
                                  r"satellite"],
}

METRIC_COLS   = ["rouge_l", "meteor", "bleu_4", "bertscore_f1"]
METRIC_LABELS = ["ROUGE-L", "METEOR", "BLEU-4", "BERTScore-F1"]


# ── Utilities ─────────────────────────────────────────────────────────────────

def _strip_header(text: str) -> str:
    """Remove the # comment header that run_inference.py prepends to reports."""
    return "\n".join(l for l in text.strip().splitlines()
                     if not l.startswith("#")).strip()

def _has_concept(text: str, patterns: list) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)

def _short(pid: str) -> str:
    """'BraTS-GLI-00000-000' → '00000'"""
    return pid.replace("BraTS-GLI-", "").replace("-000", "")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_reports() -> dict:
    patients = sorted(p.stem for p in DIRS["reference"].glob("*.txt"))
    data = {}
    for pid in patients:
        entry = {}
        for model, d in DIRS.items():
            path = d / f"{pid}.txt"
            entry[model] = _strip_header(path.read_text()) if path.exists() else ""
        data[pid] = entry
    return data


def load_metrics():
    dfs = {}
    for model, path in EVAL_FILES.items():
        dfs[model] = (pd.read_csv(path).set_index("patient_id")[METRIC_COLS]
                      if path.exists() else pd.DataFrame())
    return dfs["base"], dfs["finetuned"]


def build_concept_df(reports: dict) -> pd.DataFrame:
    rows = []
    for pid, versions in reports.items():
        for model, text in versions.items():
            row = {"patient": pid, "model": model}
            for concept, patterns in CONCEPTS.items():
                row[concept] = int(_has_concept(text, patterns))
            rows.append(row)
    return pd.DataFrame(rows)


# ── Plot 1: metric comparison ─────────────────────────────────────────────────

def plot_metric_comparison(base_df: pd.DataFrame, ft_df: pd.DataFrame):
    if base_df.empty or ft_df.empty:
        return
    patients = base_df.index.tolist()
    short    = [_short(p) for p in patients]
    x, w     = np.arange(len(patients)), 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Evaluation Metrics: Base vs Finetuned Model", fontsize=14, fontweight="bold")

    for ax, col, label in zip(axes.flat, METRIC_COLS, METRIC_LABELS):
        b = base_df[col].reindex(patients).values
        f = ft_df[col].reindex(patients).values
        ax.bar(x - w/2, b, w, label="Base",      color="#4472C4", alpha=0.85)
        ax.bar(x + w/2, f, w, label="Finetuned", color="#ED7D31", alpha=0.85)
        ax.set_title(label, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(short, rotation=35, ha="right")
        ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.legend(fontsize=8)
        for offset, vals in [(-w/2, b), (w/2, f)]:
            for xi, v in zip(x + offset, vals):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(OUT / "metric_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] metric_comparison.png")


# ── Plot 2: finetuning delta ──────────────────────────────────────────────────

def plot_metric_delta(base_df: pd.DataFrame, ft_df: pd.DataFrame):
    if base_df.empty or ft_df.empty:
        return
    patients = base_df.index.tolist()
    short    = [_short(p) for p in patients]
    delta    = ft_df[METRIC_COLS] - base_df[METRIC_COLS]
    x, w     = np.arange(len(patients)), 0.18
    colors   = ["#4472C4", "#70AD47", "#FFC000", "#7030A0"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (col, label, color) in enumerate(zip(METRIC_COLS, METRIC_LABELS, colors)):
        vals = delta[col].reindex(patients).values
        ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=color, alpha=0.85)
        for xi, v in zip(x + (i - 1.5) * w, vals):
            if abs(v) > 0.001:
                ax.text(xi, v + (0.002 if v >= 0 else -0.009), f"{v:+.3f}",
                        ha="center", va="bottom", fontsize=6.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=35, ha="right")
    ax.set_ylabel("Δ Score  (Finetuned − Base)")
    ax.set_title("Effect of QLoRA Finetuning per Patient  (positive = improvement)",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "finetuning_delta.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] finetuning_delta.png")


# ── Plot 3: medical concept heatmap ──────────────────────────────────────────

def plot_concept_heatmap(concept_df: pd.DataFrame):
    concept_names = list(CONCEPTS.keys())
    patients      = sorted(concept_df["patient"].unique())
    short         = [_short(p) for p in patients]
    cmap          = LinearSegmentedColormap.from_list("cov", ["#F2F2F2", "#1F4E79"])

    fig, axes = plt.subplots(1, 3, figsize=(22, 9), sharey=True)
    fig.suptitle("Medical Concept Coverage by Model", fontsize=14, fontweight="bold")

    for ax, model, title in zip(
        axes,
        ["reference", "base", "finetuned"],
        ["Reference (DeepSeek R1)", "Base Model (Qwen2.5-VL-3B)", "Finetuned (QLoRA)"],
    ):
        sub = concept_df[concept_df["model"] == model].set_index("patient")
        mat = sub[concept_names].reindex(patients).T.values.astype(float)
        ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(patients)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(concept_names)))
        if model == "reference":
            ax.set_yticklabels([c.replace("_", " ") for c in concept_names], fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_title(title, fontweight="bold", pad=8)
        for i in range(len(concept_names)):
            for j in range(len(patients)):
                ax.text(j, i, "✓" if mat[i, j] else "·",
                        ha="center", va="center", fontsize=10,
                        color="#1F4E79" if mat[i, j] else "#BFBFBF")

    plt.tight_layout()
    plt.savefig(OUT / "concept_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] concept_heatmap.png")


# ── Plot 4: concept recall ────────────────────────────────────────────────────

def plot_concept_recall(concept_df: pd.DataFrame):
    concept_names = list(CONCEPTS.keys())
    patients      = sorted(concept_df["patient"].unique())
    short         = [_short(p) for p in patients]

    base_recall, ft_recall = [], []
    for pid in patients:
        ref  = concept_df[(concept_df["patient"] == pid) & (concept_df["model"] == "reference")][concept_names].values[0].astype(bool)
        base = concept_df[(concept_df["patient"] == pid) & (concept_df["model"] == "base")][concept_names].values[0]
        ft   = concept_df[(concept_df["patient"] == pid) & (concept_df["model"] == "finetuned")][concept_names].values[0]
        n    = ref.sum()
        base_recall.append(float(base[ref].sum() / n) if n else float("nan"))
        ft_recall.append(  float(ft[ref].sum()   / n) if n else float("nan"))

    x, w = np.arange(len(patients)), 0.30
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, base_recall, w, label="Base Model", color="#4472C4", alpha=0.85)
    ax.bar(x + w/2, ft_recall,   w, label="Finetuned",  color="#ED7D31", alpha=0.85)
    ax.axhline(1.0, color="#7F7F7F", linestyle="--", linewidth=0.9, label="Reference (100%)")
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=35, ha="right")
    ax.set_ylim(0, 1.3); ax.set_ylabel("Recall of Reference Medical Concepts")
    ax.set_title("How many reference medical concepts does each model capture?",
                 fontweight="bold")
    ax.legend()
    for xi, (b, f) in zip(x, zip(base_recall, ft_recall)):
        if not np.isnan(b):
            ax.text(xi - w/2, b + 0.02, f"{b:.0%}", ha="center", fontsize=9,
                    color="#4472C4", fontweight="bold")
        if not np.isnan(f):
            ax.text(xi + w/2, f + 0.02, f"{f:.0%}", ha="center", fontsize=9,
                    color="#ED7D31", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "concept_recall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] concept_recall.png")


# ── Plot 5: template collapse ─────────────────────────────────────────────────

def plot_template_collapse(reports: dict):
    patients = sorted(reports.keys())
    n        = len(patients)
    short    = [_short(p) for p in patients]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Within-Model Report Similarity  (1.0 = identical — template collapse)",
                 fontsize=13, fontweight="bold")

    for ax, model, title in zip(
        axes, ["base", "finetuned"], ["Base Model", "Finetuned (QLoRA)"]
    ):
        mat = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                s = difflib.SequenceMatcher(
                    None, reports[patients[i]][model], reports[patients[j]][model]
                ).ratio()
                mat[i, j] = mat[j, i] = s

        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(short, rotation=45, ha="right")
        ax.set_yticklabels(short)
        ax.set_title(title, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if mat[i, j] > 0.65 else "black")

        off_diag  = [mat[i, j] for i in range(n) for j in range(n) if i != j]
        mean_sim  = np.mean(off_diag)
        ax.set_xlabel(f"Mean pairwise similarity: {mean_sim:.3f}", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT / "template_collapse.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] template_collapse.png")


# ── Qualitative per-patient analysis ─────────────────────────────────────────

CHECKED_CONCEPTS = [
    ("mass_effect",             "Mass effect / herniation"),
    ("midline_shift_mentioned", "Midline shift mentioned"),
    ("midline_shift_quantified","Midline shift quantified (mm)"),
    ("tumor_dimensions",        "Tumor dimensions (cm)"),
    ("necrosis",                "Necrosis"),
    ("ventricle_effect",        "Ventricle compression / entrapment"),
    ("ependymal_invasion",      "Ependymal invasion"),
    ("brats_segments",          "BraTS segments (NCR / ET / ED)"),
    ("diffusion_restriction",   "Diffusion restriction"),
    ("hemorrhage",              "Hemorrhage / susceptibility"),
    ("edema_quantified",        "Edema quantified"),
]

def qualitative_analysis(reports: dict, concept_df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("QUALITATIVE ANALYSIS  —  Per-Patient Error Profile")
    print("=" * 72)

    for pid in sorted(reports):
        short  = _short(pid)
        ref_r  = concept_df[(concept_df["patient"] == pid) & (concept_df["model"] == "reference")].iloc[0]
        base_r = concept_df[(concept_df["patient"] == pid) & (concept_df["model"] == "base")].iloc[0]
        ft_r   = concept_df[(concept_df["patient"] == pid) & (concept_df["model"] == "finetuned")].iloc[0]

        print(f"\n┌─ Patient {short} " + "─" * 55)
        print(f"  {'Feature':<38} {'Ref':>5} {'Base':>6} {'FT':>6}  Note")
        print("  " + "─" * 68)

        s = lambda v: "✓" if v else "✗"
        for concept, label in CHECKED_CONCEPTS:
            r, b, f = int(ref_r[concept]), int(base_r[concept]), int(ft_r[concept])
            if   r and not b and not f: note = "⚠ OMITTED by both models"
            elif r and not b and f:     note = "↑ Finetuning fixed this"
            elif r and b and not f:     note = "↓ Finetuning regressed"
            elif not r and (b or f):    note = "⚠ Possible hallucination"
            elif r and b and f:         note = "✓ Both models correct"
            else:                       note = "— Not in reference"
            print(f"  {label:<38} {s(r):>5} {s(b):>6} {s(f):>6}  {note}")

    # Template collapse summary
    patients = sorted(reports.keys())
    print(f"\n{'─' * 72}")
    print("TEMPLATE COLLAPSE SUMMARY")
    for model in ["base", "finetuned"]:
        texts = [reports[p][model] for p in patients]
        sims  = [
            difflib.SequenceMatcher(None, texts[i], texts[j]).ratio()
            for i in range(len(texts)) for j in range(i + 1, len(texts))
        ]
        mean  = np.mean(sims) if sims else 0.0
        level = ("CRITICAL" if mean > 0.70 else "MODERATE" if mean > 0.50 else "OK")
        print(f"  {model:<12}  mean pairwise similarity = {mean:.3f}  [{level}]")
        if mean > 0.50:
            print(f"  {'':12}  ⚠ Model generates near-identical reports for different patients.")


# ── Summary CSV ───────────────────────────────────────────────────────────────

def save_summary_csv(concept_df: pd.DataFrame, base_df: pd.DataFrame, ft_df: pd.DataFrame):
    concept_names = list(CONCEPTS.keys())
    rows = []
    for pid in sorted(concept_df["patient"].unique()):
        ref_r  = concept_df[(concept_df["patient"] == pid) & (concept_df["model"] == "reference")].iloc[0]
        base_r = concept_df[(concept_df["patient"] == pid) & (concept_df["model"] == "base")].iloc[0]
        ft_r   = concept_df[(concept_df["patient"] == pid) & (concept_df["model"] == "finetuned")].iloc[0]
        ref_present  = np.array([ref_r[c]  for c in concept_names]).astype(bool)
        base_present = np.array([base_r[c] for c in concept_names])
        ft_present   = np.array([ft_r[c]   for c in concept_names])
        n = ref_present.sum()

        row = {
            "patient_id":         pid,
            "ref_concepts_found": int(n),
            "base_concept_recall": float(base_present[ref_present].sum() / n) if n else float("nan"),
            "ft_concept_recall":   float(ft_present[ref_present].sum()   / n) if n else float("nan"),
        }
        for col in METRIC_COLS:
            row[f"base_{col}"] = base_df.loc[pid, col] if (not base_df.empty and pid in base_df.index) else float("nan")
            row[f"ft_{col}"]   = ft_df.loc[pid, col]   if (not ft_df.empty   and pid in ft_df.index)   else float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "analysis_summary.csv", index=False)
    print("[✓] analysis_summary.csv")
    print()
    print(df[["patient_id", "ref_concepts_found", "base_concept_recall",
              "ft_concept_recall", "base_rouge_l", "ft_rouge_l",
              "base_bertscore_f1", "ft_bertscore_f1"]].to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("━" * 72)
    print("BraTS Report Analyzer")
    print("━" * 72)

    reports = load_reports()
    print(f"\nPatients loaded: {len(reports)}")
    for pid in sorted(reports):
        print(f"  {_short(pid)}  "
              f"ref={'✓' if reports[pid]['reference'] else '✗'}  "
              f"base={'✓' if reports[pid]['base'] else '✗'}  "
              f"finetuned={'✓' if reports[pid]['finetuned'] else '✗'}")

    concept_df      = build_concept_df(reports)
    base_df, ft_df  = load_metrics()

    print("\nGenerating plots...")
    plot_metric_comparison(base_df, ft_df)
    plot_metric_delta(base_df, ft_df)
    plot_concept_heatmap(concept_df)
    plot_concept_recall(concept_df)
    plot_template_collapse(reports)

    qualitative_analysis(reports, concept_df)
    save_summary_csv(concept_df, base_df, ft_df)

    print(f"\n✓ All outputs → {OUT}/")


if __name__ == "__main__":
    main()
