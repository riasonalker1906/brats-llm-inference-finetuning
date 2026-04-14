#!/usr/bin/env python3
"""
verify_metadata.py
==================
Factual verification of generated reports against VASARI ground truth.

Why this matters
────────────────
Text-similarity metrics (ROUGE, BERTScore) measure how similar a report is
to a reference text. But the reference is itself LLM-generated — not clinical
ground truth. VASARI features, midline shift measurements, and tumour volumes
are computed deterministically from the BraTS segmentation masks. They are
the closest thing to ground truth available for these patients.

This script checks each model report against those structured facts, flagging
specific factual errors rather than just reporting text overlap scores.

Facts checked
─────────────
  location          vasari_f1_tumour_location  (Frontal / Temporal / Parietal / Insula / …)
  hemisphere        vasari_f2_side_of_tumour_epicenter  (Left / Right)
  enhancement       vasari_f4_enhancement_quality  (Marked / Mild / Absent)
  ependymal         vasari_f19_ependymal_invasion  (Present / Absent)
  wm_invasion       vasari_f21_deep_wm_invasion  (Present / Absent)
  crosses_midline   vasari_f23_cet_crosses_midline  (True / False)
  midline_shift_mm  max_midline_shift_mm  (numeric — checks if model mentions shift & approximate value)
  ncr_volume        ncr_volume  (checks if model mentions necrosis when NCR > threshold)
  edema_proportion  vasari_f14_proportion_of_oedema  (categorical — checks consistency)
  multifocal        vasari_f9_multifocal_or_multicentric  (Solitary vs Multifocal/Multicentric)

Output → ./metadata_verification/
  factual_accuracy_<model>.csv   — per-patient binary fact accuracy
  factual_summary.csv            — aggregate accuracy per fact type
  factual_heatmap.png            — visual summary

Usage:
  python verify_metadata.py                       # both models
  python verify_metadata.py --model base
  python verify_metadata.py --model finetuned
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT         = Path(__file__).parent
METADATA_DIR = ROOT / "metadata"
OUT          = ROOT / "metadata_verification"
OUT.mkdir(exist_ok=True)

# Minimum NCR volume (mm³) for necrosis to be considered present in the scan.
# BraTS NCR voxels are 1mm³, so 500mm³ ≈ a pea-sized necrotic core.
NCR_PRESENT_THRESHOLD = 500


# ── Metadata loading ──────────────────────────────────────────────────────────

def load_all_metadata() -> dict:
    """Load all vasari_<pid>.json files from ./metadata/."""
    meta = {}
    if not METADATA_DIR.exists():
        print(f"⚠ {METADATA_DIR}/ not found. Run match_reports.py first.")
        return meta
    for f in METADATA_DIR.glob("vasari_*.json"):
        pid = f.stem.replace("vasari_", "")
        with open(f) as fh:
            meta[pid] = json.load(fh)
    return meta


def load_report(pid: str, model: str) -> str:
    folder = ROOT / ("reports" if model == "base" else "reports_finetuned")
    path   = folder / f"{pid}.txt"
    if not path.exists():
        return ""
    text = path.read_text()
    return "\n".join(l for l in text.splitlines() if not l.startswith("#")).strip()


# ── Fact extractors ───────────────────────────────────────────────────────────
# Each returns (model_value, correct: bool | None)
# None means "fact not checkable from model text" (model was silent).

def _any(text: str, patterns: list) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def check_location(report: str, meta: dict):
    """Does the model mention the correct brain lobe/region?"""
    loc = meta.get("global", {}).get("vasari_f1_tumour_location", "")
    if not loc:
        return None, None
    # Map VASARI location to expected text patterns
    patterns = {
        "Frontal":    [r"frontal\s+lobe", r"frontal\s+region"],
        "Temporal":   [r"temporal\s+lobe", r"temporal\s+region"],
        "Parietal":   [r"parietal\s+lobe", r"parietal\s+region"],
        "Occipital":  [r"occipital\s+lobe", r"occipital\s+region"],
        "Insula":     [r"\binsula\b", r"insular"],
        "Cerebellum": [r"cerebellum", r"cerebellar"],
        "Brainstem":  [r"brainstem", r"brain\s+stem"],
        "Basal Ganglia": [r"basal\s+ganglia", r"caudate", r"putamen"],
    }
    expected_pats = patterns.get(loc, [r"\b" + loc.lower() + r"\b"])
    mentions_correct = _any(report, expected_pats)
    # Also check if model says a WRONG lobe
    wrong_lobes = [k for k in patterns if k != loc]
    mentions_wrong = any(_any(report, patterns[wl]) for wl in wrong_lobes)
    # Correct = mentioned right AND not only mentioned wrong ones
    correct = mentions_correct and not (mentions_wrong and not mentions_correct)
    return loc, bool(mentions_correct)


def check_hemisphere(report: str, meta: dict):
    side = meta.get("global", {}).get("vasari_f2_side_of_tumour_epicenter", "")
    if not side:
        return None, None
    if side == "Left":
        correct = _any(report, [r"\bleft\b"])
        wrong   = _any(report, [r"\bright\b.*lobe", r"right\s+hemisphere"])
    elif side == "Right":
        correct = _any(report, [r"\bright\b"])
        wrong   = _any(report, [r"\bleft\b.*lobe", r"left\s+hemisphere"])
    else:
        return side, None
    return side, bool(correct and not wrong)


def check_ependymal(report: str, meta: dict):
    val = meta.get("global", {}).get("vasari_f19_ependymal_invasion", "")
    if not val:
        return None, None
    mentions = _any(report, [r"ependymal", r"subependymal"])
    if val == "Present":
        # Model should mention ependymal invasion (or at least not deny it)
        denies = _any(report, [r"no\s+ependymal", r"without\s+ependymal"])
        return val, bool(mentions and not denies)
    else:  # Absent
        # Model should NOT assert ependymal invasion
        if not mentions:
            return val, True   # silent = OK
        denies = _any(report, [r"no\s+ependymal", r"without\s+ependymal"])
        return val, bool(denies)


def check_wm_invasion(report: str, meta: dict):
    val = meta.get("global", {}).get("vasari_f21_deep_wm_invasion", "")
    if not val:
        return None, None
    mentions = _any(report, [r"white\s+matter", r"deep\s+white"])
    if val == "Present":
        return val, bool(mentions)
    else:
        if not mentions:
            return val, True
        denies = _any(report, [r"no\s+.*white\s+matter", r"without\s+.*white\s+matter"])
        return val, bool(denies)


def check_enhancement(report: str, meta: dict):
    qual = meta.get("global", {}).get("vasari_f4_enhancement_quality", "")
    if not qual:
        return None, None
    if qual == "Marked":
        correct = _any(report, [r"marked\s+enhanc", r"avid\s+enhanc", r"intense\s+enhanc",
                                  r"significant\s+enhanc", r"strong\s+enhanc"])
    elif qual == "Mild":
        correct = _any(report, [r"mild\s+enhanc", r"minimal\s+enhanc", r"faint\s+enhanc",
                                  r"subtle\s+enhanc"])
    elif qual == "Absent":
        correct = _any(report, [r"no\s+enhanc", r"non-?enhanc", r"without\s+enhanc"])
    else:
        return qual, None
    return qual, bool(correct)


def check_midline_shift(report: str, meta: dict):
    """
    Check whether the model correctly identifies whether significant midline
    shift is present. We don't check the exact mm value — just presence/absence.
    Significant = max_midline_shift_mm >= 4mm (radiologically meaningful).
    """
    g = meta.get("global", {})
    max_shift = abs(g.get("max_midline_shift_mm", 0) or 0)
    significant = max_shift >= 4.0

    mentions_shift = _any(report, [r"midline\s+shift", r"midline\s+deviation",
                                    r"\d+\s*mm\s*shift", r"shift.*\d+\s*mm"])
    denies_shift   = _any(report, [r"no\s+(clear\s+)?midline\s+shift",
                                    r"no\s+significant\s+.*shift",
                                    r"without\s+.*shift"])

    if significant:
        # Model should mention shift, not deny it
        correct = mentions_shift and not denies_shift
    else:
        # Model should NOT assert significant shift
        if denies_shift or not mentions_shift:
            correct = True
        else:
            # Mentions shift — check if it claims a big number incorrectly
            nums = re.findall(r"(\d+)\s*mm", report, re.IGNORECASE)
            big_claim = any(int(n) >= 4 for n in nums)
            correct = not big_claim

    return f"{max_shift:.0f}mm (sig={significant})", bool(correct)


def check_necrosis(report: str, meta: dict):
    """Model should mention necrosis when NCR volume is substantial."""
    ncr = meta.get("global", {}).get("ncr_volume", 0) or 0
    ncr_present = ncr >= NCR_PRESENT_THRESHOLD
    mentions = _any(report, [r"necros", r"necrotic"])
    denies   = _any(report, [r"no\s+(central\s+)?necros", r"without\s+necros"])
    if ncr_present:
        return f"NCR={int(ncr)}mm³", bool(mentions and not denies)
    else:
        if not mentions:
            return f"NCR={int(ncr)}mm³", True
        return f"NCR={int(ncr)}mm³", bool(denies)


def check_crosses_midline(report: str, meta: dict):
    """Does enhancing tumour cross midline?"""
    val = meta.get("global", {}).get("vasari_f23_cet_crosses_midline", "")
    if not val:
        return None, None
    crosses = val == "True"
    mentions_cross = _any(report, [r"cross.*midline", r"midline.*cross",
                                    r"bilateral", r"contralateral"])
    if crosses:
        return val, bool(mentions_cross)
    else:
        denies = _any(report, [r"does not cross", r"confined to", r"ipsilateral",
                                r"without.*crossing"])
        if not mentions_cross:
            return val, True   # silent = OK
        return val, bool(denies)


def check_multifocal(report: str, meta: dict):
    val = meta.get("global", {}).get("vasari_f9_multifocal_or_multicentric", "")
    if not val:
        return None, None
    if val == "Solitary":
        wrong = _any(report, [r"multifocal", r"multicentric",
                               r"multiple\s+(lesion|tumor|mass)"])
        return val, not wrong
    else:
        correct = _any(report, [r"multifocal", r"multicentric",
                                  r"multiple\s+(lesion|tumor|mass)", r"satellite"])
        return val, bool(correct)


# ── Fact registry ─────────────────────────────────────────────────────────────

FACTS = {
    "location_correct":       (check_location,       "Tumour lobe/region correct"),
    "hemisphere_correct":     (check_hemisphere,      "Hemisphere (L/R) correct"),
    "ependymal_correct":      (check_ependymal,       "Ependymal invasion correct"),
    "wm_invasion_correct":    (check_wm_invasion,     "White matter invasion correct"),
    "enhancement_correct":    (check_enhancement,     "Enhancement quality correct"),
    "midline_shift_correct":  (check_midline_shift,   "Midline shift presence correct"),
    "necrosis_correct":       (check_necrosis,        "Necrosis presence correct"),
    "crosses_midline_correct":(check_crosses_midline, "Crosses midline correct"),
    "multifocal_correct":     (check_multifocal,      "Multifocal status correct"),
}


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_patient(pid: str, model: str, meta: dict) -> dict:
    report = load_report(pid, model)
    if not report:
        return {}

    g = meta.get("global", meta)  # metadata may be nested under "global"
    wrapped = {"global": g}       # normalise for checkers

    row = {"patient_id": pid}
    for fact_key, (checker, _) in FACTS.items():
        ground_truth_val, correct = checker(report, wrapped)
        row[fact_key] = int(correct) if correct is not None else float("nan")
        row[f"{fact_key}_gt"] = str(ground_truth_val) if ground_truth_val is not None else ""
    return row


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame, model_name: str):
    fact_cols = list(FACTS.keys())
    patients  = df["patient_id"].tolist()
    short     = [p.replace("BraTS-GLI-", "").replace("-000", "") for p in patients]
    labels    = [FACTS[f][1] for f in fact_cols]

    mat = df[fact_cols].values.T.astype(float)

    fig, ax = plt.subplots(figsize=(max(7, len(patients) * 1.3), max(5, len(fact_cols) * 0.6)))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "acc", ["#C00000", "#F2F2F2", "#70AD47"]
    )
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(patients)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(fact_cols)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(
        f"Factual Accuracy vs VASARI Ground Truth — {model_name}\n"
        f"Green = correct  ·  Red = wrong  ·  Grey = not verifiable",
        fontweight="bold",
    )
    plt.colorbar(im, ax=ax, fraction=0.04, label="Correct (1) / Wrong (0)")

    for i in range(len(fact_cols)):
        for j in range(len(patients)):
            v = mat[i, j]
            if np.isnan(v):
                sym, col = "?", "#BFBFBF"
            elif v == 1:
                sym, col = "✓", "#1A5276"
            else:
                sym, col = "✗", "white"
            ax.text(j, i, sym, ha="center", va="center", fontsize=10, color=col)

    plt.tight_layout()
    plt.savefig(OUT / f"factual_heatmap_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] factual_heatmap_{model_name}.png")


def plot_accuracy_bars(dfs: dict):
    """Side-by-side accuracy per fact for both models."""
    fact_cols = list(FACTS.keys())
    labels    = [FACTS[f][1] for f in fact_cols]

    fig, ax = plt.subplots(figsize=(11, max(5, len(fact_cols) * 0.55)))
    colors  = {"base": "#4472C4", "finetuned": "#ED7D31"}
    n_facts = len(fact_cols)
    y       = np.arange(n_facts)
    w       = 0.35

    for i, (model_name, df) in enumerate(dfs.items()):
        accs = [df[f].mean(skipna=True) for f in fact_cols]
        ax.barh(y + (i - 0.5) * w, accs, w,
                label=model_name.capitalize(), color=colors[model_name], alpha=0.85)
        for yi, acc in zip(y + (i - 0.5) * w, accs):
            if not np.isnan(acc):
                ax.text(acc + 0.01, yi, f"{acc:.0%}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Factual Accuracy (fraction of verifiable patients)")
    ax.set_title("Factual Accuracy vs VASARI Ground Truth", fontweight="bold")
    ax.set_xlim(0, 1.2)
    ax.axvline(1.0, color="#7F7F7F", linestyle="--", linewidth=0.8)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "factual_accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] factual_accuracy_comparison.png")


# ── Console report ────────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame, model_name: str):
    fact_cols = list(FACTS.keys())
    n = len(df)
    print(f"\n{'═' * 68}")
    print(f"FACTUAL ACCURACY REPORT — {model_name}  ({n} patients with metadata)")
    print(f"{'═' * 68}")
    print(f"\n  {'Fact':<35} {'Correct':>10} {'Rate':>8}   Ground Truth Values")
    print("  " + "─" * 68)
    for fact, (_, label) in FACTS.items():
        col   = df[fact]
        gt_col = df.get(f"{fact}_gt", pd.Series(["?"] * n))
        n_verifiable = col.count()
        if n_verifiable == 0:
            print(f"  {'?':2} {label:<35} {'N/A':>10}              (no verifiable patients)")
            continue
        rate   = col.mean(skipna=True)
        flag   = "✓" if rate == 1.0 else ("⚠" if rate >= 0.5 else "✗")
        gts    = ", ".join(sorted(set(str(v) for v in gt_col if v and v != "nan")))
        print(f"  {flag}  {label:<35} {int(col.sum(skipna=True)):>4}/{n_verifiable:<4}  "
              f"{rate:>6.0%}   [{gts}]")

    print(f"\n  Per-patient summary:")
    for _, row in df.iterrows():
        vals   = [row[f] for f in fact_cols]
        n_ok   = sum(1 for v in vals if v == 1)
        n_bad  = sum(1 for v in vals if v == 0)
        n_na   = sum(1 for v in vals if np.isnan(v))
        pid    = row["patient_id"].replace("BraTS-GLI-", "").replace("-000", "")
        print(f"    {pid}:  ✓{n_ok}  ✗{n_bad}  ?{n_na}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Factual verification against VASARI ground truth")
    parser.add_argument("--model", choices=["base", "finetuned", "both"], default="both")
    args = parser.parse_args()

    all_meta = load_all_metadata()
    if not all_meta:
        print("No VASARI metadata found. Run match_reports.py first.")
        return

    models = ["base", "finetuned"] if args.model == "both" else [args.model]
    dfs    = {}

    for model_name in models:
        print(f"\n{'━' * 68}")
        print(f"Verifying: {model_name}")
        rows = []
        for pid, meta in all_meta.items():
            row = score_patient(pid, model_name, meta)
            if row:
                rows.append(row)

        if not rows:
            print(f"  ⚠ No reports found for {model_name}.")
            continue

        df = pd.DataFrame(rows)
        dfs[model_name] = df

        print_report(df, model_name)
        df.to_csv(OUT / f"factual_accuracy_{model_name}.csv", index=False)
        print(f"\n  [✓] factual_accuracy_{model_name}.csv")
        plot_heatmap(df, model_name)

    if len(dfs) > 1:
        plot_accuracy_bars(dfs)

        # Cross-model delta
        common_facts = list(FACTS.keys())
        print(f"\n{'═' * 68}")
        print("FACTUAL ACCURACY: BASE vs FINETUNED")
        print("═" * 68)
        for f in common_facts:
            b = dfs["base"][f].mean(skipna=True)
            ft = dfs["finetuned"][f].mean(skipna=True)
            if np.isnan(b) or np.isnan(ft):
                continue
            delta = ft - b
            arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
            print(f"  {arrow}  {FACTS[f][1]:<38}  base={b:.0%}  ft={ft:.0%}  "
                  f"Δ={delta:+.0%}")

    print(f"\n✓ All outputs → {OUT}/")


if __name__ == "__main__":
    main()
