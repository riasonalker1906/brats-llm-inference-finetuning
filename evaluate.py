#!/usr/bin/env python3
"""
evaluate.py
===========
Pipeline position: STEP 4 of 4  (final step)
Depends on: run_inference.py  (./reports/ must exist)
            match_reports.py  (./reference_reports/ must exist)
Run after:  match_reports.py

Computes the 7 radiology-report evaluation metrics from the ReMIND paper
for each patient, saves per-patient results to ./evaluation/results.csv,
and prints a mean ± std summary table.

Metrics
-------
  1. ROUGE-L       — Longest common subsequence recall/precision/F1
                     Package: rouge-score
  2. METEOR        — Unigram matching with synonym/stem support
                     Package: nltk (wordnet, punkt_tab)
  3. BLEU-4        — 4-gram precision with brevity penalty
                     Package: nltk
  4. BERTScore-F1  — Contextual token similarity (DeBERTa-XL backbone)
                     Package: bert-score
  5. RadGraph-F1   — Entity+relation F1 over the RadGraph clinical NLP graph
                     Package: f1radgraph
                     NOTE: f1radgraph requires Java ≥ 11 on PATH.  If Java is
                     absent the score is set to NaN and a warning is printed.
  6. RaTEScore     — Official implementation attempted via
                     transformers pipeline "YtongXie/RaTEScore".
                     APPROXIMATION FALLBACK: BERTScore-F1 computed with the
                     radiology-specific backbone
                     "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
                     is used if the official model is unavailable.  The column
                     header is suffixed with "(approx)" in that case.
  7. GREEN         — Official implementation attempted via
                     transformers pipeline "StanfordMIMI/GREEN".
                     APPROXIMATION FALLBACK: Same BiomedBERT BERTScore as above
                     if the official model is unavailable.  Column suffixed with
                     "(approx)" accordingly.

Output
------
  ./evaluation/results.csv   — one row per patient, one column per metric
  Console                    — mean ± std table across all patients
"""

import csv
import sys
import warnings
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from bert_score import score as bertscore_fn
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Download required NLTK data (no-ops if already present)
nltk.download("wordnet",    quiet=True)
nltk.download("punkt",      quiet=True)
nltk.download("punkt_tab",  quiet=True)
nltk.download("omw-1.4",    quiet=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPORTS_DIR  = Path("./reports")
REF_DIR      = Path("./reference_reports")
EVAL_DIR     = Path("./evaluation")
RESULTS_CSV  = EVAL_DIR / "results.csv"

# BERTScore backbone for standard BERTScore-F1
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"

# Radiology-specific BERTScore backbone used as fallback for RaTEScore / GREEN
BIOMEDBERT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_rouge_l(hypothesis: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, hypothesis)["rougeL"].fmeasure


def compute_meteor(hypothesis: str, reference: str) -> float:
    # meteor_score expects tokenised lists
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    ref_tokens = nltk.word_tokenize(reference.lower())
    return meteor_score([ref_tokens], hyp_tokens)


def compute_bleu4(hypothesis: str, reference: str) -> float:
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    ref_tokens = nltk.word_tokenize(reference.lower())
    sf = SmoothingFunction().method1   # add-one smoothing avoids zero for short texts
    return sentence_bleu(
        [ref_tokens],
        hyp_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=sf,
    )


def compute_bertscore(hypothesis: str, reference: str, model_type: str) -> float:
    import bert_score.utils as bsu
    from bert_score.utils import model2layers

    # Some models (e.g. DeBERTa-XL, BiomedBERT) set model_max_length = 1e30,
    # which causes an OverflowError in the Rust tokenizer's enable_truncation
    # call (int too large to convert).  Temporarily cap it to 512 during
    # encoding, then restore, so the model object is not permanently mutated.
    _orig_sent_encode = bsu.sent_encode

    def _safe_sent_encode(tokenizer, sent):
        original_max = tokenizer.model_max_length
        if original_max > 100_000:
            tokenizer.model_max_length = 512
        result = _orig_sent_encode(tokenizer, sent)
        tokenizer.model_max_length = original_max
        return result

    bsu.sent_encode = _safe_sent_encode
    try:
        # model2layers doesn't include domain-specific models (e.g. BiomedBERT);
        # default to 9 (BERT-base standard) for any unknown model.
        num_layers = model2layers.get(model_type, 9)
        _, _, F1 = bertscore_fn(
            [hypothesis],
            [reference],
            lang="en",
            model_type=model_type,
            num_layers=num_layers,
            verbose=False,
        )
        return F1[0].item()
    finally:
        bsu.sent_encode = _orig_sent_encode


def compute_radgraph_f1(hypothesis: str, reference: str) -> float:
    """
    Compute RadGraph F1 using the `radgraph` package (PhysioNet-gated).
    Returns NaN with a warning if the package is not installed.

    To enable: request access at https://physionet.org/content/radgraph/
    then run:  pip install radgraph
    """
    try:
        from radgraph import F1RadGraph
        scorer = F1RadGraph(reward_level="partial")
        _, _, result, _ = scorer(hyps=[hypothesis], refs=[reference])
        return float(result[0]) if hasattr(result, "__len__") else float(result)
    except ImportError:
        warnings.warn(
            "RadGraph-F1: `radgraph` package not installed (PhysioNet-gated). "
            "See requirements.txt for installation instructions. Returning NaN."
        )
        return float("nan")
    except Exception as exc:
        warnings.warn(f"RadGraph-F1 failed ({exc}). Returning NaN.")
        return float("nan")


# ── RaTEScore ───────────────────────────────────────────────────────────────

_ratescore_pipe   = None   # cached pipeline
_ratescore_approx = False  # True if using the BiomedBERT approximation


def _init_ratescore() -> None:
    global _ratescore_pipe, _ratescore_approx
    if _ratescore_pipe is not None:
        return
    try:
        from transformers import pipeline
        # Official model hosted at HuggingFace by the RaTEScore authors
        _ratescore_pipe = pipeline(
            "text-classification",
            model="YtongXie/RaTEScore",
            trust_remote_code=True,
        )
        _ratescore_approx = False
        print("  RaTEScore: using official YtongXie/RaTEScore model.")
    except Exception as exc:
        warnings.warn(
            f"Could not load YtongXie/RaTEScore ({exc}).\n"
            "  APPROXIMATION: RaTEScore will be computed as BERTScore-F1 "
            f"with {BIOMEDBERT_MODEL}."
        )
        _ratescore_pipe = "approx"
        _ratescore_approx = True


def compute_ratescore(hypothesis: str, reference: str) -> tuple[float, bool]:
    """
    Returns (score, is_approximation).
    Official model takes a concatenated hypothesis+reference string and
    outputs a similarity score in [0, 1].
    """
    _init_ratescore()

    if _ratescore_approx:
        # Approximation: radiology-specific BERTScore
        score = compute_bertscore(hypothesis, reference, BIOMEDBERT_MODEL)
        return score, True

    try:
        # Official pipeline: pass hypothesis and reference as a pair
        result = _ratescore_pipe(
            {"text": hypothesis, "text_pair": reference},
            truncation=True,
        )
        # The model returns a label ("SIMILAR"/"DISSIMILAR") and a score;
        # normalise to [0,1]: use the score directly if label is positive,
        # else 1 - score.
        label = result["label"].upper()
        raw   = result["score"]
        score = raw if "SIMILAR" in label else 1.0 - raw
        return score, False
    except Exception as exc:
        warnings.warn(f"RaTEScore inference failed ({exc}). Falling back to BiomedBERT approx.")
        return compute_bertscore(hypothesis, reference, BIOMEDBERT_MODEL), True


# ── GREEN ────────────────────────────────────────────────────────────────────

_green_pipe   = None
_green_approx = False


def _init_green() -> None:
    global _green_pipe, _green_approx
    if _green_pipe is not None:
        return
    try:
        from transformers import pipeline
        # Official Stanford MIMI GREEN model
        _green_pipe = pipeline(
            "text2text-generation",
            model="StanfordMIMI/GREEN",
            trust_remote_code=True,
        )
        _green_approx = False
        print("  GREEN: using official StanfordMIMI/GREEN model.")
    except Exception as exc:
        warnings.warn(
            f"Could not load StanfordMIMI/GREEN ({exc}).\n"
            "  APPROXIMATION: GREEN will be computed as BERTScore-F1 "
            f"with {BIOMEDBERT_MODEL}."
        )
        _green_pipe = "approx"
        _green_approx = True


def compute_green(hypothesis: str, reference: str) -> tuple[float, bool]:
    """
    Returns (score, is_approximation).
    GREEN uses a fine-tuned generative model to assess clinical accuracy of
    the hypothesis against the reference, returning a score in [0, 1].
    """
    _init_green()

    if _green_approx:
        score = compute_bertscore(hypothesis, reference, BIOMEDBERT_MODEL)
        return score, True

    try:
        # GREEN expects "hypothesis [SEP] reference" or similar concatenation;
        # exact format depends on the model's preprocessing — pass both texts.
        prompt = f"Hypothesis: {hypothesis}\nReference: {reference}"
        result = _green_pipe(prompt, max_new_tokens=10, truncation=True)
        generated = result[0]["generated_text"].strip()
        # The model outputs a numeric score string or a label; parse accordingly
        try:
            score = float(generated.split()[0])
            score = max(0.0, min(1.0, score))  # clamp to [0,1]
        except (ValueError, IndexError):
            # If the output isn't parseable, fall back to BiomedBERT
            warnings.warn(f"GREEN output not parseable ({generated!r}). Using BiomedBERT approx.")
            score = compute_bertscore(hypothesis, reference, BIOMEDBERT_MODEL)
            return score, True
        return score, False
    except Exception as exc:
        warnings.warn(f"GREEN inference failed ({exc}). Falling back to BiomedBERT approx.")
        return compute_bertscore(hypothesis, reference, BIOMEDBERT_MODEL), True


# ---------------------------------------------------------------------------
# Load reports
# ---------------------------------------------------------------------------

def load_reports(directory: Path) -> dict[str, str]:
    """
    Load all .txt files from directory.
    Returns {patient_id: report_text}.
    """
    reports = {}
    for p in sorted(directory.glob("*.txt")):
        # Strip the synthetic-reference header lines (lines starting with '#')
        raw = p.read_text(encoding="utf-8")
        text = "\n".join(
            line for line in raw.splitlines() if not line.startswith("#")
        ).strip()
        reports[p.stem] = text
    return reports


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Sanity checks ─────────────────────────────────────────────────────
    for d, label in [(REPORTS_DIR, "run_inference.py"), (REF_DIR, "match_reports.py")]:
        if not d.exists() or not list(d.glob("*.txt")):
            sys.exit(f"ERROR: {d}/ is empty or missing. Run {label} first.")

    EVAL_DIR.mkdir(exist_ok=True)

    hypotheses = load_reports(REPORTS_DIR)
    references = load_reports(REF_DIR)

    # Patients present in both directories
    patients = sorted(set(hypotheses) & set(references))
    if not patients:
        sys.exit(
            "ERROR: No patient IDs in common between ./reports/ and "
            "./reference_reports/. Check that both directories contain "
            "matching filenames."
        )

    only_hyp = set(hypotheses) - set(references)
    only_ref = set(references) - set(hypotheses)
    if only_hyp:
        print(f"WARNING: {len(only_hyp)} patient(s) in reports/ but not reference_reports/: {only_hyp}")
    if only_ref:
        print(f"WARNING: {len(only_ref)} patient(s) in reference_reports/ but not reports/: {only_ref}")

    print(f"\nEvaluating {len(patients)} patient(s) …\n")

    # ── Initialise lazy-loaded models once (avoids repeated downloads) ────
    _init_ratescore()
    _init_green()

    rows: list[dict] = []

    for pid in patients:
        hyp = hypotheses[pid]
        ref = references[pid]

        print(f"  {pid}")

        # 1. ROUGE-L
        rouge_l = compute_rouge_l(hyp, ref)
        print(f"    ROUGE-L          = {rouge_l:.4f}")

        # 2. METEOR
        meteor = compute_meteor(hyp, ref)
        print(f"    METEOR           = {meteor:.4f}")

        # 3. BLEU-4
        bleu4 = compute_bleu4(hyp, ref)
        print(f"    BLEU-4           = {bleu4:.4f}")

        # 4. BERTScore-F1
        bscore = compute_bertscore(hyp, ref, BERTSCORE_MODEL)
        print(f"    BERTScore-F1     = {bscore:.4f}")

        # 5. RadGraph-F1
        radgraph = compute_radgraph_f1(hyp, ref)
        rg_str = f"{radgraph:.4f}" if not np.isnan(radgraph) else "NaN (Java missing)"
        print(f"    RadGraph-F1      = {rg_str}")

        # 6. RaTEScore
        ratescore, rate_approx = compute_ratescore(hyp, ref)
        rate_label = "RaTEScore(approx)" if rate_approx else "RaTEScore"
        print(f"    {rate_label:<22} = {ratescore:.4f}")

        # 7. GREEN
        green, green_approx = compute_green(hyp, ref)
        green_label = "GREEN(approx)" if green_approx else "GREEN"
        print(f"    {green_label:<22} = {green:.4f}\n")

        rows.append({
            "patient_id":     pid,
            "rouge_l":        rouge_l,
            "meteor":         meteor,
            "bleu_4":         bleu4,
            "bertscore_f1":   bscore,
            "radgraph_f1":    radgraph,
            "ratescore":      ratescore,
            "ratescore_approx": rate_approx,
            "green":          green,
            "green_approx":   green_approx,
        })

    # ── Save CSV ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Per-patient results saved → {RESULTS_CSV}")

    # ── Summary table ─────────────────────────────────────────────────────
    metric_cols = ["rouge_l", "meteor", "bleu_4", "bertscore_f1", "radgraph_f1", "ratescore", "green"]

    # Build display names (mark approximations)
    display_names = {
        "rouge_l":      "ROUGE-L",
        "meteor":       "METEOR",
        "bleu_4":       "BLEU-4",
        "bertscore_f1": "BERTScore-F1",
        "radgraph_f1":  "RadGraph-F1",
        "ratescore":    "RaTEScore" + (" (approx)" if _ratescore_approx else ""),
        "green":        "GREEN"     + (" (approx)" if _green_approx     else ""),
    }

    print("\n" + "=" * 52)
    print(f"{'Metric':<28}  {'Mean':>8}  {'Std':>8}")
    print("=" * 52)
    for col in metric_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            print(f"  {display_names[col]:<26}  {'N/A':>8}  {'N/A':>8}")
        else:
            print(f"  {display_names[col]:<26}  {vals.mean():>8.4f}  {vals.std():>8.4f}")
    print("=" * 52)
    print(f"\nEvaluation complete. Results in {EVAL_DIR}/")


if __name__ == "__main__":
    main()
