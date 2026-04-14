# BraTS LLM Inference & Finetuning

Generates structured radiology reports from BraTS-2023 brain MRI scans using a vision-language model (VLM). Includes a QLoRA finetuning pipeline and a full evaluation suite.

## Overview

The pipeline takes multi-modal MRI volumes (T1c, T1n, T2f, T2w) in NIfTI format, converts them into a video-frame sequence, and passes them to **Qwen2.5-VL-3B-Instruct** to generate a structured FINDINGS + IMPRESSION radiology report. A finetuned QLoRA adapter (`qlora_adapter/`) is included for improved report quality.

```
brats_data/ (MRI volumes)
    → run_inference.py       → reports/
    → match_reports.py       → reference_reports/
    → evaluate.py            → evaluation/results.csv
```

## Repo Structure

```
├── brats_data/                   # BraTS-2023 NIfTI volumes (5 sample patients)
├── qlora_adapter/                # QLoRA adapter weights for Qwen2.5-VL-3B-Instruct
├── reports/                      # Generated reports — base model
├── reports_finetuned/            # Generated reports — finetuned model
├── reference_reports/            # Reference reports from KurtLabUW BTReport
├── evaluation/                   # Evaluation results — base model
├── evaluation_finetuned/         # Evaluation results — finetuned model
├── run_inference.py              # Step 2: MRI → VLM → report
├── match_reports.py              # Step 3: fetch reference reports
├── evaluate.py                   # Step 4: compute evaluation metrics
├── brats_finetuning_colab.ipynb  # Finetuning notebook (Colab)
├── brats_inference_colab_new_fixed.ipynb  # Inference notebook (Colab)
├── btreport_brats23.json         # Cached BTReport metadata
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. On Apple Silicon, PyTorch uses the MPS backend automatically; on GPU machines it will use CUDA via `device_map="auto"`.

> **Note:** `RadGraph-F1` requires Java ≥ 11 and gated PhysioNet access. See `requirements.txt` for details. The script falls back to `NaN` if unavailable.

## Pipeline

### Step 1 — Get data
Download BraTS-2023 glioma data from [Kaggle](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection) or [Synapse](https://www.synapse.org/) and place patient folders under `brats_data/`. Five sample patients are already included.

### Step 2 — Run inference

```bash
# Base model
python run_inference.py

# Finetuned model (uses qlora_adapter/)
python run_inference.py --adapter qlora_adapter
```

Reports are written to `reports/<patient_id>.txt` (or `reports_finetuned/` with the adapter).

### Step 3 — Fetch reference reports

```bash
python match_reports.py
```

Pulls reference reports from [KurtLabUW/BTReport](https://github.com/KurtLabUW/BTReport) (DeepSeek R1-generated reports grounded in VASARI quantitative features) and saves them to `reference_reports/`.

### Step 4 — Evaluate

```bash
python evaluate.py
```

Computes 7 metrics against reference reports and saves results to `evaluation/results.csv`:

| Metric | Description |
|---|---|
| ROUGE-L | Longest common subsequence F1 |
| METEOR | Unigram matching with synonym/stem support |
| BLEU-4 | 4-gram precision |
| BERTScore-F1 | Contextual token similarity (DeBERTa-XL) |
| RadGraph-F1 | Clinical entity + relation F1 |
| RaTEScore | Radiology text evaluation (BiomedBERT fallback) |
| GREEN | Clinically-grounded report quality (BiomedBERT fallback) |

## Colab Notebooks

- **`brats_finetuning_colab.ipynb`** — QLoRA finetuning of Qwen2.5-VL-3B-Instruct on BraTS reports
- **`brats_inference_colab_new_fixed.ipynb`** — End-to-end inference on Colab (GPU)

## QLoRA Adapter

The included adapter (`qlora_adapter/`) was trained with:
- Base model: `Qwen/Qwen2.5-VL-3B-Instruct`
- PEFT type: LoRA (r=16, alpha=32, dropout=0.05)
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
