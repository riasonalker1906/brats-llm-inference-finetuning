# Analysis Notes

This document explains what the three analysis scripts do, why each design decision was made, and how to interpret the outputs. Intended for someone reading the repo cold.

---

## What the scripts produce

| Script | Purpose | Outputs |
|---|---|---|
| `analyze_reports.py` | Compare reference / base / finetuned reports | `analysis_output/` — 5 plots + CSV |
| `error_analysis.py` | Categorise model errors, scale with N patients | `error_analysis/` — per-patient CSV, frequency charts |
| `download_new_patients.py` | Fetch 7 new patients from Kaggle | adds to `brats_data/` |

---

## analyze_reports.py — Design rationale

### Why regex-based medical concept extraction?

The standard evaluation metrics (ROUGE-L, METEOR, BERTScore) measure surface-level text similarity. They cannot tell you *which* clinically important findings the model missed. A report that gets ROUGE-L = 0.17 may be missing mass effect quantification entirely while scoring the same as one that mentions it.

The 18 medical concepts (defined in the `CONCEPTS` dict) were chosen to cover the key findings a neuroradiologist would expect in a glioma report:
- **Structural effects**: midline shift (mentioned + quantified separately), mass effect, ventricle compression/entrapment
- **Tumour characteristics**: necrosis, ring enhancement, haemorrhage, dimensions
- **Invasion patterns**: ependymal invasion, white matter invasion
- **BraTS-specific**: NCR/ET/ED segmentation references
- **Functional**: diffusion restriction, vascularity
- **Epidemiology**: multifocality

Regex is deliberately conservative — patterns require the actual clinical term so false positives are rare. The downside is false negatives (a model that says "the mass displaces midline structures" might miss the `midline_shift_mentioned` pattern). This is acceptable for now; LLM-based concept extraction would be the next step.

### Why is "concept recall" more informative than ROUGE-L for this task?

ROUGE-L penalises paraphrasing equally to actual omissions. Concept recall only asks: *did the model mention the things the reference mentioned?* It rewards inclusion of clinically relevant content regardless of phrasing. A model that says "mass effect is present" and one that says "there is midline displacement with contralateral shift" both score 1 for `mass_effect`.

Note: concept recall is computed **relative to what the reference contained** — it's not measuring absolute ground truth, it's measuring agreement with the reference. The reference reports have their own quality problems (see below).

### Template collapse detection

The pairwise similarity heatmap uses `difflib.SequenceMatcher` (character-level longest-common-subsequence ratio). A score of 1.0 means two reports are character-identical. A score above 0.7 for cross-patient pairs indicates the model is essentially copying a template rather than reading the scan.

This is one of the most important diagnostics in this pipeline. If a model can generate a plausible-sounding report without having looked at the scan, it provides zero diagnostic value.

---

## error_analysis.py — Design rationale

### Why a fixed taxonomy of 10 error types?

Free-text error analysis doesn't scale. With 5 patients you can read every report; with 100 you cannot. The taxonomy converts qualitative judgement into binary signals that aggregate naturally.

The 10 categories cover three fundamentally different failure modes:

**Omission errors** (model fails to mention something the reference contains):
- `OMIT_MASS_EFFECT`, `OMIT_QUANTIFICATION`, `OMIT_NECROSIS`, `OMIT_HEMORRHAGE`, `OMIT_VENTRICLE_EFFECT`, `OMIT_INVASION`
- These are the most medically dangerous — a radiologist not mentioning mass effect in a patient with 10mm midline shift is a significant miss.

**Hallucination / confabulation errors** (model asserts something that contradicts or isn't in the reference):
- `HALLUCINATED_NEGATIVE`: The model says "no mass effect" when the reference describes significant mass effect. This is the inverse of omission — actively wrong rather than silent.
- Note: this is harder to detect with precision. The current implementation checks for explicit negation phrases ("no clear mass effect", "no midline shift") co-occurring with positive reference evidence. It will miss subtle contradictions.

**Systemic / structural errors** (the model's generation process is broken at a higher level):
- `TEMPLATE_COLLAPSE`: The model is generating the same text for different patients. Detectable by cross-patient similarity ≥ 0.70.
- `SPECIFICITY_LOSS`: The model always writes "left frontal lobe" for every patient. As the cohort grows, this fires if ALL patients get the same location phrase.
- `GENERIC_IMPRESSION`: The IMPRESSION paragraph (the clinical conclusion) is ≥ 85% identical across patients. This is particularly bad because the impression is what the referring clinician reads.

### Why these thresholds (0.70, 0.85)?

- **0.70 for template collapse**: Below this, reports can differ just by modality-level bullet points while being structurally identical — that's still meaningful variation. Above 0.70, the boilerplate dominates the content.
- **0.85 for generic impression**: The IMPRESSION is shorter than FINDINGS, so a higher threshold avoids false positives from shared boilerplate ("biopsy is recommended").

Both thresholds can be adjusted in the `ERRORS` dict if you find them too tight or too loose for your patient population.

### Scaling to more patients

`error_analysis.py` requires no changes when you add new patients. It scans `reference_reports/` and `reports/` automatically. Run it after each inference batch:

```bash
python error_analysis.py --model both
```

The aggregate CSVs accumulate over the full cohort, letting you track whether problems improve or worsen as the dataset grows.

---

## What the analysis revealed (5-patient cohort)

### Finding 1: Complete quantification failure (100% of patients, both models)

Neither the base model nor the finetuned model ever produces a numeric measurement (mm, cm, %). The reference reports mention midline shift in mm and tumour volumes in cm³. This is not a small omission — quantification is the primary reason a radiologist reads an MRI report rather than just the clinical summary.

**Root cause hypothesis**: The model is generating from visual features (MRI frames), but Qwen2.5-VL-3B-Instruct was never trained to produce radiology measurements. It knows what findings *look like* but not how to translate visual extent into spatial measurements. This would require either supervised fine-tuning on measurement-annotated data or a post-processing step that runs a segmentation model and injects measurements into the prompt.

### Finding 2: Moderate template collapse (mean similarity 0.56–0.66)

The base model's reports are 56% text-similar to each other across patients; the finetuned model worsens this to 66%. This means the model has learned a radiology report *format* but not patient-specific content extraction. Reports 00002, 00003, 00005 from the base model are character-for-character identical in their bullet structure.

**Why finetuning made it worse**: The QLoRA adapter was presumably trained on a small set of report examples with consistent structure. This reinforced the template further. To fix this, fine-tuning data needs high *diversity* of findings, not just consistent formatting.

### Finding 3: Finetuning trade-off — fixes mass effect, introduces hallucinated negatives

Finetuning reduced `OMIT_MASS_EFFECT` from 2 → 0 patients and `GENERIC_IMPRESSION` from 4 → 2. However, it increased `HALLUCINATED_NEGATIVE` from 2 → 4 patients. The finetuned model mentions mass effect more often — but sometimes when it isn't in the reference, and simultaneously uses negation phrases like "no clear mass effect" more aggressively elsewhere.

This is a classic precision/recall trade-off in a model that doesn't truly *see* mass effect but has learned that glioma reports should mention it.

### Finding 4: Location specificity never differentiates

Every model report for every patient says "left frontal lobe". BraTS-2023 contains tumours across all brain regions. This likely reflects a mode collapse during inference: the model sees a large enhancing mass and outputs the statistically most common glioma location in its training data.

### Finding 5: Reference quality is variable

The reference reports (from KurtLabUW BTReport, generated by DeepSeek R1 1.5B from VASARI metadata) contain garbled passages, mixed-language text, and template artefacts. This means concept recall figures are lower bounds — some "missed" concepts may simply not have been well-expressed in the reference either. Any evaluation that treats these references as gold standard should be interpreted with caution.

---

## How to expand to new patients

```bash
# 1. Download 7 new patients (Kaggle credentials required)
python download_new_patients.py --n 7

# If Kaggle dataset is unavailable, download from Synapse manually:
# https://www.synapse.org/#!Synapse:syn51514105
# then:
python download_new_patients.py --skip_kaggle --src /path/to/extracted/

# 2. Run inference on all patients in brats_data/
python run_inference.py

# 3. Fetch reference reports
python match_reports.py

# 4. Compute NLP metrics
python evaluate.py

# 5. Re-run analyses (they auto-detect all available patients)
python analyze_reports.py
python error_analysis.py
```

The next 7 patient IDs available (not yet inferred, present in btreport):
- BraTS-GLI-00008-000
- BraTS-GLI-00009-000
- BraTS-GLI-00011-000
- BraTS-GLI-00012-000
- BraTS-GLI-00014-000
- BraTS-GLI-00016-000
- BraTS-GLI-00017-000

---

## Suggested next steps based on findings

1. **Fix quantification**: Add a segmentation-derived metadata injection step — run a lightweight segmentation model on each patient, extract tumour volume + midline shift in mm, and prepend this as structured text to the VLM prompt. The model then just needs to report the numbers it's given.

2. **Reduce template collapse**: Curate fine-tuning examples with high diversity of tumour location, size, and finding type. Add a "dissimilarity" regularisation objective during training.

3. **Improve location specificity**: Fine-tune on examples that span all brain regions, or add a classification head that predicts location and injects it as a prompt prefix.

4. **Better reference reports**: Replace the DeepSeek R1 generated references with radiologist-annotated reports for at least a subset of patients, and use those as the evaluation gold standard.

5. **Increase cohort to ≥ 30 patients** before drawing conclusions about error frequency rates — with 5 patients, a single patient difference is 20 percentage points.
