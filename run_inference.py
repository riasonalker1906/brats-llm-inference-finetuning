#!/usr/bin/env python3
"""
run_inference.py
================
End-to-end pipeline: BraTS-2023 NIfTI volumes → video frame sequence →
Qwen3.5-0.8B → structured radiology report (.txt).

Pipeline per patient
--------------------
  1. Load each of the 4 MRI volumes with nibabel.
  2. Infer the through-plane (depth) axis from voxel spacing — the axis with
     the largest voxel dimension is through-plane.
  3. Drop all-zero slices.
  4. Uniformly sample up to MAX_SLICES_PER_VOLUME slices from the remainder.
  5. Min-max normalise the full volume, then cast to uint8.
  6. Resize each slice to SLICE_SIZE and save as PNG.
  7. Concatenate frames across all 4 modalities (t1c → t1n → t2f → t2w).
  8. Pass the full frame sequence to Qwen3.5-0.8B as a single video block.
  9. Generate a structured radiology report (FINDINGS + IMPRESSION).
 10. Write the report to ./reports/<patient_id>.txt.

Model architecture note
-----------------------
  Qwen3.5-0.8B is a hybrid VLM: 24 layers arranged as 6 × (3 × Gated-
  DeltaNet + 1 × Gated-Attention).  Gated DeltaNet is a linear-attention
  variant (O(1) per token at inference), making it memory-efficient while
  retaining full image/video understanding.  The architecture is not yet
  built into mainline transformers, so trust_remote_code=True is required.

Device strategy
---------------
  • MPS (Apple Silicon GPU) is preferred with bfloat16.
  • If MPS is unavailable the model falls back to CPU with float32.
  • device_map="auto" is used so Hugging Face accelerate handles placement.
  • Any individual unsupported MPS operations fall back silently to CPU
    via PyTorch's MPS fallback mechanism (PYTORCH_ENABLE_MPS_FALLBACK=1).

Memory note
-----------
  Qwen3.5-0.8B in bfloat16 requires ~1.6 GB — comfortably fits in 8 GB M1
  unified memory with ample headroom for activations and frame tensors.
"""

import os
import glob
import warnings
import numpy as np
import nibabel as nib
from pathlib import Path
from PIL import Image

import torch

# Enable silent CPU fallback for any MPS-unsupported ops
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BRATS_DIR            = Path("./brats_data")
REPORTS_DIR          = Path("./reports")
FRAMES_DIR           = Path("./frames")
MODEL_ID             = "Qwen/Qwen2.5-VL-3B-Instruct"
# Note: Qwen3.5-0.8B is text-only; Qwen2.5-VL-3B-Instruct is the VLM with video support.

# Modality processing order — must match the order described in the prompt
MODALITIES           = ("t1c", "t1n", "t2f", "t2w")

MAX_SLICES_PER_VOL   = 48       # max slices sampled from each modality volume
SLICE_SIZE           = (128, 128)  # (width, height) for each saved frame

# Qwen2.5-VL visual token budget: min/max pixels per frame.
# 128×128 = 16 384 → kept exactly at SLICE_SIZE; cap prevents upscaling.
FRAME_MIN_PIXELS     = 128 * 128
FRAME_MAX_PIXELS     = 128 * 128

# Maximum new tokens for the generated report
MAX_NEW_TOKENS       = 600


# ---------------------------------------------------------------------------
# Step 1 – Device and model loading
# ---------------------------------------------------------------------------

def setup_device() -> tuple[str, torch.dtype]:
    """
    Choose the best available device (CUDA > MPS > CPU) and matching dtype.
    Returns (device_str, dtype).

    MPS bfloat16 requires macOS 14+; older systems use float16 instead.
    """
    if torch.cuda.is_available():
        print(f"Using CUDA ({torch.cuda.get_device_name(0)}) with bfloat16.")
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        import platform
        mac_ver = tuple(int(x) for x in platform.mac_ver()[0].split(".")[:2])
        if mac_ver >= (14, 0):
            print("Using MPS (Apple Silicon) with bfloat16.")
            return "mps", torch.bfloat16
        else:
            print(f"Using MPS (Apple Silicon) with float16 (macOS {'.'.join(str(x) for x in mac_ver)} < 14; bfloat16 unsupported).")
            return "mps", torch.float16
    print("CUDA/MPS unavailable — falling back to CPU with float32.")
    return "cpu", torch.float32


def load_model(device: str, dtype: torch.dtype):
    """
    Load Qwen3.5-0.8B in the specified dtype.

    trust_remote_code=True is required because Qwen3.5's Gated-DeltaNet
    hybrid architecture is not yet registered in mainline transformers;
    the model repo ships the layer code and registers it at load time.

    device_map="auto" lets accelerate decide layer placement.  On Apple
    Silicon this resolves to MPS when PYTORCH_ENABLE_MPS_FALLBACK=1 is set;
    otherwise to CPU.  We explicitly move the model to `device` afterward
    to guarantee placement.
    """
    print(f"Loading {MODEL_ID} …  (first run downloads ~1.6 GB from HuggingFace)")

    # trust_remote_code required for Gated-DeltaNet architecture registration
    # Load to CPU first to avoid device_map overriding our dtype choice,
    # then move explicitly — this is the only safe path for MPS on macOS < 14.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    print(f"  Moving model to {device} with {dtype} …")
    model = model.to(dtype).to(device)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f"Model loaded on {next(model.parameters()).device}.")
    return model, processor


# ---------------------------------------------------------------------------
# Steps 2–6 – NIfTI volume → list of PIL frames
# ---------------------------------------------------------------------------

def infer_depth_axis(nii_img: nib.Nifti1Image) -> int:
    """
    Return the axis index with the largest voxel spacing.
    The through-plane (slice-selection) direction has the biggest voxel
    dimension in typical clinical MRI acquisitions.
    """
    zooms = np.array(nii_img.header.get_zooms()[:3], dtype=float)
    return int(np.argmax(zooms))


def volume_to_frames(nii_path: str) -> list[Image.Image]:
    """
    Load a single NIfTI volume and return a list of preprocessed PIL frames.

    Steps applied:
      • Move depth axis to dim-0 for easy iteration.
      • Drop all-zero slices (background / padded slices).
      • Uniformly sample up to MAX_SLICES_PER_VOL slices.
      • Min-max normalise the *whole* volume (not per-slice) to preserve
        relative contrast, then cast to uint8.
      • Resize each slice to SLICE_SIZE as a greyscale → RGB image.
    """
    nii  = nib.load(nii_path)
    data = nii.get_fdata(dtype=np.float32)      # always float32 from nibabel

    # ── Infer and re-orient depth axis to dim-0 ───────────────────────────
    depth_axis = infer_depth_axis(nii)
    data = np.moveaxis(data, depth_axis, 0)     # shape: (D, H, W)

    # ── Remove all-zero slices ───────────────────────────────────────────
    nonzero_mask = data.reshape(data.shape[0], -1).any(axis=1)
    data = data[nonzero_mask]

    if data.shape[0] == 0:
        warnings.warn(f"All slices are zero in {nii_path}; returning empty list.")
        return []

    # ── Uniform sampling ─────────────────────────────────────────────────
    n = min(data.shape[0], MAX_SLICES_PER_VOL)
    indices = np.round(np.linspace(0, data.shape[0] - 1, n)).astype(int)
    data = data[indices]                        # shape: (n, H, W)

    # ── Volume-level min-max normalisation → uint8 ───────────────────────
    vmin, vmax = data.min(), data.max()
    if vmax > vmin:
        data = (data - vmin) / (vmax - vmin) * 255.0
    else:
        data = np.zeros_like(data)
    data = data.astype(np.uint8)                # shape: (n, H, W)

    # ── Convert each slice to resized RGB PIL image ───────────────────────
    frames: list[Image.Image] = []
    for i in range(data.shape[0]):
        img = Image.fromarray(data[i], mode="L").convert("RGB")
        img = img.resize(SLICE_SIZE, Image.LANCZOS)
        frames.append(img)

    return frames


# ---------------------------------------------------------------------------
# Step 7 – Pack all 4 modalities into one video frame sequence
# ---------------------------------------------------------------------------

def pack_patient_video(patient_dir: Path) -> list[str]:
    """
    For one patient, load all 4 modality volumes in canonical order,
    process each into frames, save them as PNGs, and return the ordered
    list of frame file paths.

    Frame naming: frame_<global_index:04d>_<modality>.png
    Frames from all modalities are concatenated in MODALITIES order so the
    VLM sees them as one continuous video block.
    """
    pid = patient_dir.name
    frames_out_dir = FRAMES_DIR / pid
    frames_out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[str] = []
    global_idx = 0

    for mod in MODALITIES:
        # Accept both .nii and .nii.gz
        candidates = (
            list(patient_dir.glob(f"*-{mod}.nii"))
            + list(patient_dir.glob(f"*-{mod}.nii.gz"))
        )
        if not candidates:
            warnings.warn(f"No {mod} file found for {pid}; skipping modality.")
            continue

        nii_path = str(candidates[0])
        print(f"    [{mod}]  {Path(nii_path).name}")

        frames = volume_to_frames(nii_path)
        print(f"           → {len(frames)} slices sampled")

        for frame in frames:
            fname = frames_out_dir / f"frame_{global_idx:04d}_{mod}.png"
            # Only write if not already present (allows resuming interrupted runs)
            if not fname.exists():
                frame.save(str(fname))
            frame_paths.append(str(fname))
            global_idx += 1

    return frame_paths


# ---------------------------------------------------------------------------
# Step 8 – Structured radiology prompt
# ---------------------------------------------------------------------------

MODALITY_DESCRIPTIONS = {
    "t1c": "T1-weighted post-contrast (T1c)",
    "t1n": "T1-weighted pre-contrast / native (T1n)",
    "t2f": "T2-weighted FLAIR (T2-FLAIR)",
    "t2w": "T2-weighted (T2w)",
}


def build_prompt() -> str:
    """
    Return the zero-shot prompt.

    The prompt:
      • Declares the expert role.
      • Lists exactly which sequences are present (and in what frame order).
      • Instructs the model NOT to reference absent sequences.
      • Specifies the exact two-section output format.
    """
    # Approximate frame ranges (≤ MAX_SLICES_PER_VOL per modality)
    n = MAX_SLICES_PER_VOL
    modality_legend = "\n".join(
        f"  {i+1}. Frames ~{i*n+1}–{(i+1)*n}: {MODALITY_DESCRIPTIONS[mod]}"
        for i, mod in enumerate(MODALITIES)
    )

    return f"""You are an expert neuroradiologist reviewing a multi-parametric brain MRI study for a glioma patient from the BraTS 2023 dataset.

The video contains axial slices from exactly four MRI sequences, presented in this order:
{modality_legend}

Instructions:
- Base your report solely on the visual information in these four sequences.
- Do NOT mention or reference any sequence not listed above (e.g. DWI, SWI, perfusion, ADC).
- Describe signal characteristics for each relevant sequence where findings are present.
- Use standard neuroradiology terminology (e.g. T2/FLAIR hyperintensity, ring enhancement, mass effect, midline shift).
- If a structure appears normal on a given sequence, state that explicitly.

Generate your report using EXACTLY the following two-section format — no other sections:

FINDINGS:
[Provide a thorough, sequence-by-sequence description of the imaging findings. Include: lesion location (lobe, hemisphere, specific anatomy), morphology (solid vs. cystic vs. necrotic), signal intensity on each available sequence, presence or absence of contrast enhancement on T1c, surrounding T2/FLAIR signal abnormality, mass effect, oedema, midline shift, and involvement of eloquent cortex or deep structures.]

IMPRESSION:
[Provide a concise 2–4 sentence summary of the key findings, most likely diagnosis or differential diagnoses with relative likelihood, and any recommended next steps if applicable.]"""


# ---------------------------------------------------------------------------
# Step 9 – Inference
# ---------------------------------------------------------------------------

def run_inference(
    model,
    processor,
    frame_paths: list[str],
    patient_id: str,
    device: str,
    dtype: torch.dtype,
) -> str:
    """
    Run Qwen3.5-0.8B inference on the ordered frame sequence and return the
    generated report text.

    The full frame list is passed as a single 'video' block so the model
    attends to all modalities in one forward pass.  fps=1 is nominal — the
    model treats each frame as one timestep.
    """
    prompt = build_prompt()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    # Pass the list of PNG paths as a single video block.
                    # qwen-vl-utils will load each path as a frame and
                    # pack them into the visual token sequence.
                    "type":       "video",
                    "video":      frame_paths,
                    "fps":        1.0,
                    "min_pixels": FRAME_MIN_PIXELS,
                    "max_pixels": FRAME_MAX_PIXELS,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    # ── Apply chat template ───────────────────────────────────────────────
    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # ── Extract image / video tensors ─────────────────────────────────────
    image_inputs, video_inputs = process_vision_info(messages)

    # ── Build model input tensors ─────────────────────────────────────────
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # ── Move tensors to device with fallback ──────────────────────────────
    try:
        inputs = inputs.to(device)
    except (RuntimeError, NotImplementedError) as exc:
        warnings.warn(
            f"Could not move inputs to {device} ({exc}); falling back to CPU."
        )
        device = "cpu"
        model  = model.to(torch.float32).to("cpu")
        inputs = inputs.to("cpu")

    # ── Generate ──────────────────────────────────────────────────────────
    print(f"    Generating report for {patient_id} …")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,         # deterministic — important for medical text
        )

    # ── Decode (strip prompt tokens) ──────────────────────────────────────
    # Slice off the input portion so we only decode the newly generated tokens
    generated_ids = [
        out_seq[len(in_seq):]
        for in_seq, out_seq in zip(inputs.input_ids, output_ids)
    ]
    report = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return report


# ---------------------------------------------------------------------------
# Step 10 – Main orchestration loop
# ---------------------------------------------------------------------------

def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    FRAMES_DIR.mkdir(exist_ok=True)

    # ── Discover patient directories ──────────────────────────────────────
    patient_dirs = sorted(
        d for d in BRATS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("BraTS-")
    )

    if not patient_dirs:
        print(
            f"No patient directories found under {BRATS_DIR}/.\n"
            "Run download_data.py first."
        )
        return

    print(f"Found {len(patient_dirs)} patient(s) under {BRATS_DIR}/")

    # ── Load model once — expensive, keep in memory for all patients ──────
    device, dtype = setup_device()
    model, processor = load_model(device, dtype)

    # ── Process each patient ──────────────────────────────────────────────
    for patient_dir in patient_dirs:
        pid         = patient_dir.name
        report_path = REPORTS_DIR / f"{pid}.txt"

        print(f"\n{'='*64}")
        print(f"  Patient: {pid}")
        print(f"{'='*64}")

        if report_path.exists():
            print("  Report already exists — skipping.")
            continue

        # ── Step 2-7: build video frame sequence ──────────────────────────
        print("  Packing MRI volumes into video frames …")
        frame_paths = pack_patient_video(patient_dir)

        if not frame_paths:
            print("  ERROR: no frames produced — skipping patient.")
            continue

        print(f"  Total frames: {len(frame_paths)}  "
              f"(≤{MAX_SLICES_PER_VOL} per modality × {len(MODALITIES)} modalities)")

        # ── Steps 8-9: VLM inference ─────────────────────────────────────
        print("  Running VLM inference …")
        report = run_inference(model, processor, frame_paths, pid, device, dtype)

        # ── Step 10: persist report ───────────────────────────────────────
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(f"# Patient ID : {pid}\n")
            fh.write(f"# Model      : {MODEL_ID}\n")
            fh.write(f"# Modalities : {', '.join(MODALITIES)}\n")
            fh.write("# " + "=" * 62 + "\n\n")
            fh.write(report)
            fh.write("\n")

        print(f"  Report saved → {report_path}")
        print(f"\n  --- Preview (first 400 chars) ---\n{report[:400]}\n  ---")

    print(f"\n{'='*64}")
    print(f"  Done.  Reports written to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
