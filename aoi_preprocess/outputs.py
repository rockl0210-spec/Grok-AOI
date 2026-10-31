"""Stage 5 - outputs.

Written per input image <stem>:
  <stem>_cleaned.png          single-channel gray PNG, SAME bit depth as the input (16-bit in -> 16-bit out; else 8-bit)
  <stem>_preview.png          8-bit single-channel contrast-stretched copy for quick viewing (optional)
  <stem>_mask.png             single-channel 8-bit, 255 = valid ROI, 0 = excluded
  <stem>_overlay_preview.png  RGB(BGR) colour overlay - PREVIEW ONLY, not for downstream algorithms
  <stem>_report.json          parameters + ROI stats + metadata
  (optional) <stem>_corrected.png / <stem>_denoised.png intermediates, <stem>_cleaned16.tif
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .io_utils import to_uint, write_png


def stretch_u8(img: np.ndarray, valid: np.ndarray | None, lo_pct: float, hi_pct: float) -> np.ndarray:
    ref = img[valid > 0] if valid is not None and (valid > 0).any() else img.ravel()
    lo, hi = np.percentile(ref, [lo_pct, hi_pct])
    return np.clip((img - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)


def make_overlay(preview_u8: np.ndarray, comps: dict, alpha: float) -> np.ndarray:
    """BGR overlay: excluded regions tinted (outside=blue, edge band=yellow, fixture=red), valid untouched,
    wafer boundary in green. PREVIEW ONLY."""
    bgr = cv2.cvtColor(preview_u8, cv2.COLOR_GRAY2BGR).astype(np.float32)
    layers = [
        ((comps["wafer"] == 0), (255, 80, 0)),                                            # outside wafer
        (((comps["wafer"] > 0) & (comps["edge_ok"] == 0)), (0, 220, 255)),                # edge band
        (((comps["corner_excl"] > 0) & (comps["wafer"] > 0) & (comps["edge_ok"] > 0)), (0, 140, 255)),  # corner discs
        (((comps["fixture"] > 0) & (comps["wafer"] > 0) & (comps["edge_ok"] > 0)), (0, 0, 255)),  # fixture
    ]
    for m, col in layers:
        bgr[m] = (1 - alpha) * bgr[m] + alpha * np.array(col, np.float32)
    out = np.clip(bgr, 0, 255).astype(np.uint8)
    cs, _ = cv2.findContours(comps["wafer"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, (0, 255, 0), 1)
    cv2.putText(out, "PREVIEW ONLY", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def write_outputs(out_dir: Path, stem: str, *, cleaned, corrected, denoised, comps, bit_depth, cfg_out) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    img = cleaned
    if cfg_out["zero_outside_valid"]:
        img = np.where(comps["valid"] > 0, cleaned, 0).astype(np.float32)
    depth = 16 if bit_depth in (16, 32) else 8   # float input -> 16-bit output
    p = out_dir / f"{stem}_cleaned.png"
    write_png(p, to_uint(img, depth))
    files["cleaned"] = p.name
    if cfg_out["write_tiff16"]:  # optional extra, off by default
        import tifffile
        t = out_dir / f"{stem}_cleaned16.tif"
        tifffile.imwrite(str(t), to_uint(img, 16))
        files["cleaned_tiff16"] = t.name
    prev = stretch_u8(cleaned, comps["valid"], cfg_out["preview_percentiles"][0], cfg_out["preview_percentiles"][1])
    if cfg_out["write_preview"]:
        p = out_dir / f"{stem}_preview.png"
        write_png(p, prev)
        files["preview"] = p.name
    p = out_dir / f"{stem}_mask.png"
    write_png(p, (comps["valid"] * 255).astype(np.uint8))
    files["mask"] = p.name
    p = out_dir / f"{stem}_overlay_preview.png"
    write_png(p, make_overlay(prev, comps, float(cfg_out["overlay_alpha"])))
    files["overlay_preview"] = p.name
    if cfg_out["save_intermediates"]:
        for name, a in (("corrected", corrected), ("denoised", denoised)):
            p = out_dir / f"{stem}_{name}.png"
            write_png(p, to_uint(np.clip(a, 0, 1), depth))
            files[name] = p.name
    return files


def write_json(path: Path, report: dict) -> None:
    def default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=default), encoding="utf-8")


# ----------------------------------------------------------------------------------------- real-data (per-mode) outputs
def class_overlay(preview_u8: np.ndarray, cls: np.ndarray, colors: dict, alpha: float) -> np.ndarray:
    bgr = cv2.cvtColor(preview_u8, cv2.COLOR_GRAY2BGR).astype(np.float32)
    for k, col in colors.items():
        m = cls == k
        if m.any():
            bgr[m] = (1 - alpha) * bgr[m] + alpha * np.array(col, np.float32)
    out = np.clip(bgr, 0, 255).astype(np.uint8)
    cv2.putText(out, "PREVIEW ONLY", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def write_outputs_real(out_dir: Path, debug_dir: Path | None, stem: str, *, cleaned, corrected, cls, bit_depth,
                       cfg_out, colors) -> dict:
    """Agreed contract: <out_dir>/<stem>_cleaned.png (input bit depth, same size) + <stem>_mask.png (0/255, ROI=255).
    Everything else (class map, overlay, corrected) is debug and goes to debug_dir only."""
    out_dir.mkdir(parents=True, exist_ok=True)
    valid = (cls == 1)
    img = np.where(valid, cleaned, 0).astype(np.float32) if cfg_out["zero_outside_valid"] else cleaned
    depth = 16 if bit_depth in (16, 32) else 8
    files = {}
    write_png(out_dir / f"{stem}_cleaned.png", to_uint(img, depth))
    files["cleaned"] = f"{stem}_cleaned.png"
    write_png(out_dir / f"{stem}_mask.png", (valid * 255).astype(np.uint8))
    files["mask"] = f"{stem}_mask.png"
    if debug_dir is not None and cfg_out["write_debug"]:
        debug_dir.mkdir(parents=True, exist_ok=True)
        want = set(cfg_out["debug_files"])
        if "corrected" in want:
            write_png(debug_dir / f"{stem}_corrected.png", to_uint(np.clip(corrected, 0, 1), depth))
        if "classes" in want:
            write_png(debug_dir / f"{stem}_classes.png", cls.astype(np.uint8))
        if "overlay" in want:
            prev = stretch_u8(cleaned, valid.astype(np.uint8), cfg_out["preview_percentiles"][0],
                              cfg_out["preview_percentiles"][1])
            write_png(debug_dir / f"{stem}_overlay_preview.png",
                      class_overlay(prev, cls, colors, float(cfg_out["overlay_alpha"])))
        files["debug_dir"] = str(debug_dir)
    return files
