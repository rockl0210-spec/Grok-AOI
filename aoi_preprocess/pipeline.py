"""Orchestration: load -> illumination -> denoise -> ROI -> outputs."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .denoise import denoise, estimate_noise_sigma
from .illumination import correct_illumination
from .io_utils import list_inputs, load_calibration_frame, load_image
from .outputs import write_json, write_outputs
from .roi import build_roi, detect_wafer

log = logging.getLogger("aoi")


def _load_mask_png(path, shape_hw):
    buf = np.fromfile(str(path), dtype=np.uint8)
    m = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if m is None or m.shape != tuple(shape_hw):
        raise ValueError(f"fixture.mask_png {path} must be a grayscale PNG of size {shape_hw[::-1]} (WxH)")
    return (m > 127).astype(np.uint8)


def _background_weight(substrate, cfg):
    """Pixels trusted for the background fit: inside the (rectangular / circular) substrate mask, eroded a
    little so the edge roll-off does not bias it, minus the polygon fixtures given in the config."""
    icfg = cfg["illumination"]
    w = substrate.copy()
    e = int(icfg["weight_erode_px"])
    if e > 0:
        w = cv2.erode(w, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1)))
    if icfg["exclude_fixture_polygons"]:
        from .roi import polygon_mask
        w &= (1 - cv2.dilate(polygon_mask(w.shape, cfg["roi"]["fixture"]), np.ones((7, 7), np.uint8))).astype(np.uint8)
    return w


def _roi_intensity_stats(img, valid):
    v = img[valid > 0]
    if v.size == 0:
        return {}
    p = np.percentile(v, [1, 50, 99])
    return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max()),
            "p01": float(p[0]), "median": float(p[1]), "p99": float(p[2])}


def process_image(path: Path, cfg, out_dir: Path, flat=None, dark=None, return_arrays=False):
    t0 = time.time()
    li = load_image(path, cfg["input"])                                   # Stage 1
    raw = li.data
    wcfg = cfg["roi"]["wafer"]
    detect_on = wcfg.choice("detect_on", ("raw", "corrected"))

    wafer = winfo = None
    weight = None
    if detect_on == "raw":                      # boundary from raw image -> can weight the background fit
        wafer, winfo = detect_wafer(raw, wcfg)
        if cfg["illumination"]["use_wafer_weight"]:
            weight = _background_weight(wafer, cfg)
    corrected, iinfo = correct_illumination(raw, cfg["illumination"], flat, dark, weight)   # Stage 2
    bg = iinfo.pop("_bg", None)
    den = denoise(corrected, cfg["denoise"])                               # Stage 3
    if wafer is None:
        wafer, winfo = detect_wafer(den, wcfg)
    comps, rstats = build_roi(wafer, winfo, den, cfg["roi"], _load_mask_png)               # Stage 4
    files = write_outputs(out_dir, path.stem, cleaned=den, corrected=corrected, denoised=den,  # Stage 5
                          comps=comps, bit_depth=li.meta["bit_depth"], cfg_out=cfg["output"])
    v = comps["valid"]
    report = {
        "tool": {"name": "aoi_preprocess", "version": __version__},
        "input": li.meta,
        "illumination": iinfo,
        "output_bit_depth": 16 if li.meta["bit_depth"] in (16, 32) else 8,
        "roi": rstats,
        "stats_valid_roi": {
            "raw": _roi_intensity_stats(raw, v),
            "corrected": _roi_intensity_stats(corrected, v),
            "cleaned": _roi_intensity_stats(den, v),
            "noise_sigma_corrected_est": estimate_noise_sigma(corrected, v),
            "noise_sigma_cleaned_est": estimate_noise_sigma(den, v),
        },
        "files": files,
        "elapsed_s": round(time.time() - t0, 3),
        "parameters": cfg.to_dict(),
    }
    write_json(out_dir / f"{path.stem}_report.json", report)
    files["report"] = f"{path.stem}_report.json"
    if return_arrays:
        return report, {"raw": raw, "corrected": corrected, "denoised": den, "bg": bg, **comps}
    return report


def run(input_path, cfg, out_dir, flat_path=None, dark_path=None):
    out_dir = Path(out_dir)
    icfg = cfg["illumination"]
    flat_path = flat_path or icfg["flat_path"]
    dark_path = dark_path or icfg["dark_path"]
    flat = load_calibration_frame(flat_path, cfg["input"]) if flat_path else None
    dark = load_calibration_frame(dark_path, cfg["input"]) if dark_path else None
    if dark is not None and flat is None:
        log.warning("dark frame given without flat frame: ignored (background estimation used instead)")
        dark = None
    inputs = list_inputs(input_path, bool(cfg["input"]["recursive"]))
    if not inputs:
        raise FileNotFoundError(f"no PNG/BMP/TIFF files in {input_path}")
    reports, failed = [], []
    for p in inputs:
        try:
            r = process_image(p, cfg, out_dir, flat, dark)
            log.info("%s: valid ROI %.1f%% of wafer, %s, %.1fs", p.name, 100 * r["roi"]["valid_frac_of_wafer"],
                     r["illumination"]["method"], r["elapsed_s"])
            reports.append(r)
        except Exception as e:  # keep batch going
            log.error("%s FAILED: %s", p.name, e, exc_info=log.isEnabledFor(logging.DEBUG))
            failed.append(p.name)
    return reports, failed
