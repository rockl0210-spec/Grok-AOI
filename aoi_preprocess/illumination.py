"""Stage 2 - illumination correction.

Two paths:
  A) flat-field with a measured flat frame (+ optional dark frame):
         corrected = (img - dark) / (flat - dark)            -> rescaled so ROI median == target_level
  B) no flat available -> estimate the slowly varying background from the image itself
         (gaussian | morphological | polynomial) and divide (or subtract) it.

Background estimation is "mask aware": pixels outside the wafer (weight 0) and pixels that
look like defects (robust clipping) do not pull the estimate. The estimate is computed on a
down-scaled copy (cheap, and the background is smooth by definition) and up-sampled.

Rule of thumb: the background scale (sigma / kernel) must be MUCH larger than the largest
defect you care about, otherwise big defects get absorbed into the background and vanish.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger("aoi")


# ----------------------------------------------------------------------------- helpers
def _downscale(img: np.ndarray, w: np.ndarray | None, max_side: int):
    h, wd = img.shape
    s = min(1.0, max_side / max(h, wd))
    if s >= 1.0:
        return img, w, 1.0
    size = (max(8, int(round(wd * s))), max(8, int(round(h * s))))
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    ws = None if w is None else cv2.resize(w, size, interpolation=cv2.INTER_AREA)
    return small, ws, s


def _masked_gaussian(img, w, sigma):
    """Normalised convolution: gaussian average using only pixels with weight w."""
    num = cv2.GaussianBlur(img * w, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(w, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    med = float(np.median(img[w > 0.5])) if (w > 0.5).any() else float(np.median(img))
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), med).astype(np.float32)


def _poly_terms(h, w, degree):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    x = x / max(w - 1, 1) * 2 - 1
    y = y / max(h - 1, 1) * 2 - 1
    terms = [(x ** i) * (y ** j) for i in range(degree + 1) for j in range(degree + 1 - i)]
    return np.stack(terms, axis=-1)  # H,W,K


# ----------------------------------------------------------------------------- B) estimated background
def estimate_background(img: np.ndarray, cfg, weight: np.ndarray | None = None) -> np.ndarray:
    """Estimate smooth background of `img` (float32 HxW). `weight` in {0,1}: pixels to trust.

    cfg = illumination.background section. Returns float32 HxW background (same scale as img).
    """
    method = cfg.choice("method", ("gaussian", "morphological", "polynomial"))
    small, ws, s = _downscale(img.astype(np.float32), None if weight is None else weight.astype(np.float32),
                              int(cfg["work_max_side"]))
    if ws is None:
        ws = np.ones_like(small)
    ws = (ws > 0.5).astype(np.float32)
    h, w = small.shape

    def px(v):  # full-res px -> work-res px
        return max(1.0, float(v) * s)

    robust_iters = int(cfg["robust_iters"])
    clip_sigma = float(cfg["robust_clip_sigma"])

    if method == "gaussian":
        sig = cfg["gaussian_sigma_px"] or 0.10 * min(img.shape)
        bg = _masked_gaussian(small, ws, px(sig))
        for _ in range(robust_iters):  # drop outliers (defects / clamps), re-estimate
            r = small - bg
            sd = 1.4826 * np.median(np.abs(r[ws > 0.5] - np.median(r[ws > 0.5])))
            keep = ws * (np.abs(r) < clip_sigma * max(sd, 1e-6))
            bg = _masked_gaussian(small, keep.astype(np.float32), px(sig))
    elif method == "morphological":
        k = int(round(px(cfg["morph_kernel_px"]))) | 1
        med = np.median(small[ws > 0.5]) if (ws > 0.5).any() else np.median(small)
        filled = np.where(ws > 0.5, small, med).astype(np.float32)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        # opening removes bright specks < kernel, closing removes dark specks < kernel
        bg = cv2.morphologyEx(cv2.morphologyEx(filled, cv2.MORPH_OPEN, ker), cv2.MORPH_CLOSE, ker)
        bg = cv2.GaussianBlur(bg, (0, 0), px(cfg["morph_smooth_sigma_px"]), borderType=cv2.BORDER_REFLECT)
    else:  # polynomial surface, weighted LSQ with iterative outlier rejection
        deg = int(cfg["poly_degree"])
        A = _poly_terms(h, w, deg)
        Af = A.reshape(-1, A.shape[-1])
        yv = small.reshape(-1)
        keep = (ws.reshape(-1) > 0.5)
        coef = None
        for _ in range(robust_iters + 1):
            coef, *_ = np.linalg.lstsq(Af[keep], yv[keep], rcond=None)
            fit = Af @ coef
            r = yv - fit
            sd = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
            keep = (ws.reshape(-1) > 0.5) & (np.abs(r) < clip_sigma * max(sd, 1e-6))
        bg = (Af @ coef).reshape(h, w).astype(np.float32)

    if s < 1.0:
        bg = cv2.resize(bg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    return bg.astype(np.float32)


def _roi_median(x: np.ndarray, weight: np.ndarray | None) -> float:
    if weight is not None and (weight > 0).any():
        return float(np.median(x[weight > 0]))
    return float(np.median(x))


def correct_with_background(img: np.ndarray, bg: np.ndarray, cfg, weight=None) -> np.ndarray:
    mode = cfg.choice("mode", ("divide", "subtract"))
    target = cfg["target_level"]
    ref = float(target) if target is not None else _roi_median(img, weight)
    if mode == "divide":
        floor = float(cfg["bg_floor_frac"]) * max(_roi_median(bg, weight), 1e-6)
        out = img / np.maximum(bg, floor) * ref
    else:
        out = img - bg + ref
    return out.astype(np.float32)


# ----------------------------------------------------------------------------- A) flat field
def flat_field_correct(img, flat, dark=None, cfg=None, weight=None) -> np.ndarray:
    """(img - dark) / (flat - dark), rescaled so that the ROI median becomes target_level.

    All frames must have identical shape. A dark frame removes sensor offset / fixed pattern;
    without it, the flat's own offset slightly distorts the ratio (acceptable for a first pass).
    """
    if flat.shape != img.shape or (dark is not None and dark.shape != img.shape):
        raise ValueError(f"flat/dark shape must match image {img.shape}; got flat {flat.shape}"
                         + (f", dark {dark.shape}" if dark is not None else ""))
    d = dark if dark is not None else 0.0
    den = flat - d
    floor = float(cfg["bg_floor_frac"]) * max(_roi_median(den, weight), 1e-6)
    ratio = (img - d) / np.maximum(den, floor)
    target = cfg["target_level"]
    ref = float(target) if target is not None else _roi_median(img, weight)
    ratio_med = max(_roi_median(ratio, weight), 1e-6)
    return (ratio * (ref / ratio_med)).astype(np.float32)


# ----------------------------------------------------------------------------- stage entry
def correct_illumination(img, cfg, flat=None, dark=None, weight=None):
    """Stage 2 entry point.

    img    : float32 HxW in [0,1]
    flat   : optional float32 HxW (same convention)   dark: optional float32 HxW
    weight : optional {0,1} HxW (wafer mask) used to ignore the outside region
    Returns (corrected float32 HxW, info dict). If `flat` is given it is used; otherwise the
    background is estimated from the image.
    """
    if flat is not None:
        out = flat_field_correct(img, flat, dark, cfg, weight)
        return out, {"method": "flat_field", "dark_used": dark is not None}
    bgcfg = cfg["background"]
    bg = estimate_background(img, bgcfg, weight)
    out = correct_with_background(img, bg, cfg, weight)
    return out, {"method": "estimated_" + bgcfg["method"], "bg_min": float(bg.min()), "bg_max": float(bg.max()),
                 "dark_used": False, "_bg": bg}
