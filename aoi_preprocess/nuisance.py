"""v2 nuisance bit-flags (NEVER used to exclude anything - they only label regions inside the ROI).

bit0 (1)  specular / saturation      saturated pixels + large bright (specular) components with their glow
bit1 (2)  uneven illumination / dim  smooth level map deviates from the frame reference (dim zone, shadow, haze, hot spot)
bit2 (4)  rough texture band         dense fine texture attached to the edge (Edge mode)
bit3 (8)  suspected water mark       low-frequency, large-area, low-gradient, BOUNDED patch (low-level marking only)
bit4 (16) suspected oil film         low-frequency, very large / diffuse, low-gradient patch (low-level marking only)
Flags may overlap; everything is 0 outside the ROI. The map is uint8.
"""
from __future__ import annotations

import cv2
import numpy as np

BIT_SPECULAR, BIT_UNEVEN, BIT_ROUGH, BIT_WATER, BIT_OIL = 1, 2, 4, 8, 16
BIT_TABLE = {
    "1": {"bit": 0, "name": "specular_saturation", "meaning": "saturated pixels and large bright specular components (+glow margin)"},
    "2": {"bit": 1, "name": "uneven_illumination_dim_zone", "meaning": "smooth illumination level deviates from the frame reference (dim / shadow / haze / hot zone)"},
    "4": {"bit": 2, "name": "rough_texture_band", "meaning": "dense fine-texture band attached to the edge (Edge mode)"},
    "8": {"bit": 3, "name": "suspected_water_mark", "meaning": "SUSPECTED: low-frequency, large-area, low-gradient bounded patch (marking only, nothing is removed)"},
    "16": {"bit": 4, "name": "suspected_oil_film", "meaning": "SUSPECTED: very large / diffuse low-frequency low-gradient patch (marking only, nothing is removed)"},
}
BIT_KEYS = {"specular": BIT_SPECULAR, "uneven": BIT_UNEVEN, "rough": BIT_ROUGH, "water": BIT_WATER, "oil": BIT_OIL}


def _ell(px):
    k = int(px) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _small(a, s, interp=cv2.INTER_AREA):
    return cv2.resize(a, (max(8, int(round(a.shape[1] * s))), max(8, int(round(a.shape[0] * s)))), interpolation=interp)


def _up(a, hw, interp=cv2.INTER_LINEAR):
    return cv2.resize(a, (hw[1], hw[0]), interpolation=interp)


def masked_gauss(x, w, sigma):
    num = cv2.GaussianBlur(x * w, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(w, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    return num / np.maximum(den, 1e-4), den


def _drop_small(b, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
    if n <= 1:
        return b.astype(bool)
    keep = st[:, cv2.CC_STAT_AREA] >= int(min_area)
    keep[0] = False
    return keep[lab]


# --------------------------------------------------------------------------------------------- bit0
def specular_flag(u8: np.ndarray, c) -> np.ndarray:
    """saturated pixels (>= sat_level, dilated a few px) | large bright components (>= bright_level, opened, area filter) + glow."""
    h, w = u8.shape
    out = np.zeros((h, w), bool)
    if c["sat_level"] is not None:
        s = (u8 >= int(c["sat_level"])).astype(np.uint8)
        s = _drop_small(s, int(c["sat_min_area_px"])).astype(np.uint8)
        if s.any():
            out |= cv2.dilate(s, _ell(2 * int(c["sat_margin_px"]) + 1)) > 0
    if c["bright_level"] is not None:
        b = (u8 >= int(c["bright_level"])).astype(np.uint8)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, _ell(c["bright_open_px"]))
        b = _drop_small(b, int(c["bright_min_area_px"])).astype(np.uint8)
        if b.any():
            out |= cv2.dilate(b.astype(np.uint8), _ell(2 * int(c["glow_margin_px"]) + 1)) > 0
    return out


# --------------------------------------------------------------------------------------------- bit1
def level_map(u8, weight, c, scale=0.25):
    """Smooth level map (median-5 on a 1/4 copy, masked gaussian that ignores `weight`==0 pixels)."""
    h, w = u8.shape
    sm = cv2.medianBlur(_small(u8, scale), 5).astype(np.float32)
    ws = (_small(weight.astype(np.float32), scale) > 0.5).astype(np.float32)
    g, den = masked_gauss(sm, ws, float(c["level_sigma_px"]) * scale)
    ref_sel = (ws > 0.5) & (den > 0.2)
    ref = float(np.percentile(g[ref_sel], c["ref_percentile"])) if ref_sel.sum() > 50 else float(np.median(sm))
    return _up(g.astype(np.float32), (h, w)), ref


def uneven_flag(L, ref, roi, edge_d, c):
    """dim (L < dim_frac*ref) and hot (L > hot_frac*ref) zones. Dim components must attach to the edge band or be
    larger than dim_min_area_px (isolated dark blobs are objects/defects, not illumination)."""
    lo = L < float(c["dim_frac"]) * ref
    hi = (L > float(c["hot_frac"]) * ref) if c["hot_frac"] else np.zeros_like(lo)
    if c["min_abs_dev_levels"]:
        dev = np.abs(L - ref) >= float(c["min_abs_dev_levels"])
        lo &= dev
        hi &= dev
    zone = (lo | hi) & roi
    n, lab, st, _ = cv2.connectedComponentsWithStats(zone.astype(np.uint8), connectivity=8)
    if n <= 1:
        return zone
    keep = st[:, cv2.CC_STAT_AREA] >= int(c["min_area_px"])
    if edge_d is not None:
        near = cv2.dilate((edge_d <= float(c["attach_px"])).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        touch = np.zeros(n, bool)
        touch[np.unique(lab[near & zone])] = True
        keep &= touch | (st[:, cv2.CC_STAT_AREA] >= int(c["isolated_min_area_px"]))
    keep[0] = False
    return keep[lab]


# --------------------------------------------------------------------------------------------- bit3 / bit4
def _grad_full(xm, sigma_work, s):
    """gradient magnitude in gray levels per FULL-RES px of a work-scale (scale s) image (Sobel 3x3 gain = 8)."""
    g = cv2.GaussianBlur(xm, (0, 0), sigma_work)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)) / 8.0 * s


def lowfreq_marks(corr, roi, exclude, c, ref_level):
    """Low-level marking of water-mark / oil-film candidates on the illumination-corrected image.
    R = gaussian(corr, s_fine) - gaussian(corr, s_coarse) (robust: defects clipped), candidate = |R| > max(k*sigma_R, abs_min)
    with low gradient. Component classes by area / boundary sharpness:
        water : bounded patch (boundary gradient / contrast above `sharp_min`) with area in [water_min, water_max]
        oil   : area >= oil_min and diffuse (below sharp_min) or area > water_max
    Returns (water bool, oil bool, info)."""
    h, w = corr.shape
    s = float(c["work_scale"])
    x = _small(corr.astype(np.float32) * 255.0, s)
    wgt = (_small((roi & ~exclude).astype(np.float32), s) > 0.5).astype(np.float32)
    # clip impulsive defects before smoothing (median-ish: 5x5 median on the 1/4 copy)
    xm = cv2.medianBlur(np.clip(x, 0, 255).astype(np.uint8), 5).astype(np.float32)
    kd = max(3, int(float(c["defect_suppress_px"]) * s)) | 1       # remove compact defects / dots (< kd) before smoothing
    xm = cv2.morphologyEx(cv2.morphologyEx(xm, cv2.MORPH_CLOSE, _ell(kd)), cv2.MORPH_OPEN, _ell(kd))
    fine, d1 = masked_gauss(xm, wgt, float(c["fine_sigma_px"]) * s)
    coarse, d2 = masked_gauss(xm, wgt, float(c["coarse_sigma_px"]) * s)
    R = fine - coarse
    valid = (wgt > 0.5) & (d1 > 0.3) & (d2 > 0.3)
    r = R[valid]
    if r.size < 100:
        z = np.zeros((h, w), bool)
        return z, z, {"sigma_R": None}
    sigR = float(1.4826 * np.median(np.abs(r - np.median(r))))
    thr = max(float(c["k_sigma"]) * sigR, float(c["abs_min_levels"]))
    cand = (np.abs(R) > thr) & valid
    G = _grad_full(xm, float(c["grad_sigma_px"]) * s, s)
    cand = cand & (G < float(c["max_grad_levels_per_px"]))
    cand = cand.astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, _ell(max(3, int(c["close_px"] * s))))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, _ell(max(3, int(c["open_px"] * s))))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    water = np.zeros_like(cand, bool)
    oil = np.zeros_like(cand, bool)
    comps = []
    a_s = 1.0 / (s * s)
    Gb = _grad_full(xm, 2.0 * s, s)
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA] * a_s
        if area < float(c["water_min_area_px"]):
            continue
        comp = lab == i
        thick = 2.0 * float(cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3).max()) / s     # full-res px
        if thick < float(c["min_thickness_px"]):                  # thin strips along a boundary = roll-off, not a mark
            continue
        ring = (cv2.dilate(comp.astype(np.uint8), _ell(7)) > 0) & ~(cv2.erode(comp.astype(np.uint8), _ell(7)) > 0)
        contrast = float(np.abs(R[comp]).mean())
        bsharp = float(Gb[ring].mean())
        ratio = bsharp / max(contrast, 1e-3)          # boundary gradient per level of contrast (1/px)
        info = {"area_px": int(area), "thickness_px": int(thick), "contrast_levels": round(contrast, 2), "boundary_sharpness": round(ratio * 1000, 2)}
        if area >= float(c["oil_min_area_px"]) and (ratio < float(c["sharp_min"]) or area > float(c["water_max_area_px"])):
            oil |= comp
            info["class"] = "oil"
        elif area <= float(c["water_max_area_px"]) and ratio >= float(c["sharp_min"]):
            water |= comp
            info["class"] = "water"
        else:
            info["class"] = "none"
        comps.append(info)
    return (_up(water.astype(np.float32), (h, w)) > 0.5) & roi, (_up(oil.astype(np.float32), (h, w)) > 0.5) & roi, \
        {"sigma_R_levels": round(sigR, 3), "threshold_levels": round(thr, 3), "components": comps[:12]}
