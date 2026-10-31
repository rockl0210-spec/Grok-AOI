"""v3: carrier-body contamination (oil film / water marks / ink) separation.

Chosen by evidence (dev_v3/stain_methods.json, dev_v3/stain_eval.json); see report.json -> stain_separation for the numbers.
  * method "tv_gated" (default): R = corrected - TV(corrected, weight) is the fine-scale texture layer (droplet / ring / feather texture
    + sensor noise).  The part of R that is SMALL (|R| <= c1 gray levels, soft cut-off to 0 at c2 levels) is estimated as stain and
    subtracted, only inside the dilated oil|water|ink region (feathered).  Anything with |R| > c2 levels (real defects, strong
    scratches, dark droplets) is never touched; weaker defects are partially attenuated (retention numbers are reported, not hidden).
  * method "oriented_line_streak" (kept, NOT default): oriented line-integral streak layer; measured to remove ~0-3 % of stain energy
    and to attenuate scratches (retention 0.86 -> 0.65), so it was rejected.
  * rejected (measured): band-stop (kills defects equally), rolling median / strong TV (3-px retention 0.2-0.3), morphological
    haze (does nothing - the stain is smaller than any SE that keeps defects), multi-frame median / RPCA (stain texture is
    uncorrelated frame to frame on Edge1/090, see dev_v3/multiframe_check.json).
"""
from __future__ import annotations
import cv2
import numpy as np

DEFAULTS = {
    "method": "tv_gated",
    "tv_weight": 0.02,             # skimage denoise_tv_chambolle weight on the [0,1] image
    "tv_pad_px": 40,
    "hp_sigma": 6.0, "n_orient": 16, "length_px": 41, "sigma_across": 1.0, "sigma_along": 11.0,
    "aniso_ratio": 2.2,            # best / median orientation response required
    "amp_c1": 6.0, "amp_c2": 12.0,  # gray levels (8-bit scale): full subtraction below c1, none above c2
    "min_resp": 1.0,               # ignore responses below this (levels)
    "region_dilate_px": 25, "feather_px": 15,
    "stain_gain": 8.0,             # NN_stain.png = clip(128 + gain*stain_levels)
}


def _kernels(n, L, sa, sl):
    ks = []
    r = L // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    for i in range(n):
        th = np.pi * i / n
        u = xx * np.cos(th) + yy * np.sin(th)          # along
        v = -xx * np.sin(th) + yy * np.cos(th)         # across
        k = np.exp(-0.5 * (u / sl) ** 2 - 0.5 * (v / sa) ** 2)
        ks.append((k / k.sum()).astype(np.float32))
    return ks


_KCACHE: dict = {}


def streak_layer(img255: np.ndarray, c: dict = DEFAULTS, work: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """img255 float32 (0..255 scale). Returns (S signed levels, anisotropy ratio, best orientation index) at full resolution, no region gating."""
    key = (c["n_orient"], c["length_px"], c["sigma_across"], c["sigma_along"])
    if key not in _KCACHE:
        _KCACHE[key] = _kernels(*key)
    ks = _KCACHE[key]
    hp = img255 - cv2.GaussianBlur(img255, (0, 0), c["hp_sigma"])
    resp = np.stack([cv2.filter2D(hp, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT) for k in ks])   # (n,H,W)
    a = np.abs(resp)
    best = a.argmax(0)
    rb = np.take_along_axis(resp, best[None], 0)[0]
    med = np.median(a, axis=0) + 0.25
    ratio = np.abs(rb) / med
    return rb, ratio, best


def _separate_tv(corrected01, roi, region, c):
    from skimage.restoration import denoise_tv_chambolle
    img = corrected01.astype(np.float32)
    H, Wd = img.shape
    d = int(c["region_dilate_px"]) * 2 + 1
    reg = cv2.dilate(region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    ys, xs = np.where(reg.any(1))[0], np.where(reg.any(0))[0]
    z = np.zeros(img.shape, np.float32)
    if len(ys) == 0:
        return {"separated": img.copy(), "stain_levels": z, "weight": z, "S_raw": z, "ratio": z, "theta": z.astype(np.int16)}
    pad = int(c["tv_pad_px"])
    y0, y1, x0, x1 = max(0, ys[0] - pad), min(H, ys[-1] + pad + 1), max(0, xs[0] - pad), min(Wd, xs[-1] + pad + 1)
    crop = np.ascontiguousarray(img[y0:y1, x0:x1])
    R = np.zeros(img.shape, np.float32)
    R[y0:y1, x0:x1] = (crop - denoise_tv_chambolle(crop, weight=float(c["tv_weight"]))) * 255.0
    a = np.abs(R)
    w_amp = np.clip((c["amp_c2"] - a) / max(c["amp_c2"] - c["amp_c1"], 1e-3), 0, 1)
    wr = cv2.GaussianBlur(reg.astype(np.float32), (0, 0), c["feather_px"] / 2.0) * roi.astype(np.float32)
    W = (w_amp * wr).astype(np.float32)
    stain = (R * W).astype(np.float32)
    sep = np.clip(img - stain / 255.0, 0, 1).astype(np.float32)
    return {"separated": sep, "stain_levels": stain, "weight": W, "S_raw": R, "ratio": z, "theta": z.astype(np.int16)}


def separate(corrected01: np.ndarray, roi: np.ndarray, region: np.ndarray, c: dict = DEFAULTS) -> dict:
    """corrected01: float32 [0,1] (after illumination correction).  region: bool, oil|water|ink.
    Returns separated image (float 0..1), stain layer in gray levels (float32), and weights."""
    img = corrected01 * 255.0
    H, Wd = img.shape
    if c.get("method", "tv_gated") == "tv_gated":
        return _separate_tv(corrected01, roi, region, c)
    d0 = int(c["region_dilate_px"]) * 2 + 1
    reg0 = cv2.dilate(region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d0, d0)))
    ys, xs = np.where(reg0.any(1))[0], np.where(reg0.any(0))[0]
    if len(ys) == 0:
        z = np.zeros_like(img)
        return {"separated": corrected01.copy(), "stain_levels": z, "weight": z, "S_raw": z, "ratio": z, "theta": z.astype(np.int16)}
    pad = int(c["length_px"]) + int(c["hp_sigma"] * 4)
    y0, y1, x0, x1 = max(0, ys[0] - pad), min(H, ys[-1] + pad + 1), max(0, xs[0] - pad), min(Wd, xs[-1] + pad + 1)
    Sc, rc, bc = streak_layer(np.ascontiguousarray(img[y0:y1, x0:x1]), c)
    S = np.zeros_like(img); ratio = np.zeros_like(img)
    S[y0:y1, x0:x1], ratio[y0:y1, x0:x1] = Sc, rc
    theta = np.zeros(img.shape, np.int16)
    theta[y0:y1, x0:x1] = bc
    a = np.abs(S)
    w_amp = np.clip((c["amp_c2"] - a) / max(c["amp_c2"] - c["amp_c1"], 1e-3), 0, 1)
    w_an = np.clip((ratio - c["aniso_ratio"] * 0.7) / (c["aniso_ratio"] * 0.6), 0, 1)       # soft gate around aniso_ratio
    w_min = (a >= c["min_resp"]).astype(np.float32)
    d = int(c["region_dilate_px"]) * 2 + 1
    reg = cv2.dilate(region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    wr = cv2.GaussianBlur(reg.astype(np.float32), (0, 0), c["feather_px"] / 2.0)
    wr *= roi.astype(np.float32)
    W = w_amp * w_an * w_min * wr
    stain = (S * W).astype(np.float32)
    sep = np.clip((img - stain) / 255.0, 0, 1).astype(np.float32)
    return {"separated": sep, "stain_levels": stain, "weight": W, "S_raw": S, "ratio": ratio, "theta": theta}


def stain_u8(stain_levels: np.ndarray, roi: np.ndarray, gain: float) -> np.ndarray:
    v = np.clip(128.0 + gain * stain_levels, 0, 255).astype(np.uint8)
    v[~roi] = 0
    return v
