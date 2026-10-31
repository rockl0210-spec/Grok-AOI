"""Stage 3 - noise suppression (selectable: median | bilateral | nlm | none).

TRADE-OFF (read this before tuning): every denoiser lowers noise by averaging, and small
particle defects are exactly "small, isolated, few-pixel" structures - the thing an averaging
filter removes best. So:
  * median   : kernel k removes isolated features smaller than ~k/2 px in radius. k=3 keeps
               particles >= ~2-3 px wide but deletes 1-px hot spots. Larger k = flatter noise but
               kills small particles. Good against salt&pepper, mediocre against gaussian noise.
  * bilateral: averages only neighbours with similar intensity (sigma_color). A defect whose
               contrast is >> sigma_color is left nearly untouched while noise (< sigma_color) is
               smoothed. Keep sigma_color a few x the noise sigma but BELOW the faintest defect
               contrast you must detect. Too large -> turns into a plain gaussian blur.
  * nlm      : patch-based; best noise/detail trade-off, but slowest, and tiny defects smaller
               than the patch can be treated as noise when h is too high. Keep h ~ 0.6-1.0 x noise
               sigma and patch_size small (3-5).
Always verify on real samples that the weakest defect contrast / CNR survives (see README).
"""
from __future__ import annotations

import cv2
import numpy as np


def estimate_noise_sigma(img: np.ndarray, valid: np.ndarray | None = None, hp_sigma_px: float = 8.0) -> float:
    """Robust noise std: MAD of (img - gaussian_blur(img)) over `valid` (eroded to avoid mask-edge effects).
    Small defects barely move the MAD, so this tracks the noise, not the particles."""
    res = img - cv2.GaussianBlur(img, (0, 0), hp_sigma_px, borderType=cv2.BORDER_REFLECT)
    if valid is not None and (valid > 0).any():
        k = int(2 * hp_sigma_px + 1)
        v = cv2.erode(valid.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        if v.any():
            res = res[v]
    r = res.ravel()
    # the high-pass removes a (1 - small) fraction of the variance; correct for it
    corr = 1.0 / np.sqrt(1.0 - 1.0 / (4 * np.pi * hp_sigma_px ** 2))
    return float(1.4826 * np.median(np.abs(r - np.median(r))) * corr)


def _median(img, k):
    k = int(k) | 1
    if k <= 5:  # cv2.medianBlur supports float32 only for ksize 3/5
        return cv2.medianBlur(img, k)
    u = np.clip(np.rint(img * 65535.0), 0, 65535).astype(np.uint16)
    return cv2.medianBlur(u, k).astype(np.float32) / 65535.0


def _bilateral(img, p):
    # sigma_color is in normalised [0,1] intensity units (0.05 ~ 5% of full scale)
    return cv2.bilateralFilter(img, int(p["diameter"]), float(p["sigma_color"]), float(p["sigma_space"]))


def _nlm(img, p, sigma=None):
    from skimage.restoration import denoise_nl_means

    sig = float(sigma) if sigma is not None else estimate_noise_sigma(img)
    h = float(p["h_factor"]) * sig
    return denoise_nl_means(img, patch_size=int(p["patch_size"]), patch_distance=int(p["patch_distance"]),
                            h=h, sigma=sig, fast_mode=True).astype(np.float32)


def denoise(img: np.ndarray, cfg, sigma: float | None = None) -> np.ndarray:
    """Stage 3 entry. img float32 HxW -> float32 HxW (same range)."""
    method = cfg.choice("method", ("none", "median", "bilateral", "nlm"))
    if method == "none":
        return img.copy()
    if method == "median":
        return _median(img, cfg["median"]["ksize"]).astype(np.float32)
    if method == "bilateral":
        return _bilateral(img, cfg["bilateral"]).astype(np.float32)
    return _nlm(img, cfg["nlm"], sigma)
