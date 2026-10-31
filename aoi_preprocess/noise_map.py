"""NN_noise.png : local robust noise sigma (MAD based) of the cleaned image, 16-bit, value = sigma[8-bit gray levels] * scale_x.

sigma(x,y) = 1.4826 * median(|hp|) over a block neighbourhood, hp = img - gaussian(img, hp_sigma) (correction for the variance
the high-pass removes is applied). Defects barely move a block median, so the map follows the noise (and fine texture such as the
rough band, which really is 'noise' for a defect detector). Pixels outside the ROI are 0 (not meaningful there).
"""
from __future__ import annotations

import cv2
import numpy as np


def noise_map(img01: np.ndarray, roi: np.ndarray, c) -> np.ndarray:
    """img01 float [0,1] (cleaned, un-quantised). Returns float32 sigma map in 8-bit levels (0 outside the ROI)."""
    h, w = img01.shape
    s = float(c["hp_sigma_px"])
    x = img01 * 255.0
    hp = x - cv2.GaussianBlur(x, (0, 0), s, borderType=cv2.BORDER_REFLECT)
    corr = 1.0 / np.sqrt(1.0 - 1.0 / (4 * np.pi * s ** 2))
    B = int(c["block_px"])
    ny, nx = -(-h // B), -(-w // B)
    a = np.full((ny * B, nx * B), np.nan, np.float32)
    a[:h, :w] = np.where(roi, np.abs(hp), np.nan)
    blk = a.reshape(ny, B, nx, B).transpose(0, 2, 1, 3).reshape(ny, nx, B * B)
    frac = np.isfinite(blk).mean(-1)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(blk, axis=-1)
    sig = (1.4826 * corr * med).astype(np.float32)
    ok = (frac >= float(c["min_valid_frac"])) & np.isfinite(sig)
    num = cv2.GaussianBlur(np.where(ok, sig, 0).astype(np.float32), (0, 0), float(c["smooth_blocks"]))
    den = cv2.GaussianBlur(ok.astype(np.float32), (0, 0), float(c["smooth_blocks"]))
    fill = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0).astype(np.float32)
    if (den < 1e-3).any() and ok.any():                       # far from any valid block: use the global median
        fill[den < 1e-3] = float(np.median(sig[ok]))
    big = cv2.resize(fill, (nx * B, ny * B), interpolation=cv2.INTER_LINEAR)[:h, :w]
    return np.where(roi, big, 0).astype(np.float32)


def noise_u16(sig_levels: np.ndarray, scale_x: float) -> np.ndarray:
    return np.clip(np.rint(sig_levels * scale_x), 0, 65535).astype(np.uint16)
