"""Sequence (per angle folder) stationary-illumination estimate for Edge mode.

Adjacent Edge frames show DIFFERENT places along the edge under the same stationary illumination (phase correlation
found no usable shift, the edge line only drifts ~10 px over 46 frames), so the per-pixel median over a handful of
evenly spaced frames removes all specks/fibres/texture and leaves the fixed illumination pattern (lens dust, vignetting,
uneven lighting). Only `n_frames` NORMAL (non-flagged) frames are read - never the whole folder.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .io_utils import load_image
from .roi_modes import edge_flags, edge_structure

log = logging.getLogger("aoi")


def build_sequence_background(folder: Path, mcfg, files: list[Path] | None = None):
    """Median of up to n_frames normal frames of `folder`. Returns (bg float32 [0,1] or None, info dict)."""
    sc = mcfg["illumination"]["sequence"]
    n_want, n_min = int(sc["n_frames"]), int(sc["min_frames"])
    files = files or sorted(Path(folder).glob("*.png"))
    info = {"frames_used": [], "frames_skipped_abnormal": [], "folder": str(folder)}
    if len(files) < max(n_min, 2) or n_want <= 0:
        info["status"] = "not_enough_frames"
        return None, info
    ecfg = mcfg["roi"]
    cand_idx = np.unique(np.linspace(0, len(files) - 1, min(len(files), 3 * n_want)).round().astype(int))
    # order candidates so that the most evenly spread ones come first
    order = list(cand_idx[::max(1, len(cand_idx) // n_want)]) + [i for i in cand_idx]
    seen, stack = set(), []
    for i in order:
        if i in seen:
            continue
        seen.add(i)
        p = files[i]
        raw = load_image(p, mcfg["input"]).data
        st = edge_structure(raw, ecfg["structure"])
        fl = edge_flags(st["info"], mcfg["abnormal"])
        if fl["abnormal"]:
            info["frames_skipped_abnormal"].append(p.stem)
            continue
        stack.append((np.clip(raw * 255 + 0.5, 0, 255).astype(np.uint8)))
        info["frames_used"].append(p.stem)
        if len(stack) >= n_want:
            break
    if len(stack) < n_min:
        info["status"] = "too_few_normal_frames"
        return None, info
    med = np.median(np.stack(stack, 0), axis=0).astype(np.float32) / 255.0
    sp = float(sc["smooth_px"])
    if sp > 0:
        med = cv2.GaussianBlur(med, (0, 0), sp)
    info["status"] = "ok"
    return med.astype(np.float32), info
