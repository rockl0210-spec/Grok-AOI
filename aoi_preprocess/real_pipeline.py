"""Real-data orchestration (per-mode): load -> structure/ROI -> illumination -> denoise -> fixtures -> outputs.

Mode (edge | side) comes from the folder name (config `mode_rules`) or `--mode`. Output layout
(mirrors the input folder structure below `root`):
    <out>/<Edge1|Edge2|Side>/<angle>/NN_cleaned.png   NN_mask.png   report.json
    <out>/_debug/<same>/NN_corrected.png NN_classes.png NN_overlay_preview.png   (preview only)
"""
from __future__ import annotations

import copy
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .config import Cfg, ConfigError
from .denoise import denoise, estimate_noise_sigma
from .illumination import correct_illumination, correct_with_background, estimate_background
from .io_utils import load_image
from .outputs import write_json, write_outputs_real
from .roi_modes import (CLASS_COLORS, CLASS_NAMES, CLS_VALID, apply_fixture_class, edge_flags, edge_roi,
                        edge_structure, side_flags, side_roi)
from .sequence import build_sequence_background

log = logging.getLogger("aoi")


# ------------------------------------------------------------------------------------------------ config / mode
def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def is_multimode(cfg) -> bool:
    return "modes" in cfg


def mode_config(cfg, mode: str) -> Cfg:
    modes = cfg["modes"]
    if mode not in modes:
        raise ConfigError(f"unknown mode '{mode}' (available: {list(modes)})")
    return Cfg(_deep_merge(cfg["common"].to_dict(), modes[mode].to_dict()))


def resolve_mode(path: Path, cfg, forced: str | None = None) -> str:
    if forced:
        return forced
    parts = [p.lower() for p in Path(path).resolve().parts]
    for mode, pats in cfg["mode_rules"].to_dict().items():
        if any(str(pt).lower() in part for pt in pats for part in parts):
            return mode
    if cfg["default_mode"]:
        return cfg["default_mode"]
    raise ConfigError(f"cannot derive the mode from '{path}' (no mode_rules match); pass --mode edge|side")


# ------------------------------------------------------------------------------------------------ illumination
def correct_real(raw, icfg, weight, seq_bg=None):
    """source: self -> large-kernel masked background of the frame itself
               sequence -> frame / (per-folder median image), scaled to the ROI median
               hybrid -> sequence median x residual large-scale gain estimated from the frame itself."""
    source = icfg.choice("source", ("self", "sequence", "hybrid"))
    if source == "self" or seq_bg is None:
        out, info = correct_illumination(raw, icfg, None, None, weight)
        info["source_used"] = "self" + ("" if source == "self" else " (sequence unavailable)")
        return out, info
    floor = float(icfg["bg_floor_frac"]) * max(float(np.median(seq_bg)), 1e-6)
    seq = np.maximum(seq_bg, floor)
    if source == "sequence":
        bg = seq
    else:
        ratio = (raw / seq).astype(np.float32)
        bgc = Cfg(dict(icfg["background"].to_dict(), gaussian_sigma_px=float(icfg["sequence"]["gain_sigma_px"])))
        bg = seq * estimate_background(ratio, bgc, weight)
    out = correct_with_background(raw, bg, icfg, weight)
    return out, {"method": "sequence_" + source, "source_used": source, "bg_min": float(bg.min()),
                 "bg_max": float(bg.max()), "dark_used": False, "_bg": bg}


def _roi_stats(img, valid):
    v = img[valid]
    if v.size == 0:
        return {}
    p = np.percentile(v[::7], [1, 50, 99])
    return {"mean": float(v.mean()), "std": float(v.std()), "p01": float(p[0]), "median": float(p[1]), "p99": float(p[2])}


# ------------------------------------------------------------------------------------------------ one frame
def process_frame(path: Path, mcfg, mode: str, out_dir: Path, debug_dir: Path | None = None, seq_bg=None,
                  return_arrays: bool = False) -> dict:
    t = {}
    t0 = time.perf_counter()
    li = load_image(path, mcfg["input"])                                                       # Stage 1
    raw = li.data
    h, w = raw.shape
    t["load"] = time.perf_counter() - t0
    rcfg = mcfg["roi"]
    t1 = time.perf_counter()
    if mode == "edge":                                                                         # Stage 4a: structure
        st = edge_structure(raw, rcfg["structure"])
        fl = edge_flags(st["info"], mcfg["abnormal"])
        cls, sinfo = edge_roi(raw, rcfg, st, abnormal=fl["abnormal"])
        fl = edge_flags(sinfo, mcfg["abnormal"])
        sinfo = {k: v for k, v in sinfo.items() if k != "line"} | {"line": st["info"].get("line")}
    else:
        cls, sinfo = side_roi(raw, rcfg)
        fl = side_flags(sinfo, mcfg["abnormal"])
    t["structure_roi"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    icfg = mcfg["illumination"]
    e = float(icfg["weight_erode_px"])
    vpre = (cls == CLS_VALID).astype(np.uint8)
    weight = (cv2.distanceTransform(vpre, cv2.DIST_L2, 3) > e).astype(np.float32) if e > 0 else vpre.astype(np.float32)
    use_seq = seq_bg if (mode == "edge" and not fl["abnormal"]) else None                      # Stage 2
    corrected, iinfo = correct_real(raw, icfg, weight, use_seq)
    bg = iinfo.pop("_bg", None)
    t["illumination"] = time.perf_counter() - t2
    t3 = time.perf_counter()
    den = denoise(corrected, mcfg["denoise"])                                                  # Stage 3
    t["denoise"] = time.perf_counter() - t3
    t4 = time.perf_counter()
    cls = apply_fixture_class(cls, den, rcfg)                                                  # Stage 4b: fixtures
    t["fixture"] = time.perf_counter() - t4
    t5 = time.perf_counter()
    files = write_outputs_real(out_dir, debug_dir, path.stem, cleaned=den, corrected=corrected, cls=cls,
                               bit_depth=li.meta["bit_depth"], cfg_out=mcfg["output"], colors=CLASS_COLORS)  # Stage 5
    t["write"] = time.perf_counter() - t5
    valid = cls == CLS_VALID
    counts = {CLASS_NAMES[k]: round(float((cls == k).mean() * 100), 3) for k in CLASS_NAMES if (cls == k).any()}
    rep = {
        "frame": path.stem, "mode": mode, "size_wh": [w, h], "bit_depth": li.meta["bit_depth"],
        "abnormal": fl["abnormal"], "flags": fl["flags"], "notes": fl.get("notes", []),
        "valid_roi_percent": round(float(valid.mean() * 100), 3),
        "class_percent": counts,
        "structure": sinfo,
        "illumination": iinfo,
        "noise_sigma_levels8": {"corrected": round(estimate_noise_sigma(corrected, valid.astype(np.uint8)) * 255, 3),
                                "cleaned": round(estimate_noise_sigma(den, valid.astype(np.uint8)) * 255, 3)},
        "stats_valid_roi_levels8": {k: {a: round(b * 255, 3) for a, b in _roi_stats(x, valid).items()}
                                    for k, x in (("raw", raw), ("cleaned", den))},
        "files": files,
        "runtime_s": {k: round(v, 3) for k, v in t.items()} | {"total": round(time.perf_counter() - t0, 3)},
    }
    if return_arrays:
        return rep, {"raw": raw, "corrected": corrected, "denoised": den, "cls": cls, "bg": bg}
    return rep


# ------------------------------------------------------------------------------------------------ folders
def _select(files: list[Path], frames: list[str] | None):
    if not frames:
        return files
    want = {f"{int(f):02d}" if str(f).isdigit() else str(f) for f in frames}
    return [p for p in files if p.stem in want]


def run_real(input_path, cfg, out_root, mode=None, frames=None, run_all=False, root=None,
             allow_protected=False, debug_root=None) -> tuple[list, list]:
    input_path, out_root = Path(input_path).resolve(), Path(out_root).resolve()
    if input_path.is_dir():
        base = input_path if root is None else Path(root).resolve()
        pngs = sorted(input_path.rglob("*.png") if cfg["common"]["input"]["recursive"] else input_path.glob("*.png"))
    else:
        base = input_path.parent if root is None else Path(root).resolve()
        pngs = [input_path]
    if out_root == base or base in out_root.parents or input_path in out_root.parents:
        raise ConfigError(f"refusing to write into the input tree: out={out_root} root={base}")
    for bad in ([] if allow_protected else cfg["safety"]["protected_out_roots"]):
        bp = Path(bad).resolve()
        if out_root == bp or bp in out_root.parents:
            raise ConfigError(f"refusing to write under protected path {bp} (config safety.protected_out_roots; pass --allow-protected-out once the full run is approved)")
    sel = _select(pngs, frames)
    if not sel:
        raise FileNotFoundError(f"no frames selected in {input_path} (frames={frames})")
    if not frames and not run_all and len(sel) > int(cfg["safety"]["max_files_without_flag"]):
        raise ConfigError(f"{len(sel)} files selected: pass --frames 00,01,... or --all to confirm a bigger run")
    by_dir: dict[Path, list[Path]] = {}
    for p in sel:
        by_dir.setdefault(p.parent, []).append(p)
    reports, failed = [], []
    for d, plist in by_dir.items():
        m = resolve_mode(d if d != base else plist[0], cfg, mode)
        mcfg = mode_config(cfg, m)
        rel = d.relative_to(base) if base in (d, *d.parents) else Path(d.name)
        out_dir = out_root / rel
        dbg = (Path(debug_root).resolve() if debug_root else out_root / "_debug") / rel
        seq_bg, seq_info = None, {"status": "not_used"}
        if m == "edge" and mcfg["illumination"]["source"] in ("sequence", "hybrid"):
            t = time.perf_counter()
            seq_bg, seq_info = build_sequence_background(d, mcfg, sorted(d.glob("*.png")))
            seq_info["build_s"] = round(time.perf_counter() - t, 2)
            log.info("%s: sequence background %s (%s frames) %.1fs", rel, seq_info["status"], len(seq_info["frames_used"]),
                     seq_info["build_s"])
        frep = {}
        for p in plist:
            try:
                r = process_frame(p, mcfg, m, out_dir, dbg, seq_bg)
                log.info("%s/%s: valid ROI %.1f%% flags=%s %.2fs", rel, p.stem, r["valid_roi_percent"], r["flags"],
                         r["runtime_s"]["total"])
                frep[p.stem] = r
                reports.append(r)
            except Exception as ex:  # keep batch going
                log.error("%s FAILED: %s", p, ex, exc_info=log.isEnabledFor(logging.DEBUG))
                failed.append(str(p))
        expected = {str(x) for x in (mcfg["abnormal"].to_dict().get("expected_frames") or [])}
        abn = {k for k, v in frep.items() if v["abnormal"]} | (expected & set(frep))
        write_json(out_dir / "report.json", {
            "tool": {"name": "aoi_preprocess", "version": __version__}, "mode": m, "folder": str(rel),
            "frames_processed": sorted(frep), "frames_selected_by_user": bool(frames),
            "sequence_background": seq_info,
            "abnormal_frames": sorted(abn),
            "abnormal_frames_detail": {k: {"flags": frep[k]["flags"], "detected_by_stats": frep[k]["abnormal"],
                                           "listed_in_config": k in expected,
                                           "reason": frep[k]["flags"] if frep[k]["abnormal"] else
                                           ["listed_in_config (integration spec); stats flags did not trigger"]}
                                       for k in sorted(abn)},
            "frames": frep, "parameters": mcfg.to_dict()})
    return reports, failed
