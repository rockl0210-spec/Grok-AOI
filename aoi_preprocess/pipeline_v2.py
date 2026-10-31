"""v2 real-data orchestration. ROI = everything physically on the part (only external background, boundary margin and
fixtures are excluded); nuisance regions are labelled, never excluded.

Per frame NN:  NN_cleaned.png (uint8) NN_mask.png (uint8 0/255) NN_nuisance.png (uint8 bit-flags, 0 outside the ROI)
               NN_noise.png (uint16 = sigma[levels] * noise_map.scale_x, 0 outside the ROI)
Per angle folder: report.json.  All images have the raw frame size (no crop / resize) - original pixel coordinates.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .config import Cfg, ConfigError
from .denoise import denoise, estimate_noise_sigma
from .illumination import estimate_background
from .io_utils import load_image, write_png
from .noise_map import noise_map, noise_u16
from .nuisance import (BIT_OIL, BIT_ROUGH, BIT_SPECULAR, BIT_SPECULAR as _B0, BIT_TABLE, BIT_UNEVEN, BIT_WATER, BIT_KEYS,
                       level_map, lowfreq_marks, specular_flag, uneven_flag)
from .outputs import write_json
from .real_pipeline import _roi_stats, _select, correct_real, mode_config, resolve_mode
from .registration import prep, register_pair
from .roi_modes import edge_flags, edge_roughness, edge_structure, side_rows, to_u8
from .roi_v2 import (C_FIXTURE, C_MARGIN, C_OUT, C_VALID, V2_CLASS_NAMES, apply_fixture_v2, edge_roi_v2,
                     side_roi_v2, side_flags_v2, side_rows_consensus)
from .sequence import build_sequence_background

log = logging.getLogger("aoi")
NOISE_SCALE = 100


# ------------------------------------------------------------------------------------------------ local normalisation
def local_normalise(x: np.ndarray, weight: np.ndarray, lc, mode_div: bool, ref: float) -> tuple[np.ndarray, dict]:
    """2nd illumination stage: remove the residual large-scale level (sigma ~60 px, robust against defects) so dim /
    specular-adjacent / uneven areas are comparable. Subtractive for Side (additive haze), divisive for Edge. The scale is
    ~20x a 3-px defect, so a 3-px spot is not part of the estimated background."""
    bgc = Cfg({"method": "gaussian", "gaussian_sigma_px": float(lc["sigma_px"]), "work_max_side": 1024,
               "robust_iters": int(lc["robust_iters"]), "robust_clip_sigma": float(lc["robust_clip_sigma"])})
    bg = estimate_background(x, bgc, weight)
    if mode_div:
        gain = ref / np.maximum(bg, 0.05 * ref)
        out = x * gain
        info = {"gain_min": float(gain.min()), "gain_max": float(gain.max())}
    else:
        out = x - bg + ref
        info = {"offset_min": float((ref - bg).min()), "offset_max": float((ref - bg).max())}
    return np.clip(out, 0, 1).astype(np.float32), info


def _pct(mask, roi_n):
    return round(float(mask.sum() / max(1, roi_n) * 100), 3)


# ------------------------------------------------------------------------------------------------ one frame
def prepare_folder(d: Path, mcfg, mode: str, files: list[Path]) -> dict:
    ctx = {"seq_bg": None, "seq_info": {"status": "not_used"}, "rows_consensus": None}
    t = time.perf_counter()
    if mode == "edge" and mcfg["illumination"]["source"] in ("sequence", "hybrid"):
        ctx["seq_bg"], ctx["seq_info"] = build_sequence_background(d, mcfg, sorted(d.glob("*.png")))
    if mode == "side":
        rows = [side_rows(load_image(p, mcfg["input"]).data, mcfg["roi"]["strip"]) for p in sorted(d.glob("*.png"))]
        ctx["rows_consensus"] = side_rows_consensus(rows, mcfg["roi"]["strip"])
        ctx["seq_info"] = {"status": "not_used", "rows_consensus": ctx["rows_consensus"]}
    ctx["seq_info"]["prepare_s"] = round(time.perf_counter() - t, 2)
    return ctx


def process_array(raw: np.ndarray, mcfg, mode: str, ctx: dict, name: str = "") -> tuple[dict, dict]:
    """raw float32 [0,1] -> (report dict, arrays dict). arrays: cleaned_u8, mask, nuisance, noise16, plus intermediates."""
    h, w = raw.shape
    t = {}
    t0 = time.perf_counter()
    rcfg, ncfg, icfg = mcfg["roi"], mcfg["nuisance"], mcfg["illumination"]
    u8 = to_u8(raw)
    struct = None
    # ---- ROI (physical boundary / external background / fixtures only)
    if mode == "edge":
        struct = edge_structure(raw, rcfg["structure"])
        cls, rinfo, struct["M"], struct["d"] = edge_roi_v2(struct["d"], struct["M"], rcfg)
        fl = edge_flags(struct["info"], mcfg["abnormal"])
        rinfo["line"] = struct["info"].get("line")
    else:
        cls, rinfo = side_roi_v2(raw, rcfg, ctx["rows_consensus"])
        fl = side_flags_v2(rinfo)
    cls, fix_px = apply_fixture_v2(cls, raw, rcfg)
    roi = cls == C_VALID
    roi_n = int(roi.sum())
    t["roi"] = time.perf_counter() - t0
    # ---- nuisance bits (labels only)
    t1 = time.perf_counter()
    nu = np.zeros((h, w), np.uint8)
    b0 = specular_flag(u8, ncfg["specular"]) & roi
    nu[b0] |= BIT_SPECULAR
    if mode == "edge":
        L, ref = struct["L"], struct["ref"]
        b1 = uneven_flag(L, ref, roi, struct["d"], ncfg["uneven"])
        rc = ncfg["rough"]
        b2 = (edge_roughness(raw, struct["M"], struct["d"], rc)[0] & roi) if rc["enabled"] else np.zeros_like(roi)
    else:
        L, ref = level_map(u8, roi & ~b0, ncfg["uneven"])
        b1 = uneven_flag(L, ref, roi, None, ncfg["uneven"])
        b2 = np.zeros_like(roi)
    nu[b1] |= BIT_UNEVEN
    nu[b2] |= BIT_ROUGH
    t["nuisance_a"] = time.perf_counter() - t1
    # ---- illumination: stage 1 (sequence/self) + stage 2 (local normalisation)
    t2 = time.perf_counter()
    e = float(icfg["weight_erode_px"])
    wsrc = (roi & ~b0).astype(np.uint8)
    weight = (cv2.distanceTransform(wsrc, cv2.DIST_L2, 3) > e).astype(np.float32) if e > 0 else wsrc.astype(np.float32)
    if weight.sum() < 20000:
        weight = wsrc.astype(np.float32)
    use_seq = ctx["seq_bg"] if (mode == "edge" and not fl["abnormal"]) else None
    corrected, iinfo = correct_real(raw, icfg, weight, use_seq)
    iinfo.pop("_bg", None)
    lc = icfg["local_norm"]
    if lc["enabled"]:
        wv = roi.astype(np.float32)
        ref_lvl = float(np.median(corrected[roi])) if roi_n else 0.5
        corrected, linfo = local_normalise(corrected, (wv * (~b0)).astype(np.float32), lc, mcfg["illumination"]["mode"] == "divide",
                                           ref_lvl)
        iinfo["local_norm"] = linfo
    t["illumination"] = time.perf_counter() - t2
    # ---- water / oil (low-frequency marks on the corrected image)
    t3 = time.perf_counter()
    lf = ncfg["lowfreq"]
    lfinfo = {}
    if lf["enabled"] and roi_n > 50000:
        k31 = np.ones((31, 31), np.uint8)
        excl = b0 | (cv2.dilate((b2 | b1).astype(np.uint8), k31) > 0)     # texture / dim zones / saturation dominate there
        b3, b4, lfinfo = lowfreq_marks(corrected, roi, excl, lf, ref)
    else:
        b3 = b4 = np.zeros_like(roi)
    nu[b3] |= BIT_WATER
    nu[b4] |= BIT_OIL
    t["lowfreq"] = time.perf_counter() - t3
    # ---- denoise (conservative NLM) + noise map
    t4 = time.perf_counter()
    den = denoise(corrected, mcfg["denoise"])
    t["denoise"] = time.perf_counter() - t4
    t5 = time.perf_counter()
    sig = noise_map(den, roi, mcfg["noise_map"])
    t["noise_map"] = time.perf_counter() - t5
    cleaned = to_u8(den)
    mask = (roi * 255).astype(np.uint8)
    nu[~roi] = 0
    # ---- report
    pb = {k: _pct(((nu & bit) > 0), roi_n) for k, bit in BIT_KEYS.items()}
    pbf = {k: round(float(((nu & bit) > 0).sum() / (h * w) * 100), 3) for k, bit in BIT_KEYS.items()}
    sv = sig[roi]
    rep = {
        "frame": name, "mode": mode, "size_wh": [w, h],
        "abnormal": fl["abnormal"], "flags": fl["flags"], "notes": fl.get("notes", []),
        "roi_percent": round(roi_n / (h * w) * 100, 3),
        "class_percent": {V2_CLASS_NAMES[k]: round(float((cls == k).mean() * 100), 3) for k in V2_CLASS_NAMES if (cls == k).any()},
        "fixture_px": fix_px,
        "nuisance_percent_of_roi": pb, "nuisance_percent_of_frame": pbf,
        "any_nuisance_percent_of_roi": _pct(nu > 0, roi_n),
        "structure": rinfo, "illumination": iinfo, "lowfreq": lfinfo,
        "noise_sigma_levels8": {
            "map_median": round(float(np.median(sv)), 3) if sv.size else None,
            "map_p05": round(float(np.percentile(sv, 5)), 3) if sv.size else None,
            "map_p95": round(float(np.percentile(sv, 95)), 3) if sv.size else None,
            "global_mad_cleaned": round(estimate_noise_sigma(den, roi.astype(np.uint8)) * 255, 3),
        },
        "stats_roi_levels8": {k: {a: round(b * 255, 3) for a, b in _roi_stats(x, roi).items()}
                              for k, x in (("raw", raw), ("cleaned", den))},
        "runtime_s": {k: round(v, 3) for k, v in t.items()} | {"total": round(time.perf_counter() - t0, 3)},
    }
    arrs = {"raw": raw, "cleaned_f": den, "cleaned": cleaned, "mask": mask, "nuisance": nu, "noise16": noise_u16(sig, NOISE_SCALE),
            "noise_levels": sig, "cls": cls, "struct": struct, "bits": {"specular": b0, "uneven": b1, "rough": b2, "water": b3, "oil": b4}}
    return rep, arrs


def process_frame(path: Path, mcfg, mode: str, out_dir: Path, ctx: dict) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    li = load_image(path, mcfg["input"])
    rep, arrs = process_array(li.data, mcfg, mode, ctx, path.stem)
    rep["bit_depth"] = li.meta["bit_depth"]
    t1 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_png(out_dir / f"{path.stem}_cleaned.png", arrs["cleaned"])
    write_png(out_dir / f"{path.stem}_mask.png", arrs["mask"])
    write_png(out_dir / f"{path.stem}_nuisance.png", arrs["nuisance"])
    write_png(out_dir / f"{path.stem}_noise.png", arrs["noise16"])
    rep["files"] = {k: f"{path.stem}_{k}.png" for k in ("cleaned", "mask", "nuisance", "noise")}
    rep["runtime_s"]["write"] = round(time.perf_counter() - t1, 3)
    rep["runtime_s"]["total"] = round(time.perf_counter() - t0, 3)
    return rep, arrs


# ------------------------------------------------------------------------------------------------ folders
def _reg_item(arrs, name, abnormal, rc):
    st = arrs["struct"]
    line = st["info"].get("line") if st else None
    return {"hp": prep(arrs["raw"], rc), "line": line, "inlier": float(st["info"].get("line_inlier_frac", 0.0)) if st else 0.0,
            "abnormal": abnormal, "name": name}


def run_v2(input_path, cfg, out_root, mode=None, frames=None, run_all=False, root=None, allow_protected=False, **_) -> tuple[list, list]:
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
            raise ConfigError(f"refusing to write under protected path {bp} (pass --allow-protected-out once the full run is approved)")
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
        t_f = time.perf_counter()
        ctx = prepare_folder(d, mcfg, m, plist)
        log.info("%s: prepared (%s) %.1fs", rel, ctx["seq_info"].get("status"), ctx["seq_info"]["prepare_s"])
        frep, regs, prev = {}, [], None
        rcfg = mcfg["registration"]
        for p in plist:
            try:
                r, arrs = process_frame(p, mcfg, m, out_dir, ctx)
                log.info("%s/%s: ROI %.1f%% nuisance(of ROI) %s flags=%s %.2fs", rel, p.stem, r["roi_percent"],
                         r["nuisance_percent_of_roi"], r["flags"], r["runtime_s"]["total"])
                frep[p.stem] = r
                reports.append(r)
                if m == "edge" and rcfg["enabled"]:
                    cur = _reg_item(arrs, p.stem, r["abnormal"], rcfg)
                    if prev is not None:
                        regs.append(register_pair(prev, cur, rcfg))
                    prev = cur
            except Exception as ex:
                log.error("%s FAILED: %s", p, ex, exc_info=log.isEnabledFor(logging.DEBUG))
                failed.append(str(p))
                prev = None
        expected = {str(x) for x in (mcfg["abnormal"].to_dict().get("expected_frames") or [])}
        abn = {k for k, v in frep.items() if v["abnormal"]} | (expected & set(frep))
        write_json(out_dir / "report.json", {
            "tool": {"name": "aoi_preprocess", "version": __version__, "pipeline": "v2"}, "mode": m, "folder": str(rel),
            "frames_processed": sorted(frep), "frames_selected_by_user": bool(frames),
            "roi_rule": "mask excludes ONLY external background, the physical boundary (+margin) and fixtures; "
                        "specular / uneven-dim / rough band / water / oil stay inside and are labelled in *_nuisance.png",
            "nuisance_bit_encoding": BIT_TABLE,
            "nuisance_note": "uint8 bit-flags (value = sum of bits), flags may overlap, 0 outside the ROI; water / oil are SUSPECTED, low-level region marking only",
            "noise_map": {"file_suffix": "_noise.png", "dtype": "uint16", "scale": NOISE_SCALE,
                          "definition": "sigma_levels8 = value / scale; local robust sigma (1.4826*median|hp| per 32px block, hp = cleaned - gaussian(1.5px)) of the cleaned image in 8-bit gray levels; 0 outside ROI"},
            "mask_encoding": {"255": "ROI (inspect)", "0": "external background / physical boundary margin / fixture / end background"},
            "sequence_background": ctx["seq_info"],
            "abnormal_frames": sorted(abn),
            "abnormal_frames_detail": {k: {"flags": frep[k]["flags"], "detected_by_stats": frep[k]["abnormal"],
                                           "listed_in_config": k in expected, "roi_percent": frep[k]["roi_percent"],
                                           "nuisance_percent_of_roi": frep[k]["nuisance_percent_of_roi"],
                                           "stats_roi_levels8": frep[k]["stats_roi_levels8"],
                                           "structure": {kk: vv for kk, vv in frep[k]["structure"].items() if kk != "rows_detected"}}
                                       for k in sorted(abn) if k in frep},
            "registration": {"enabled": bool(m == "edge" and rcfg["enabled"]),
                             "sign": "dx,dy = shift of frame k+1 content relative to frame k (px, +x right, +y down)",
                             "pairs": regs} if m == "edge" else {"enabled": False},
            "folder_runtime_s": round(time.perf_counter() - t_f, 2),
            "frames": frep, "parameters": mcfg.to_dict()})
    return reports, failed
