"""v3 real-data orchestration = v2 + (1) template-derived oil / water / ink pattern detectors (flags overlap freely, rough
band narrowed to the residual), (2) carrier-body contamination separation (NN_cleaned = stain-removed, defect-preserving;
NN_stain.png = estimated contamination layer), (3) Side end-fixture detection (fixtures are EXCLUDED from the ROI),
(4) per-frame overlap indicator and a labels section (Willson-confirmed vs auto-suspected).

Per frame NN: NN_cleaned.png NN_mask.png NN_nuisance.png NN_noise.png NN_stain.png ; per angle folder: report.json.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .config import ConfigError
from .denoise import denoise, estimate_noise_sigma
from .io_utils import load_image, write_png
from .noise_map import noise_map, noise_u16
from .nuisance import BIT_ROUGH, BIT_SPECULAR, BIT_UNEVEN, BIT_WATER, BIT_OIL, BIT_TABLE, level_map, specular_flag, uneven_flag
from .outputs import write_json
from .patterns import detect as detect_patterns, side_end_fixture
from .pipeline_v2 import NOISE_SCALE, _pct, _reg_item, local_normalise, prepare_folder
from .real_pipeline import _roi_stats, _select, correct_real, mode_config, resolve_mode
from .registration import register_pair
from .roi_modes import edge_flags, edge_roughness, edge_structure, to_u8
from .roi_v2 import (C_FIXTURE, C_VALID, V2_CLASS_NAMES, apply_fixture_v2, edge_roi_v2, side_flags_v2, side_roi_v2)
from .stain import separate, stain_u8

log = logging.getLogger("aoi")
BIT_INK = 32
V3_BITS = {"specular": 1, "uneven": 2, "rough": 4, "water": 8, "oil": 16, "ink": 32}
V3_BIT_TABLE = {
    "1": BIT_TABLE["1"],
    "2": BIT_TABLE["2"],
    "4": {"bit": 2, "name": "rough_texture_residual",
          "meaning": "v3: dense fine-texture band attached to the edge, NARROWED - it no longer includes oil / water / ink regions "
                     "(v2 rough band was mostly oil film + water marks on Edge1/090 17-26)"},
    "8": {"bit": 3, "name": "water_mark",
          "meaning": "v3: dried-droplet / ring / puddle region (dense small dark droplets, rings, bright puddles) inside or next to oil; "
                     "learned from Willson's labels (Edge1/090 18-24, Edge1/000 25). Flag only, overlaps other flags"},
    "16": {"bit": 4, "name": "oil_film",
           "meaning": "v3: brush-like oriented streak / mottled texture region (clipped high-pass energy E > 3.2); learned from Willson's "
                      "labels (Edge1/000 20,21,24,25; Edge1/090 17-26,28). Flag only, overlaps other flags"},
    "32": {"bit": 5, "name": "ink_dark_patch_suspected",
           "meaning": "v3, LOW CONFIDENCE: large (>=150 px) very dark solid patch inside / touching the contaminated band - suspected ink / "
                      "dirt on the carrier body. Small dark blobs (<150 px) are NOT distinguishable from particle defects and are not flagged "
                      "(they are covered by the oil/water bits; see report 'ink_note')"},
}


# ------------------------------------------------------------------------------------------------ stage A (ROI + illumination)
def stage_a(raw: np.ndarray, mcfg, mode: str, ctx: dict) -> dict:
    h, w = raw.shape
    t = {}
    t0 = time.perf_counter()
    rcfg, ncfg, icfg = mcfg["roi"], mcfg["nuisance"], mcfg["illumination"]
    u8 = to_u8(raw)
    struct = None
    if mode == "edge":
        struct = edge_structure(raw, rcfg["structure"])
        cls, rinfo, struct["M"], struct["d"] = edge_roi_v2(struct["d"], struct["M"], rcfg)
        fl = edge_flags(struct["info"], mcfg["abnormal"])
        rinfo["line"] = struct["info"].get("line")
    else:
        cls, rinfo = side_roi_v2(raw, rcfg, ctx["rows_consensus"])
        fl = side_flags_v2(rinfo)
    cls, fix_px = apply_fixture_v2(cls, raw, rcfg)
    fixtures = []
    if mode == "side" and mcfg["v3"]["fixture"]["enabled"]:
        cls, fixtures = side_end_fixture(raw * 255.0, cls, dict(mcfg["v3"]["fixture"].to_dict()))
    roi = cls == C_VALID
    t["roi"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    b0 = specular_flag(u8, ncfg["specular"]) & roi
    if mode == "edge":
        L, ref = struct["L"], struct["ref"]
        b1 = uneven_flag(L, ref, roi, struct["d"], ncfg["uneven"])
        rc = ncfg["rough"]
        b2 = (edge_roughness(raw, struct["M"], struct["d"], rc)[0] & roi) if rc["enabled"] else np.zeros_like(roi)
    else:
        L, ref = level_map(u8, roi & ~b0, ncfg["uneven"])
        b1 = uneven_flag(L, ref, roi, None, ncfg["uneven"])
        b2 = np.zeros_like(roi)
    t["nuisance_a"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    roi_n = int(roi.sum())
    e = float(icfg["weight_erode_px"])
    wsrc = (roi & ~b0).astype(np.uint8)
    weight = (cv2.distanceTransform(wsrc, cv2.DIST_L2, 3) > e).astype(np.float32) if e > 0 else wsrc.astype(np.float32)
    if weight.sum() < 20000:
        weight = wsrc.astype(np.float32)
    fl_abn = fl["abnormal"]
    use_seq = ctx["seq_bg"] if (mode == "edge" and not fl_abn) else None
    corrected, iinfo = correct_real(raw, icfg, weight, use_seq)
    iinfo.pop("_bg", None)
    lc = icfg["local_norm"]
    if lc["enabled"]:
        wv = roi.astype(np.float32)
        ref_lvl = float(np.median(corrected[roi])) if roi_n else 0.5
        corrected, linfo = local_normalise(corrected, (wv * (~b0)).astype(np.float32), lc, icfg["mode"] == "divide", ref_lvl)
        iinfo["local_norm"] = linfo
    t["illumination"] = time.perf_counter() - t2
    return {"raw": raw, "u8": u8, "cls": cls, "roi": roi, "rinfo": rinfo, "fl": fl, "fix_px": fix_px, "fixtures": fixtures,
            "struct": struct, "b0": b0, "b1": b1, "b2": b2, "corrected": corrected, "iinfo": iinfo, "t": t, "t0": t0}


# ------------------------------------------------------------------------------------------------ frame-level rules
def frame_rules(meas: dict, fr: dict) -> dict:
    out = {}
    if meas["oil_pct"] >= fr["oil_min_pct"]:
        out["oil"] = {"area_pct_of_roi": round(meas["oil_pct"], 2),
                      "confidence": round(float(min(1.0, 0.3 + 0.7 * meas["oil_pct"] / fr["oil_conf_full_pct"])), 2)}
    wp = meas["water_pct"]
    if wp >= fr["water_min_pct"]:
        out["water"] = {"area_pct_of_roi": round(wp, 2),
                        "confidence": round(float(min(1.0, 0.3 + 0.7 * wp / fr["water_conf_full_pct"])), 2)}
    elif wp >= fr["water_puddle_min_pct"] and meas["puddle_frac"] >= fr["water_puddle_frac"] and "oil" in out:
        out["water"] = {"area_pct_of_roi": round(wp, 2), "confidence": 0.5, "via": "bright puddles inside oil"}
    if meas["ink_pct"] >= fr["ink_min_pct"]:
        out["ink"] = {"area_pct_of_roi": round(meas["ink_pct"], 3), "confidence": 0.3}
    return out


# ------------------------------------------------------------------------------------------------ stage B
def stage_b(st: dict, mcfg, mode: str, name: str = "", folder: str = "") -> tuple[dict, dict]:
    raw, roi, cls, b0, b1, b2 = st["raw"], st["roi"], st["cls"], st["b0"], st["b1"], st["b2"]
    h, w = raw.shape
    roi_n = int(roi.sum())
    v3 = mcfg["v3"]
    t = dict(st["t"])
    nu = np.zeros((h, w), np.uint8)
    z = np.zeros_like(roi)
    t3 = time.perf_counter()
    det = None
    oil = water = ink = z
    if mode in v3["oil_water_modes"] and roi_n > 50000:
        det = detect_patterns(raw * 255.0, roi, dict(v3["patterns"].to_dict()))
        oil, water, ink = det["oil"], det["water"], det["ink"]
    t["patterns"] = time.perf_counter() - t3
    # ---- rough band narrowed: remove everything that is oil / water / ink (dilated)
    kd = int(v3["rough"]["oil_exclude_dilate_px"]) | 1
    contam = oil | water | ink
    contam_d = cv2.dilate(contam.astype(np.uint8), np.ones((kd, kd), np.uint8)) > 0 if contam.any() else z
    b2n = b2 & ~contam_d
    nu[b0] |= 1
    nu[b1] |= 2
    nu[b2n] |= 4
    nu[water] |= 8
    nu[oil] |= 16
    nu[ink] |= 32
    nu[~roi] = 0
    # ---- stain separation (Edge, only inside the contaminated region; never elsewhere)
    t4 = time.perf_counter()
    scfg = dict(v3["stain"].to_dict())
    if contam.any():
        sp = separate(st["corrected"], roi, contam, scfg)
        sep, stain_lv, wgt = sp["separated"], sp["stain_levels"], sp["weight"]
        sinfo = {"applied": True, "weight_mean_in_region": round(float(wgt[contam].mean()), 3),
                 "stain_abs_mean_levels_in_region": round(float(np.abs(stain_lv[contam]).mean()), 3),
                 "stain_abs_p99_levels_in_region": round(float(np.percentile(np.abs(stain_lv[contam]), 99)), 3),
                 "stain_max_abs_levels": round(float(np.abs(stain_lv).max()), 3)}
    else:
        sep, stain_lv = st["corrected"], np.zeros((h, w), np.float32)
        sinfo = {"applied": False, "reason": "no oil / water / ink region in this frame" if mode == "edge" else "Side mode: no templates"}
    t["stain_separation"] = time.perf_counter() - t4
    t5 = time.perf_counter()
    # NLM strength from the PRE-separation image, so clean regions are denoised exactly as in v2 (separation lowers the global sigma estimate)
    den = denoise(sep, mcfg["denoise"], sigma=estimate_noise_sigma(st["corrected"]))
    t["denoise"] = time.perf_counter() - t5
    t6 = time.perf_counter()
    sig = noise_map(den, roi, mcfg["noise_map"])
    t["noise_map"] = time.perf_counter() - t6
    cleaned = to_u8(den)
    mask = (roi * 255).astype(np.uint8)
    stain8 = stain_u8(stain_lv, roi, float(scfg["stain_gain"]))
    # ---- overlap indicator
    nbits = np.zeros((h, w), np.uint8)
    for b in (1, 2, 4, 8, 16, 32):
        nbits += ((nu & b) > 0).astype(np.uint8)
    ov = {"any_overlap_pct_of_roi": _pct(nbits >= 2, roi_n),
          "oil_and_water_pct_of_roi": _pct(((nu & 16) > 0) & ((nu & 8) > 0), roi_n),
          "oil_and_ink_pct_of_roi": _pct(((nu & 16) > 0) & ((nu & 32) > 0), roi_n),
          "oil_or_water_or_ink_pct_of_roi": _pct(contam, roi_n),
          "oil_and_rough_pct_of_roi": _pct(((nu & 16) > 0) & ((nu & 4) > 0), roi_n),
          "max_flags_on_one_pixel": int(nbits.max()) if roi_n else 0}
    meas = det["meas"] if det else {"oil_pct": 0.0, "water_pct": 0.0, "ink_pct": 0.0, "E_median": None, "dark_p95": None, "puddle_frac": 0.0}
    rules = frame_rules(meas, v3["frame_rules"].to_dict()) if det else {}
    # left/right side of the contamination (Edge1 band is on the left, Edge2 on the right)
    side_info = {}
    if oil.any():
        xs = np.where(oil.any(0))[0]
        side_info = {"oil_x_min": int(xs[0]), "oil_x_max": int(xs[-1]),
                     "oil_fraction_in_left_half": round(float(oil[:, : w // 2].sum() / oil.sum()), 3)}
    pb = {k: _pct(((nu & bit) > 0), roi_n) for k, bit in V3_BITS.items()}
    pbf = {k: round(float(((nu & bit) > 0).sum() / (h * w) * 100), 3) for k, bit in V3_BITS.items()}
    sv = sig[roi]
    fix_cls = (cls == C_FIXTURE)
    rep = {
        "frame": name, "mode": mode, "size_wh": [w, h],
        "abnormal": st["fl"]["abnormal"], "flags": st["fl"]["flags"], "notes": st["fl"].get("notes", []),
        "roi_percent": round(roi_n / (h * w) * 100, 3),
        "class_percent": {V2_CLASS_NAMES[k]: round(float((cls == k).mean() * 100), 3) for k in V2_CLASS_NAMES if (cls == k).any()},
        "fixture_px": st["fix_px"],
        "fixtures": st["fixtures"],
        "nuisance_percent_of_roi": pb, "nuisance_percent_of_frame": pbf,
        "any_nuisance_percent_of_roi": _pct(nu > 0, roi_n),
        "rough_v2_percent_of_roi": _pct(b2, roi_n), "rough_v3_percent_of_roi": _pct(b2n, roi_n),
        "rough_removed_as_contamination_percent_of_roi": _pct(b2 & contam_d, roi_n),
        "overlap": ov,
        "patterns": {"measurements": {k: (round(float(v), 4) if v is not None else None) for k, v in meas.items()},
                     "frame_level": rules, "geometry": side_info},
        "stain_separation": sinfo,
        "structure": st["rinfo"], "illumination": st["iinfo"],
        "noise_sigma_levels8": {
            "map_median": round(float(np.median(sv)), 3) if sv.size else None,
            "map_p05": round(float(np.percentile(sv, 5)), 3) if sv.size else None,
            "map_p95": round(float(np.percentile(sv, 95)), 3) if sv.size else None,
            "global_mad_cleaned": round(estimate_noise_sigma(den, roi.astype(np.uint8)) * 255, 3)},
        "stats_roi_levels8": {k: {a: round(b * 255, 3) for a, b in _roi_stats(x, roi).items()} for k, x in (("raw", raw), ("cleaned", den))},
    }
    # ---- labels section (confirmed vs auto)
    lab = {}
    conf_lab = v3["labels"].to_dict().get(folder, {}) if folder else {}
    k = int(name) if name.isdigit() else None
    classes = {"oil", "water", "ink", "fixture"}
    for cl in sorted(classes):
        confirmed = k is not None and k in [int(x) for x in conf_lab.get(cl, [])]
        if cl == "fixture":
            auto = any(f.get("detected") for f in st["fixtures"])
            fx = [f for f in st["fixtures"] if f.get("detected")]
            area = round(sum(f["area_px"] for f in fx) / max(roi_n + sum(f["area_px"] for f in fx), 1) * 100, 2) if fx else 0.0
            cf = max((f["confidence"] for f in fx), default=0.0)
        else:
            r = rules.get(cl)
            auto = r is not None
            area = r["area_pct_of_roi"] if r else round(float(meas.get(cl + "_pct", 0.0)), 3)
            cf = r["confidence"] if r else 0.0
        if confirmed or auto:
            lab[cl] = {"willson_confirmed": bool(confirmed), "auto_detected": bool(auto),
                       "status": ("confirmed_and_detected" if confirmed and auto else "confirmed_NOT_detected" if confirmed else "auto_suspected"),
                       "area_pct_of_roi": area, "confidence": cf}
    rep["labels"] = lab
    rep["runtime_s"] = {kk: round(v, 3) for kk, v in t.items()} | {"total": round(time.perf_counter() - st["t0"], 3)}
    arrs = {"raw": raw, "cleaned_f": den, "cleaned": cleaned, "mask": mask, "nuisance": nu, "noise16": noise_u16(sig, NOISE_SCALE),
            "stain": stain8, "stain_levels": stain_lv, "noise_levels": sig, "cls": cls, "struct": st["struct"],
            "bits": {"specular": b0, "uneven": b1, "rough": b2n, "rough_v2": b2, "water": water, "oil": oil, "ink": ink},
            "corrected": st["corrected"]}
    return rep, arrs


def process_array(raw, mcfg, mode, ctx, name="", folder=""):
    return stage_b(stage_a(raw, mcfg, mode, ctx), mcfg, mode, name, folder)


def process_frame(path: Path, mcfg, mode: str, out_dir: Path, ctx: dict, folder: str = ""):
    t0 = time.perf_counter()
    li = load_image(path, mcfg["input"])
    rep, arrs = process_array(li.data, mcfg, mode, ctx, path.stem, folder)
    rep["bit_depth"] = li.meta["bit_depth"]
    t1 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_png(out_dir / f"{path.stem}_cleaned.png", arrs["cleaned"])
    write_png(out_dir / f"{path.stem}_mask.png", arrs["mask"])
    write_png(out_dir / f"{path.stem}_nuisance.png", arrs["nuisance"])
    write_png(out_dir / f"{path.stem}_noise.png", arrs["noise16"])
    write_png(out_dir / f"{path.stem}_stain.png", arrs["stain"])
    rep["files"] = {k: f"{path.stem}_{k}.png" for k in ("cleaned", "mask", "nuisance", "noise", "stain")}
    rep["runtime_s"]["write"] = round(time.perf_counter() - t1, 3)
    rep["runtime_s"]["total"] = round(time.perf_counter() - t0, 3)
    return rep, arrs


# ------------------------------------------------------------------------------------------------ folders
def run_v3(input_path, cfg, out_root, mode=None, frames=None, run_all=False, root=None, allow_protected=False, **_):
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
        folder_key = "/".join(rel.parts[-2:]) if len(rel.parts) >= 2 else str(rel)
        out_dir = out_root / rel
        t_f = time.perf_counter()
        ctx = prepare_folder(d, mcfg, m, plist)
        log.info("%s: prepared (%s) %.1fs", rel, ctx["seq_info"].get("status"), ctx["seq_info"]["prepare_s"])
        frep, regs, prev = {}, [], None
        rcfg = mcfg["registration"]
        for p in plist:
            try:
                r, arrs = process_frame(p, mcfg, m, out_dir, ctx, folder_key)
                log.info("%s/%s: ROI %.1f%% flags %s overlap %.2f%% frame-level %s %.2fs", rel, p.stem, r["roi_percent"],
                         r["nuisance_percent_of_roi"], r["overlap"]["any_overlap_pct_of_roi"], list(r["patterns"]["frame_level"]),
                         r["runtime_s"]["total"])
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
        summary = {}
        for cl in ("oil", "water", "ink", "fixture"):
            summary[cl] = {"willson_confirmed_frames": sorted(k for k, v in frep.items() if v["labels"].get(cl, {}).get("willson_confirmed")),
                           "confirmed_and_detected": sorted(k for k, v in frep.items() if v["labels"].get(cl, {}).get("status") == "confirmed_and_detected"),
                           "confirmed_not_detected": sorted(k for k, v in frep.items() if v["labels"].get(cl, {}).get("status") == "confirmed_NOT_detected"),
                           "auto_suspected_new": sorted(k for k, v in frep.items() if v["labels"].get(cl, {}).get("status") == "auto_suspected")}
        write_json(out_dir / "report.json", {
            "tool": {"name": "aoi_preprocess", "version": __version__, "pipeline": "v3"}, "mode": m, "folder": str(rel),
            "frames_processed": sorted(frep), "frames_selected_by_user": bool(frames),
            "roi_rule": "mask excludes ONLY external background, the physical boundary (+margin) and fixtures (v3: incl. the Side end-frame holder band); "
                        "specular / uneven-dim / rough / water / oil / ink stay INSIDE and are labelled in *_nuisance.png (flags overlap freely)",
            "nuisance_bit_encoding": V3_BIT_TABLE,
            "nuisance_note": "uint8 bit-flags (value = sum of bits); flags OVERLAP freely (a pixel can be oil+water+ink+rough); 0 outside the ROI. "
                             "oil / water / ink are SUSPECTED region labels learned from frame-level templates, nothing is excluded",
            "ink_note": "carrier-body ink cannot be told apart from particle defects by shape (dark-blob density is ~12x higher inside the contaminated band but the "
                        "blobs look like isolated particles); bit 32 only marks large (>=150 px) dark patches. Dark blobs <150 px inside the band are covered by the oil/water bits.",
            "noise_map": {"file_suffix": "_noise.png", "dtype": "uint16", "scale": NOISE_SCALE,
                          "definition": "sigma_levels8 = value / scale; local robust sigma of the cleaned image (32px blocks), 0 outside ROI. "
                                        "USE IT as the local threshold scale: inside contamination it is higher (residual stain texture is not removed)"},
            "stain_file": {"file_suffix": "_stain.png", "dtype": "uint8",
                           "definition": "stain_levels8 = (value - 128) / gain ; value 128 = no stain removed, 0 outside the ROI. cleaned = corrected - stain_levels8/255 (before NLM)",
                           "gain": float(mcfg["v3"]["stain"]["stain_gain"]),
                           "method": mcfg["v3"]["stain"].to_dict()},
            "mask_encoding": {"255": "ROI (inspect)", "0": "external background / boundary margin / fixture / end background"},
            "sequence_background": ctx["seq_info"],
            "abnormal_frames": sorted(abn),
            "abnormal_frames_detail": {k: {"flags": frep[k]["flags"], "detected_by_stats": frep[k]["abnormal"], "listed_in_config": k in expected,
                                           "roi_percent": frep[k]["roi_percent"], "nuisance_percent_of_roi": frep[k]["nuisance_percent_of_roi"],
                                           "stats_roi_levels8": frep[k]["stats_roi_levels8"],
                                           "structure": {kk: vv for kk, vv in frep[k]["structure"].items() if kk != "rows_detected"}}
                                       for k in sorted(abn) if k in frep},
            "labels_summary": summary,
            "labels_note": "frame-level labels from Willson (not pixel masks, not exhaustive): willson_confirmed vs auto_detected per frame in frames[*].labels",
            "registration": {"enabled": bool(m == "edge" and rcfg["enabled"]),
                             "sign": "dx,dy = shift of frame k+1 content relative to frame k (px, +x right, +y down)",
                             "pairs": regs} if m == "edge" else {"enabled": False},
            "folder_runtime_s": round(time.perf_counter() - t_f, 2),
            "frames": frep, "parameters": mcfg.to_dict()})
    return reports, failed
