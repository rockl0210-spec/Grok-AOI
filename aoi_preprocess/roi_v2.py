"""v2 ROI: ONLY external background, the physical boundary (+ small margin) and fixtures are excluded.

Specular reflection, uneven / dim illumination, rough texture bands, water marks and oil films stay INSIDE the ROI
(they are reported by NN_nuisance.png instead). Class codes (debug only):
  0 outside (external background)   1 valid   2 boundary margin   3 fixture   4 end background (Side end frames)
mask = (class == 1) * 255
"""
from __future__ import annotations

import cv2
import numpy as np

from .roi import auto_fixture_mask
from .roi_modes import _ell, side_rows, to_u8

C_OUT, C_VALID, C_MARGIN, C_FIXTURE, C_ENDBG = 0, 1, 2, 3, 4
V2_CLASS_NAMES = {0: "external_background", 1: "valid", 2: "boundary_margin", 3: "fixture", 4: "end_background"}


# ------------------------------------------------------------------------------------------------ EDGE
def edge_roi_v2(d: np.ndarray, M: np.ndarray, c) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
    """d = distance (px) to the external background (0 outside), M = substrate mask.
    Dark objects that merely touch the frame border (debris / stains lying ON the part) are NOT external background:
    a background component must span >= bg_min_extent_frac of the frame width or height (the real edge runs through the
    frame); smaller dark components are filled back into the substrate so they stay inspectable.
    ROI = substrate beyond the edge margin. Returns (class map, info, M_used, d_used)."""
    h, w = M.shape
    em = float(c["structure"]["edge_margin_px"])
    frac = float(c["structure"]["bg_min_extent_frac"])
    inv = (M == 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    filled = []
    M2 = M.copy()
    for i in range(1, n):
        bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if bw < frac * w and bh < frac * h:
            M2[lab == i] = 1
            filled.append({"bbox_xywh": [int(st[i, 0]), int(st[i, 1]), int(bw), int(bh)], "area_px": int(st[i, cv2.CC_STAT_AREA])})
    if filled:
        d = cv2.distanceTransform((M2 > 0).astype(np.uint8), cv2.DIST_L2, 5)
    cls = np.full(M.shape, C_VALID, np.uint8)
    cls[d <= em] = C_MARGIN
    cls[M2 == 0] = C_OUT
    b = int(c["border_margin_px"])
    if b > 0:
        for sl in (np.s_[:b, :], np.s_[-b:, :], np.s_[:, :b], np.s_[:, -b:]):
            seg = cls[sl]
            seg[seg == C_VALID] = C_MARGIN
    return cls, {"edge_margin_used_px": em, "internal_dark_objects_kept_in_roi": filled[:10]}, M2, d


def apply_fixture_v2(cls, raw, c):
    """Auto fixtures: large bright/dark blobs inside the ROI. By default a blob only counts as a fixture if it touches
    the frame border (a clamp enters from outside; free-floating dark objects are debris = defects and must stay)."""
    n_rect = 0
    try:
        rects = c["fixture_rects"]
    except Exception:
        rects = None
    if rects:                                           # manual fixtures: [[x0, y0, x1, y1], ...] in original px
        cls = cls.copy()
        for x0, y0, x1, y1 in rects:
            sel = np.zeros(cls.shape, bool)
            sel[int(y0):int(y1), int(x0):int(x1)] = True
            sel &= cls == C_VALID
            n_rect += int(sel.sum())
            cls[sel] = C_FIXTURE
    f = c["fixture_auto"]
    if not f["enabled"]:
        return cls, n_rect
    valid = (cls == C_VALID).astype(np.uint8)
    fx = auto_fixture_mask(raw, valid, f).astype(np.uint8)
    if f["require_border_touch"] and fx.any():
        n, lab, st, _ = cv2.connectedComponentsWithStats(fx, connectivity=8)
        h, w = fx.shape
        keep = np.zeros(n, bool)
        for i in range(1, n):
            x, y, bw, bh = (st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
            keep[i] = x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2
        fx = keep[lab].astype(np.uint8)
    if not fx.any():
        return cls, n_rect
    fx = cv2.dilate(fx, _ell(2 * int(f["margin_px"]) + 1)) > 0
    cls = cls.copy()
    n_px = int((fx & (cls == C_VALID)).sum())
    cls[fx & (cls == C_VALID)] = C_FIXTURE
    return cls, n_px + n_rect


# ------------------------------------------------------------------------------------------------ SIDE
def side_rows_consensus(raws_or_rows: list[dict], c) -> dict:
    """Folder-level strip rows: median of the confident per-frame detections (camera / stage are stationary)."""
    out = {}
    for k in ("top", "bottom"):
        v = [r[k]["row"] for r in raws_or_rows if r[k]["row"] is not None and r[k]["confidence"] >= float(c["min_confidence"])]
        out[k] = {"row": int(np.median(v)) if v else None, "n": len(v),
                  "spread_px": float(np.ptp(v)) if len(v) > 1 else 0.0}
    return out


def side_roi_v2(raw: np.ndarray, c, cons: dict | None = None) -> tuple[np.ndarray, dict]:
    h, w = raw.shape
    sc = c["strip"]
    r = side_rows(raw, sc)
    fb = (int(sc["fallback_rows_frac"][0] * h), int(sc["fallback_rows_frac"][1] * h))
    rows, src = {}, {}
    for k, fbv in (("top", fb[0]), ("bottom", fb[1])):
        det, conf = r[k]["row"], r[k]["confidence"]
        cv = cons[k]["row"] if cons and cons.get(k, {}).get("row") is not None else None
        if det is not None and conf >= float(sc["min_confidence"]) and (cv is None or abs(det - cv) <= float(sc["consensus_tol_px"])):
            rows[k], src[k] = det, "frame"
        elif cv is not None:
            rows[k], src[k] = cv, "folder_consensus"
        else:
            rows[k], src[k] = fbv, "fallback_fraction"
    top, bot = int(rows["top"]), int(rows["bottom"])
    m = int(sc["boundary_margin_px"])
    info = {"strip_top_row": top, "strip_bottom_row": bot, "rows_source": src, "rows_detected": r,
            "boundary_margin_px": m}
    cls = np.full((h, w), C_VALID, np.uint8)
    cls[:top + m] = C_OUT
    cls[bot - m:] = C_OUT
    cls[top:top + m] = C_MARGIN
    cls[bot - m:bot] = C_MARGIN
    cls[:top] = C_OUT
    cls[bot:] = C_OUT
    # ---- strip END frames: the part ends, a smooth out-of-focus background fills the rest of the row range
    info.update(_side_end(raw, top, bot, c, cls))
    return cls, info


def _side_end(raw, top, bot, c, cls) -> dict:
    h, w = raw.shape
    u8 = to_u8(raw)
    e = c["end"]
    out = {"end_side": None, "end_x": None, "full_height_bands": []}
    # full-height saturated bands (the end-chamfer reflection)
    hi = (u8 >= int(e["band_level"])).astype(np.uint8)
    hi = cv2.morphologyEx(hi, cv2.MORPH_OPEN, _ell(e["band_open_px"]))
    n, lab, st, _ = cv2.connectedComponentsWithStats(hi, connectivity=8)
    bands = []
    for i in range(1, n):
        bw, bh, x0, a = (st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT], st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_AREA])
        if a >= int(e["band_min_area_px"]) and bh >= float(e["band_min_height_frac"]) * max(1, bot - top) \
                and bw <= float(e["band_max_width_frac"]) * w:
            bands.append({"x0": int(x0), "x1": int(x0 + bw), "area": int(a)})
    out["full_height_bands"] = bands
    if not bands:
        return out
    bnd = max(bands, key=lambda b: b["area"])
    r0, r1 = top + int(0.2 * (bot - top)), bot - int(0.2 * (bot - top))
    mid = u8[r0:r1].astype(np.float32)
    med = np.median(mid, axis=0)
    mrg = int(e["band_margin_px"])
    left = med[: max(1, bnd["x0"] - mrg)]
    right = med[min(w - 1, bnd["x1"] + mrg):]
    ml, mr = float(np.median(left)), float(np.median(right)) if right.size else 0.0
    out["end_side_medians"] = [ml, mr]
    if max(ml, mr) < float(e["min_ratio"]) * max(min(ml, mr), 1.0):
        return out
    side = "left" if ml > mr else "right"
    out["end_side"] = side
    prof = cv2.GaussianBlur(med.reshape(1, -1), (0, 0), float(e["profile_sigma_px"]))[0]
    g = np.abs(np.gradient(prof))
    thr, run = float(e["step_levels_per_px"]), int(e["step_run_px"])
    if side == "left":
        rng = range(0, max(1, bnd["x0"] - mrg))
    else:
        rng = range(w - 1, min(w - 1, bnd["x1"] + mrg), -1)
    xs = None
    cnt = 0
    for x in rng:
        cnt = cnt + 1 if g[x] > thr else 0
        if cnt >= run:
            xs = x - (run - 1) if side == "left" else x + (run - 1)      # outermost column of the step
            break
    out["end_boundary_found"] = xs is not None
    if xs is None:                                        # fall back to the band edge
        xs = (bnd["x0"] - mrg) if side == "left" else (bnd["x1"] + mrg)
    out["end_x"] = int(xs)
    me = int(e["boundary_margin_px"])
    if side == "left":
        a, b = slice(0, xs), slice(xs, xs + me)
    else:
        a, b = slice(xs + 1, w), slice(xs + 1 - me, xs + 1)
    seg = cls[:, a]
    seg[seg != C_OUT] = C_ENDBG
    seg = cls[:, b]
    seg[seg == C_VALID] = C_MARGIN
    return out


def side_flags_v2(info: dict) -> dict:
    f = []
    if info["end_side"] is not None:
        f.append("strip_end_frame")
    if any(v != "frame" for v in info["rows_source"].values()):
        f.append("rows_from_consensus_or_fallback")
    abn = "strip_end_frame" in f
    return {"flags": f, "abnormal": abn}
