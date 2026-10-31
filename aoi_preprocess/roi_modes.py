"""Stage 4 (real data) - per-mode ROI definitions.

Two inspection modes, each with its own structure model (selected from the folder name or by --mode):

EDGE  (bright-field close-up of the substrate edge; Edge1 / Edge2)
    structure : substrate (bright) vs background (dark), vertical-ish edge line; optional rough textured
                band along the edge; vignetted / shadowed corners in the first/last frame of a scan.
    valid ROI = substrate
                - margin around the edge line            (class 2  edge_margin)
                - dim zone: smoothed level < level_frac * plateau  (edge roll-off, vignette, corner shadow; class 3)
                - rough band (local-texture measure, hysteresis, attached to the edge; class 4)
                - large fixture-like blobs               (class 5)
                - frame border margin                    (class 9)

SIDE  (dark-field side-face strip: black above/below, bright specular lines)
    structure : horizontal strip between two row boundaries (found by column voting), a chamfer/bevel band
                just inside each boundary, saturated vertical specular bands and (strip-end frames) the
                end cap / bright background beyond the band.
    valid ROI = main face between the bevel bands
                - specular components (opening removes thin scratches, area filter keeps particles; class 6)
                - bevel bands                            (class 7)
                - end region beyond a full-height specular band (class 8)
                - fixture-like blobs (class 5) and frame border margin (class 9)

Everything here works on a uint8 view of the frame for the structure steps (fast, 8-bit data is 8-bit anyway).
Class map codes: see CLASS_NAMES.  mask = (class == 1) * 255.
"""
from __future__ import annotations

import cv2
import numpy as np

from .roi import _tls_line, auto_fixture_mask

CLS_OUTSIDE, CLS_VALID, CLS_EDGE_MARGIN, CLS_DIM, CLS_ROUGH, CLS_FIXTURE, CLS_SPECULAR, CLS_BEVEL, CLS_END, \
    CLS_BORDER = range(10)
CLASS_NAMES = {0: "outside_substrate", 1: "valid", 2: "edge_margin", 3: "dim_zone", 4: "rough_band",
               5: "fixture_like", 6: "specular", 7: "bevel_band", 8: "end_region", 9: "frame_border"}
# BGR colours for the preview overlay
CLASS_COLORS = {0: (160, 60, 0), 2: (0, 220, 255), 3: (200, 0, 200), 4: (0, 140, 255), 5: (0, 0, 255),
                6: (255, 255, 0), 7: (0, 200, 120), 8: (255, 120, 120), 9: (120, 120, 120)}


def to_u8(img01: np.ndarray) -> np.ndarray:
    return np.clip(img01 * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _ell(px):
    k = int(px) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _fill_holes(b):
    h, w = b.shape
    inv = (1 - b).astype(np.uint8)
    pad = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    ff = pad.copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2)
    holes = (ff[1:-1, 1:-1] == 1)
    out = b.copy()
    out[holes] = 1
    return out


def _largest(b):
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    if n <= 1:
        return b
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == i).astype(np.uint8)


def _small(u8, s):
    return cv2.resize(u8, (max(8, int(round(u8.shape[1] * s))), max(8, int(round(u8.shape[0] * s)))),
                      interpolation=cv2.INTER_AREA)


def _up(a, shape_hw, interp=cv2.INTER_LINEAR):
    return cv2.resize(a, (shape_hw[1], shape_hw[0]), interpolation=interp)


# =================================================================================================== EDGE
def edge_structure(raw: np.ndarray, c) -> dict:
    """Substrate / background segmentation, edge-line fit and shadow-level map for an Edge frame.
    Returns dict with arrays (substrate M, dist d, level L, ref) and scalar features for the abnormal-frame test."""
    u8 = to_u8(raw)
    h, w = u8.shape
    sm = cv2.GaussianBlur(u8, (0, 0), float(c["blur_px"]))
    sub = sm[::4, ::4]
    hi = float(np.percentile(sub, c["plateau_percentile"]))
    lo = float(np.percentile(sub, c["background_percentile"]))
    info = {"plateau_level": hi, "background_level": lo}
    if hi - lo < float(c["min_contrast_levels"]):     # no recognisable edge in this frame
        M = np.ones((h, w), np.uint8)
        info["substrate_found"] = False
    else:
        thr = lo + float(c["threshold_frac"]) * (hi - lo)
        M = (sm > thr).astype(np.uint8)
        k = _ell(c["morph_px"])
        M = cv2.morphologyEx(M, cv2.MORPH_CLOSE, k)
        M = cv2.morphologyEx(M, cv2.MORPH_OPEN, k)
        M = _fill_holes(_largest(M))
        info["substrate_found"] = True
        info["threshold_level"] = thr
    bg = (M == 0)
    info["bg_frac"] = float(bg.mean())
    # which image borders touch background (>= border_touch_frac of the border pixels)
    tf = float(c["border_touch_frac"])
    sides = {"left": bg[:, 0].mean(), "right": bg[:, -1].mean(), "top": bg[0, :].mean(), "bottom": bg[-1, :].mean()}
    info["bg_touch_sides"] = [k for k, v in sides.items() if v >= tf]
    # straight edge line (robust TLS on the boundary pixels), fraction of boundary points on the line
    bnd = (M - cv2.erode(M, np.ones((3, 3), np.uint8))) > 0
    bnd[:4, :] = bnd[-4:, :] = False
    bnd[:, :4] = bnd[:, -4:] = False
    ys, xs = np.nonzero(bnd)
    info["line"] = None
    info["line_inlier_frac"] = 1.0 if len(xs) == 0 else 0.0
    if len(xs) > 50:
        P = np.stack([xs, ys], 1).astype(np.float64)[:: max(1, len(xs) // 4000)]
        cen, d0 = _tls_line(P)
        for _ in range(4):
            nrm = np.array([-d0[1], d0[0]])
            r = (P - cen) @ nrm
            sd = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
            keep = np.abs(r - np.median(r)) < max(3 * sd, 2.0)
            if keep.sum() < 20:
                break
            cen, d0 = _tls_line(P[keep])
        nrm = np.array([-d0[1], d0[0]])
        r = (P - cen) @ nrm
        info["line_inlier_frac"] = float((np.abs(r) <= float(c["line_tol_px"])).mean())
        ang = float(np.degrees(np.arctan2(d0[0], d0[1])))          # 0 = line along +y (vertical)
        a = ang if abs(ang) <= 90 else ang - np.sign(ang) * 180
        info["line"] = {"point": [float(cen[0]), float(cen[1])], "dir": [float(d0[0]), float(d0[1])],
                        "angle_from_vertical_deg": float(a) if abs(a) < 45 else float(a - np.sign(a) * 90)}
    d = cv2.distanceTransform(M, cv2.DIST_L2, 3)
    # shadow / vignette level map (median + gaussian on a 1/4 scale copy)
    s = 0.25
    med = cv2.medianBlur(_small(u8, s), 5)
    g = cv2.GaussianBlur(med.astype(np.float32), (0, 0), float(c["level_sigma_px"]) * s)
    L = _up(g, (h, w))
    inner = (M > 0) & (d > float(c["edge_margin_px"]) + 50)
    sel = inner if inner.sum() > 20000 else (M > 0)
    ref = float(np.percentile(L[sel][::37], c["level_ref_percentile"])) if sel.any() else hi
    info["level_ref"] = ref
    return {"M": M, "d": d, "L": L, "ref": ref, "info": info, "u8": u8}


def edge_roughness(raw: np.ndarray, M: np.ndarray, d: np.ndarray, c) -> tuple[np.ndarray, dict]:
    """Rough textured band along the edge. T = fraction of pixels with |hp| > t in a box window, where
    hp = img - gaussian_blur(img, hp_sigma). Hysteresis (T>hi seeds, T>lo grows) and attachment to the edge."""
    h, w = raw.shape
    img = raw * 255.0
    hp = img - cv2.GaussianBlur(img, (0, 0), float(c["hp_sigma_px"]))
    b = (np.abs(hp) > float(c["hp_threshold_levels"])).astype(np.uint8)
    s = float(c["work_scale"])
    bs = cv2.resize(b.astype(np.float32), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    k = int(round(c["window_px"] * s)) | 1
    T = cv2.boxFilter(bs, -1, (k, k), borderType=cv2.BORDER_REFLECT)
    Ms = _small(M, s) > 0
    ds = cv2.resize(d, (bs.shape[1], bs.shape[0]), interpolation=cv2.INTER_AREA)
    lo_m = ((T > float(c["thr_low"])) & Ms).astype(np.uint8)
    hi_m = ((T > float(c["thr_high"])) & Ms & (ds < float(c["attach_px"]) * s)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(lo_m, connectivity=8)
    keep = np.zeros(n, bool)
    if n > 1:
        ids = np.unique(lab[hi_m > 0])
        keep[ids[ids > 0]] = True
    rough_s = keep[lab].astype(np.uint8) if n > 1 else np.zeros_like(lo_m)
    rough_s = cv2.morphologyEx(rough_s, cv2.MORPH_CLOSE, _ell(max(3, int(c["close_px"] * s))))
    rough = _up(rough_s.astype(np.float32), (h, w)) > 0.5
    mg = int(c["margin_px"])
    if mg > 0:
        rough = cv2.dilate(rough.astype(np.uint8), _ell(2 * mg + 1)) > 0
    ar = rough & (M > 0)
    info = {"rough_area_frac_of_substrate": float(ar.sum() / max(1, (M > 0).sum())),
            "texture_T_smooth_median": float(np.median(T[Ms & ~(rough_s > 0)])) if (Ms & ~(rough_s > 0)).any() else None}
    # depth of band in px from the edge (max over rows) for the report
    if ar.any():
        info["rough_max_depth_px"] = float(d[ar].max())
    else:
        info["rough_max_depth_px"] = 0.0
    return rough, info


def edge_flags(info: dict, c) -> dict:
    f = []
    if not info["substrate_found"]:
        f.append("edge_not_found")
    if info["bg_frac"] > float(c["max_bg_frac"]):
        f.append("large_background_area")
    if len(info["bg_touch_sides"]) >= 2:
        f.append("corner_frame")
    notes = []
    if info.get("dim_frac_of_substrate", 0.0) > float(c["max_dim_frac"]):
        notes.append("large_dim_zone")
    if info["substrate_found"] and info["line_inlier_frac"] < float(c["min_line_inlier_frac"]) and "corner_frame" not in f:
        notes.append("edge_not_straight")
    f = [x for x in f if x != "edge_not_straight"]
    return {"flags": f, "notes": notes, "abnormal": len(f) > 0}


def edge_roi(raw: np.ndarray, c, struct=None, abnormal: bool = False):
    """Edge-mode class map (before the optional fixture step). `struct` = result of edge_structure (reused)."""
    h, w = raw.shape
    st = struct or edge_structure(raw, c["structure"])
    M, d, L, ref, info = st["M"], st["d"], st["L"], st["ref"], st["info"]
    em = float(c["structure"]["edge_margin_px"]) + (float(c["abnormal_extra_margin_px"]) if abnormal else 0.0)
    cls = np.full((h, w), CLS_VALID, np.uint8)
    dim = (L < float(c["structure"]["level_frac"]) * ref) & (M > 0)
    # isolated dark blobs that do not belong to the edge roll-off / corner shadow are NOT a "dim zone":
    # they are dark particles / fibres and must stay inspectable
    n, lab, stt, _ = cv2.connectedComponentsWithStats(dim.astype(np.uint8), connectivity=8)
    if n > 1:
        near_edge = cv2.dilate((d <= em + float(c["structure"]["dim_attach_px"])).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        touch = np.zeros(n, bool)
        touch[np.unique(lab[near_edge & dim])] = True
        big = stt[:, cv2.CC_STAT_AREA] >= int(c["structure"]["dim_min_area_px"])
        keep = touch | big
        keep[0] = False
        dim_removed = int((~keep[1:]).sum())
        dim = keep[lab]
    else:
        dim_removed = 0
    rough, rinfo = edge_roughness(raw, M, d, c["rough_band"]) if c["rough_band"]["enabled"] else \
        (np.zeros((h, w), bool), {})
    info["dim_frac_of_substrate"] = float((dim & (M > 0)).sum() / max(1, (M > 0).sum()))
    info.update(rinfo)
    info["isolated_dark_blobs_released"] = dim_removed
    cls[rough] = CLS_ROUGH      # lowest precedence first, later assignments win
    cls[dim] = CLS_DIM
    cls[d <= em] = CLS_EDGE_MARGIN
    cls[M == 0] = CLS_OUTSIDE
    b = int(c["border_margin_px"])
    if b > 0:
        edge = np.zeros((h, w), bool)
        edge[:b, :] = edge[-b:, :] = True
        edge[:, :b] = edge[:, -b:] = True
        cls[edge & (cls == CLS_VALID)] = CLS_BORDER
    info["edge_margin_used_px"] = em
    return cls, info


# =================================================================================================== SIDE
def _vote(positions, weights, tol):
    """Mode of a list of integer positions (+/- tol) -> (mode, support share)."""
    if len(positions) == 0:
        return None, 0.0
    p = np.asarray(positions)
    best, bs = None, 0.0
    for v in np.unique(p):
        s = float(weights[np.abs(p - v) <= tol].sum())
        if s > bs:
            best, bs = int(v), s
    sel = np.abs(p - best) <= tol
    return int(round(np.average(p[sel], weights=weights[sel]))), bs / max(float(weights.sum()), 1e-9)


def side_rows(raw: np.ndarray, c) -> dict:
    """Row boundaries of the strip via per-column-block voting of the strongest vertical step inside a search window."""
    u8 = to_u8(raw).astype(np.float32)
    h, w = u8.shape
    nb = int(c["col_blocks"])
    edges = np.linspace(0, w, nb + 1).astype(int)
    P = np.stack([np.median(u8[:, a:b], axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)  # H x nb
    P = cv2.GaussianBlur(P, (0, 0), float(c["row_smooth_px"]))
    s = int(c["step_px"])
    G = np.zeros_like(P)
    G[s:-s] = P[2 * s:] - P[:-2 * s]
    res = {}
    for name, (f0, f1), sign in (("top", c["top_search_frac"], 0), ("bottom", c["bottom_search_frac"], -1)):
        r0, r1 = int(f0 * h), int(f1 * h)
        seg = G[r0:r1]
        score = np.abs(seg) if sign == 0 else -seg
        idx = score.argmax(0)
        val = score.max(0)
        ok = val >= float(c["min_step_levels"])
        pos, conf = _vote(idx[ok] + r0, val[ok], float(c["vote_tol_px"]))
        if pos is not None:
            conf *= ok.mean()
        res[name] = {"row": pos, "confidence": float(conf)}
    return res


def side_roi(raw: np.ndarray, c, rows=None):
    """Side-mode class map + structure info. `rows` = optional (top, bottom) override (folder-level fallback)."""
    h, w = raw.shape
    u8 = to_u8(raw)
    sc = c["strip"]
    r = side_rows(raw, sc)
    fb_top, fb_bot = (int(sc["fallback_rows_frac"][0] * h), int(sc["fallback_rows_frac"][1] * h))
    info = {"rows_detected": r}
    top, bot = r["top"]["row"], r["bottom"]["row"]
    lowc = []
    if top is None or r["top"]["confidence"] < float(sc["min_confidence"]):
        lowc.append("top")
        top = rows[0] if rows else fb_top
    if bot is None or r["bottom"]["confidence"] < float(sc["min_confidence"]):
        lowc.append("bottom")
        bot = rows[1] if rows else fb_bot
    info.update({"strip_top_row": int(top), "strip_bottom_row": int(bot), "rows_low_confidence": lowc,
                 "rows_fallback_used": bool(lowc)})
    cls = np.full((h, w), CLS_VALID, np.uint8)
    cls[:top] = CLS_OUTSIDE
    cls[bot:] = CLS_OUTSIDE
    tb, bb = int(sc["top_bevel_px"]), int(sc["bottom_bevel_px"])
    cls[top:min(h, top + tb)] = CLS_BEVEL
    cls[max(0, bot - bb):bot] = CLS_BEVEL
    # specular components
    spc = c["specular"]
    hi = (u8 >= int(spc["threshold_level"])).astype(np.uint8)
    hi = cv2.morphologyEx(hi, cv2.MORPH_OPEN, _ell(spc["open_px"]))
    n, lab, st, _ = cv2.connectedComponentsWithStats(hi, connectivity=8)
    spec = np.zeros((h, w), np.uint8)
    bands = []
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a >= int(spc["min_area_px"]):
            comp = lab == i
            spec[comp] = 1
            bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            x0 = st[i, cv2.CC_STAT_LEFT]
            if bh >= float(spc["band_min_height_frac"]) * max(1, bot - top) and bw <= float(spc["band_max_width_frac"]) * w:
                bands.append({"x0": int(x0), "x1": int(x0 + bw), "area": int(a)})
    spec_d = cv2.dilate(spec, _ell(2 * int(spc["margin_px"]) + 1)) > 0
    info["specular_components"] = int(sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= int(spc["min_area_px"])))
    info["full_height_bands"] = bands
    end = np.zeros((h, w), bool)
    info["end_side"] = None
    if bands:
        bnd = max(bands, key=lambda b: b["area"])
        mrg = int(spc["margin_px"])
        mid = u8[top + tb: max(top + tb + 1, bot - bb)]
        left = mid[:, : max(1, bnd["x0"] - mrg)]
        right = mid[:, min(w - 1, bnd["x1"] + mrg):]
        ml = float(np.median(left)) if left.size > 50 else 0.0
        mr = float(np.median(right)) if right.size > 50 else 0.0
        info["end_side_medians"] = [ml, mr]
        if max(ml, mr) >= float(spc["end_min_ratio"]) * max(min(ml, mr), 1.0):
            if ml > mr:
                end[:, : bnd["x0"] - mrg] = True; info["end_side"] = "left"
            else:
                end[:, bnd["x1"] + mrg:] = True; info["end_side"] = "right"
    cls[(spec_d) & (cls == CLS_VALID)] = CLS_SPECULAR
    cls[end & (cls != CLS_OUTSIDE)] = CLS_END
    b = int(c["border_margin_px"])
    if b > 0:
        edge = np.zeros((h, w), bool)
        edge[:b, :] = edge[-b:, :] = True
        edge[:, :b] = edge[:, -b:] = True
        cls[edge & (cls == CLS_VALID)] = CLS_BORDER
    return cls, info


def side_flags(info: dict, c) -> dict:
    f = []
    if info["end_side"] is not None:
        f.append("strip_end_frame")
    elif info["full_height_bands"]:
        f.append("full_height_specular_band")
    if info["rows_low_confidence"]:
        f.append("rows_low_confidence")
    return {"flags": f, "abnormal": len(f) > 0}


def apply_fixture_class(cls, denoised, c):
    """Optional auto-detection of large fixture-like blobs inside the still-valid region (both modes)."""
    fcfg = c["fixture_auto"]
    if not fcfg["enabled"]:
        return cls
    valid = (cls == CLS_VALID).astype(np.uint8)
    fx = auto_fixture_mask(denoised, valid, fcfg) > 0
    fx = cv2.dilate(fx.astype(np.uint8), _ell(2 * int(fcfg["margin_px"]) + 1)) > 0
    cls = cls.copy()
    cls[fx & (cls == CLS_VALID)] = CLS_FIXTURE
    return cls
