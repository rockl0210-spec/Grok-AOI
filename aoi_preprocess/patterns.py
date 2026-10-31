"""v3: template-derived detectors for the carrier-body contamination patterns (oil film / water marks / ink-like dark
blobs) and the Side end-fixture.  Everything works on the RAW 8-bit-scale gray image (float 0..255) and the ROI.

Features (all derived from a high-pass hp = R - gauss(R, 4)):
  E  : local RMS of the amplitude-clipped high-pass (clip +-8)  -> 'brush / mottled texture energy'  (oil)
       clipping makes isolated spots / blobs contribute almost nothing, so E responds to *textured area*, not to particles
  D  : local density of dark droplets (hp < -20)                                   -> water droplets / rings
  P  : local density of bright blobs (hp > +30) that lie inside the oil region      -> bright puddles
  K  : large very dark compact blobs (area >= ink_min_area) inside / hugging the contaminated band -> ink-like (suspected)
"""
from __future__ import annotations
import cv2
import numpy as np

DEFAULTS = {
    "scale": 4,                # feature maps are computed at 1/scale resolution
    "hp_sigma": 4.0, "clip": 8.0, "e_sigma": 16.0,
    "oil_E": 3.2,              # E threshold (clean baseline ~2.2-2.4)
    "oil_open_px": 72, "oil_min_area_px": 20000,
    "frame_oil_min_pct": 2.0,  # frame-level rule (% of ROI)
    "dark_thr": -20.0, "dark_sigma": 30.0, "water_D": 0.028, "water_open_px": 48,
    "water_frame_p95": 0.04,   # frame-level: p95 of D inside ROI
    "bright_thr": 30.0, "puddle_min_area_px": 60, "puddle_sigma": 30.0, "puddle_frac": 0.02,
    "dilate_px": 64,
    "ink_thr": -45.0, "ink_min_area_px": 150, "ink_min_solidity": 0.0,
}


def _ds(a, s):
    return cv2.resize(a, (a.shape[1] // s, a.shape[0] // s), interpolation=cv2.INTER_AREA)


def _us(a, shape, interp=cv2.INTER_LINEAR):
    return cv2.resize(a, (shape[1], shape[0]), interpolation=interp)


def features(R: np.ndarray, roi: np.ndarray, c: dict = DEFAULTS) -> dict:
    """R float32 0..255, roi bool.  Returns small-scale maps + full-res hp."""
    s = int(c["scale"])
    hp = R - cv2.GaussianBlur(R, (0, 0), c["hp_sigma"])
    hpc = np.clip(hp, -c["clip"], c["clip"])
    w = roi.astype(np.float32)
    ws = _ds(w, s)
    e2 = _ds(hpc * hpc * w, s)
    sg = c["e_sigma"] / s
    den = cv2.GaussianBlur(ws, (0, 0), sg) + 1e-6
    E = np.sqrt(cv2.GaussianBlur(e2, (0, 0), sg) / den)
    dk = _ds(((hp < c["dark_thr"]) & roi).astype(np.float32), s)
    D = cv2.GaussianBlur(dk, (0, 0), c["dark_sigma"] / s) / (cv2.GaussianBlur(ws, (0, 0), c["dark_sigma"] / s) + 1e-6)
    return {"hp": hp, "E": E, "D": D, "ws": ws, "s": s}


def _open(b, px, s):
    k = max(3, int(px / s) | 1)
    return cv2.morphologyEx(b.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


def _drop_small(b, min_px, s):
    n, lab, st, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_px / (s * s)
    return keep[lab]


def detect(R: np.ndarray, roi: np.ndarray, c: dict = DEFAULTS) -> dict:
    """oil / water / ink full-resolution boolean regions (all inside roi) + per-frame measurements."""
    f = features(R, roi, c)
    s, hp = f["s"], f["hp"]
    shp = roi.shape
    ws = f["ws"] > 0.5
    oil_s = _drop_small(_open((f["E"] > c["oil_E"]) & ws, c["oil_open_px"], s), c["oil_min_area_px"], s)
    dil = max(3, int(c["dilate_px"] / s) | 1)
    oil_d = cv2.dilate(oil_s.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil))) > 0
    # ---- water: dark-droplet density core (anywhere textured) + bright puddles inside the oil region
    wat_s = _open((f["D"] > c["water_D"]) & ws & oil_d, c["water_open_px"], s)
    # bright puddles: bright blobs of a minimum size inside the oil region
    br = ((hp > c["bright_thr"]) & roi).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(br, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= c["puddle_min_area_px"]
    bigbr = keep[lab]
    oil_full = _us(oil_s.astype(np.uint8), shp, cv2.INTER_NEAREST) > 0
    pud = bigbr & cv2.dilate(oil_full.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    pud_s = _ds(pud.astype(np.float32), s)
    P = cv2.GaussianBlur(pud_s, (0, 0), c["puddle_sigma"] / s) / (cv2.GaussianBlur(f["ws"], (0, 0), c["puddle_sigma"] / s) + 1e-6)
    pud_reg = (P > c["puddle_frac"] / 4) & oil_d
    water_s = wat_s | pud_reg
    water = _us(water_s.astype(np.uint8), shp, cv2.INTER_NEAREST) > 0
    oil = oil_full & roi
    water &= roi
    # ---- ink-like: big very dark compact blobs lying in/near the contaminated band
    dk = ((hp < c["ink_thr"]) & roi).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(dk, connectivity=8)
    ink = np.zeros(shp, bool)
    near = cv2.dilate(((oil | water)).astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < c["ink_min_area_px"]:
            continue
        x, y, ww, hh = st[i, :4]
        sub = lab[y:y + hh, x:x + ww] == i
        if near[y:y + hh, x:x + ww][sub].mean() < 0.5:
            continue
        cnt, _ = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.contourArea(cv2.convexHull(max(cnt, key=cv2.contourArea)))
        if a / max(hull, 1) >= c["ink_min_solidity"]:
            ink[y:y + hh, x:x + ww] |= sub
    roi_n = max(int(roi.sum()), 1)
    Dv = f["D"][ws]
    meas = {
        "oil_pct": 100.0 * oil.sum() / roi_n, "water_pct": 100.0 * water.sum() / roi_n, "ink_pct": 100.0 * ink.sum() / roi_n,
        "E_median": float(np.median(f["E"][ws])) if ws.any() else 0.0,
        "dark_p95": float(np.percentile(Dv, 95)) if Dv.size else 0.0,
        "puddle_frac": float(pud.sum() / roi_n * 100) if roi_n else 0.0,
    }
    return {"oil": oil, "water": water, "ink": ink, "E": f["E"], "D": f["D"], "meas": meas, "features": f}


# ------------------------------------------------------------------------------------------------ Side end fixture
FIXTURE_DEFAULTS = {
    "enabled": True,"search_px": 600, "min_drop_levels": 30.0, "max_dark_level": 70.0, "win_px": 52, "gap_px": 8,
                    "margin_px": 6, "min_grad": 3.0, "row_frac": [0.1, 0.9], "min_roi_px": 20000, "profile_sigma": 3.0}


def side_end_fixture(raw255: np.ndarray, cls: np.ndarray, c: dict = FIXTURE_DEFAULTS, valid: int = 1, fixture: int = 3):
    """Side END frames show the part end (dark body, sharp step) preceded by an out-of-focus ribbed holder ('fixture').
    v2 started the ROI at the outer edge of that holder ghost; this finds the sharp body step and re-classes the strip between
    the ROI start and the step (+margin) as fixture.  Applies only if the ROI does not reach the image border (end frames).
    Returns (new cls, list of fixture dicts)."""
    roi = cls == valid
    h, w = roi.shape
    out = []
    if roi.sum() < c["min_roi_px"]:
        return cls, out
    cols = np.where(roi.any(0))[0]
    rows = np.where(roi.any(1))[0]
    r0 = int(rows[0] + c["row_frac"][0] * (rows[-1] - rows[0]))
    r1 = int(rows[0] + c["row_frac"][1] * (rows[-1] - rows[0]))
    p = np.median(raw255[r0:r1], axis=0).astype(np.float32)
    p = cv2.GaussianBlur(p.reshape(1, -1), (0, 0), c["profile_sigma"]).ravel()
    g = np.gradient(p)
    win, gap = int(c["win_px"]), int(c["gap_px"])
    newcls = cls.copy()
    mg = float(c["min_grad"])

    def pick(rng, sign):
        """outermost sharp step (local extremum of sign*g, >= min_grad) whose two sides satisfy the contrast / dark rules"""
        best = None
        gs = sign * g
        for j in rng:
            if gs[j] < mg or gs[j] < gs[max(j - 10, 0):j + 11].max():
                continue
            if sign < 0:
                A, B = float(p[j - win:j - gap].mean()), float(p[j + gap:j + win].mean())
            else:
                A, B = float(p[j + gap:j + win].mean()), float(p[j - win:j - gap].mean())
            if (A - B) >= c["min_drop_levels"] and B <= c["max_dark_level"]:
                best = (int(j), A - B)
                if sign < 0:
                    return best                                   # left end: outermost = smallest x
        return best                                               # right end: outermost = largest x

    for side in ("left", "right"):
        if side == "left" and cols[0] > 40:
            rng = np.arange(cols[0] + 20, min(w - win - gap - 1, cols[0] + c["search_px"]))
            r = pick(rng, -1)
            x0 = int(cols[0])
        elif side == "right" and cols[-1] < w - 41:
            rng = np.arange(max(win + gap + 1, cols[-1] - c["search_px"]), cols[-1] - 20)
            r = pick(rng, +1)
            x1 = int(cols[-1])
        else:
            continue
        ok = r is not None
        if ok:
            j, contrast = r
            if side == "left":
                x1 = j + int(c["margin_px"])
            else:
                x0 = j - int(c["margin_px"])
        else:
            j, contrast = -1, 0.0
        if not ok:
            out.append({"side": side, "detected": False, "step_x": j, "contrast_levels": round(contrast, 1)})
            continue
        sub = newcls[:, x0:x1 + 1]
        m = sub == valid
        n = int(m.sum())
        if n == 0:
            continue
        sub[m] = fixture
        yy = np.where(m.any(1))[0]
        bx = [x0, int(yy[0]), x1, int(yy[-1])]
        out.append({"side": side, "detected": True, "step_x": j, "bbox_xyxy": bx,
                    "polygon_xy": [[bx[0], bx[1]], [bx[2], bx[1]], [bx[2], bx[3]], [bx[0], bx[3]]],
                    "area_px": n, "contrast_levels": round(contrast, 1),
                    "confidence": round(float(min(1.0, contrast / 80.0)), 2),
                    "description": "out-of-focus ribbed holder band between the v2 ROI start and the sharp part-body step"})
    return newcls, out
