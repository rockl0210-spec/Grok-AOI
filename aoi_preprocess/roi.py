"""Stage 4 - ROI masks.

    substrate mask: rotated RECTANGLE (default; minAreaRect + robust per-side line fits, optional rounded /
                    chamfered corners) or fitted CIRCLE (contour / Hough); `manual` overrides both
  - edge band    : pixels closer than `edge_exclusion_px` to the wafer boundary
  - fixtures     : polygons from config (+ optional external mask PNG) + optional auto-detection
                   (bright/dark intensity threshold, only large blobs near the edge)
  = valid ROI    : where defect inspection is allowed

All masks are uint8 {0,1} internally (written as 0/255 PNG), shape HxW.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger("aoi")


# ----------------------------------------------------------------------------- wafer boundary
def _binary_for_boundary(gray, cfg):
    """Smoothed, down-scaled threshold -> binary object mask. Returns (bin, scale)."""
    h, w = gray.shape
    s = min(1.0, float(cfg["work_max_side"]) / max(h, w))   # 1.0 = full resolution (best edge accuracy)
    small = cv2.resize(gray, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else gray
    k = int(cfg["blur_px"]) | 1
    small = cv2.GaussianBlur(small, (k, k), 0)
    u8 = np.clip(small * 255.0, 0, 255).astype(np.uint8)
    thr = cfg["threshold"]
    if thr == "otsu":
        t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        t = float(thr) * 255.0
    b = (u8 > t) if cfg["object_brighter_than_background"] else (u8 <= t)
    b = b.astype(np.uint8)
    # close small holes (particles / texture inside wafer), drop specks outside
    kk = max(3, int(round(cfg["morph_close_px"] * s))) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, ker)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, ker)
    return b, s


def _largest_contour(b):
    cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        raise RuntimeError("wafer boundary: no object found (check threshold / polarity in config)")
    return max(cs, key=cv2.contourArea)


def fit_circle_robust(pts: np.ndarray, iters=8, k=3.0):
    """Algebraic (Kasa) circle fit with MAD-based outlier trimming.

    Robust to clamp bumps / notches that make parts of the contour deviate from the circle.
    pts: Nx2 (x,y). Returns (cx, cy, r, n_inliers).
    """
    pts = pts.astype(np.float64)
    keep = np.ones(len(pts), bool)
    cx = cy = r = 0.0
    for _ in range(iters):
        p = pts[keep]
        A = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]
        b = (p ** 2).sum(1)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = sol[0], sol[1]
        r = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1e-9)))
        res = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        mad = 1.4826 * np.median(res[keep])
        new = res < max(1.0, k * mad)
        if new.sum() < 10 or (new == keep).all():
            break
        keep = new
    return cx, cy, r, int(keep.sum())


def _circle_contour(gray, cfg):
    b, s = _binary_for_boundary(gray, cfg)
    c = _largest_contour(b)[:, 0, :].astype(np.float64)
    cx, cy, r, n = fit_circle_robust(c, k=float(cfg["circle"]["outlier_sigma"]))
    # small-image px -> full-res px (pixel-centre convention)
    return ((cx + 0.5) / s - 0.5, (cy + 0.5) / s - 0.5, r / s, {"inlier_pts": n, "contour_pts": len(c)})


def _circle_hough(gray, cfg):
    h, w = gray.shape
    s = min(1.0, 800 / max(h, w))
    small = cv2.resize(gray, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else gray
    u8 = cv2.GaussianBlur(np.clip(small * 255, 0, 255).astype(np.uint8), (0, 0), float(cfg["blur_px"]) / 2)
    m = min(u8.shape)
    hc = cfg["circle"]["hough"]
    rmin, rmax = (int(m * f) for f in hc["radius_frac"])
    cs = cv2.HoughCircles(u8, cv2.HOUGH_GRADIENT, dp=1.0, minDist=m, param1=float(hc["canny_high"]),
                          param2=float(hc["accumulator_thr"]), minRadius=rmin, maxRadius=rmax)
    if cs is None:
        raise RuntimeError("Hough found no circle (relax accumulator_thr / radius_frac, or use method: contour)")
    cx, cy, r = cs[0, 0]
    return (cx + 0.5) / s - 0.5, (cy + 0.5) / s - 0.5, r / s, {"hough_candidates": int(len(cs[0]))}


def detect_wafer(gray: np.ndarray, cfg) -> tuple[np.ndarray, dict]:
    """Return (wafer_mask uint8 {0,1}, shape_info dict)."""
    h, w = gray.shape
    shape = cfg.choice("shape", ("circle", "rect"))
    yy, xx = np.mgrid[0:h, 0:w]
    manual = cfg["manual"]
    if shape == "circle":
        if manual["enabled"]:
            cx, cy = manual["center_xy"]
            r = manual["radius"]
            info = {"source": "manual"}
        else:
            method = cfg["circle"].choice("method", ("contour", "hough"))
            try:
                cx, cy, r, extra = (_circle_contour if method == "contour" else _circle_hough)(gray, cfg)
                info = {"source": method, **extra}
            except RuntimeError as e:
                if not cfg["circle"]["fallback_to_other"]:
                    raise
                log.warning("%s failed (%s); trying the other circle method", method, e)
                cx, cy, r, extra = (_circle_hough if method == "contour" else _circle_contour)(gray, cfg)
                info = {"source": "fallback", **extra}
        r -= float(cfg["boundary_shrink_px"])
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r).astype(np.uint8)
        info.update({"shape": "circle", "center_xy": [float(cx), float(cy)], "radius_px": float(r)})
        return mask, info
    # ---- rectangle / square substrate (possibly slightly rotated)
    rc = cfg["rect"]
    if manual["enabled"]:
        rect = (tuple(manual["center_xy"]), tuple(manual["size_wh"]), float(manual["angle_deg"]))
        corners = cv2.boxPoints(rect).astype(np.float64)
        info = {"source": "manual"}
    else:
        b, s = _binary_for_boundary(gray, cfg)
        pts = _largest_contour(b)[:, 0, :].astype(np.float64)
        pts = (pts + 0.5) / s - 0.5
        corners, info = fit_rect_robust(pts, rc)
    corners = _shrink_quad(corners, float(cfg["boundary_shrink_px"]))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.rint(corners * 16).astype(np.int32)], 1, shift=4)   # sub-pixel polygon fill
    geo = quad_geometry(corners)
    cs = rc["corner"]
    ctype = cs.choice("type", ("none", "round", "chamfer"))
    if ctype != "none" and float(cs["size_px"]) > 0:
        mask &= corner_cut_mask((h, w), geo, ctype, float(cs["size_px"]))
    info.update({"shape": "rect", "corners": corners.round(2).tolist(), "corner_type": ctype,
                 "corner_size_px": float(cs["size_px"]), **geo_to_info(geo)})
    lim = float(rc["warn_rotation_deg"])
    if abs(info["angle_deg"]) > lim:
        log.warning("fitted substrate rotation %.2f deg exceeds warn_rotation_deg=%.1f", info["angle_deg"], lim)
    return mask, info


# ----------------------------------------------------------------------------- rectangle fitting
def _tls_line(p):
    """Total-least-squares line through Nx2 points -> (point, unit direction)."""
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    return c, vt[0]


def _fit_side(P, n_hint, rc, fixed_dir=None):
    """Robustly fit one side from contour points P (Nx2) belonging to that side.

    n_hint: initial outward unit normal (from minAreaRect). Clamps / notches put a *minority* of
    points far from the true edge, so: (1) take the histogram MODE of the signed distance along the
    hint normal as the edge position, (2) keep points within mode_tol_px, (3) TLS line fit,
    (4) iterate with MAD-based trimming of the residuals.
    """
    d = P @ n_hint
    lo, hi = d.min(), d.max()
    hist, edges = np.histogram(d, bins=max(8, int(hi - lo) + 1))
    hist = np.convolve(hist, np.ones(3), "same")
    m = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
    keep = np.abs(d - m) < float(rc["mode_tol_px"])
    if keep.sum() < int(rc["min_points"]):
        raise RuntimeError("too few contour points on one side of the substrate")
    t_hint = np.array([-n_hint[1], n_hint[0]])
    for _ in range(6):
        if fixed_dir is None:
            c, t = _tls_line(P[keep])
        else:
            t = np.asarray(fixed_dir, np.float64)
            c = P[keep].mean(0)
        nrm = np.array([-t[1], t[0]])
        res = (P - c) @ nrm
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        new = np.abs(res) < max(1.0, float(rc["outlier_sigma"]) * mad)
        if new.sum() < int(rc["min_points"]) or (new == keep).all():
            break
        keep = new
    if t @ t_hint < 0:
        t = -t
    return c, t, {"inlier_pts": int(keep.sum()), "side_pts": int(len(P)), "rms_px": float(np.sqrt(np.mean(res[keep] ** 2)))}


def _intersect(l1, l2):
    (p1, t1), (p2, t2) = l1, l2
    A = np.array([t1, -t2]).T
    return p1 + np.linalg.solve(A, p2 - p1)[0] * t1


def fit_rect_robust(pts: np.ndarray, rc) -> tuple[np.ndarray, dict]:
    """Rotated-rectangle fit of an outline (Nx2 contour points).

    1) cv2.minAreaRect gives an initial centre / axes / size (biased outward by clamps),
    2) each side is fitted separately and robustly (_fit_side) from the central `span_frac` of the side
       (corners are skipped so rounded / chamfered corners don't matter),
    3) the 4 lines are intersected -> 4 corners (a general quadrilateral; handles small rotation and
       slight non-orthogonality).
    """
    (cx, cy), (w0, h0), ang0 = cv2.minAreaRect(pts.astype(np.float32))
    box = cv2.boxPoints(((cx, cy), (w0, h0), ang0)).astype(np.float64)
    c0 = box.mean(0)
    ub = (box[1] - box[0]); a0 = np.linalg.norm(ub) / 2; ub /= 2 * a0
    vb = (box[3] - box[0]); b0 = np.linalg.norm(vb) / 2; vb /= 2 * b0
    du, dv = (pts - c0) @ ub, (pts - c0) @ vb
    span = float(rc["span_frac"])
    sides = {  # name: (mask of points, outward normal)
        "u+": ((np.abs(du) / a0 >= np.abs(dv) / b0) & (du > 0) & (np.abs(dv) < span * b0), ub),
        "v+": ((np.abs(dv) / b0 > np.abs(du) / a0) & (dv > 0) & (np.abs(du) < span * a0), vb),
        "u-": ((np.abs(du) / a0 >= np.abs(dv) / b0) & (du < 0) & (np.abs(dv) < span * b0), -ub),
        "v-": ((np.abs(dv) / b0 > np.abs(du) / a0) & (dv < 0) & (np.abs(du) < span * a0), -vb),
    }
    fixed = {}
    if not rc["allow_rotation"]:  # axis-aligned substrate: force every side parallel to an image axis
        snap = lambda vec: np.array([1.0, 0.0]) if abs(vec[0]) >= abs(vec[1]) else np.array([0.0, 1.0])
        fixed = {"u+": snap(vb), "u-": snap(vb), "v+": snap(ub), "v-": snap(ub)}
    lines, sinfo = {}, {}
    for k, (m, nh) in sides.items():
        c, t, inf = _fit_side(pts[m], nh, rc, fixed.get(k))
        lines[k] = (c, t)
        sinfo[k] = inf
    order = ["u+", "v+", "u-", "v-"]
    corners = np.array([_intersect(lines[order[i]], lines[order[(i + 1) % 4]]) for i in range(4)])
    area = cv2.contourArea(corners.astype(np.float32))
    if not (0.6 * w0 * h0 < area < 1.05 * w0 * h0):
        log.warning("rect line fit looks degenerate (area %.0f vs minAreaRect %.0f); using minAreaRect", area, w0 * h0)
        return box, {"source": "minAreaRect_fallback", "side_fits": sinfo}
    return corners, {"source": "contour_minAreaRect+robust_side_lines", "minAreaRect_size_wh": [float(w0), float(h0)],
                     "minAreaRect_angle_deg": float(ang0), "side_fits": sinfo}


def _shrink_quad(corners, px):
    """Move every side inward by px (exact offset of the 4 lines) and re-intersect."""
    if px == 0:
        return corners
    c = corners.mean(0)
    lines = []
    for i in range(4):
        p, q = corners[i], corners[(i + 1) % 4]
        t = (q - p) / np.linalg.norm(q - p)
        n = np.array([-t[1], t[0]])
        if n @ (c - p) < 0:
            n = -n
        lines.append((p + n * px, t))
    return np.array([_intersect(lines[(i - 1) % 4], lines[i]) for i in range(4)])


def quad_geometry(corners):
    """centre, orthonormal axes (unit vectors, u ~ horizontal side), half sizes, angle (deg, image coords,
    y down, wrapped to (-45, 45]) and side-wise stats."""
    c = corners.mean(0)
    e = [corners[(i + 1) % 4] - corners[i] for i in range(4)]
    angs = []
    for v in e:
        a = np.degrees(np.arctan2(v[1], v[0]))
        angs.append((a + 45) % 90 - 45)           # orientation mod 90 deg -> (-45, 45]
    ang = float(np.mean(angs))
    th = np.radians(ang)
    u = np.array([np.cos(th), np.sin(th)]); v = np.array([-np.sin(th), np.cos(th)])
    # sides closer to u-direction are "width" sides
    lens = [np.linalg.norm(x) for x in e]
    wl = [lens[i] for i in range(4) if abs((e[i] / lens[i]) @ u) > 0.5]
    hl = [lens[i] for i in range(4) if abs((e[i] / lens[i]) @ u) <= 0.5]
    return {"center": c, "u": u, "v": v, "hw": float(np.mean(wl)) / 2, "hh": float(np.mean(hl)) / 2,
            "angle": ang, "side_angles": angs}


def geo_to_info(g):
    return {"center_xy": [float(g["center"][0]), float(g["center"][1])], "size_wh": [2 * g["hw"], 2 * g["hh"]],
            "angle_deg": g["angle"], "side_angles_deg": [float(a) for a in g["side_angles"]]}


def corner_cut_mask(shape_hw, g, ctype, size):
    """1 = keep. Rounded corners (radius=size) or 45-deg chamfers (leg length=size) of the fitted rectangle."""
    h, w = shape_hw
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    px, py = xx - g["center"][0], yy - g["center"][1]
    u = np.abs(px * g["u"][0] + py * g["u"][1])
    v = np.abs(px * g["v"][0] + py * g["v"][1])
    a, b = g["hw"], g["hh"]
    if ctype == "round":
        qx, qy = u - (a - size), v - (b - size)
        sdf = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - size
        return (sdf <= 0).astype(np.uint8)
    return (((a - u) + (b - v)) >= size).astype(np.uint8)


# ----------------------------------------------------------------------------- edge band
def edge_valid_mask(wafer: np.ndarray, edge_px: float) -> np.ndarray:
    """1 where the pixel is at least `edge_px` away from the wafer boundary (and inside)."""
    if edge_px <= 0:
        return wafer.copy()
    dist = cv2.distanceTransform(wafer, cv2.DIST_L2, 5)
    return (dist > edge_px).astype(np.uint8)


# ----------------------------------------------------------------------------- fixtures
def polygon_mask(shape_hw, cfg_fix) -> np.ndarray:
    h, w = shape_hw
    m = np.zeros((h, w), np.uint8)
    units = cfg_fix.choice("polygon_units", ("px", "norm"))
    for poly in cfg_fix["polygons"] or []:
        p = np.asarray(poly, np.float64)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
            raise ValueError(f"fixture polygon needs >=3 [x,y] points: {poly}")
        if units == "norm":
            p = p * [w - 1, h - 1]
        cv2.fillPoly(m, [np.rint(p).astype(np.int32)], 1)
    return m


def auto_fixture_mask(img: np.ndarray, wafer: np.ndarray, cfg_auto) -> np.ndarray:
    """Detect clamps/fixtures as LARGE bright and/or dark blobs inside the wafer mask.

    img should be the illumination-corrected image so that a single ratio threshold works across
    the field. Thresholds are ratios to the wafer median. min_area_px keeps particles out;
    edge_zone_px (>0) restricts detection to a band near the boundary so that big real
    contamination in the wafer interior is NOT silently excluded.
    """
    h, w = img.shape
    inside = wafer > 0
    med = float(np.median(img[inside])) if inside.any() else float(np.median(img))
    cand = np.zeros((h, w), bool)
    if cfg_auto["bright_ratio"]:
        cand |= img > med * float(cfg_auto["bright_ratio"])
    if cfg_auto["dark_ratio"]:
        cand |= img < med * float(cfg_auto["dark_ratio"])
    cand &= inside
    zone = float(cfg_auto["edge_zone_px"])
    if zone > 0:
        cand &= cv2.distanceTransform(wafer, cv2.DIST_L2, 5) < zone
    b = cand.astype(np.uint8)
    k = int(cfg_auto["open_px"]) | 1
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    k = int(cfg_auto["close_px"]) | 1
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    out = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= int(cfg_auto["min_area_px"]):
            out[lab == i] = 1
    return out


def _dilate(m, px):
    if px <= 0:
        return m
    k = int(2 * round(px) + 1)
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def rect_sides(info: dict) -> dict:
    """name -> (point on side, inward unit normal) for the fitted quadrilateral.
    Names top/right/bottom/left are by position in the image (valid for small rotation)."""
    P = np.asarray(info["corners"], np.float64)
    c = P.mean(0)
    sides = []
    for i in range(4):
        p, q = P[i], P[(i + 1) % 4]
        t = (q - p) / np.linalg.norm(q - p)
        n = np.array([-t[1], t[0]])
        if n @ (c - p) < 0:
            n = -n
        sides.append(((p + q) / 2, n, p))
    names = {}
    for key, fn in (("top", lambda m: m[1]), ("bottom", lambda m: -m[1]), ("left", lambda m: m[0]), ("right", lambda m: -m[0])):
        i = min(range(4), key=lambda j: fn(sides[j][0]))
        names[key] = (sides[i][2], sides[i][1])
    return names


def build_roi(wafer: np.ndarray, wafer_info: dict, detect_img: np.ndarray, cfg, mask_png_loader=None):
    """Combine everything. Returns (components dict of uint8 masks, stats dict).

    components: wafer, edge_ok, corner_excl, fixture_poly, fixture_auto, fixture_ext, fixture (union), valid
    valid = substrate AND inside every edge band AND NOT corner-exclusion discs AND NOT fixture
    """
    h, w = wafer.shape
    is_rect = wafer_info.get("shape") == "rect"
    edge_px = float(cfg["edge_exclusion_px"])
    edge_ok = edge_valid_mask(wafer, edge_px)       # distance to nearest boundary point, inward from EVERY side
    per_side_applied = {}
    corner_excl = np.zeros((h, w), np.uint8)
    if is_rect:
        sides = rect_sides(wafer_info)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        for name, val in (cfg["edge_exclusion_per_side_px"] or {}).items():   # optional per-side override (>= global)
            if val is None or name not in sides:
                continue
            p, n = sides[name]
            d = (xx - p[0]) * n[0] + (yy - p[1]) * n[1]
            edge_ok &= (d > float(val)).astype(np.uint8)
            per_side_applied[name] = float(val)
        r = float(cfg["corner_exclusion_radius_px"])
        if r > 0:      # extra exclusion discs centred on the 4 nominal corners
            for cx, cy in wafer_info["corners"]:
                cv2.circle(corner_excl, (int(round(cx * 16)), int(round(cy * 16))), int(round(r * 16)), 1, -1, cv2.LINE_8, 4)
    fx = cfg["fixture"]
    margin = float(fx["margin_px"])
    poly = _dilate(polygon_mask((h, w), fx), margin)
    auto = np.zeros((h, w), np.uint8)
    auto_found = []
    if fx["auto"]["enabled"]:
        # searches the whole band `edge_zone_px` inside the boundary => along all four edges AND the corners
        raw_auto = auto_fixture_mask(detect_img, wafer, fx["auto"])
        auto = _dilate(raw_auto, margin)
        n, lab, st, cen = cv2.connectedComponentsWithStats(raw_auto, connectivity=8)
        for i in range(1, n):
            item = {"area_px": int(st[i, cv2.CC_STAT_AREA]), "centroid_xy": [float(cen[i][0]), float(cen[i][1])]}
            if is_rect:
                sd = {k: float(((cen[i] - p) @ nn)) for k, (p, nn) in rect_sides(wafer_info).items()}
                near = sorted(sd, key=sd.get)[:2]
                item["nearest_side"] = near[0] if sd[near[1]] > 1.5 * max(sd[near[0]], 1.0) + 60 else "corner:" + "+".join(sorted(near))
            auto_found.append(item)
    ext = np.zeros((h, w), np.uint8)
    if fx["mask_png"]:
        ext = mask_png_loader(fx["mask_png"], (h, w))
    fixture = ((poly | auto | ext) > 0).astype(np.uint8)
    valid = ((edge_ok > 0) & (fixture == 0) & (corner_excl == 0)).astype(np.uint8)

    area = lambda m: int(m.sum())
    a_img = h * w
    a_w = area(wafer)
    stats = {
        "image_px": a_img,
        "wafer_px": a_w,
        "wafer_frac_of_image": a_w / a_img,
        "edge_exclusion_px_width": edge_px,
        "edge_exclusion_per_side_applied": per_side_applied,
        "edge_band_px": a_w - area(edge_ok),
        "corner_exclusion_radius_px": float(cfg["corner_exclusion_radius_px"]),
        "corner_exclusion_px": area(corner_excl & wafer & edge_ok),   # extra pixels beyond the edge band
        "fixture_polygon_px": area(poly & wafer),
        "fixture_auto_px": area(auto & wafer),
        "fixture_auto_components": auto_found,
        "fixture_external_px": area(ext & wafer),
        "fixture_union_inside_wafer_px": area(fixture & wafer),
        "valid_px": area(valid),
        "valid_frac_of_wafer": area(valid) / max(a_w, 1),
        "valid_frac_of_image": area(valid) / a_img,
        "wafer": wafer_info,
    }
    comps = {"wafer": wafer, "edge_ok": edge_ok, "corner_excl": corner_excl, "fixture_poly": poly,
             "fixture_auto": auto, "fixture_ext": ext, "fixture": fixture, "valid": valid}
    return comps, stats
