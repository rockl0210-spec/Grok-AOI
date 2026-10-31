"""Synthetic wafer image generator (for the demo / smoke test).

Model: image = reflectance(x,y) * illumination(x,y) + gaussian noise
  * round wafer (dark chuck outside), slight low-freq mottling, darker bevel ring at the edge
  * vignetting (radial) + linear tilt of the illumination
  * 8 small particle defects (bright/dark, sigma 1-3 px) inside the wafer + 2 'decoy' defects that
    must be masked out (one in the edge band, one on a clamp)
  * two clamps at the edge: A bright metal (listed as a polygon in config_panel.yaml),
    B dark (left to auto-detection)
(`make_synthetic_panel` below builds the square-panel variant, which is the default demo.)
Writes: synthetic_wafer_u16.png (16-bit gray, main demo input), synthetic_wafer_u8.png (8-bit gray),
        flat_u16.png / dark_u16.png (calibration frames), ground_truth.json
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .io_utils import write_png

W = H = 1600
CX, CY, R = 812.0, 792.0, 690.0
NOISE_SIGMA = 0.012          # fraction of full scale (~790 counts @16 bit)
DARK_LEVEL = 0.01


def _clamp_poly(angle_deg, r_in, r_out, half_w):
    a = np.deg2rad(angle_deg)
    u = np.array([np.cos(a), np.sin(a)])
    t = np.array([-u[1], u[0]])
    c = np.array([CX, CY])
    pts = [c + u * r_in + t * half_w, c + u * r_out + t * half_w, c + u * r_out - t * half_w, c + u * r_in - t * half_w]
    return np.array(pts)


def _polar(angle_deg, r):
    a = np.deg2rad(angle_deg)
    return float(CX + r * np.cos(a)), float(CY + r * np.sin(a))


def make_synthetic_wafer(out_dir, seed=7) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot(xx - CX, yy - CY)

    # reflectance
    refl = np.full((H, W), 0.08, np.float32)                       # chuck
    wafer = r <= R
    mott = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 60)
    mott = mott / mott.std() * 0.003
    refl[wafer] = 0.60 + mott[wafer]
    bevel = np.clip((r - (R - 10)) / 10.0, 0, 1)                   # last 10 px: ramps down to 0.8x
    refl = np.where(wafer, refl * (1 - 0.2 * bevel), refl)

    # clamps
    polyA = _clamp_poly(20, R - 55, R + 90, 75)     # bright metal
    polyB = _clamp_poly(200, R - 45, R + 90, 65)    # dark
    clampA = np.zeros((H, W), np.uint8); cv2.fillPoly(clampA, [np.rint(polyA).astype(np.int32)], 1)
    clampB = np.zeros((H, W), np.uint8); cv2.fillPoly(clampB, [np.rint(polyB).astype(np.int32)], 1)
    refl[clampA > 0] = 0.95
    refl[clampB > 0] = 0.15

    # defects (added to reflectance so that they see the same illumination as the surface)
    spec = [  # x, y, sigma, delta, expected_valid, note
        (500, 600, 1.5, +0.25, True, "bright"), (1100, 500, 2.0, +0.18, True, "bright"),
        (900, 1000, 1.0, +0.30, True, "bright small"), (700, 1200, 2.5, +0.12, True, "bright faint"),
        (600, 900, 1.5, -0.25, True, "dark"), (1000, 800, 2.0, -0.20, True, "dark"),
        (850, 450, 1.0, -0.30, True, "dark small"), (1150, 1050, 3.0, -0.12, True, "dark faint"),
    ]
    ex, ey = _polar(100, R - 6)    # decoy in edge band
    spec.append((ex, ey, 1.5, +0.25, False, "decoy: edge band"))
    cx_, cy_ = _polar(20, R - 25)  # decoy on clamp A (inside the wafer circle)
    spec.append((cx_, cy_, 1.5, -0.30, False, "decoy: on clamp A"))
    gt = []
    for x, y, s, d, ok, note in spec:
        rad = int(np.ceil(5 * s))
        x0, x1 = int(max(0, x - rad)), int(min(W, x + rad + 1))
        y0, y1 = int(max(0, y - rad)), int(min(H, y + rad + 1))
        g = np.exp(-((xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2) / (2 * s * s))
        refl[y0:y1, x0:x1] += d * g.astype(np.float32)
        gt.append({"x": float(x), "y": float(y), "sigma_px": s, "delta_reflectance": d,
                   "expected_in_valid_roi": ok, "note": note})

    # illumination: radial vignetting + tilt
    rho = np.hypot(xx - W / 2, yy - H / 2) / np.hypot(W / 2, H / 2)
    illum = (1 - 0.45 * rho ** 2) * (1 + 0.12 * (xx / W - 0.5) * 2)
    illum = illum.astype(np.float32)

    def to_u16(a):
        return np.clip(np.rint(a * 65535), 0, 65535).astype(np.uint16)

    clean = refl * illum + DARK_LEVEL
    img = clean + rng.normal(0, NOISE_SIGMA, (H, W)).astype(np.float32)
    flat = 0.8 * illum + DARK_LEVEL + rng.normal(0, 0.002, (H, W)).astype(np.float32)   # uniform white target
    dark = np.full((H, W), DARK_LEVEL, np.float32) + rng.normal(0, 0.002, (H, W)).astype(np.float32)

    p16 = out / "synthetic_wafer_u16.png"
    write_png(p16, to_u16(img))
    write_png(out / "synthetic_wafer_u8.png", np.clip(np.rint(img * 255), 0, 255).astype(np.uint8))
    write_png(out / "flat_u16.png", to_u16(flat))
    write_png(out / "dark_u16.png", to_u16(dark))
    meta = {"kind": "wafer", "size_wh": [W, H], "wafer_center_xy": [CX, CY], "wafer_radius_px": R, "noise_sigma_frac": NOISE_SIGMA,
            "clamp_A_polygon_bright": polyA.round(0).astype(int).tolist(),
            "clamp_B_polygon_dark_(auto_detect)": polyB.round(0).astype(int).tolist(),
            "defects": gt}
    (out / "ground_truth.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"input_16bit": str(p16), "ground_truth": str(out / "ground_truth.json"),
            "clamp_A_polygon": meta["clamp_A_polygon_bright"]}


# ============================================================================ square / rectangular panel
PANEL = dict(cx=805.0, cy=795.0, A=590.0, B=520.0, angle_deg=1.4, corner_r=30.0)   # half sizes A (u), B (v)


def _to_img(u, v, th):
    c, s = np.cos(th), np.sin(th)
    return PANEL["cx"] + u * c - v * s, PANEL["cy"] + u * s + v * c


def make_synthetic_panel(out_dir, seed=11) -> dict:
    """Slightly rotated rectangular panel (rounded corners), vignetting, gaussian noise, 8 real particles
    (3 of them close to the edge but outside the 20 px edge band), 4 decoys (2 in the edge band / corner,
    1 on each clamp), 2 edge clamps (A bright on the top edge, B dark on the left edge).
    Local panel frame: u along the 'width' side, v along the 'height' side; image = rotation by +angle (y down).
    Files: synthetic_panel_u16.png / _u8.png, flat_u16.png, dark_u16.png, ground_truth.json"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    A, B, Rc = PANEL["A"], PANEL["B"], PANEL["corner_r"]
    th = np.deg2rad(PANEL["angle_deg"])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    px, py = xx - PANEL["cx"], yy - PANEL["cy"]
    u = px * np.cos(th) + py * np.sin(th)          # inverse rotation -> local coords
    v = -px * np.sin(th) + py * np.cos(th)
    qx, qy = np.abs(u) - (A - Rc), np.abs(v) - (B - Rc)
    sdf = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - Rc   # <0 inside
    inside = sdf <= 0
    depth = -sdf                                    # distance to the edge (inside)

    refl = np.full((H, W), 0.08, np.float32)
    mott = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 60)
    mott = mott / mott.std() * 0.003
    bevel = np.clip(1 - depth / 8.0, 0, 1)          # last 8 px ramp to 0.8x
    refl[inside] = ((0.60 + mott) * (1 - 0.2 * bevel))[inside]

    def local_poly(u0, u1, v0, v1):
        return np.array([_to_img(a, b, th) for a, b in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]).T.T
    # clamp A (bright, top edge): outside by 70, inside by 50.  clamp B (dark, left edge): outside 70, inside 45
    polyA = local_poly(-295, -145, -B - 70, -B + 50)
    polyB = local_poly(-A - 70, -A + 45, 50, 190)
    clampA = np.zeros((H, W), np.uint8); cv2.fillPoly(clampA, [np.rint(polyA).astype(np.int32)], 1)
    clampB = np.zeros((H, W), np.uint8); cv2.fillPoly(clampB, [np.rint(polyB).astype(np.int32)], 1)
    refl[clampA > 0] = 0.95
    refl[clampB > 0] = 0.15

    # (u, v, sigma, delta, expected_valid, note)   edge distances: right 28, bottom 36, corner 70/70
    spec = [
        (-100, -150, 1.5, +0.25, True, "bright"), (250, 80, 2.0, -0.20, True, "dark"),
        (50, 250, 1.0, +0.30, True, "bright small"), (-350, 200, 3.0, -0.12, True, "dark faint"),
        (400, -250, 1.0, -0.30, True, "dark small"),
        (A - 28, -200, 1.5, +0.25, True, "bright, 28px from right edge"),
        (100, B - 36, 1.5, -0.25, True, "dark, 36px from bottom edge"),
        (-A + 70, -B + 70, 2.0, +0.15, True, "bright faint, 70px from top-left corner"),
        (-100, -B + 8, 1.5, +0.25, False, "decoy: edge band (top)"),
        (A - 12, B - 12, 1.5, -0.30, False, "decoy: corner / edge band"),
        (-220, -B + 25, 1.5, -0.30, False, "decoy: on clamp A"),
        (-A + 22, 120, 1.5, +0.30, False, "decoy: on clamp B"),
    ]
    gt = []
    for uu, vv, s_, d, ok, note in spec:
        x, y = _to_img(uu, vv, th)
        rad = int(np.ceil(5 * s_))
        x0, x1 = int(max(0, x - rad)), int(min(W, x + rad + 1))
        y0, y1 = int(max(0, y - rad)), int(min(H, y + rad + 1))
        g = np.exp(-((xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2) / (2 * s_ * s_))
        refl[y0:y1, x0:x1] += d * g.astype(np.float32)
        gt.append({"x": float(x), "y": float(y), "sigma_px": s_, "delta_reflectance": d,
                   "expected_in_valid_roi": ok, "note": note})

    rho = np.hypot(xx - W / 2, yy - H / 2) / np.hypot(W / 2, H / 2)
    illum = ((1 - 0.45 * rho ** 2) * (1 + 0.12 * (xx / W - 0.5) * 2)).astype(np.float32)
    u16 = lambda a: np.clip(np.rint(a * 65535), 0, 65535).astype(np.uint16)
    img = refl * illum + DARK_LEVEL + rng.normal(0, NOISE_SIGMA, (H, W)).astype(np.float32)
    flat = 0.8 * illum + DARK_LEVEL + rng.normal(0, 0.002, (H, W)).astype(np.float32)
    dark = np.full((H, W), DARK_LEVEL, np.float32) + rng.normal(0, 0.002, (H, W)).astype(np.float32)
    write_png(out / "synthetic_panel_u16.png", u16(img))
    write_png(out / "synthetic_panel_u8.png", np.clip(np.rint(img * 255), 0, 255).astype(np.uint8))
    write_png(out / "flat_u16.png", u16(flat))
    write_png(out / "dark_u16.png", u16(dark))
    corners = [list(map(float, _to_img(a, b, th))) for a, b in ((A, -B), (A, B), (-A, B), (-A, -B))]
    meta = {"kind": "panel", "size_wh_image": [W, H], "panel_center_xy": [PANEL["cx"], PANEL["cy"]],
            "panel_size_wh": [2 * A, 2 * B], "panel_angle_deg": PANEL["angle_deg"],
            "corner_radius_px": Rc, "nominal_corners_xy": corners, "noise_sigma_frac": NOISE_SIGMA,
            "clamp_A_polygon_bright": np.rint(polyA).astype(int).tolist(),
            "clamp_B_polygon_dark_(auto_detect)": np.rint(polyB).astype(int).tolist(),
            "defects": gt}
    (out / "ground_truth.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"input_16bit": str(out / "synthetic_panel_u16.png"), "ground_truth": str(out / "ground_truth.json"),
            "clamp_A_polygon": meta["clamp_A_polygon_bright"]}


def make_synthetic(out_dir, seed=None, kind="panel") -> dict:
    if kind == "wafer":
        return make_synthetic_wafer(out_dir, 7 if seed is None else seed)
    return make_synthetic_panel(out_dir, 11 if seed is None else seed)
