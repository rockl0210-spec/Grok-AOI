"""Inter-frame registration (Edge sequence): shift of frame k+1 relative to frame k, in full-res px.

Two independent estimates are reported for every consecutive pair:
  * phase_correlation : high-passed 1/4-scale frames, Hann window, sub-pixel peak; confidence = peak response (0..1).
                        Only trusted when response >= min_response. NOTE a stationary illumination / lens-dust pattern
                        also correlates (it gives dx=dy=0 with a high response) - so the high-pass removes the large scale
                        pattern and the result is listed together with the second estimate.
  * edge_line         : offset of the fitted substrate edge line along its normal (only the component perpendicular to the
                        edge is observable); confidence = min of the two line inlier fractions (0 when a frame is a corner frame).
The chosen (dx, dy, method, confidence) is phase_correlation if it is trusted, else the edge-line normal shift, else null.
Sign: positive dx = image content of frame k+1 sits dx px to the right of the same content in frame k.
"""
from __future__ import annotations

import cv2
import numpy as np


def prep(raw: np.ndarray, c) -> np.ndarray:
    s = float(c["work_scale"])
    sm = cv2.resize(raw.astype(np.float32), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    hp = sm - cv2.GaussianBlur(sm, (0, 0), float(c["hp_sigma_px"]) * s)
    return hp.astype(np.float32)


def line_normal_offset(line):
    """Signed offset of the edge line along its unit normal oriented to +x (or +y if the line is horizontal)."""
    if not line:
        return None
    p, d = np.array(line["point"], float), np.array(line["dir"], float)
    n = np.array([-d[1], d[0]])
    n /= np.linalg.norm(n) + 1e-12
    if abs(n[0]) < abs(n[1]):
        n = n if n[1] > 0 else -n
    else:
        n = n if n[0] > 0 else -n
    return float(p @ n), n


def register_pair(prev: dict, cur: dict, c) -> dict:
    """prev/cur = {'hp': prep(raw), 'line': edge line dict or None, 'inlier': float, 'abnormal': bool, 'name': str}"""
    s = float(c["work_scale"])
    win = cv2.createHanningWindow((prev["hp"].shape[1], prev["hp"].shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(prev["hp"], cur["hp"], win)
    pc = {"dx": float(dx / s), "dy": float(dy / s), "response": float(resp)}
    out = {"from": prev["name"], "to": cur["name"], "phase_correlation": pc}
    a, b = line_normal_offset(prev["line"]), line_normal_offset(cur["line"])
    el = None
    max_n = float(c["max_line_shift_px"]) if "max_line_shift_px" in c else 300.0
    if a and b and not prev["abnormal"] and not cur["abnormal"] and abs(b[0] - a[0]) <= max_n:
        n = a[1]
        el = {"normal_shift_px": float(b[0] - a[0]), "normal": [float(n[0]), float(n[1])],
              "dx": float((b[0] - a[0]) * n[0]), "dy": float((b[0] - a[0]) * n[1]),
              "confidence": float(min(prev["inlier"], cur["inlier"]))}
    out["edge_line"] = el
    el_ok = el is not None and el["confidence"] >= 0.5
    tol = float(c["consistency_tol_px"]) if "consistency_tol_px" in c else 20.0
    if resp >= float(c["min_response"]):
        if el_ok:
            n = el["normal"]
            pn = pc["dx"] * n[0] + pc["dy"] * n[1]
            if abs(pn - el["normal_shift_px"]) > tol:      # the edge did not move but phase correlation says it did -> spurious peak
                out["phase_correlation_rejected"] = f"normal component {pn:.1f}px disagrees with edge-line shift {el['normal_shift_px']:.1f}px"
            else:
                out.update({"dx": pc["dx"], "dy": pc["dy"], "method": "phase_correlation", "confidence": pc["response"], "trusted": True,
                            "note": "consistent with the edge-line shift"})
                return out
        else:
            big = float(np.hypot(pc["dx"], pc["dy"])) > float(c.get("max_line_shift_px", 300))
            out.update({"dx": pc["dx"], "dy": pc["dy"], "method": "phase_correlation", "confidence": pc["response"],
                        "trusted": bool(resp >= 0.5 and not big),
                        "note": "not cross-checked (no usable edge line: corner frame / broken line)" + ("; |shift| > max_line_shift_px, implausible -> untrusted" if big else "")})
            return out
    if el is not None:
        out.update({"dx": el["dx"], "dy": el["dy"], "method": "edge_line_normal_only", "confidence": el["confidence"],
                    "trusted": bool(el_ok),
                    "note": "only the shift perpendicular to the edge is observable (the other component is reported as 0 = unknown)"})
    else:
        out.update({"dx": None, "dy": None, "method": "none", "confidence": 0.0, "trusted": False})
    return out
