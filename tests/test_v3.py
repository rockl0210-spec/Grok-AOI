"""Tests for the v3 additions (oil/water/ink patterns, stain separation, Side end fixture, encoding). Fully synthetic."""
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aoi_preprocess.config import load_config                                    # noqa: E402
from aoi_preprocess.patterns import detect, side_end_fixture, DEFAULTS as PD, FIXTURE_DEFAULTS as FD   # noqa: E402
from aoi_preprocess.pipeline_v3 import V3_BITS, V3_BIT_TABLE, run_v3              # noqa: E402
from aoi_preprocess.stain import separate, DEFAULTS as SD                         # noqa: E402

RNG = np.random.default_rng(3)
CFG = load_config(ROOT / "grok_aoi" / "configs" / "preprocess_v3.yaml")


def oil_like(h=800, w=900):
    """clean substrate + oriented feathery low-contrast streaks in the left part"""
    img = np.full((h, w), 170.0, np.float32) + RNG.normal(0, 1.5, (h, w)).astype(np.float32)
    n = RNG.normal(0, 1, (h, w)).astype(np.float32)
    k = np.zeros((1, 61), np.float32); k[0, :] = 1 / 61
    streak = cv2.GaussianBlur(cv2.filter2D(n, -1, k), (0, 0), 1.2)
    streak = streak / streak.std() * 7.0
    img[:, :450] += streak[:, :450]
    return np.clip(img, 0, 255)


def test_bit_table_v3():
    assert V3_BITS["ink"] == 32 and V3_BITS["oil"] == 16 and V3_BITS["water"] == 8 and V3_BITS["rough"] == 4
    assert {int(k) for k in V3_BIT_TABLE} >= {1, 2, 4, 8, 16, 32} if isinstance(V3_BIT_TABLE, dict) else len(V3_BIT_TABLE) >= 6


def test_oil_detector_fires_only_on_textured_area():
    img = oil_like()
    roi = np.ones(img.shape, bool)
    d = detect(img, roi, PD)
    assert d["oil"][:, :400].mean() > 0.5, d["oil"][:, :400].mean()
    assert d["oil"][:, 520:].mean() < 0.02, "clean half must not fire"


def test_fixture_side_end():
    h, w = 1000, 1400
    img = np.full((h, w), 20.0, np.float32)
    img[:, 200:330] = 110 + 25 * np.sin(np.arange(h))[:, None] * 0 + 20 * np.sin(np.arange(130) * 0.9)[None, :]   # ribbed ghost band
    img += RNG.normal(0, 1.0, (h, w)).astype(np.float32)
    cls = np.zeros((h, w), np.uint8); cls[100:900, 200:1100] = 1
    new, fx = side_end_fixture(img, cls, FD)
    left = [f for f in fx if f["side"] == "left"]
    assert left and left[0]["detected"] and 310 <= left[0]["step_x"] <= 345, fx
    assert (new[500, 250] == 3) and (new[500, 600] == 1)
    new2, fx2 = side_end_fixture(np.full((h, w), 20.0, np.float32), cls, FD)
    assert not any(f.get("detected") for f in fx2) and (new2 == 1).sum() == (cls == 1).sum()


def test_stain_separation_keeps_defects():
    img = oil_like()
    roi = np.ones(img.shape, bool)
    base = (img / 255.0).astype(np.float32)
    region = np.zeros(img.shape, bool); region[:, :450] = True
    pts = [(100, 200), (250, 400), (350, 600)]
    inj = img.copy()
    for x, y in pts:
        yy, xx = np.mgrid[-4:5, -4:5]
        inj[y - 4:y + 5, x - 4:x + 5] -= 45 * (np.hypot(xx, yy) <= 1.5)          # 3-px, 45-level spot (> c2)
    inj = np.clip(inj, 0, 255)
    s0 = separate(base, roi, region, SD); s1 = separate((inj / 255.0).astype(np.float32), roi, region, SD)
    diff = (s0["separated"] - s1["separated"]) * 255
    resp = [diff[y - 2:y + 3, x - 2:x + 3].max() for x, y in pts]
    assert min(resp) > 30, f"strong 3-px spots must survive the separation untouched: {resp}"
    e0 = (base[:, :450] - cv2.GaussianBlur(base, (0, 0), 3)[:, :450]).std()
    e1 = (s0["separated"][:, :450] - cv2.GaussianBlur(s0["separated"], (0, 0), 3)[:, :450]).std()
    assert e1 < 0.85 * e0, "separation must reduce fine stain texture energy"
    assert np.abs(s0["stain_levels"]).max() <= SD["amp_c2"] + 1e-3, "removed layer is capped at c2 levels"
    assert np.abs(s0["stain_levels"][:, 600:]).max() == 0, "nothing removed outside the contaminated region"


def test_run_v3_report_and_files():
    h, w = 1028, 1232
    imgs = []
    for _ in range(2):
        im = np.full((h, w), 190.0, np.float32); im[:, :100] = 15
        im += RNG.normal(0, 2.2, (h, w)).astype(np.float32)
        imgs.append(np.clip(im, 0, 255).astype(np.uint8))
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "db/Edge1/090"; d.mkdir(parents=True)
        for i, im in enumerate(imgs):
            cv2.imwrite(str(d / f"{i:02d}.png"), im)
        out = Path(tmp) / "out"
        reports, failed = run_v3(d, CFG, out, root=Path(tmp) / "db", frames=["00", "01"])
        assert not failed and len(reports) == 2
        od = out / "Edge1/090"
        for n in ("00", "01"):
            for suf in ("cleaned", "mask", "nuisance", "noise", "stain"):
                assert (od / f"{n}_{suf}.png").exists()
        assert cv2.imread(str(od / "01_stain.png"), cv2.IMREAD_UNCHANGED).dtype == np.uint8
        rj = json.loads((od / "report.json").read_text())
        for key in ("nuisance_bit_encoding", "registration", "abnormal_frames", "labels_summary", "stain_file"):
            assert key in rj, key
        f = rj["frames"]["01"]
        assert "overlap" in f and "labels" in f and "stain_separation" in f and "fixtures" in f
        nu = cv2.imread(str(od / "01_nuisance.png"), cv2.IMREAD_UNCHANGED)
        assert nu.max() < 64


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
