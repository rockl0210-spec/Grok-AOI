#!/usr/bin/env python3
"""AOI defect-candidate detector (classical CV, single-channel gray). See README.md."""
import os, sys, json, glob, math, time, csv
import numpy as np, cv2
from scipy import ndimage as ndi
from multiprocessing import Pool

PRE = os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/preprocess'
DB = os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_db'
OUT = os.environ.get('AOI_OUT', os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/detect')
FLIP = bool(os.environ.get('AOI_FLIP'))   # null-model test: swap the polarity of both modes
MODES = {'Edge1': 'edge', 'Edge2': 'edge', 'Side': 'side'}
ANGLES = {'Edge1': ['000', '090'], 'Edge2': ['000', '090'], 'Side': ['000', '090', '180', '270']}
ANOM = {'edge': {'00', '45'}, 'side': {'00', '15'}}
BITN = {1: 'specular_glare', 2: 'uneven_illumination', 4: 'rough_texture', 8: 'suspected_water_mark', 16: 'suspected_oil_film'}

# ---------------- parameters (per mode, per nuisance zone) ----------------
# zone precedence (highest first): rough > oil > water > specular > uneven > normal
ZONES = ['rough', 'oil', 'water', 'specular', 'uneven', 'normal']
ZBIT = {'rough': 4, 'oil': 16, 'water': 8, 'specular': 1, 'uneven': 2}
PARAMS = {
 'edge': {  # bright field, dark defects.  v3: lower seeds (k_hi 4.5-6), cnr/snr relaxed, rough floors 25/10 -> 18/9, rough min_area 8 -> 4
   'polarity': 'dark', 'sigma_floor': 0.6,
   'zone': {  # k_hi seed, k_lo grow, abs_hi / abs_lo min contrast (gray levels), min_area px, cnr_min per blob
     'normal':   dict(k_hi=4.5, k_lo=2.2, abs_hi=8.0,  abs_lo=4.0, min_area=3, cnr_min=4.5, snr_min=6.0),
     'uneven':   dict(k_hi=5.0, k_lo=2.5, abs_hi=8.0,  abs_lo=4.0, min_area=3, cnr_min=4.5, snr_min=6.0),
     'specular': dict(k_hi=6.0, k_lo=3.0, abs_hi=10.0, abs_lo=5.0, min_area=3, cnr_min=6.0, snr_min=7.0),
     'water':    dict(k_hi=5.5, k_lo=2.8, abs_hi=8.0,  abs_lo=4.0, min_area=3, cnr_min=5.0, snr_min=6.0),
     'oil':      dict(k_hi=5.5, k_lo=2.8, abs_hi=8.0,  abs_lo=4.0, min_area=3, cnr_min=5.0, snr_min=6.0),
     'rough':    dict(k_hi=4.5, k_lo=2.2, abs_hi=9.0, abs_lo=5.0, min_area=4, cnr_min=4.5, snr_min=7.0)}},
 'side': {  # dark field, bright defects
   'polarity': 'bright', 'sigma_floor': 0.4,
   'zone': {
     'normal':   dict(k_hi=4.5, k_lo=2.2, abs_hi=8.0,  abs_lo=3.5, min_area=3, cnr_min=4.5, snr_min=6.0),
     'uneven':   dict(k_hi=4.5, k_lo=2.2, abs_hi=8.0,  abs_lo=3.5, min_area=3, cnr_min=4.5, snr_min=6.0),  # weak flag: same as normal
     'specular': dict(k_hi=7.0, k_lo=3.5, abs_hi=12.0, abs_lo=6.0, min_area=3, cnr_min=7.0, snr_min=7.0),
     'water':    dict(k_hi=5.5, k_lo=2.8, abs_hi=8.0,  abs_lo=3.5, min_area=3, cnr_min=5.0, snr_min=6.0),
     'oil':      dict(k_hi=5.5, k_lo=2.8, abs_hi=8.0,  abs_lo=3.5, min_area=3, cnr_min=5.0, snr_min=6.0),
     'rough':    dict(k_hi=6.0, k_lo=3.0, abs_hi=10.0, abs_lo=5.0, min_area=4, cnr_min=5.0, snr_min=6.0)}},
}
BG = dict(block=4, sigma_ds_px=12, iters=3, outlier_k=3.0, outlier_abs=2.0)
CLS = dict(scratch_min_len=12.0, scratch_min_lw=4.0, scratch_max_width=8.0, scratch_min_aspect=3.0,
           long_scratch_min_len=40.0, long_scratch_min_lw=8.0, long_scratch_max_width=20.0,
           stain_min_area=150, stain_irregular_area=40, stain_irregular_solidity=0.65, particle_max_area=150)
# compact-dot path for rough zones (relative to texture-sigma, not the k_hi-8 streak-safe path)
DOT = {'edge': dict(k_hi=3.5, k_lo=2.0, abs_hi=9.0, abs_lo=5.0, min_mean_contrast=7.0, strong_contrast=40.0, strong_min_area=30, tighten=0.4, min_area=4, cnr_min=2.5, snr_min=6.0,
                    min_solidity=0.72, max_aspect=3.0, max_lw=6.0, max_area=6000),
       'side': dict(k_hi=4.5, k_lo=2.5, abs_hi=12.0, abs_lo=6.0, min_mean_contrast=8.0, strong_contrast=20.0, strong_min_area=30, tighten=0.4, min_area=4, cnr_min=3.5, snr_min=6.0,
                    min_solidity=0.72, max_aspect=3.0, max_lw=6.0, max_area=6000)}
# v3 path C: multi-scale blob (min-eigenvalue-of-Hessian) detector on a SMALL robust background (median 31px), all zones
BLOB = dict(med_k=31, scales=(1.0, 1.6, 2.5, 4.0), k_sig=4.5, nms=7, frac=0.5, min_area=3, max_area=2500,
            min_solidity=0.6, max_aspect=3.5, max_lw=7.0, cnr_min=3.0, snr_min=5.0,
            # seed floor on smoothed contrast (gray levels) and min mean contrast of the grown blob, per zone
            edge=dict(normal=(5.0, 6.0), uneven=(5.0, 6.0), specular=(8.0, 9.0), water=(5.0, 6.0), oil=(5.0, 6.0), rough=(6.0, 7.0)),
            side=dict(normal=(7.0, 8.0), uneven=(7.0, 8.0), specular=(9.0, 10.0), water=(7.0, 8.0), oil=(7.0, 8.0), rough=(8.0, 10.0)))
SENS = 'v3.1'
BLOB['roi_ext'] = 16; BLOB['roi_keep_frac'] = 0.3
BLOB.update(inpaint_k=4.0, inpaint_abs={'edge': 6.0, 'side': 4.0}, side_low_snr=24.0)
# v3.1 straddle keep rule: a component with < roi_keep_frac inside the ROI is still kept when it touches the ROI (>= min_in px) and is a high-contrast dot
STR = dict(min_frac=0.05, peak=dict(edge=40.0, side=25.0), min_mean=dict(edge=15.0, side=10.0), cnr=12.0, a_strong_contrast=25.0)
# v3.1 consolidation: regrow ROI-edge dots to their full connected component and merge fragments; watershed split of touching dots
REG = dict(near=20.0, win=20, fracs=(0.3, 0.4, 0.5, 0.6), floor=6.0, max_ratio=10.0, max_area=3000, min_peak=20.0)
SPLIT = dict(min_area=250, max_area=6000, h_frac=0.2, h_min=6.0, min_piece=12, smooth=1.5, min_rel_peak=0.45)
# v3.1 band refinement of big faint band-like A components (illumination / specular glare bands)
CORE = dict(min_area=300, max_area=6000, min_peak=40.0, frac=0.35, max_core_ratio=0.5, min_piece=12)   # trim a bright/dark core out of its low-contrast haze
BAND = dict(min_area=3000, min_asp=12.0, max_mean=45.0, factors=(1.5, 2.0, 3.0, 4.0), max_keep=1500, line_len=1000.0, line_asp=40.0, spec_min_area=1500, spec_asp=10.0)
DEBUG = False
if FLIP:
    PARAMS['edge']['polarity'], PARAMS['side']['polarity'] = 'bright', 'dark'
PERSIST = dict(window=3, tol_n=3.0, tol_t=6.0, area_ratio=(0.2, 5.0), vote_normal_band=10.0, vote_min=3, vote_ratio=2.0, tan_min=1650.0, tan_max=2060.0)


def zone_maps(nuis, mode):
    """per-pixel zone index (0..5 in ZONES order) from nuisance bits (precedence)."""
    zi = np.full(nuis.shape, len(ZONES) - 1, np.uint8)
    for i in range(len(ZONES) - 2, -1, -1):  # lowest precedence first, overwritten by higher
        zi[(nuis & ZBIT[ZONES[i]]) > 0] = i
    return zi


def estimate_bg(c, roi, sig):
    """robust local background: block-median downsample + iterative outlier-excluding normalized gaussian."""
    b = BG['block']; H, W = c.shape
    H4, W4 = H // b * b, W // b * b
    cd = np.median(c[:H4, :W4].reshape(H4 // b, b, W4 // b, b).transpose(0, 2, 1, 3).reshape(H4 // b, W4 // b, b * b), axis=2)
    rd = roi[:H4, :W4].reshape(H4 // b, b, W4 // b, b).mean(axis=(1, 3)) > 0.5
    sd = sig[:H4, :W4].reshape(H4 // b, b, W4 // b, b).mean(axis=(1, 3))
    gm = np.median(cd[rd]) if rd.any() else float(np.median(c))
    bg = np.full(cd.shape, gm, np.float32)
    w = rd.astype(np.float32)
    for _ in range(BG['iters']):
        thr = np.maximum(BG['outlier_k'] * sd, BG['outlier_abs'])
        w = (rd & (np.abs(cd - bg) < thr)).astype(np.float32)
        num = cv2.GaussianBlur(cd * w, (0, 0), BG['sigma_ds_px'])
        den = cv2.GaussianBlur(w, (0, 0), BG['sigma_ds_px'])
        bg = np.where(den > 1e-3, num / np.maximum(den, 1e-3), gm).astype(np.float32)
    bgf = cv2.resize(bg, (W4, H4), interpolation=cv2.INTER_LINEAR)
    out = np.full((H, W), gm, np.float32); out[:H4, :W4] = bgf
    if H4 < H: out[H4:] = out[H4 - 1]
    if W4 < W: out[:, W4:] = out[:, W4 - 1:W4]
    return out


def residual_sigma(d, roi, block=32):
    """local robust sigma of the residual d (= defect-positive bg-subtracted cleaned image), estimated from the NON-defect
    (lower) tail only: sigma = median(d) - p16(d) per 32px block. Big/strong defects live in the upper tail so they cannot
    inflate it. Captures residual texture the preprocess noise map under-estimates (e.g. Side machining marks)."""
    H, W = d.shape
    H4, W4 = H // block * block, W // block * block
    a = d[:H4, :W4].reshape(H4 // block, block, W4 // block, block).transpose(0, 2, 1, 3).reshape(H4 // block, W4 // block, block * block)
    rr = roi[:H4, :W4].reshape(H4 // block, block, W4 // block, block).mean(axis=(1, 3))
    p = np.percentile(a, [16, 50], axis=2)
    sg = np.maximum(p[1] - p[0], 0)
    gmed = np.median(sg[rr > 0.25]) if (rr > 0.25).any() else 0.0
    sg = np.where(rr > 0.25, sg, gmed).astype(np.float32)
    sg = cv2.GaussianBlur(sg, (0, 0), 1.0)
    out = cv2.resize(sg, (W4, H4), interpolation=cv2.INTER_LINEAR)
    full = np.full((H, W), gmed, np.float32); full[:H4, :W4] = out
    return full



def signed(a, b, pol):
    """defect-positive residual: dark -> a - b (bg - c) ; bright -> b - a (c - bg). a=bg, b=c"""
    return (a - b) if pol == 'dark' else (b - a)


def two_sided_median_bg(c8, m, idx_in, idx_out, sig, pol, iabs):
    """v3.1 small-scale background for BOTH sides of the ROI boundary (31px median, 2 passes).
    inside: outside pixels are replaced by the nearest inside pixel; outside: inside pixels are replaced by the nearest OUTSIDE
    pixel, so the ROI step edge cannot create a false band. pass 2 in-paints pixels flagged defect-like in pass 1 so a dot touching the
    boundary / a big dot does not bias its own background."""
    cf = c8.astype(np.float32)
    def one(rep, sg):
        b0 = cv2.GaussianBlur(cv2.medianBlur(rep, BLOB['med_k']).astype(np.float32), (0, 0), 4)
        d0 = signed(b0, rep.astype(np.float32), pol)
        mk = d0 > np.maximum(BLOB['inpaint_k'] * sg, iabs)
        if not mk.any(): return b0
        mk = cv2.dilate(mk.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        rep2 = rep.copy(); rep2[mk] = np.clip(b0[mk] + 0.5, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(cv2.medianBlur(rep2, BLOB['med_k']).astype(np.float32), (0, 0), 4)
    rin = c8[idx_in[0], idx_in[1]]
    b_in = one(rin, sig)
    if idx_out is None: return b_in, b_in
    rout = c8[idx_out[0], idx_out[1]]
    b_out = one(rout, sig)
    return b_in, b_out


def blob_maps(d, m_ext):
    """path-C maps from the two-sided residual d (defect positive, already 0 outside m_ext): for each scale s the smoothed contrast L_s
    and the scale-normalised min-eigenvalue of the Hessian B_s (>0 only for compact blobs; ~0 for streaks/edges)."""
    kxx = np.array([[1, -2, 1]], np.float32); kyy = kxx.T.copy(); kxy = 0.25 * np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], np.float32)
    Bs, Ls = [], []
    for s in BLOB['scales']:
        g = cv2.GaussianBlur(d, (0, 0), s)
        a = -cv2.filter2D(g, -1, kxx) * s * s; cc = -cv2.filter2D(g, -1, kyy) * s * s; b = -cv2.filter2D(g, -1, kxy) * s * s
        Bs.append((a + cc) / 2 - np.sqrt(((a - cc) / 2) ** 2 + b * b)); Ls.append(g)
    return np.array(Bs), np.array(Ls)



def shape_feats(patch):
    """patch: bool mask (filled). returns dict of shape features."""
    cnts, _ = cv2.findContours(patch.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea) if cnts else None
    area = int(patch.sum())
    ys, xs = np.nonzero(patch)
    if area >= 2:
        cov = np.cov(np.vstack([xs, ys]).astype(np.float64)) + np.eye(2) / 12.0  # pixel-size regulariser
        ev, evec = np.linalg.eigh(cov)
        major = 4 * math.sqrt(max(ev[1], 1e-9)); minor = 4 * math.sqrt(max(ev[0], 1e-9))
        v = evec[:, 1]; ori = math.degrees(math.atan2(v[1], v[0])) % 180.0
        elong = major / max(minor, 1e-6)
    else:
        major = minor = 1.0; ori = 0.0; elong = 1.0
    if cnt is not None and len(cnt) >= 3:
        hull = cv2.convexHull(cnt); ha = cv2.contourArea(hull)
        solidity = min(1.0, area / ha) if ha > 0 else 1.0
        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        long_, short_ = max(rw, rh), min(rw, rh)
        pts = hull[:, 0, :].astype(np.float32)
        feret = float(np.max(np.linalg.norm(pts[:, None] - pts[None], axis=2))) + 1.0 if len(pts) < 400 else float(long_)
    else:
        solidity = 1.0; long_ = short_ = float(math.sqrt(area)); feret = float(math.sqrt(area))
    aspect = (long_ + 1.0) / (short_ + 1.0)  # +1 px: pixel extent
    width = area / max(feret, 1.0)  # mean thickness
    return dict(aspect_ratio=round(aspect, 3), elongation=round(elong, 3), solidity=round(solidity, 3),
                orientation_deg=round(ori, 1), major_len=round(major, 2), feret=round(feret, 2), thickness=round(width, 2))


def classify(a, f):
    lw = f['feret'] / max(f['thickness'], 0.5)
    if (f['feret'] >= CLS['scratch_min_len'] and f['thickness'] <= CLS['scratch_max_width'] and
            lw >= CLS['scratch_min_lw'] and f['aspect_ratio'] >= CLS['scratch_min_aspect']):
        return 'scratch'
    if f['feret'] >= CLS['long_scratch_min_len'] and lw >= CLS['long_scratch_min_lw'] and f['thickness'] <= CLS['long_scratch_max_width']:
        return 'scratch'  # long, thicker / curved scratch
    if a >= CLS['stain_min_area']:
        return 'stain'
    if a >= CLS['stain_irregular_area'] and f['solidity'] < CLS['stain_irregular_solidity']:
        return 'stain'
    return 'particle'


def confidence(cls, a, f, contrast, cnr, kz, zone, dist):
    s_cnr = 1 - math.exp(-0.5 * cnr / kz)
    s_con = 1 - math.exp(-abs(contrast) / 8.0)
    s_size = float(np.clip(0.4 + 0.6 * math.log10(max(a, 3) / 3.0) / math.log10(10), 0.4, 1.0))
    if cls == 'particle':
        s_shape = 0.5 * min(1.0, f['solidity'] / 0.85) + 0.5 * min(1.0, 2.0 / max(f['aspect_ratio'], 1.0) + 0.0)
        s_shape = float(np.clip(0.3 + 0.7 * s_shape, 0.3, 1))
    elif cls == 'scratch':
        lw = f['feret'] / max(f['thickness'], 0.5)
        s_shape = float(np.clip(0.4 + 0.6 * min(1.0, (lw - 3) / 10.0) * min(1.0, f['feret'] / 25.0) ** 0.5, 0.4, 1))
    else:
        s_shape = float(np.clip(0.5 + 0.5 * min(1.0, a / 400.0), 0.5, 1))
    s_zone = {'normal': 1.0, 'uneven': 0.97, 'specular': 0.8, 'water': 0.9, 'oil': 0.9, 'rough': 0.7}[zone]
    s_edge = 0.85 if dist < 15 else 1.0
    conf = (s_cnr ** 0.35) * (s_con ** 0.25) * (s_size ** 0.15) * (s_shape ** 0.15) * (s_zone ** 0.1) * (s_edge ** 0.1)
    return round(float(np.clip(conf, 0, 1)), 3)


def detect_frame(args):
    src, ang, fr = args
    t0 = time.time()
    mode = MODES[src]; P = PARAMS[mode]; pol = P['polarity']
    pdir = f'{PRE}/{src}/{ang}'
    cpath = f'{pdir}/{fr}_cleaned.png'; opath = f'{DB}/{src}/{ang}/{fr}.png'
    c = cv2.imread(cpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
    o = cv2.imread(opath, cv2.IMREAD_UNCHANGED).astype(np.float32)
    m = cv2.imread(f'{pdir}/{fr}_mask.png', cv2.IMREAD_UNCHANGED) > 0
    nu0 = cv2.imread(f'{pdir}/{fr}_nuisance.png', cv2.IMREAD_UNCHANGED)
    ns = cv2.imread(f'{pdir}/{fr}_noise.png', cv2.IMREAD_UNCHANGED).astype(np.float32) / 100.0
    assert c.shape == o.shape == m.shape == nu0.shape == ns.shape, 'shape mismatch'
    H, W = c.shape
    sig0 = np.maximum(ns, P['sigma_floor'])
    bg = estimate_bg(c, m, sig0)
    sig_r = residual_sigma(signed(bg, c, pol), m)
    sig = np.maximum(sig0, np.minimum(sig_r, 2.5 * sig0))  # effective sigma = max(noise map, floor, residual-texture sigma capped at 2.5x noise-based sigma so dense texture cannot hide strong defects)
    # raw-image noise (frame level, hp MAD in non-rough ROI) for blob-level SNR
    hp = o - cv2.GaussianBlur(o, (0, 0), 1.5)
    nr = m & ((nu0 & 4) == 0)
    raw_sig = float(1.4826 * np.median(np.abs(hp[nr] - np.median(hp[nr])))) if nr.sum() > 1000 else 1.0
    raw_sig = max(raw_sig, 0.3)
    # ---- v3.1 two-sided residual: detection runs on the ROI dilated by roi_ext px. Inside the ROI the background is the robust
    # large-scale estimate; OUTSIDE the ROI it is a small-scale median of the real outside pixels (inside replaced by the nearest outside
    # pixel) re-anchored to the inside background, so the ROI step edge / the dark rim outside cannot leak into d as a false band.
    has_out = bool((~m).any())
    if has_out:
        idx_in = ndi.distance_transform_edt(~m, return_distances=False, return_indices=True)
        idx_out = ndi.distance_transform_edt(m, return_distances=False, return_indices=True)
    else:
        yy_, xx_ = np.indices(m.shape); idx_in = np.stack([yy_, xx_]); idx_out = None
    ii = (idx_in[0], idx_in[1])
    sig_x = sig[ii]; sig0_x = sig0[ii]
    nu = nu0[ii]
    zi = zone_maps(nu, mode)
    c8 = np.clip(c, 0, 255).astype(np.uint8)
    b_in, b_out = two_sided_median_bg(c8, m, idx_in, idx_out, sig_x, pol, BLOB['inpaint_abs'][mode])
    off = cv2.GaussianBlur(bg - b_in, (0, 0), 6)
    bg_ext = np.where(m, bg, b_out + off[ii]).astype(np.float32)
    bm_ext = np.where(m, b_in, b_out).astype(np.float32)
    del b_in, b_out, off, idx_out
    m_ext = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BLOB['roi_ext'] + 1,) * 2)).astype(bool) if has_out else m
    d = signed(bg_ext, c, pol); d[~m_ext] = 0
    d_l = signed(bm_ext, c, pol); d_l[~m_ext] = 0
    tbl = P['zone']
    kh = np.zeros(zi.shape, np.float32); kl = np.zeros_like(kh); ah = np.zeros_like(kh); al = np.zeros_like(kh)
    for i, z in enumerate(ZONES):
        sel = zi == i; q = tbl[z]
        kh[sel] = q['k_hi']; kl[sel] = q['k_lo']; ah[sel] = q['abs_hi']; al[sel] = q['abs_lo']
    thi = np.maximum(kh * sig_x, ah); tlo = np.maximum(kl * sig_x, al)
    hi = (d > thi) & m_ext
    lo = (d > tlo) & m_ext
    # hysteresis
    n, lab, st, cen = cv2.connectedComponentsWithStats(lo.astype(np.uint8), connectivity=8)
    seeds = np.bincount(lab[hi], minlength=n) > 0
    seeds[0] = False
    keep_ids = np.nonzero(seeds & (st[:, 4] >= 3))[0]
    # path B: compact dark/bright dots in rough zones, lower relative threshold, shape-gated afterwards
    DP = DOT[mode]
    rough_i = ZONES.index('rough')
    rz = (zi == rough_i) & m_ext
    hiB = (d > np.maximum(DP['k_hi'] * sig_x, DP['abs_hi'])) & rz
    loB = (d > np.maximum(DP['k_lo'] * sig_x, DP['abs_lo'])) & rz
    labB = stB = None
    accA = np.zeros((H, W), bool)
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    anomaly = fr in ANOM[mode]
    acc = []          # accepted records
    bands = []        # suppressed (refined-away) band components, audit

    def evaluate(path, x0, y0, patch, gate=True, claim=True):
        """metrics + gates + ROI rule for one component (patch = bool array whose [0,0] is global (x0,y0)). Returns record dict or None."""
        a = int(patch.sum())
        if a < 3: return None
        y1, x1 = y0 + patch.shape[0], x0 + patch.shape[1]
        sl = (slice(y0, y1), slice(x0, x1))
        n_in = int((patch & m[sl]).sum()); roi_frac = n_in / a
        if n_in < 3: return None  # never output components (nearly) fully outside the ROI
        ys, xs = np.nonzero(patch)
        cc = c[sl][patch]; oo = o[sl][patch]
        zc = np.bincount(zi[sl][patch], minlength=len(ZONES))
        zone = 'normal'
        for zz in range(len(ZONES)):
            if zc[zz] >= 0.25 * a: zone = ZONES[zz]; break
        q = tbl[zone]
        if path == 'C':
            bb = bm_ext[sl][patch]; dsel = d_l[sl][patch]
            sg = float(sig0_x[sl][patch].mean())  # noise-map sigma (not texture-inflated)
        else:
            bb = bg_ext[sl][patch]; dsel = d[sl][patch]
            sg = float(sig_x[sl][patch].mean())
        contrast = float(cc.mean() - bb.mean())
        cnr = abs(contrast) / sg
        snr = abs(contrast) * math.sqrt(a) / raw_sig
        peak = float(np.percentile(dsel, 90))
        bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
        sub = patch[by0:by1 + 1, bx0:bx1 + 1]
        f = shape_feats(sub)
        if gate:
            if path == 'A':
                strongA = abs(contrast) >= STR['a_strong_contrast'] and snr >= 8 * q.get('snr_min', 6.0) and cnr >= 0.6 * q['cnr_min']
                if a < q['min_area'] or (cnr < q['cnr_min'] and not strongA) or snr < q.get('snr_min', 6.0): return None
            elif path == 'C':
                if claim and (accA[sl] & patch).sum() > 0.3 * a: return None
                lwC = f['feret'] / max(f['thickness'], 0.5)
                cmin_ = BLOB[mode][zone][1]
                if (a < BLOB['min_area'] or abs(contrast) < cmin_ or cnr < BLOB['cnr_min'] or snr < BLOB['snr_min']): return None
                if mode == 'side' and abs(contrast) < 20 and snr < BLOB['side_low_snr']: return None  # v3.1: Side path-C gray_low needs strong raw SNR
                if f['solidity'] < BLOB['min_solidity'] or f['aspect_ratio'] > BLOB['max_aspect'] or lwC > BLOB['max_lw']: return None
            else:
                if claim and (accA[sl] & patch).sum() > 0.3 * a: return None
                lwB = f['feret'] / max(f['thickness'], 0.5)
                strong = abs(contrast) >= DP['strong_contrast'] and a >= DP['strong_min_area']
                mmc = DP['min_mean_contrast'] * (1.45 if a < 20 else (1.2 if a < 40 else (1.0 if a < 80 else 1.3)))
                if (a < DP['min_area'] or abs(contrast) < mmc or cnr < DP['cnr_min'] or snr < DP['snr_min']): return None
                if strong:
                    if lwB > DP['max_lw'] * 1.5: return None
                elif f['solidity'] < DP['min_solidity'] or f['aspect_ratio'] > DP['max_aspect'] or lwB > DP['max_lw']: return None
        # ROI rule: keep if >= roi_keep_frac inside; else keep when it touches the ROI (>=3 px inside) and is a high-contrast dot
        if roi_frac >= BLOB['roi_keep_frac']:
            reason = 'inside' if roi_frac >= 0.999 else 'frac'
        elif (roi_frac >= STR['min_frac'] and peak >= STR['peak'][mode] and abs(contrast) >= STR['min_mean'][mode]) or (roi_frac >= STR['min_frac'] and cnr >= STR['cnr'] and abs(contrast) >= STR['min_mean'][mode] * 2):
            reason = 'high_contrast'
        else:
            return None
        bits = int(np.bitwise_or.reduce(nu[sl][patch]))
        dmin = float(dist[sl][patch].min())
        return dict(path=path, x0=x0, y0=y0, patch=patch, a=a, roi_frac=roi_frac, reason=reason, zone=zone, q=q, contrast=contrast, cnr=cnr, snr=snr, sg=sg,
                    peak=peak, f=f, cc=cc, oo=oo, bb=bb, bits=bits, dmin=dmin, xs=xs, ys=ys, bx0=bx0, bx1=bx1, by0=by0, by1=by1, cons='none', merged_n=1, dsrc='C' if path == 'C' else 'A')

    def accept(r):
        acc.append(r)
        sl = (slice(r['y0'], r['y0'] + r['patch'].shape[0]), slice(r['x0'], r['x0'] + r['patch'].shape[1]))
        accA[sl] |= r['patch']

    def band_refine(x0, y0, patch):
        """v3.1: a big, faint, band-like A component (e.g. a 2464x30 illumination / specular glare band) is NOT output as one box: re-segment it with
        raised thresholds (1.5x..4x the zone low threshold) so only genuinely stronger sub-structures survive; the original is logged in the audit list."""
        a = int(patch.sum()); y1, x1 = y0 + patch.shape[0], x0 + patch.shape[1]
        sl = (slice(y0, y1), slice(x0, x1))
        dm = float(d[sl][patch].mean())
        ys, xs = np.nonzero(patch)
        f = shape_feats(patch[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
        asp = f['feret'] / max(f['thickness'], 0.5)
        line = a >= BAND['min_area'] and f['feret'] >= BAND['line_len'] and asp >= BAND['line_asp']   # very long straight thin line = glare stripe
        if dm >= BAND['max_mean'] and not line: return None
        spec = float((zi[sl][patch] == ZONES.index('specular')).mean())
        if not ((a >= BAND['min_area'] and asp >= BAND['min_asp']) or (a >= BAND['spec_min_area'] and spec >= 0.5 and asp >= BAND['spec_asp'])): return None
        tl = tlo[sl]
        for fct in BAND['factors']:
            sub = patch & (d[sl] > fct * tl)
            lb, nl = ndi.label(sub, structure=np.ones((3, 3)))
            if nl == 0: break
            ar = np.bincount(lb.ravel())[1:]
            if ar.max() <= BAND['max_keep']: break
        out = []
        if nl:
            for j, s_ in enumerate(ndi.find_objects(lb), start=1):
                if ar[j - 1] < 3: continue
                pj = ndi.binary_fill_holes(lb[s_] == j)
                if line:
                    fj = shape_feats(pj)
                    if fj['feret'] >= BAND['line_len'] and fj['feret'] / max(fj['thickness'], 0.5) >= BAND['line_asp']: continue   # the stripe itself is dropped
                out.append((x0 + s_[1].start, y0 + s_[0].start, pj))
        bands.append(dict(bbox=[int(x0 + xs.min()), int(y0 + ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)], area_px=a, mean_contrast=round(dm, 2),
                          aspect=round(asp, 1), specular_frac=round(spec, 2), raise_factor=fct, n_substructures=len(out), reason=('glare-stripe line (feret>=%d, aspect>=%d) re-segmented; stripe itself dropped' % (BAND['line_len'], BAND['line_asp'])) if line else 'faint band-like component re-segmented'))
        return out

    # order by position (y,x) for stable ids
    order = sorted(keep_ids, key=lambda i: (st[i, 1] // 64, st[i, 0]))
    items = [('A', i) for i in order] + [('B0', -1)]
    for path, i in items:
        if path == 'B0':
            # path B is built AFTER path A so that pixels already claimed by A candidates are removed first (separates
            # dark dots that touch a bigger neighbour); then B components are relabelled
            loB2 = loB & ~cv2.dilate(accA.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            nB, labB, stB, _ = cv2.connectedComponentsWithStats(loB2.astype(np.uint8), connectivity=8)
            seedB = np.bincount(labB[hiB & loB2], minlength=nB) > 0
            seedB[0] = False
            keepB = np.nonzero(seedB & (stB[:, 4] >= DP['min_area']) & (stB[:, 4] <= DP['max_area']))[0]
            finalB = np.zeros((H, W), np.uint8)
            for j in keepB:  # tighten each component to >= tighten*peak, so touching dots/halo separate
                bx, by, bw_, bh_, _ = stB[j]
                sl = (slice(by, by + bh_), slice(bx, bx + bw_))
                pj = labB[sl] == j
                pk = float(np.percentile(d[sl][pj], 90))
                finalB[sl][pj & (d[sl] >= DP['tighten'] * pk)] = 1
            finalB = cv2.morphologyEx(finalB, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # close ring/halo-shaped dots (holes filled later)
            nB, labB, stB, _ = cv2.connectedComponentsWithStats(finalB, connectivity=8)
            keepB = np.nonzero(stB[:, 4] >= DP['min_area'])[0]; keepB = keepB[keepB > 0]
            items.extend(('B', int(j)) for j in sorted(keepB, key=lambda j: (stB[j, 1] // 64, stB[j, 0])))
            items.append(('C0', -1))
            continue
        if path == 'C0':
            # path C: multi-scale compact-blob seeds on the two-sided small-scale robust (median) background, all zones; built after A and B
            Bs, Ls = blob_maps(d_l, m_ext)
            sigL = np.stack([residual_sigma(Ls[k], m) for k in range(len(Ls))])
            ks = np.argmax(Bs, axis=0)[None]
            Bmax = np.take_along_axis(Bs, ks, 0)[0]; Lsel = np.take_along_axis(Ls, ks, 0)[0]; Ssel = np.take_along_axis(sigL, ks, 0)[0]
            fl = np.zeros((H, W), np.float32)
            for zi_, z in enumerate(ZONES):
                sel = zi == zi_; fl[sel] = BLOB[mode][z][0]
            ok = (Bmax > 0.12 * Lsel) & (Lsel > np.maximum(BLOB['k_sig'] * Ssel, fl)) & m_ext
            mx = cv2.dilate((Bmax * ok).astype(np.float32), np.ones((BLOB['nms'], BLOB['nms']), np.uint8))
            seedm = ok & (Bmax >= mx)
            claimed7 = cv2.dilate(accA.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            claimed3 = cv2.dilate(accA.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            sy, sx = np.nonzero(seedm & ~claimed7)
            ordr = np.argsort(-Lsel[sy, sx]); sy, sx = sy[ordr], sx[ordr]
            labC = np.zeros((H, W), np.int32); nC = 0; L0 = Ls[0]
            for yy, xx in zip(sy, sx):
                if labC[yy, xx]: continue
                R = int(3 * BLOB['scales'][int(ks[0, yy, xx])]) + 3
                wy0, wy1, wx0, wx1 = max(yy - R, 0), min(yy + R + 1, H), max(xx - R, 0), min(xx + R + 1, W)
                g = L0[wy0:wy1, wx0:wx1]
                pk = float(g[max(yy - wy0 - 1, 0):yy - wy0 + 2, max(xx - wx0 - 1, 0):xx - wx0 + 2].max())
                thr_ = max(BLOB['frac'] * pk, 0.5 * float(fl[yy, xx]))
                rg = (g > thr_) & m_ext[wy0:wy1, wx0:wx1] & ~claimed3[wy0:wy1, wx0:wx1] & (labC[wy0:wy1, wx0:wx1] == 0)
                lb, nl = ndi.label(rg, structure=np.ones((3, 3)))
                l0 = lb[yy - wy0, xx - wx0]
                if l0 == 0: continue
                comp = lb == l0
                if comp.sum() < BLOB['min_area']: continue
                nC += 1
                labC[wy0:wy1, wx0:wx1][comp] = nC
            stC = np.zeros((nC + 1, 5), np.int32)
            for j, sl in enumerate(ndi.find_objects(labC), start=1):
                if sl is None: continue
                stC[j] = [sl[1].start, sl[0].start, sl[1].stop - sl[1].start, sl[0].stop - sl[0].start, int((labC[sl] == j).sum())]
            keepC = [j for j in range(1, nC + 1) if stC[j, 4] >= BLOB['min_area'] and stC[j, 4] <= BLOB['max_area']]
            items.extend(('C', int(j)) for j in sorted(keepC, key=lambda j: (stC[j, 1] // 64, stC[j, 0])))
            continue
        lab_, st_ = (lab, st) if path == 'A' else ((labB, stB) if path == 'B' else (labC, stC))
        x, y, w, h, a0 = st_[i]
        x0, y0, x1, y1 = max(x - 2, 0), max(y - 2, 0), min(x + w + 2, W), min(y + h + 2, H)
        patch = lab_[y0:y1, x0:x1] == i
        if a0 > 1: patch = ndi.binary_fill_holes(patch)
        if path == 'A' and not anomaly and a0 >= BAND['spec_min_area']:
            subs = band_refine(x0, y0, patch)
            if subs is not None:
                accA[y0:y1, x0:x1] |= patch   # the band itself stays claimed so B/C do not re-detect its texture
                for sx0, sy0, sp in subs:
                    r = evaluate('A', sx0, sy0, sp, gate=True, claim=False)
                    if r is not None:
                        acc.append(r)
                continue
        r = evaluate(path, x0, y0, patch)
        if r is not None: accept(r)

    # ---------------- consolidation (v3.1): one dot = one candidate ----------------
    own = np.zeros((H, W), np.int32)
    for k_, r in enumerate(acc):
        own[r['y0']:r['y0'] + r['patch'].shape[0], r['x0']:r['x0'] + r['patch'].shape[1]][r['patch']] = k_ + 1
    alive = [True] * len(acc)

    def dsrc_of(r): return d_l if r['dsrc'] == 'C' else d
    rank = {'A': 0, 'B': 1, 'C': 2}
    for i0 in sorted(range(len(acc)), key=lambda k: -acc[k]['peak']):
        if not alive[i0]: continue
        r = acc[i0]
        if not (r['dmin'] <= REG['near'] or r['roi_frac'] < 1.0): continue
        if r['a'] > REG['max_area'] // 2 or r['peak'] < REG['min_peak']: continue
        pad = REG['win']; ph, pw = r['patch'].shape
        wy0, wx0 = max(r['y0'] - pad, 0), max(r['x0'] - pad, 0); wy1, wx1 = min(r['y0'] + ph + pad, H), min(r['x0'] + pw + pad, W)
        rp = np.zeros((wy1 - wy0, wx1 - wx0), bool); rp[r['y0'] - wy0:r['y0'] - wy0 + ph, r['x0'] - wx0:r['x0'] - wx0 + pw] = r['patch']
        dw = np.maximum(d[wy0:wy1, wx0:wx1], d_l[wy0:wy1, wx0:wx1]) if r['dsrc'] == 'C' else d[wy0:wy1, wx0:wx1]
        mw = m_ext[wy0:wy1, wx0:wx1]; ow = own[wy0:wy1, wx0:wx1]
        best = None
        for frac in REG['fracs']:
            thr = max(frac * r['peak'], REG['floor'])
            bwm = (dw > thr) & mw
            lb, nl = ndi.label(bwm, structure=np.ones((3, 3)))
            ids = np.unique(lb[rp & bwm]); ids = ids[ids > 0]
            if not len(ids): break
            U = np.isin(lb, ids) | rp
            U = ndi.binary_fill_holes(U)
            mem = np.unique(ow[U]); mem = mem[mem > 0] - 1
            # members not entirely inside the window or giant are not merged (their pixels are excluded)
            bad = [k for k in mem if k != i0 and (not alive[k] or acc[k]['a'] > REG['max_area'] // 2 or acc[k]['x0'] < wx0 or acc[k]['y0'] < wy0 or
                   acc[k]['x0'] + acc[k]['patch'].shape[1] > wx1 or acc[k]['y0'] + acc[k]['patch'].shape[0] > wy1)]
            for k in bad: U &= ~(ow == k + 1)
            mem = [k for k in mem if k not in bad]
            if i0 not in mem: mem.append(i0)
            amax = max(acc[k]['a'] for k in mem)
            if U.sum() <= min(REG['max_ratio'] * amax, REG['max_area']):
                best = (U, mem); break
        if best is None: continue
        U, mem = best
        if U.sum() <= r['a'] and len(mem) == 1: continue
        lead = max(mem, key=lambda k: (acc[k]['a'] >= 12, -rank[acc[k]['path']], acc[k]['a']))
        nr_ = evaluate(acc[lead]['path'], wx0, wy0, U, gate=False)
        if nr_ is None: continue
        nr_['cons'] = 'regrown' if (U.sum() > r['a'] or len(mem) > 1) else 'none'; nr_['merged_n'] = len(mem)
        nr_['path'] = acc[lead]['path']
        for k in mem: alive[k] = False
        acc.append(nr_); alive.append(True)
        own[wy0:wy1, wx0:wx1][U] = len(acc)
    # ---- split touching dots (watershed on the smoothed residual, h-maxima markers) ----
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed
    newrecs = []
    for k in range(len(acc)):
        if not alive[k]: continue
        r = acc[k]
        if anomaly or not (SPLIT['min_area'] <= r['a'] <= SPLIT['max_area']): continue
        ph, pw = r['patch'].shape; x0, y0 = r['x0'], r['y0']
        sl = (slice(y0, y0 + ph), slice(x0, x0 + pw))
        dd = dsrc_of(r)[sl]
        g = cv2.GaussianBlur(np.where(r['patch'], dd, 0).astype(np.float32), (0, 0), SPLIT['smooth'])
        g = np.where(r['patch'], g, g.min() - 1)
        if r['peak'] >= CORE['min_peak'] and CORE['min_area'] <= r['a'] <= CORE['max_area']:
            core = r['patch'] & (g >= CORE['frac'] * g[r['patch']].max())
            core = ndi.binary_opening(core, structure=np.ones((3, 3))) if core.sum() > 30 else core
            if 0 < core.sum() <= CORE['max_core_ratio'] * r['a']:
                cd = ndi.binary_dilation(core, structure=np.ones((3, 3))) & r['patch']
                cl, ncl = ndi.label(cd, structure=np.ones((3, 3)))
                rest = ndi.binary_opening(r['patch'] & ~cd, structure=np.ones((3, 3)))
                rl, nrl = ndi.label(rest, structure=np.ones((3, 3)))
                masks_ = [cl == j for j in range(1, ncl + 1)] + [rl == j for j in range(1, nrl + 1)]
                pcs = []
                for pp in masks_:
                    if pp.sum() < CORE['min_piece']: continue
                    rr = evaluate(r['path'], x0, y0, pp, gate=False)
                    if rr is not None: rr['cons'] = 'split'; pcs.append(rr)
                if len(pcs) >= 2 or (len(pcs) == 1 and pcs[0]['a'] < 0.8 * r['a']):
                    alive[k] = False; newrecs += pcs; continue
        hh = max(SPLIT['h_frac'] * r['peak'], SPLIT['h_min'])
        mk = h_maxima(g, hh)
        ml, nm = ndi.label(mk & r['patch'], structure=np.ones((3, 3)))
        if nm < 2: continue
        gp = g.max()
        keepm = [j for j in range(1, nm + 1) if g[ml == j].max() >= SPLIT['min_rel_peak'] * gp]
        if len(keepm) < 2: continue
        ml2 = np.zeros_like(ml)
        for t_, j in enumerate(keepm, start=1): ml2[ml == j] = t_
        ws = watershed(-g, ml2, mask=r['patch'])
        for _ in range(2):
            ar = np.bincount(ws.ravel(), minlength=len(keepm) + 1)[1:]
            small = [t_ + 1 for t_, a_ in enumerate(ar) if a_ < SPLIT['min_piece']]
            if not small or len(keepm) - len(small) < 1: break
            for t_ in small: ml2[ml2 == t_] = 0
            ws = watershed(-g, ml2, mask=r['patch'])
        labs = [t_ for t_ in np.unique(ws) if t_ > 0]
        if len(labs) < 2: continue
        pcs = []
        for t_ in labs:
            pp = ws == t_
            rr = evaluate(r['path'], x0, y0, pp, gate=False)
            if rr is not None: rr['cons'] = 'split'; pcs.append(rr)
        if len(pcs) >= 2:
            alive[k] = False; newrecs += pcs
    final = [r for k, r in enumerate(acc) if alive[k]] + newrecs
    # ---------------- output ----------------
    maskdir = f'{OUT}/{mode}/{src}_{ang}/masks'
    final.sort(key=lambda r: (rank[r['path']], r['y0'] // 64, r['x0']))
    cands = []
    for cid, r in enumerate(final):
        a = r['a']; f = r['f']; q = r['q']; zone = r['zone']; contrast = r['contrast']; cnr = r['cnr']
        x0, y0 = r['x0'], r['y0']; xs, ys = r['xs'], r['ys']
        bx0, bx1, by0, by1 = r['bx0'], r['bx1'], r['by0'], r['by1']
        gx0, gy0 = x0 + int(bx0), y0 + int(by0); bw, bh = int(bx1 - bx0 + 1), int(by1 - by0 + 1)
        sub = r['patch'][by0:by1 + 1, bx0:bx1 + 1]
        cls = classify(a, f)
        conf = confidence(cls, a, f, contrast, cnr, q['k_hi'], zone, r['dmin'])
        faint_large = bool(a >= 15000 and abs(contrast) < 40)  # shading/illumination-gradient region, not a compact defect
        ac = abs(contrast)
        tier = 'gray_low' if round(ac, 2) < 20 else ('gray_mid' if round(ac, 2) < 40 else 'high')  # confidence tiered by contrast
        conf = round(conf * {'gray_low': 0.7, 'gray_mid': 0.88, 'high': 1.0}[tier], 3)
        if faint_large: conf = round(conf * 0.4, 3)
        # surrounding nuisance (context): bits in a 8px ring
        r0, r1, r2, r3 = max(gy0 - 8, 0), min(gy0 + bh + 8, H), max(gx0 - 8, 0), min(gx0 + bw + 8, W)
        sur = int(np.bitwise_or.reduce(nu[r0:r1, r2:r3].ravel()))
        cx = float(xs.mean() + x0); cy = float(ys.mean() + y0)
        cname = f'{src}_{ang}_{fr}_c{cid:04d}'
        cv2.imwrite(f'{maskdir}/{fr}_{cname}.png', (sub * 255).astype(np.uint8))
        cc, oo, bb = r['cc'], r['oo'], r['bb']
        cands.append(dict(
            path=r['path'], sensitivity_version=SENS, roi_overlap_frac=round(r['roi_frac'], 3), straddles_roi=bool(r['roi_frac'] < 1.0), roi_keep_reason=r['reason'],
            consolidation=r['cons'], merged_n=int(r['merged_n']), contrast_tier=tier, large_faint_region=faint_large, id=cname, frame=fr, mode=mode, source=src, angle=ang, image=opath, cleaned_image=cpath,
            mask_patch=f'{maskdir}/{fr}_{cname}.png',
            bbox=[int(gx0), int(gy0), bw, bh], centroid=[round(cx, 1), round(cy, 1)], area_px=a, polarity=pol,
            mean_gray=round(float(cc.mean()), 2), max_gray=round(float(cc.max()), 1), min_gray=round(float(cc.min()), 1),
            local_bg_gray=round(float(bb.mean()), 2), contrast=round(contrast, 2), cnr=round(cnr, 2),
            noise_sigma=round(r['sg'], 3), snr_raw=round(r['snr'], 2), orig_mean_gray=round(float(oo.mean()), 2),
            nuisance_bits=r['bits'], nuisance_labels=[BITN[b] for b in BITN if r['bits'] & b],
            nuisance_bits_surround=sur, zone=zone,
            aspect_ratio=f['aspect_ratio'], elongation=f['elongation'], solidity=f['solidity'],
            orientation_deg=f['orientation_deg'], major_len_px=f['major_len'], feret_px=f['feret'], thickness_px=f['thickness'],
            dist_to_roi_edge_px=round(r['dmin'], 1), **{'class': cls}, confidence=conf, anomaly_frame=anomaly,
            persistence=None, persistence_checked=None))
    stats = dict(frame=fr, n=len(cands), seeds=int(len(keep_ids)), roi_px=int(m.sum()), runtime_s=round(time.time() - t0, 2),
                 zone_px={z: int(((zi == i) & m).sum()) for i, z in enumerate(ZONES)}, bands=bands,
                 bg_mean=round(float(bg[m].mean()), 2) if m.any() else None)
    return src, ang, fr, cands, stats



def estimate_tangential(ca, cb, n_vec, normal_shift):
    """Burst frames of an Edge folder are tiles stepping ~1950 px ALONG the edge (only ~100 px overlap), and the preprocess
    registration only gives the edge-normal component. Recover the tangential shift by voting over candidate pairs:
    (content of frame k+1 position) - (frame k position) = normal_shift*n + vt*t. Returns (vt, votes, ratio) or None."""
    def pick(c):
        c = sorted(c, key=lambda x: -x['cnr'] * math.sqrt(x['area_px']))
        c = [x for x in c if x['area_px'] >= 6 and x['cnr'] >= 10][:700]
        return np.array([x['centroid'] for x in c]) if c else np.zeros((0, 2))
    A_, B_ = pick(ca), pick(cb)
    if len(A_) < 3 or len(B_) < 3: return None
    t_vec = np.array([-n_vec[1], n_vec[0]]); n_vec = np.array(n_vec)
    d = (B_[None] - A_[:, None]).reshape(-1, 2)
    vn = d @ n_vec; vt = d @ t_vec
    sel = np.abs(vn - normal_shift) <= PERSIST['vote_normal_band']
    if sel.sum() < 3: return None
    vt = vt[sel]
    # prior: tiles step ~1.9k px along the edge (overlap <= ~350 px) -> only consider |vt| in [1650, 2060]
    pr = (np.abs(vt) >= PERSIST['tan_min']) & (np.abs(vt) <= PERSIST['tan_max'])
    vt = vt[pr]
    if len(vt) < 3: return None
    bins = np.arange(-2600, 2601, 3.0)
    H, _ = np.histogram(vt, bins)
    Hs = H[:-1] + H[1:]  # 6-px window
    k = int(np.argmax(Hs)); votes = int(Hs[k])
    Hm = Hs.copy(); Hm[max(k - 10, 0):k + 11] = 0
    bg = float(Hm.max())  # strongest competing peak elsewhere
    if votes < PERSIST['vote_min'] or votes < PERSIST['vote_ratio'] * max(bg, 1.0): return None
    cen = bins[k] + 3.0
    m = np.abs(vt - cen) <= 5.0
    return float(np.mean(vt[m])), votes, float(votes / max(bg, 1.0))


def persistence(src, ang, frames, allc):
    """persistence = number of neighbouring frames (within +-window, chain of trusted/voted shifts) in which the shifted
    location is inside that frame's ROI AND a same-polarity candidate lies near it. persistence_checked = number of
    neighbour frames whose ROI covers the shifted location (only the ~100 px overlap strip qualifies for Edge tiles).
    Edge only. Never used to drop candidates."""
    rep = json.load(open(f'{PRE}/{src}/{ang}/report.json'))
    pairs = {p['from']: p for p in rep['registration']['pairs']}
    idx = {f: i for i, f in enumerate(frames)}
    ok = []; nrm = []; ns = []; vts = []; info = []
    for k, f in enumerate(frames[:-1]):
        p = pairs.get(f)
        good = bool(p and p.get('trusted') and p.get('edge_line'))
        n_ = p['edge_line']['normal'] if good else [1.0, 0.0]
        nshift = p['edge_line']['normal_shift_px'] if good else 0.0
        vt = None
        if good:
            r = estimate_tangential(allc.get(f, []), allc.get(frames[k + 1], []), n_, nshift)
            if r: vt = r[0]
        ok.append(good and vt is not None); nrm.append(np.array(n_)); ns.append(nshift); vts.append(vt if vt is not None else 0.0)
        info.append(dict(pair=f'{f}->{frames[k+1]}', trusted=good, normal_shift=round(nshift, 2),
                         tangential_shift=None if vt is None else round(vt, 1), votes=None if vt is None else r[1]))
    # shift vector (xy) for each consecutive pair, cumulative
    sv = [ns[i] * nrm[i] + vts[i] * np.array([-nrm[i][1], nrm[i][0]]) for i in range(len(ok))]
    cum = np.concatenate([[[0.0, 0.0]], np.cumsum(np.array(sv), axis=0)])
    masks = {}
    def roi_at(j, pt):
        if j not in masks:
            masks[j] = cv2.imread(f'{PRE}/{src}/{ang}/{frames[j]}_mask.png', cv2.IMREAD_UNCHANGED) > 0
        mk = masks[j]; x, y = int(round(pt[0])), int(round(pt[1]))
        return 0 <= x < mk.shape[1] and 0 <= y < mk.shape[0] and bool(mk[y, x])
    arr = {}
    for f in frames:
        cl = allc.get(f, [])
        if cl: arr[f] = (np.array([cc['centroid'] for cc in cl]), np.array([cc['area_px'] for cc in cl], float))
    W = PERSIST['window']
    for f in frames:
        k = idx[f]
        for cc in allc.get(f, []):
            cnt = 0; checked = 0
            r = math.sqrt(cc['area_px'] / math.pi); p0 = np.array(cc['centroid'])
            for dj in range(-W, W + 1):
                j = k + dj
                if dj == 0 or j < 0 or j >= len(frames): continue
                lo_, hi_ = min(k, j), max(k, j)
                if not all(ok[lo_:hi_]): continue
                pj = p0 + (cum[j] - cum[k])
                if not roi_at(j, pj): continue
                checked += 1
                if frames[j] not in arr: continue
                cen, ar = arr[frames[j]]
                n_ = nrm[min(k, j)]; t_ = np.array([-n_[1], n_[0]])
                dv = cen - pj
                dn = np.abs(dv @ n_); dt = np.abs(dv @ t_)
                ratio = ar / cc['area_px']
                sel = (dn <= PERSIST['tol_n'] + r) & (dt <= PERSIST['tol_t'] + r) & (ratio >= PERSIST['area_ratio'][0]) & (ratio <= PERSIST['area_ratio'][1])
                if sel.any(): cnt += 1
            cc['persistence'] = cnt; cc['persistence_checked'] = checked
    return info


FLAT = ['id', 'frame', 'mode', 'source', 'angle', 'image', 'cleaned_image', 'mask_patch', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
        'centroid_x', 'centroid_y', 'area_px', 'polarity', 'mean_gray', 'max_gray', 'min_gray', 'local_bg_gray', 'contrast', 'cnr',
        'noise_sigma', 'snr_raw', 'orig_mean_gray', 'nuisance_bits', 'nuisance_labels', 'nuisance_bits_surround', 'zone', 'aspect_ratio',
        'elongation', 'solidity', 'orientation_deg', 'major_len_px', 'feret_px', 'thickness_px', 'dist_to_roi_edge_px', 'class',
        'confidence', 'anomaly_frame', 'persistence', 'persistence_checked', 'path', 'large_faint_region', 'sensitivity_version', 'roi_overlap_frac', 'straddles_roi', 'roi_keep_reason', 'consolidation', 'merged_n', 'contrast_tier']


def flat(c):
    d = dict(c); b = d.pop('bbox'); ce = d.pop('centroid')
    d.update(bbox_x=b[0], bbox_y=b[1], bbox_w=b[2], bbox_h=b[3], centroid_x=ce[0], centroid_y=ce[1])
    d['nuisance_labels'] = '|'.join(d['nuisance_labels'])
    return d


def run(only=None, nproc=int(os.environ.get('AOI_NPROC', 6))):
    t0 = time.time()
    jobs = []
    fsel = {}
    for tok in filter(None, os.environ.get('AOI_FRAMES', '').split(';')):   # tuning hook, e.g. 'Edge1/090:19,20;Edge2/090:25,26'
        k_, v_ = tok.split(':'); fsel[tuple(k_.split('/'))] = set(v_.split(','))
    partial = bool(fsel)
    for src in MODES:
        for ang in ANGLES[src]:
            if only and (src, ang) not in only: continue
            mode = MODES[src]
            os.makedirs(f'{OUT}/{mode}/{src}_{ang}/masks', exist_ok=True)
            frs = sorted(os.path.basename(p)[:2] for p in glob.glob(f'{PRE}/{src}/{ang}/*_cleaned.png'))
            if fsel: frs = [f for f in frs if f in fsel.get((src, ang), set())]
            if not partial:   # full run: remove stale masks / json of the previous version
                for p in glob.glob(f'{OUT}/{mode}/{src}_{ang}/masks/*.png'): os.remove(p)
            jobs += [(src, ang, f) for f in frs]
    res = {}
    with Pool(nproc) as pool:
        for src, ang, fr, cands, stats in pool.imap_unordered(detect_frame, jobs, chunksize=1):
            res.setdefault((src, ang), {})[fr] = (cands, stats)
            print(src, ang, fr, len(cands), stats['runtime_s'], flush=True)
    summaries = {}
    for (src, ang), fd in sorted(res.items()):
        mode = MODES[src]; frames = sorted(fd)
        allc = {f: fd[f][0] for f in frames}
        pinfo = None
        if mode == 'edge' and not FLIP and not partial:
            pinfo = persistence(src, ang, frames, allc)
        od = f'{OUT}/{mode}/{src}_{ang}'
        rows = []
        for f in frames:
            json.dump(allc[f], open(f'{od}/{f}_candidates.json', 'w'), indent=1)
            rows += [flat(c) for c in allc[f]]
        json.dump({f: fd[f][1].get('bands', []) for f in frames if fd[f][1].get('bands')}, open(f'{od}/suppressed_bands.json', 'w'), indent=1)
        with open(f'{od}/candidates.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, FLAT); w.writeheader(); [w.writerow({k: r.get(k) for k in FLAT}) for r in rows]
        by_class = {}; by_zone = {}
        for r in rows:
            by_class[r['class']] = by_class.get(r['class'], 0) + 1; by_zone[r['zone']] = by_zone.get(r['zone'], 0) + 1
        per_frame = {}
        for f in frames:
            cl = allc[f]; s = fd[f][1]
            pc = {}
            zc = {}
            for c in cl: pc[c['class']] = pc.get(c['class'], 0) + 1; zc[c['zone']] = zc.get(c['zone'], 0) + 1
            per_frame[f] = dict(n=len(cl), high_count_flag=len(cl) > 1000, by_class=pc, by_zone=zc, anomaly_frame=f in ANOM[mode], runtime_s=s['runtime_s'], zone_px=s['zone_px'], roi_px=s['roi_px'])
        summ = dict(source=src, angle=ang, mode=mode, polarity=PARAMS[mode]['polarity'], n_frames=len(frames), total=len(rows),
                    by_class=by_class, by_zone=by_zone, anomaly_frame_candidates=sum(1 for r in rows if r['anomaly_frame']),
                    per_frame=per_frame, sensitivity_version=SENS, by_path={p: sum(1 for r in rows if r['path'] == p) for p in 'ABC'}, by_roi_keep_reason={k_: sum(1 for r in rows if r['roi_keep_reason'] == k_) for k_ in ('inside', 'frac', 'high_contrast')}, by_consolidation={k_: sum(1 for r in rows if r['consolidation'] == k_) for k_ in ('none', 'regrown', 'split')}, suppressed_band_components=sum(len(fd[f][1].get('bands', [])) for f in frames), params=dict(detector=PARAMS[mode], straddle_rule=STR, regrow=REG, split=SPLIT, core_trim=CORE, band_refine=BAND, rough_dot_path_B=DOT[mode], blob_path_C={k: v for k, v in BLOB.items() if k not in ('edge', 'side')} | {'zone_floors(seed_peak,min_mean_contrast)': BLOB[mode]}, background=BG, classification=CLS,
                    persistence=PERSIST if mode == 'edge' else 'not computed (Side has no registration in preprocess report)'),
                    burst_shifts=pinfo, preprocess_dir=f'{PRE}/{src}/{ang}', preprocess_report=f'{PRE}/{src}/{ang}/report.json',
                    coordinate_note='cleaned images are same size as originals (2464x2056, no crop/rescale); coordinates are original pixels')
        json.dump(summ, open(f'{od}/summary.json', 'w'), indent=1)
        summaries[f'{src}/{ang}'] = dict(total=len(rows), by_class=by_class)
    print('runtime', time.time() - t0)
    return summaries


if __name__ == '__main__':
    only = None
    if len(sys.argv) > 1:
        only = {tuple(a.split('/')) for a in sys.argv[1:]}
    print(json.dumps(run(only), indent=1))
