#!/usr/bin/env python3
import os
"""v4 raw-image dot recall (12-gray reference, no ground truth). Same reference as Integration-Eval's aoi_recall_eval.py
(bg = 100px-scale median, residual>=thr, area 3..1500, compact fill>=.6 L/W<=2.5, centroid in ROI eroded 3px) but:
 * runs on ALL non-anomaly frames (Edge: all; Side: 1..14), not a sample
 * reference blobs are labelled with the PREPROCESS-v3 nuisance bits (zone precedence: ink/oil > water > rough > specular > uneven > normal) and with
   `stain` = inside the stain-separation region (stain file != 128 dilated) => contrast of these dots is attenuated in the cleaned image
 * several detector outputs are scored against the SAME reference list: --dirs name=path ...
Writes only into --out (detect_eval_v4)."""
import argparse, glob, os, json
import cv2, numpy as np, pandas as pd
from multiprocessing import Pool
PRE = os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/preprocess'; DB = os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_db'
BINS = [12, 20, 40, 1e9]; BL = ['12-20', '20-40', '40+']
def bgmed(a, k=25, ds=4):
    s = cv2.resize(a, None, fx=1/ds, fy=1/ds, interpolation=cv2.INTER_AREA); m = cv2.medianBlur(s, k)
    return cv2.resize(m, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
def zone_of(nu):
    z = np.full(nu.shape, 'normal', object)
    for bit, nm in [(2, 'uneven'), (1, 'specular'), (4, 'rough'), (8, 'water'), (16, 'oil'), (32, 'ink')]:
        z[(nu & bit) > 0] = nm
    return z
def scan(args):
    src, ang, fr, pol, thr = args
    o = cv2.imread(f'{DB}/{src}/{ang}/{fr:02d}.png', 0); roi = cv2.imread(f'{PRE}/{src}/{ang}/{fr:02d}_mask.png', 0) > 0
    nu = cv2.imread(f'{PRE}/{src}/{ang}/{fr:02d}_nuisance.png', -1)
    st = cv2.imread(f'{PRE}/{src}/{ang}/{fr:02d}_stain.png', -1)
    stm = cv2.dilate((st != 128).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    a = o.copy(); a[~roi] = int(np.median(o[roi]))
    g = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 0.8); bg = bgmed(a)
    r = (bg - g) if pol == 'dark' else (g - bg)
    n, l, stt, ce = cv2.connectedComponentsWithStats(((r >= thr) & roi).astype(np.uint8), connectivity=8)
    er = cv2.erode(roi.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    rows = []
    for i in range(1, n):
        x, y, w, h, A = stt[i]
        if A < 3 or A > 1500: continue
        cx, cy = ce[i]
        if not er[int(cy), int(cx)]: continue
        sub = l[y:y+h, x:x+w] == i; ys, xs = np.nonzero(sub)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
        L, W = max(rw, rh) + 1, min(rw, rh) + 1; fill = A / (np.pi/4*L*W)
        if fill < 0.6 or L/W > 2.5: continue
        rr = r[y:y+h, x:x+w][sub]
        b = int(nu[int(cy), int(cx)])
        # zone precedence: ink/oil > water > rough > specular > uneven > normal
        zone = 'ink' if b & 32 else 'oil' if b & 16 else 'water' if b & 8 else 'rough' if b & 4 else 'specular' if b & 1 else 'uneven' if b & 2 else 'normal'
        rows.append(dict(src=src, ang=ang, fr=fr, cx=cx, cy=cy, area=A, L=L, W=W, fill=fill, peak=float(rr.max()), mean=float(rr.mean()), rough=bool(b & 4),
                         bits=b, zone=zone, stain=bool(stm[int(cy), int(cx)]), contam=bool(b & 56)))
    return rows
def load_cands(d):
    fs = sorted(glob.glob(d + '/*/*_*/candidates.csv'))
    c = pd.concat([pd.read_csv(f, dtype={'frame': str, 'angle': str}) for f in fs], ignore_index=True)
    c['folder'] = c.source + '/' + c.angle; c['fr'] = c.frame.astype(int)
    return c
def cover(b, c, pad=2, ratio=30):
    out = np.zeros(len(b), bool)
    if len(c) == 0 or len(b) == 0: return out
    x0 = c.bbox_x.values - pad; y0 = c.bbox_y.values - pad; x1 = c.bbox_x.values + c.bbox_w.values + pad; y1 = c.bbox_y.values + c.bbox_h.values + pad; ca = c.area_px.values
    for i, (cx, cy, ar) in enumerate(zip(b.cx.values, b.cy.values, b.area.values)):
        out[i] = ((cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1) & (ca <= ratio*ar)).any()
    return out
def jobs(thr):
    J = []
    for s in ['Edge1', 'Edge2']:
        for a in ['000', '090']: J += [(s, a, f, 'dark', thr) for f in range(1, 45)]
    J += [('Side', a, f, 'bright', thr) for a in ['000', '090', '180', '270'] for f in range(1, 15)]
    return J
def summ(g, cols):
    d = {'n': len(g)}
    for c in cols: d[c] = round(g[c].mean()*100, 1) if len(g) else np.nan
    return pd.Series(d)
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--dirs', nargs='+', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--thr', type=float, default=12); ap.add_argument('--procs', type=int, default=6); ap.add_argument('--refpkl', default=None)
    A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
    assert os.path.realpath(A.out).startswith(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/detect_eval_v41')
    refp = A.out + '/ref_blobs_base.pkl'
    if os.path.exists(refp): B = pd.read_pickle(refp)
    else:
        with Pool(A.procs) as p: res = p.map(scan, jobs(A.thr), chunksize=2)
        B = pd.DataFrame([r for rr in res for r in rr]); B['folder'] = B.src + '/' + B.ang
        B = B[~B.fr.isin([0, 45]) & ~((B.src == 'Side') & B.fr.isin([0, 15]))].reset_index(drop=True)
        B['bin'] = pd.cut(B.peak, BINS, right=False, labels=BL).astype(str)
        B['dotsize'] = np.where(B.area >= 12, 'area>=12', 'area3-11')
        B.to_pickle(refp)
    names = []
    for kv in A.dirs:
        nm, d = kv.split('=', 1); names.append(nm)
        C = load_cands(d); cols = []
        cv = pd.Series(False, index=B.index)
        for (f, fr), idx in B.groupby(['folder', 'fr']).groups.items():
            cv.loc[idx] = cover(B.loc[idx], C[(C.folder == f) & (C.fr == fr)])
        B[nm] = cv
    B.to_pickle(A.out + '/ref_blobs.pkl')
    pd.set_option('display.width', 250)
    with open(A.out + '/recall_tables.txt', 'w') as fh:
        def P(t, df):
            s = f'\n== {t}\n' + df.round(1).to_string() + '\n'; print(s); fh.write(s); df.round(1).to_csv(f'{A.out}/recall_{t.replace(" ", "_")}.csv')
        P('by_folder', B.groupby('folder').apply(lambda g: summ(g, names)))
        P('by_bin', B.groupby('bin').apply(lambda g: summ(g, names)))
        P('by_mode_zone_bin', B.assign(mode=np.where(B.src == 'Side', 'side', 'edge')).groupby(['mode', 'zone', 'bin']).apply(lambda g: summ(g, names)))
        P('by_mode_zone', B.assign(mode=np.where(B.src == 'Side', 'side', 'edge')).groupby(['mode', 'zone']).apply(lambda g: summ(g, names)))
        P('by_folder_zone', B.groupby(['folder', 'zone']).apply(lambda g: summ(g, names)))
        P('by_stainregion_bin', B.assign(mode=np.where(B.src == 'Side', 'side', 'edge')).groupby(['mode', 'stain', 'bin']).apply(lambda g: summ(g, names)))
        P('by_mode_bin_size', B.assign(mode=np.where(B.src == 'Side', 'side', 'edge')).groupby(['mode', 'bin', 'dotsize']).apply(lambda g: summ(g, names)))
        E2 = B[(B.folder == 'Edge2/000')]
        P('Edge2_000_by_bin_zone_size', E2.groupby(['bin', 'rough', 'dotsize']).apply(lambda g: summ(g, names)))
