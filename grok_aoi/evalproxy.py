"""Proxy evaluation without ground truth.

Recomputes, on the frames of this run:

* reference-dot retention (THR=12 scan; contrast bins 12–20 / 20–40 / 40+; zones)
* a polarity-flip false-real estimate

Definitions follow the historical filter v4.2 checks (coverage vs retention vs non-review).
The flip pass is an upper bound on the Side T2 term: persistence is not applied, so rules
(a) and (b) do not fire. Numbers are for this database, not the original 248-frame delivery.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BINS = [12, 20, 40, 1e9]
BL = ['12-20', '20-40', '40+']


def _bgmed(a, k=25, ds=4):
    s = cv2.resize(a, None, fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA)
    kk = min(k, s.shape[0] | 1, s.shape[1] | 1)
    kk = max(3, int(kk) | 1)
    if kk >= min(s.shape[:2]):
        kk = max(3, (min(s.shape[:2]) // 2) * 2 - 1)
    m = cv2.medianBlur(s, kk)
    return cv2.resize(m, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _zone_at(nu, st, y, x):
    contam = int(nu[y, x]) & (8 | 16 | 32)
    rough = (int(nu[y, x]) & 4) and not contam
    if contam:
        return 'contam'
    if rough:
        return 'rough'
    if st is not None and int(st[y, x]) != 128:
        return 'stain-zone'
    return 'clean'


def scan_frame(db, pre, src, ang, fr, pol, thr=12.0):
    o = cv2.imread(f'{db}/{src}/{ang}/{fr}.png', 0)
    roi_im = cv2.imread(f'{pre}/{src}/{ang}/{fr}_mask.png', 0)
    nu = cv2.imread(f'{pre}/{src}/{ang}/{fr}_nuisance.png', cv2.IMREAD_UNCHANGED)
    st = cv2.imread(f'{pre}/{src}/{ang}/{fr}_stain.png', cv2.IMREAD_UNCHANGED)
    if o is None or roi_im is None or nu is None:
        return []
    roi = roi_im > 0
    if int(roi.sum()) < 100:
        return []
    a = o.copy()
    a[~roi] = int(np.median(o[roi]))
    g = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 0.8)
    bg = _bgmed(a)
    r = (bg - g) if pol == 'dark' else (g - bg)
    n, lab, stt, ce = cv2.connectedComponentsWithStats(((r >= thr) & roi).astype(np.uint8), connectivity=8)
    er = cv2.erode(roi.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    rows = []
    for i in range(1, n):
        x, y, w, h, A = stt[i]
        if A < 3 or A > 1500:
            continue
        cx, cy = ce[i]
        iy, ix = int(cy), int(cx)
        if iy < 0 or ix < 0 or iy >= er.shape[0] or ix >= er.shape[1] or not er[iy, ix]:
            continue
        sub = lab[y:y + h, x:x + w] == i
        ys, xs = np.nonzero(sub)
        if len(xs) < 3:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
        L, W = max(rw, rh) + 1, min(rw, rh) + 1
        fill = A / (np.pi / 4 * L * W)
        if fill < 0.6 or L / max(W, 1e-6) > 2.5:
            continue
        rr = r[y:y + h, x:x + w][sub]
        rows.append(dict(folder=f'{src}_{ang}', src=src, ang=ang, fr=fr, cx=float(cx), cy=float(cy),
                         area=int(A), peak=float(rr.max()), zone=_zone_at(nu, st, iy, ix)))
    return rows


def _cover_frame(blobs, cands):
    """blobs: cx,cy,area. cands: bbox_*, area_px, decision, review_level."""
    n = len(blobs)
    cov = np.zeros(n, bool)
    real = np.zeros(n, bool)
    nr = np.zeros(n, bool)
    if n == 0 or cands is None or len(cands) == 0:
        return cov, real, nr
    bx = cands.bbox_x.values - 2
    by = cands.bbox_y.values - 2
    x1 = cands.bbox_x.values + cands.bbox_w.values + 2
    y1 = cands.bbox_y.values + cands.bbox_h.values + 2
    ca = cands.area_px.values
    is_real = (cands.decision.values == 'real')
    rev = cands.review_required.values.astype(bool) if 'review_required' in cands.columns else (cands.review_level.values > 0)
    for i, (cx, cy, ar) in enumerate(zip(blobs.cx.values, blobs.cy.values, blobs.area.values)):
        H = (cx >= bx) & (cx <= x1) & (cy >= by) & (cy <= y1) & (ca <= 30 * ar)
        cov[i] = H.any()
        real[i] = (H & is_real).any()
        nr[i] = (H & is_real & ~rev).any()
    return cov, real, nr


def _agg(g):
    c = int(g['cov'].sum())
    n = len(g)
    return pd.Series(dict(
        n=n,
        coverage_pct=round(100 * c / n, 1) if n else None,
        retention_incl_review_pct=round(100 * g['real'].sum() / c, 1) if c else None,
        retention_no_review_pct=round(100 * g['nr'].sum() / c, 1) if c else None,
    ))


def retention(db, pre, decisions, anomaly, thr=12.0):
    """Scan every non-anomaly frame that has a mask. Edge = dark dots, Side = bright dots."""
    anom = {('edge', f'{int(x):02d}') for x in anomaly.get('edge', [])} | {('side', f'{int(x):02d}') for x in anomaly.get('side', [])}
    rows = []
    frames = sorted({(r.folder, str(r.frame).zfill(2)) for r in decisions.itertuples()}) if len(decisions) else []
    # also frames present on disk under pre, so empty-candidate frames still count
    pre = Path(pre)
    seen = set(frames)
    for mask in pre.glob('*/*/*_mask.png'):
        fr = mask.name.split('_')[0]
        src, ang = mask.parent.parent.name, mask.parent.name
        if not fr.isdigit():
            continue
        seen.add((f'{src}_{ang}', fr))
    for folder, fr in sorted(seen):
        src, ang = folder.split('_', 1)
        kind = 'side' if src.startswith('Side') else 'edge'
        if (kind, fr) in anom:
            continue
        pol = 'bright' if kind == 'side' else 'dark'
        rows.extend(scan_frame(db, pre, src, ang, fr, pol, thr))
    B = pd.DataFrame(rows)
    if len(B) == 0:
        return {'n_ref': 0, 'by_bin': [], 'by_zone_bin': [], 'by_folder_bin': [], 'note': 'no reference dots'}
    B['bin'] = pd.cut(B.peak, BINS, right=False, labels=BL).astype(str)
    B['kind'] = np.where(B.src.str.startswith('Side'), 'Side', 'Edge')
    cov = np.zeros(len(B), bool)
    real = np.zeros(len(B), bool)
    nr = np.zeros(len(B), bool)
    dec = decisions.copy()
    if len(dec):
        dec['fr'] = dec.frame.astype(str).str.zfill(2)
        grouped = {k: g for k, g in dec.groupby(['folder', 'fr'])}
    else:
        grouped = {}
    for (folder, fr), idx in B.groupby(['folder', 'fr']).groups.items():
        c = grouped.get((folder, fr))
        cv, rl, nn = _cover_frame(B.loc[idx], c)
        cov[idx] = cv
        real[idx] = rl
        nr[idx] = nn
    B['cov'] = cov
    B['real'] = real
    B['nr'] = nr

    def table(df, keys):
        if len(df) == 0:
            return []
        t = df.groupby(keys, observed=True).apply(_agg, include_groups=False).reset_index()
        return json.loads(t.to_json(orient='records'))

    return {
        'n_ref': int(len(B)),
        'definition': {
            'scan': 'THR=12 residual vs 100px-scale median, area 3–1500, fill>=0.6, L/W<=2.5, centroid in ROI eroded 7px. Edge dark, Side bright. Anomaly frames skipped.',
            'coverage_pct': '參考點被任一候選框（bbox±2、候選面積≤30×）蓋住的比例',
            'retention_incl_review_pct': '在已被候選蓋住的參考點中，被 real 決策蓋住的比例',
            'retention_no_review_pct': '在已被候選蓋住的參考點中，被 real 且不需複檢蓋住的比例',
        },
        'by_bin': table(B, ['kind', 'bin']),
        'by_zone_bin': table(B, ['kind', 'zone', 'bin']),
        'by_folder_bin': table(B, ['folder', 'bin']),
    }


def _zone_bucket(z):
    z = np.asarray(z, dtype=object)
    return np.select(
        [np.isin(z, ['oil', 'water', 'ink']), z == 'rough', z == 'uneven', z == 'specular'],
        ['contam', 'rough', 'uneven', 'specular'],
        default='normal',
    )


def flip_estimate(db, pre, det_out, decisions, glare, anomaly, nproc, folders):
    """Polarity-flip detector + rules42 with dummy persistence, then a stratified false-real estimate."""
    from grok_aoi.detect import aoi_detect
    from grok_aoi.filter import feats
    from grok_aoi.filter.rules42 import run as rules_run

    flip_out = Path(det_out).parent / 'eval' / 'detect_flip'
    flip_out.mkdir(parents=True, exist_ok=True)
    only = None
    if folders:
        only = {(f.split('_', 1)[0], f.split('_', 1)[1]) for f in folders}
    prev = os.environ.get('AOI_NOMASK')
    os.environ['AOI_NOMASK'] = '1'
    try:
        aoi_detect.run(only=only, nproc=nproc, pre=str(pre), db=str(db), out=str(flip_out), anom=anomaly, flip=True)
    finally:
        if prev is None:
            os.environ.pop('AOI_NOMASK', None)
        else:
            os.environ['AOI_NOMASK'] = prev
        aoi_detect.configure(flip=False)

    R = feats.load(str(det_out), folders)
    F = feats.load(str(flip_out), folders)
    if len(F) == 0:
        return {'n_flip': 0, 'note': 'flip detector returned no candidates'}
    if 'anomaly_frame' in F.columns:
        F = F[~F.anomaly_frame.astype(bool)].copy()
    if len(F) == 0:
        return {'n_flip': 0, 'note': 'flip detector returned no candidates on normal frames'}
    F['id'] = 'F_' + F['id'].astype(str)
    R = R.copy()
    R['is_flip'] = False
    F['is_flip'] = True
    M = pd.concat([R, F], ignore_index=True)
    for col, default in (('raw_rescue', False), ('haze_texture', False), ('large_faint_region', False)):
        if col not in M.columns:
            M[col] = default
    M = feats.enrich(M, str(pre), True)
    n = len(M)
    pers = pd.DataFrame({
        'p_tested': np.zeros(n, bool),
        'p_match': np.zeros(n, bool),
        'p_ctrl': np.zeros(n, np.float32),
        'p_frames': [[] for _ in range(n)],
    })
    o = rules_run(M, pers, glare, {'rule_b': 0})
    o['is_flip'] = M.is_flip.values
    o['folder'] = M.folder.values
    flip = o[o.is_flip]
    n_flip = int(len(flip))
    n_pass = int((flip.decision == 'real').sum()) if n_flip else 0
    n_pass_nr = int(((flip.decision == 'real') & ~flip.review_required).sum()) if n_flip else 0

    def prep(df, contrast, zone, tier, kind_from_folder=True):
        d = pd.DataFrame({
            'kind': np.where(df.folder.astype(str).str.startswith('Side'), 'Side', 'Edge') if kind_from_folder else df.kind,
            'zb': _zone_bucket(zone),
            'ab': pd.cut(np.abs(contrast), [0, 12, 20, 40, 1e9], right=False, labels=['<12', '12-20', '20-40', '40+']).astype(str),
            'tr': np.asarray(tier).astype(str),
        })
        return d

    prod = decisions.copy()
    if 'anomaly_frame' in prod.columns:
        prod = prod[~prod.anomaly_frame.astype(bool)]
    prod['real'] = prod.decision == 'real'
    prod['nr'] = prod['real'] & ~(prod.review_required.astype(bool) if 'review_required' in prod.columns else prod.review_level.astype(int) > 0)
    tier_col = 'tier_prior' if 'tier_prior' in prod.columns else 'tier'
    P = prep(prod, prod.contrast.values, prod.zone.values, prod[tier_col].values)
    P['real'] = prod.real.values
    P['nr'] = prod.nr.values
    Mf = M.loc[flip.index]
    Ff = prep(flip, Mf.contrast.values, Mf.zone.values, Mf.tier.values)
    K = ['kind', 'zb', 'ab', 'tr']
    strat = []
    Fg = Ff.groupby(K).size().rename('n_flip')
    for col in ('real', 'nr'):
        p = P.groupby(K)[col].mean().rename('p')
        t = pd.concat([Fg, p], axis=1).dropna()
        e = float((t.n_flip * t.p).sum()) if len(t) else 0.0
        base = int(P[col].sum())
        if len(t):
            by_kind = t.groupby(level=0).apply(lambda g: float((g.n_flip * g.p).sum()), include_groups=False)
            by_kind = {k: round(float(v), 1) for k, v in by_kind.to_dict().items()}
        else:
            by_kind = {}
        strat.append(dict(
            column=col,
            estimated_false=round(e),
            denominator=base,
            pct=round(100 * e / base, 3) if base else None,
            by_kind=by_kind,
        ))
    return {
        'n_flip': n_flip,
        'flip_real': n_pass,
        'flip_real_pct': round(100 * n_pass / n_flip, 2) if n_flip else None,
        'flip_real_no_review': n_pass_nr,
        'flip_real_no_review_pct': round(100 * n_pass_nr / n_flip, 2) if n_flip else None,
        'false_share_pct': round(100 * n_pass / max(int((~o.is_flip & (o.decision == 'real')).sum()), 1), 3),
        'stratified': strat,
        'note': 'Flip candidates are featurized together with the real candidates, then judged by rules v4.2 with empty persistence. '
                'Side T2 rules (a) and (b) therefore do not fire; the pass rate is an upper bound on that term, same limitation as the historical union run.',
    }


def run_eval(db, pre, det_out, filter_out, work, out_dir, anomaly, nproc, folders, glare):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from grok_aoi.report.make_csv import discover_folders, _mode
    folders = folders or discover_folders(filter_out)
    parts = []
    for f in folders:
        p = Path(filter_out) / _mode(f) / f / 'decisions.csv'
        if p.is_file() and p.stat().st_size:
            d = pd.read_csv(p, dtype={'frame': str})
            if len(d):
                parts.append(d.assign(folder=f))
    decisions = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    print('eval: reference scan', flush=True)
    ret = retention(db, pre, decisions, anomaly)
    print('eval: reference dots', ret.get('n_ref'), flush=True)
    print('eval: polarity-flip detector', flush=True)
    flip = flip_estimate(db, pre, det_out, decisions, glare, anomaly, nproc, folders) if len(decisions) else {'n_flip': 0}
    metrics = {'retention': ret, 'flip': flip}
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print('eval: wrote', out_dir / 'metrics.json', flush=True)
    return metrics
