"""Per-frame annotation JPGs.

Colors (BGR): particle green, stain orange, scratch blue.
review level 0 solid, L1 dashed, L2 thick + white halo.
ROI outline = magenta (preprocess *_mask.png). Boxes smaller than 6 px are drawn at 6 px.
"""
import os
import cv2
import numpy as np
import pandas as pd
from grok_aoi.report.make_csv import discover_folders, _mode

F42 = None
PRE = None
DB = None
OUT = None
COL = {'particle': (60, 220, 60), 'stain': (0, 150, 255), 'scratch': (255, 140, 30)}
ROI = (255, 0, 255)
S = 0.5

def dashed(img, p0, p1, c, t, dl=4):
    x0, y0 = p0; x1, y1 = p1
    for x in range(x0, x1, 2 * dl):
        e = min(x + dl, x1); cv2.line(img, (x, y0), (e, y0), c, t); cv2.line(img, (x, y1), (e, y1), c, t)
    for y in range(y0, y1, 2 * dl):
        e = min(y + dl, y1); cv2.line(img, (x0, y), (x0, e), c, t); cv2.line(img, (x1, y), (x1, e), c, t)

def box(r):
    x0 = r.bbox_x * S; y0 = r.bbox_y * S; x1 = (r.bbox_x + r.bbox_w) * S; y1 = (r.bbox_y + r.bbox_h) * S
    if x1 - x0 < 6:
        m = (x0 + x1) / 2; x0, x1 = m - 3, m + 3
    if y1 - y0 < 6:
        m = (y0 + y1) / 2; y0, y1 = m - 3, m + 3
    return (int(x0), int(y0)), (int(x1), int(y1))

def do(args):
    f, fr, g, db, pre, out = args
    global DB, PRE, OUT
    DB, PRE, OUT = db, pre, out
    src, ang = f.split('_', 1)
    p = f'{DB}/{src}/{ang}/{fr}.png'
    a = cv2.imread(p, 0)
    if a is None:
        raise FileNotFoundError(p)
    img = cv2.cvtColor(cv2.resize(a, None, fx=S, fy=S, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
    mk = cv2.imread(f'{PRE}/{src}/{ang}/{fr}_mask.png', 0)
    if mk is not None:
        m = cv2.resize((mk > 0).astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, ROI, 1)
    if len(g):
        g = g.assign(o=g.review_level.values)
    else:
        g = g.copy()
        g['o'] = []
    for lvl in (0, 1, 2):
        sub = g[g.o == lvl] if len(g) else g
        for r in sub.itertuples():
            p0, p1 = box(r)
            c = COL.get(r.fine_class, (200, 200, 200))
            if lvl == 0:
                cv2.rectangle(img, p0, p1, c, 1)
            elif lvl == 1:
                dashed(img, p0, p1, c, 1)
            else:
                cv2.rectangle(img, (p0[0] - 2, p0[1] - 2), (p1[0] + 2, p1[1] + 2), (255, 255, 255), 1)
                cv2.rectangle(img, p0, p1, c, 3)
    n = len(g)
    cn = {k: int((g.fine_class == k).sum()) if len(g) else 0 for k in COL}
    hb = 40
    bar = np.zeros((hb, img.shape[1], 3), np.uint8)
    anom = bool(g.anomaly_frame.any()) if len(g) and 'anomaly_frame' in g.columns else False
    t = f'{f} frame {fr}{"  [anomaly frame]" if anom else ""}   real={n}  particle={cn["particle"]} stain={cn["stain"]} scratch={cn["scratch"]}   review L1={int((g.o == 1).sum()) if len(g) else 0} L2={int((g.o == 2).sum()) if len(g) else 0}   (filter v4.2)'
    cv2.putText(bar, t, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    x = 6
    for k, lab in [('particle', 'particle'), ('stain', 'stain'), ('scratch', 'scratch')]:
        cv2.rectangle(bar, (x, 24), (x + 14, 34), COL[k], -1)
        cv2.putText(bar, lab, (x + 18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
        x += 95
    dashed(bar, (x, 24), (x + 22, 34), (200, 200, 200), 1, 3)
    cv2.putText(bar, 'review L1 (dashed)', (x + 28, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
    x += 150
    cv2.rectangle(bar, (x, 24), (x + 22, 34), (255, 255, 255), 1)
    cv2.rectangle(bar, (x + 2, 26), (x + 20, 32), (200, 200, 200), 3)
    cv2.putText(bar, 'review L2 (thick+halo)', (x + 28, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
    x += 175
    cv2.line(bar, (x, 29), (x + 22, 29), ROI, 1)
    cv2.putText(bar, 'ROI boundary', (x + 28, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
    x += 110
    cv2.putText(bar, 'solid=no review; boxes <6px drawn 6px', (x, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
    img = np.vstack([bar, img])
    ddir = f'{OUT}/overlays/{f}'
    os.makedirs(ddir, exist_ok=True)
    cv2.imwrite(f'{ddir}/{fr}.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return f, fr, n

def write_overlays(filter_out, pre, db, out_dir, nproc=4, folders=None):
    global F42, PRE, DB, OUT
    F42, PRE, DB, OUT = str(filter_out), str(pre), str(db), str(out_dir)
    if folders is None:
        folders = discover_folders(F42)
    parts = []
    for f in folders:
        dp = f'{F42}/{_mode(f)}/{f}/decisions.csv'
        if not os.path.isfile(dp) or os.path.getsize(dp) == 0:
            continue
        d = pd.read_csv(dp, dtype={'frame': str})
        if len(d) == 0:
            continue
        parts.append(d.assign(folder=f))
    if not parts:
        print('overlays: no decisions')
        return []
    d = pd.concat(parts, ignore_index=True)
    r = d[d.decision == 'real']
    jobs = []
    for f in folders:
        frs = sorted(d[d.folder == f].frame.unique()) if f in set(d.folder) else []
        gr = {k: g for k, g in r[r.folder == f].groupby('frame')} if len(r) else {}
        for fr in frs:
            jobs.append((f, fr, gr.get(fr, r.iloc[0:0]), DB, PRE, OUT))
    print(len(jobs), 'frames', flush=True)
    if not jobs:
        return []
    nproc = max(1, min(int(nproc), len(jobs)))
    if nproc == 1:
        res = [do(j) for j in jobs]
    else:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(nproc) as p:
            res = p.map(do, jobs, chunksize=1)
    print('overlays done', len(res), flush=True)
    for f in folders:
        frs = sorted(d[d.folder == f].frame.unique()) if f in set(d.folder) else []
        if not frs:
            continue
        ncol = 8 if f.startswith('Edge') else 4
        tw = 300 if f.startswith('Edge') else 600
        tiles = []
        for fr in frs:
            im = cv2.imread(f'{OUT}/overlays/{f}/{fr}.jpg')
            if im is None:
                continue
            im = cv2.resize(im, (tw, int(im.shape[0] * tw / im.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.rectangle(im, (0, 0), (86, 16), (0, 0, 0), -1)
            cv2.putText(im, f'f{fr} n={len(r[(r.folder == f) & (r.frame == fr)])}', (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(im)
        if not tiles:
            continue
        th = max(t.shape[0] for t in tiles)
        nr = (len(tiles) + ncol - 1) // ncol
        Sh = np.zeros((nr * th + 28, ncol * tw, 3), np.uint8)
        for i, t in enumerate(tiles):
            Sh[28 + (i // ncol) * th:28 + (i // ncol) * th + t.shape[0], (i % ncol) * tw:(i % ncol + 1) * tw] = t
        cv2.putText(Sh, f'{f}: filter v4.2 real boxes (green particle / orange stain / blue scratch; dashed = review L1; thick = L2); n = real boxes in frame',
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        os.makedirs(f'{OUT}/overlays/{f}', exist_ok=True)
        cv2.imwrite(f'{OUT}/overlays/{f}/contact_sheet.jpg', Sh, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print('contact done')
    return res
