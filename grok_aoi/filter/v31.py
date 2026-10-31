"""Spatial status of each Side candidate against a legacy filter-v3.1 decisions tree.

status: real (a v3.1 real candidate covers it), nuis (only v3.1 nuisance covers),
bigreal/bignuis (only oversize v3.1 boxes cover), none.

`legacy_root` is the v3.1 filter output: either `<root>/side/<Folder>/decisions.csv`
or `<root>/<Folder>/decisions.csv`. This module does not read any fixed machine path.
"""
import os
import numpy as np
import pandas as pd

def _decisions_path(legacy_root, folder):
    cands = [
        os.path.join(legacy_root, 'side', folder, 'decisions.csv'),
        os.path.join(legacy_root, folder, 'decisions.csv'),
        os.path.join(legacy_root, 'filter', 'side', folder, 'decisions.csv'),
    ]
    for p in cands:
        if os.path.isfile(p):
            return p
    return None

def build_st31(work_dir, legacy_root):
    M = pd.read_pickle(os.path.join(work_dir, 'M.pkl'))
    S = M[M.folder.str.startswith('Side')] if len(M) else M
    out = pd.Series('none', index=M.index, dtype=object)
    for f, g0 in S.groupby('folder'):
        path = _decisions_path(legacy_root, f)
        if path is None:
            print('v31: no decisions for', f, 'under', legacy_root, flush=True)
            continue
        q = pd.read_csv(path, usecols=['frame', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'area_px', 'decision'])
        q['fr'] = q.frame.astype(int)
        qg = {k: g for k, g in q.groupby('fr')}
        for fr, g in g0.groupby('fr'):
            c = qg.get(fr)
            if c is None:
                continue
            bx, by, bw, bh = [c[k].values for k in ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']]
            qa = c.area_px.values
            qr = (c.decision == 'real').values
            x = g.cx.values[:, None]; y = g.cy.values[:, None]; a = g.area_px.values[:, None]
            H = (x >= bx - 2) & (x <= bx + bw + 2) & (y >= by - 2) & (y <= by + bh + 2)
            Hs = H & (qa <= 30 * a)
            s = np.where((Hs & qr).any(1), 'real',
                         np.where((Hs & ~qr).any(1), 'nuis',
                                  np.where((H & ~Hs & qr).any(1), 'bigreal',
                                           np.where((H & ~Hs).any(1), 'bignuis', 'none'))))
            out.loc[g.index] = s
    out.to_frame('st31').assign(id=M.id if len(M) else []).to_pickle(os.path.join(work_dir, 'v31status.pkl'))
    if len(S):
        print(out[S.index].value_counts().to_dict())
    return out
