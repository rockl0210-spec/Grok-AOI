"""same_object_group: connected components of candidate boxes within one (folder, frame).
Link two candidates if (1) boxes, each dilated by D=2 px, intersect (overlapping / containing / near-touching) and (2) same polarity (sign of contrast).
Boxes with area > BIGMAX (frame-scale haze/stain boxes) never link (they would chain unrelated dots).  Singletons get their own group id.
Columns: same_object_group (id '<folder>_<fr>_g<k>'), same_object_n (members), same_object_n_real (real members)."""
import sys, time
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def groups(D, dilate=1, bigmax=250000):
    gid = np.empty(len(D), dtype=object); pos = np.arange(len(D))
    D = D.reset_index(drop=True)
    for (f, fr), g in D.groupby(['folder', 'fr'], sort=False):
        ix = g.index.values; n = len(ix)
        x0 = g.bx.values - dilate; y0 = g.by.values - dilate
        x1 = g.bx.values + g.bw.values + dilate; y1 = g.by.values + g.bh.values + dilate
        sg = np.sign(g.contrast.values); ok = (g.bw.values * g.bh.values) <= bigmax
        o = np.argsort(x0); X0 = x0[o]; X1 = x1[o]; Y0 = y0[o]; Y1 = y1[o]; S = sg[o]; K = ok[o]
        rows = []; cols = []
        for i in range(n):
            if not K[i]:
                continue
            jmax = np.searchsorted(X0, X1[i], side='right')
            if jmax <= i + 1:
                continue
            j = np.arange(i + 1, jmax)
            m = K[j] & (S[j] == S[i]) & (X1[j] >= X0[i]) & (Y0[j] <= Y1[i]) & (Y1[j] >= Y0[i])
            if m.any():
                jj = j[m]; rows.append(np.full(len(jj), i)); cols.append(jj)
        if rows:
            r = np.concatenate(rows); c = np.concatenate(cols)
            nc, lab = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)), directed=False)
        else:
            nc, lab = n, np.arange(n)
        lab_orig = np.empty(n, int); lab_orig[o] = lab
        gid[ix] = [f'{f}_{int(fr):02d}_g{k:05d}' for k in lab_orig]
    return gid

def apply(path, bigmax=250000, dilate=1, inplace=True):
    D = pd.read_pickle(path)
    t = time.time()
    if len(D) == 0:
        D['same_object_group'] = []
        D['same_object_n'] = []
        D['same_object_n_real'] = []
        D.to_pickle(path if inplace else path.replace('.pkl', '_g.pkl'))
        print('time', round(time.time() - t), 'groups', 0)
        return D
    D['same_object_group'] = groups(D, dilate=dilate, bigmax=bigmax)
    s = D.groupby('same_object_group').size()
    D['same_object_n'] = D.same_object_group.map(s).astype(int)
    real_n = D[D.decision == 'real'].groupby('same_object_group').size()
    D['same_object_n_real'] = D.same_object_group.map(real_n).fillna(0).astype(int)
    print('time', round(time.time() - t), 'groups', len(s), 'multi', int((s > 1).sum()),
          'cands in multi', int((D.same_object_n > 1).sum()), 'max', int(s.max()),
          s[s > 1].describe().round(1).to_dict() if (s > 1).any() else {})
    D.to_pickle(path if inplace else path.replace('.pkl', '_g.pkl'))
    return D

if __name__ == '__main__':
    big = int(sys.argv[2]) if len(sys.argv) > 2 else 250000
    apply(sys.argv[1], bigmax=big, inplace=not (len(sys.argv) > 3))
