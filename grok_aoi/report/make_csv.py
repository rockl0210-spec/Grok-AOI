"""Final CSV deliverables from filter decisions."""
import os
import pandas as pd

def _mode(f):
    return 'edge' if str(f).startswith('Edge') else 'side'

def discover_folders(filter_out):
    found = []
    for mode in ('edge', 'side'):
        root = os.path.join(filter_out, mode)
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if os.path.isfile(os.path.join(root, name, 'decisions.csv')):
                found.append(name)
    return found

def write_csvs(filter_out, out_dir, folders=None):
    os.makedirs(out_dir, exist_ok=True)
    if folders is None:
        folders = discover_folders(filter_out)
    frames = []
    defs = []
    for f in folders:
        dp = f'{filter_out}/{_mode(f)}/{f}/decisions.csv'
        if not os.path.isfile(dp) or os.path.getsize(dp) == 0:
            continue
        d = pd.read_csv(dp, dtype={'frame': str})
        if len(d) == 0:
            continue
        frames.append(d.assign(folder=f))
        fp = f'{filter_out}/{_mode(f)}/{f}/defects.csv'
        if os.path.isfile(fp) and os.path.getsize(fp) > 0:
            u = pd.read_csv(fp)
            if len(u):
                defs.append(u.assign(folder=f))
    if not frames:
        pd.DataFrame().to_csv(f'{out_dir}/final_defects.csv', index=False)
        pd.DataFrame().to_csv(f'{out_dir}/final_defects_unique.csv', index=False)
        pd.DataFrame().to_csv(f'{out_dir}/defects_per_frame.csv', index=False)
        print('final_defects', 0)
        return
    d = pd.concat(frames, ignore_index=True)
    r = d[d.decision == 'real'].copy()
    r['class'] = 'defect'
    r['review_label'] = r.review_level.map({0: 'none', 1: 'L1', 2: 'L2'})
    cols = ['candidate_id', 'defect_id', 'folder', 'frame', 'image', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
            'area_px', 'contrast', 'cnr', 'snr_raw', 'class', 'fine_class', 'fine_class_zh', 'review_required',
            'review_level', 'review_label', 'tier_prior', 'zone', 'reason_code', 'reason', 'same_object_group',
            'same_object_n', 'same_object_n_real', 'anomaly_frame', 'pers_tested', 'pers_confirmed']
    x = r[cols].rename(columns={'fine_class': 'subclass', 'fine_class_zh': 'subclass_zh', 'tier_prior': 'tier'})
    x = x.sort_values(['folder', 'frame', 'candidate_id'])
    x.to_csv(f'{out_dir}/final_defects.csv', index=False)
    print('final_defects', len(x))
    if defs:
        u = pd.concat(defs, ignore_index=True)
        u['review_label'] = u.review_level.map({0: 'none', 1: 'L1', 2: 'L2'})
    else:
        u = pd.DataFrame()
    u.to_csv(f'{out_dir}/final_defects_unique.csv', index=False)
    print('unique', len(u))
    rows = []
    for (f, fr), g in d.groupby(['folder', 'frame']):
        rg = g[g.decision == 'real']
        row = dict(folder=f, frame=fr, anomaly_frame=bool(g.anomaly_frame.any()), candidates=len(g),
                   real=len(rg), nuisance=len(g) - len(rg),
                   real_no_review=int((rg.review_level == 0).sum()),
                   real_L1=int((rg.review_level == 1).sum()), real_L2=int((rg.review_level == 2).sum()),
                   nuisance_review_L1=int(((g.decision != 'real') & (g.review_level == 1)).sum()),
                   nuisance_review_L2=int(((g.decision != 'real') & (g.review_level == 2)).sum()))
        for s in ['particle', 'stain', 'scratch']:
            a = rg[rg.fine_class == s]
            row[s] = len(a)
            row[f'{s}_L1'] = int((a.review_level == 1).sum())
            row[f'{s}_L2'] = int((a.review_level == 2).sum())
        rows.append(row)
    pf = pd.DataFrame(rows)
    pf.to_csv(f'{out_dir}/defects_per_frame.csv', index=False)
    print(len(pf), int(pf.real.sum()) if len(pf) else 0, pf[['particle', 'stain', 'scratch']].sum().to_dict() if len(pf) else {})
    return pf
