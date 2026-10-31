"""Feature table for filter v4.2. Writes <work>/M.pkl. Paths are arguments, not the process cwd."""
import os, sys, time
import pandas as pd

def build_features(detect_root, pre_root, work_dir, recompute=True, folders=None):
    from grok_aoi.filter import feats
    os.makedirs(work_dir, exist_ok=True)
    t = time.time()
    M = feats.build(detect_root, pre_root, recompute, folders=folders)
    M.to_pickle(os.path.join(work_dir, 'M.pkl'))
    print('features', len(M), round(time.time() - t), 's', flush=True)
    if len(M) and 'zone' in M.columns:
        print(M.zone.value_counts().to_dict())
    return M

if __name__ == '__main__':
    det, pre, rec = sys.argv[1], sys.argv[2], sys.argv[3] == '1'
    work = sys.argv[4] if len(sys.argv) > 4 else '.'
    build_features(det, pre, work, rec)
