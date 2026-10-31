"""Per-folder 2-frame persistence. Writes <work>/persist_<folder>.pkl and reg_<folder>.json."""
import json, os, sys, time
import pandas as pd

def run_persistence(folder, work_dir, db, pre):
    from grok_aoi.filter import persist
    persist.configure(db, pre)
    M = pd.read_pickle(os.path.join(work_dir, 'M.pkl'))
    t = time.time()
    log_path = os.path.join(work_dir, f'persist_{folder}.log')
    with open(log_path, 'w') as L:
        def log(s, L=L):
            L.write(s + '\n'); L.flush()
        res, by = persist.run_folder(folder, M, log)
        json.dump(res, open(os.path.join(work_dir, f'reg_{folder}.json'), 'w'), indent=0)
        out = persist.match_all(folder, M, res, by)
    out.to_pickle(os.path.join(work_dir, f'persist_{folder}.pkl'))
    if len(out) and out.p_tested.any():
        print(folder, 'tested', round(float(out.p_tested.mean()), 3),
              'match|tested', round(float(out.loc[out.p_tested, 'p_match'].mean()), 3),
              'ctrl', round(float(out.loc[out.p_tested, 'p_ctrl'].mean()), 3),
              round(time.time() - t), 's', flush=True)
    else:
        print(folder, 'persist rows', len(out), round(time.time() - t), 's', flush=True)
    return out

if __name__ == '__main__':
    folder, work = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '.'
    run_persistence(folder, work, os.environ['GROK_AOI_DB'], os.environ['GROK_AOI_PRE'])
