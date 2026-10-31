"""Oil-film candidates with |contrast| >= 20, for review. Writes the CSV to an explicit path."""
import os
import pandas as pd

def write_oil_list(work_dir, out_csv):
    D = pd.read_pickle(os.path.join(work_dir, 'D_final.pkl'))
    if len(D) == 0:
        pd.DataFrame(columns=['ID', 'frame', 'bbox', '|c|', 'cnr', 'area', 'zone', 'reason_code', 'folder']).to_csv(out_csv, index=False)
        return
    x = D[(D.decision == 'nuisance') & (D.fine_class == 'oil_film') & (D.contrast.abs() >= 20)].copy()
    code = x.reason_code if 'reason_code' in x.columns else x.reason_codes
    o = pd.DataFrame({
        'ID': x.id, 'frame': x.fr.astype(int),
        'bbox': x.bx.astype(int).astype(str) + ',' + x.by.astype(int).astype(str) + ',' + x.bw.astype(int).astype(str) + ',' + x.bh.astype(int).astype(str),
        '|c|': x.contrast.abs().round(2), 'cnr': x.cnr.abs().round(2), 'area': x.area_px.astype(int),
        'zone': x.zone, 'reason_code': code, 'folder': x.folder,
    })
    o = o.sort_values('ID', kind='stable')
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    o.to_csv(out_csv, index=False)
    print('oil_film |c|>=20', len(o), out_csv)
