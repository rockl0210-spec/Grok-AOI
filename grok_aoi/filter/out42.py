"""Write per-folder decisions / defects / summary and one preview overlay per folder."""
import collections, datetime, json, os
import cv2
import numpy as np
import pandas as pd
from grok_aoi.filter.rules42 import TH

COL = dict(particle=(0, 200, 0), stain=(0, 140, 255), scratch=(255, 100, 0))
NCOL = dict(texture=(160, 160, 160), noise=(110, 110, 110), edge_artifact=(255, 0, 255),
            reflection=(0, 255, 255), water_mark=(255, 200, 0), oil_film=(0, 215, 255))

def _mode(f):
    return 'edge' if f.startswith('Edge') else 'side'

def write_outputs(work, out, db, det_label='v4.1', pre_label='v3', folders=None, exemplar=None):
    work, out, db = str(work), str(out), str(db)
    exemplar = exemplar or {}
    D = pd.read_pickle(os.path.join(work, 'D_final.pkl'))
    DEF = pd.read_pickle(os.path.join(work, 'DEF_final.pkl'))
    if folders is None:
        folders = list(dict.fromkeys(D.folder.tolist())) if len(D) else []
    D = D.sort_values(['folder', 'fr', 'id'], kind='stable') if len(D) else D
    summ_all = {}
    for f in folders:
        g = D[D.folder == f] if len(D) else D
        mode = _mode(f)
        d = f'{out}/{mode}/{f}'
        os.makedirs(d, exist_ok=True)
        if len(g) == 0:
            pd.DataFrame().to_csv(f'{d}/decisions.csv', index=False)
            json.dump([], open(f'{d}/decisions.json', 'w'))
            pd.DataFrame().to_csv(f'{d}/defects.csv', index=False)
            json.dump([], open(f'{d}/defects.json', 'w'))
            s = dict(folder=f, mode=mode, n_candidates=0, by_decision={}, real_by_class={}, nuisance_by_class={},
                     review_required=dict(total=0, real=0, nuisance=0), unique_defects=0, unique_defects_by_class={}, overlay=None)
            json.dump(s, open(f'{d}/summary.json', 'w'), indent=1)
            summ_all[f] = s
            print(f, 0, None, flush=True)
            continue
        c = pd.DataFrame(dict(
            candidate_id=g.id, image=g.image, frame=g.frame,
            bbox_x=g.bx, bbox_y=g.by, bbox_w=g.bw, bbox_h=g.bh, tier=g.tier, decision=g.decision,
            fine_class=g.fine_class, fine_class_zh=g.fine_class_zh,
            review_required=g.review_required, review_level=g.review_level, raw_rescue=g.raw_rescue,
            haze_texture=g.haze_texture, same_object_group=g.same_object_group, same_object_n=g.same_object_n,
            same_object_n_real=g.same_object_n_real, reason=g.reason,
            persistence_frames=g.persistence_frames.map(lambda l: ';'.join(f'{x:02d}' for x in l)),
            region=g.region, zone=g.zone, rim_zone=g.rim_zone, rim_dist_px=g.rim_dist.round(1),
            glare_row=g.glare_row, straddle_cap=g.straddle_cap, roi_keep_reason=g.roi_keep_reason,
            edge_completed=g.edge_completed, area_px=g.area_px, contrast=g.contrast, cnr=g.cnr, snr_raw=g.snr_raw,
            anomaly_frame=g.anomaly_frame, pers_tested=g.pers_tested, pers_confirmed=g.pers_confirmed,
            defect_id=g.defect_id, det_class=g.det_class, det_path=g.path, consolidation=g.consolidation,
            straddles_roi=g.straddles_roi, tier_prior=g.tier_prior, reason_code=g.reason_codes))
        c.to_csv(f'{d}/decisions.csv', index=False)
        recs = []
        for r in g.itertuples():
            recs.append(dict(
                candidate_id=r.id, image=r.image, frame=r.frame, bbox=[int(r.bx), int(r.by), int(r.bw), int(r.bh)],
                tier=r.tier, decision=r.decision, fine_class=r.fine_class, fine_class_zh=r.fine_class_zh,
                review_required=bool(r.review_required), review_level=int(r.review_level), raw_rescue=bool(r.raw_rescue),
                haze_texture=bool(r.haze_texture), same_object_group=r.same_object_group, same_object_n=int(r.same_object_n),
                same_object_n_real=int(r.same_object_n_real), reason=r.reason, reason_code=r.reason_codes,
                persistence_frames=[f'{x:02d}' for x in r.persistence_frames], region=r.region,
                anomaly_frame=bool(r.anomaly_frame), defect_id=r.defect_id or None, area_px=int(r.area_px),
                contrast=round(float(r.contrast), 2), cnr=round(float(r.cnr), 2), snr_raw=round(float(r.snr_raw), 2)))
        json.dump(recs, open(f'{d}/decisions.json', 'w'), ensure_ascii=False, separators=(',', ':'))
        df = DEF[DEF.defect_id.str.startswith(f + '_D')].copy() if len(DEF) and 'defect_id' in DEF.columns else pd.DataFrame()
        df.to_csv(f'{d}/defects.csv', index=False)
        dj = []
        for r in df.itertuples():
            x = r._asdict(); x.pop('Index', None)
            x['bbox'] = [x.pop('bbox_x'), x.pop('bbox_y'), x.pop('bbox_w'), x.pop('bbox_h')]
            x['frames'] = x['frames'].split(';'); x['candidate_ids'] = x['candidate_ids'].split(';')
            for k, v in list(x.items()):
                if isinstance(v, (np.bool_,)): x[k] = bool(v)
                elif isinstance(v, (np.integer,)): x[k] = int(v)
                elif isinstance(v, (np.floating,)): x[k] = float(v)
            dj.append(x)
        json.dump(dj, open(f'{d}/defects.json', 'w'), ensure_ascii=False, indent=0)
        real = g[g.decision == 'real']; nu = g[g.decision == 'nuisance']
        an = g[g.anomaly_frame]
        s = dict(
            folder=f, mode=mode, inputs=dict(detection=det_label, preprocess=pre_label + ' (nuisance masks + ROI masks)'),
            n_candidates=int(len(g)), by_decision=g.decision.value_counts().to_dict(),
            real_by_class=real.fine_class.value_counts().to_dict(), nuisance_by_class=nu.fine_class.value_counts().to_dict(),
            review_required=dict(total=int(g.review_required.sum()), real=int(real.review_required.sum()), nuisance=int(nu.review_required.sum())),
            review_level=dict(L1=int((g.review_level == 1).sum()), L2=int((g.review_level == 2).sum()),
                              real_L1=int((real.review_level == 1).sum()), real_L2=int((real.review_level == 2).sum()),
                              nuisance_L1=int((nu.review_level == 1).sum()), nuisance_L2=int((nu.review_level == 2).sum())),
            raw_rescue=dict(n=int(g.raw_rescue.sum()), real=int((g.raw_rescue & (g.decision == 'real')).sum())),
            haze_texture=dict(n=int(g.haze_texture.sum()), real=int((g.haze_texture & (g.decision == 'real')).sum())),
            by_tier=g.groupby(['tier', 'decision']).size().unstack(fill_value=0).to_dict('index'),
            by_region=g.groupby(['region', 'decision']).size().unstack(fill_value=0).to_dict('index'),
            by_reason={k: int(v) for k, v in g.reason_codes.str.split(';').str[0].value_counts().items()},
            real_high_confidence_no_review=int((~real.review_required).sum()),
            unique_defects=int(len(df)), unique_defects_by_class=df.fine_class.value_counts().to_dict() if len(df) else {},
            unique_defects_review=int(df.review_required.sum()) if len(df) else 0,
            multi_frame_defects=int((df.n_frames > 1).sum()) if len(df) else 0,
            anomaly_frames=dict(n_candidates=int(len(an)), real=int((an.decision == 'real').sum()), nuisance=int((an.decision == 'nuisance').sum()),
                                would_be_real_unverified=int(an.anom_would_real.sum()), confirmed_in_normal_frame=int((an.decision == 'real').sum())),
            persistence=dict(tested=int(g.pers_tested.sum()), confirmed=int(g.pers_confirmed.sum())),
            straddlers_high_contrast=dict(n=int(((g.contrast_tier == 'high') & g.straddles_roi).sum()),
                                          real=int(((g.contrast_tier == 'high') & g.straddles_roi & (g.decision == 'real')).sum()),
                                          all_review_if_real=True),
            regrown_merged_n_ge3=dict(n=int((g.merged_n >= 3).sum()), real=int(((g.merged_n >= 3) & (g.decision == 'real')).sum())),
            thresholds=TH)
        json.dump(s, open(f'{d}/summary.json', 'w'), ensure_ascii=False, indent=1, default=str)
        src, ang = f.split('_', 1)
        cnt = real[~real.anomaly_frame].groupby('fr').size() if len(real) else pd.Series(dtype=int)
        fr = None
        if f in exemplar or str(f) in exemplar:
            want = int(exemplar.get(f, exemplar.get(str(f))))
            if os.path.isfile(f'{db}/{src}/{ang}/{want:02d}.png'):
                fr = want
        if fr is None:
            if len(cnt) == 0:
                fr = int(g.fr.min())
            else:
                cand = cnt[(cnt >= 30) & (cnt <= 700)]
                if len(cand):
                    fr = int(cand.index[np.argmax([real[real.fr == k].fine_class.nunique() * 1000 + len(real[real.fr == k]) for k in cand.index])])
                else:
                    fr = int(cnt.idxmax())
        im = cv2.imread(f'{db}/{src}/{ang}/{fr:02d}.png', 0)
        if im is None:
            raise FileNotFoundError(f'{db}/{src}/{ang}/{fr:02d}.png')
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        H = im.shape[0]
        gf = g[g.fr == fr]
        for typ, gg in gf[gf.decision == 'nuisance'].groupby('fine_class'):
            if len(gg) > 40:
                gg = gg.sample(40, random_state=0)
            for r in gg.itertuples():
                col = NCOL.get(typ, (128, 128, 128)); x, y, w, h = int(r.bx), int(r.by), int(r.bw), int(r.bh)
                L = max(2, min(w, h) // 3)
                for (px, py, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
                    cv2.line(im, (px, py), (px + dx * L, py), col, 1)
                    cv2.line(im, (px, py), (px, py + dy * L), col, 1)
        for r in gf[gf.decision == 'real'].itertuples():
            col = COL[r.fine_class]; x, y, w, h = int(r.bx), int(r.by), int(r.bw), int(r.bh)
            cv2.rectangle(im, (x - 1, y - 1), (x + w + 1, y + h + 1), col, 2 if r.review_required else 1)
            cv2.putText(im, {'particle': 'P', 'stain': 'S', 'scratch': 'K'}[r.fine_class] + ('!' if r.review_level == 2 else '?' if r.review_required else ''),
                        (x, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        lg = ['real: green=particle(P) orange=stain(S) blue=scratch(K) thick: "?"=review L1, "!"=review L2 (priority)',
              'nuisance (corner ticks, <=40 per type): grey=texture/noise magenta=edge_artifact yellow=reflection cyan=water/oil']
        cv2.rectangle(im, (0, 0), (min(1200, im.shape[1]), 44), (0, 0, 0), -1)
        for i, t in enumerate(lg):
            cv2.putText(im, t, (6, 16 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(im, f'{f} frame {fr:02d}  real={int((gf.decision == "real").sum())} nuisance={int((gf.decision == "nuisance").sum())}',
                    (6, max(20, H - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        ov = f'{d}/overlay_{f}_{fr:02d}.png'
        cv2.imwrite(ov, im)
        s['overlay'] = ov
        json.dump(s, open(f'{d}/summary.json', 'w'), ensure_ascii=False, indent=1, default=str)
        summ_all[f] = s
        print(f, len(g), fr, flush=True)
    json.dump(summ_all, open(os.path.join(work, 'summ_all.json'), 'w'), default=str)
    return summ_all
