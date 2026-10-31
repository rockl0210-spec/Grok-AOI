"""Run-level summary.json. Candidate count is computed from this run; it is not asserted against a constant."""
import datetime, json, os
import numpy as np
import pandas as pd
from grok_aoi.filter.rules42 import TH

def _mode(f):
    return 'edge' if str(f).startswith('Edge') else 'side'

def write_summary(work, out, inputs, folders=None, highlight_ids=None, stain_id=None, rule_b=False, legacy_path=None):
    work, out = str(work), str(out)
    D = pd.read_pickle(os.path.join(work, 'D_final.pkl'))
    S = json.load(open(os.path.join(work, 'summ_all.json')))
    m = pd.read_pickle(os.path.join(work, 'M.pkl'))
    if folders is None:
        folders = list(S.keys())
    highlight_ids = list(highlight_ids or [])
    ids = []
    for f in folders:
        mode = _mode(f)
        cpath = f'{out}/{mode}/{f}/decisions.csv'
        if not os.path.isfile(cpath) or os.path.getsize(cpath) == 0:
            continue
        c = pd.read_csv(cpath, usecols=['candidate_id', 'decision', 'fine_class', 'review_required'])
        if len(c) == 0:
            j = json.load(open(f'{out}/{mode}/{f}/decisions.json'))
            assert len(c) == len(j)
            continue
        j = json.load(open(f'{out}/{mode}/{f}/decisions.json'))
        assert len(c) == len(j)
        assert set(c.candidate_id) == set(x['candidate_id'] for x in j)
        ids += list(c.candidate_id)
    if len(m) == 0 and len(ids) == 0:
        top = dict(created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'), inputs=inputs,
                   n_candidates=0, note='no candidates')
        json.dump(top, open(f'{out}/summary.json', 'w'), ensure_ascii=False, indent=1)
        print('ID check OK', 0)
        return top
    assert len(ids) == len(set(ids)) == len(m) and set(ids) == set(m.id), 'ID mismatch'
    print('ID check OK', len(ids))
    if os.environ.get('NEXP') and int(os.environ['NEXP']) != len(ids):
        print('note: NEXP env', os.environ['NEXP'], 'differs from this run', len(ids), '(not asserted)')
    tot = D.decision.value_counts().to_dict()
    real = D[D.decision == 'real']; nu = D[D.decision == 'nuisance']
    an = D[D.anomaly_frame]
    st = D[(D.contrast_tier == 'high') & D.straddles_roi]; rg = D[D.merged_n >= 3]
    DEF = pd.read_pickle(os.path.join(work, 'DEF_final.pkl'))
    rule_note = ('rule (b) enabled; st31 from legacy v3.1 decisions at %s' % legacy_path) if rule_b else (
        'rule (b) off by default: Side T2 conversion uses persistence rule (a) only. '
        'On the original full database about 205 of 541 Side conversions came from rule (b); '
        'pass --legacy-v31-decisions to enable it.')
    def _records(ids_):
        if not ids_ or len(D) == 0:
            return {}
        cols = [c for c in ['id', 'decision', 'fine_class', 'review_required', 'review_level', 'reason_codes'] if c in D.columns]
        return {i: D[D.id == i][cols].to_dict('records') for i in ids_}
    e1 = D.folder.str.startswith('Edge1') if len(D) else pd.Series(dtype=bool)
    pathC = D[(D.path == 'C') & e1 & D.contrast_tier.isin(['gray_mid', 'high']) & (D.decision == 'real')] if len(D) else D
    top = dict(
        created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        inputs=inputs,
        n_candidates=int(len(D)), n_unique_ids=int(D.id.nunique()) if len(D) else 0, by_decision=tot,
        real_by_class=real.fine_class.value_counts().to_dict() if len(real) else {},
        nuisance_by_class=nu.fine_class.value_counts().to_dict() if len(nu) else {},
        review_required=dict(total=int(D.review_required.sum()), real=int(real.review_required.sum()) if len(real) else 0,
                             nuisance=int(nu.review_required.sum()) if len(nu) else 0),
        real_no_review=int((~real.review_required).sum()) if len(real) else 0,
        review_level=dict(L1=int((D.review_level == 1).sum()), L2=int((D.review_level == 2).sum()),
                          real_L1=int((real.review_level == 1).sum()) if len(real) else 0,
                          real_L2=int((real.review_level == 2).sum()) if len(real) else 0,
                          nuisance_L1=int((nu.review_level == 1).sum()) if len(nu) else 0,
                          nuisance_L2=int((nu.review_level == 2).sum()) if len(nu) else 0, note='0 = no review'),
        raw_rescue=dict(n=int(D.raw_rescue.sum()) if len(D) else 0,
                        real=int((D.raw_rescue & (D.decision == 'real')).sum()) if len(D) else 0,
                        all_review=bool(D[D.raw_rescue].review_required.all()) if len(D) and D.raw_rescue.any() else True,
                        L2=int((D.raw_rescue & (D.review_level == 2)).sum()) if len(D) else 0),
        haze_texture=dict(n=int(D.haze_texture.sum()) if len(D) else 0,
                          real=int((D.haze_texture & (D.decision == 'real')).sum()) if len(D) else 0,
                          L2=int((D.haze_texture & (D.review_level == 2)).sum()) if len(D) else 0),
        side_T2_conversion=dict(total=int((D.t2_conv != '').sum()) if len(D) else 0,
                                rule_a_measurable_unconfirmed=int((D.t2_conv == 'a').sum()) if len(D) else 0,
                                rule_b_v4new_not_measurable=int((D.t2_conv == 'b').sum()) if len(D) else 0,
                                by_angle={f: int((D[D.folder == f].t2_conv != '').sum()) for f in folders if str(f).startswith('Side')} if len(D) else {},
                                all_review_L2=bool((D[D.t2_conv != ''].review_level == 2).all()) if len(D) and (D.t2_conv != '').any() else True,
                                note=rule_note),
        same_object_group=dict(
            method='per (folder,frame) connected components of boxes dilated by 1 px (<=2 px gap or overlap), same contrast polarity, boxes >250000 px2 excluded',
            groups=int(D.same_object_group.nunique()) if len(D) else 0,
            multi_member_groups=int((D.groupby('same_object_group').size() > 1).sum()) if len(D) else 0,
            candidates_in_multi=int((D.same_object_n > 1).sum()) if len(D) else 0,
            real_in_multi=int(((D.same_object_n > 1) & (D.decision == 'real')).sum()) if len(D) else 0,
            max_group=int(D.same_object_n.max()) if len(D) else 0,
            groups_with_real=int(D[D.decision == 'real'].same_object_group.nunique()) if len(real) else 0,
            real_groups_multi_real=int((D[D.decision == 'real'].groupby('same_object_group').size() > 1).sum()) if len(real) else 0),
        edge1_pathC_very_strong=dict(n=int(len(pathC)), all_review=bool(pathC.review_required.all()) if len(pathC) else True),
        special_cases=_records(highlight_ids),
        by_tier=D.groupby(['tier', 'decision']).size().unstack(fill_value=0).to_dict('index') if len(D) else {},
        by_reason={k: int(v) for k, v in D.reason_codes.str.split(';').str[0].value_counts().items()} if len(D) else {},
        unique_defects_total=int(len(DEF)),
        unique_defects_by_class=DEF.fine_class.value_counts().to_dict() if len(DEF) and 'fine_class' in DEF.columns else {},
        unique_defects_no_review=int((~DEF.review_required).sum()) if len(DEF) and 'review_required' in DEF.columns else 0,
        per_angle={f: dict(n=S[f]['n_candidates'], real=S[f]['by_decision'].get('real', 0), nuisance=S[f]['by_decision'].get('nuisance', 0),
                          real_by_class=S[f].get('real_by_class', {}), nuisance_by_class=S[f].get('nuisance_by_class', {}),
                          review_required=S[f].get('review_required', {}), unique_defects=S[f].get('unique_defects', 0),
                          unique_defects_by_class=S[f].get('unique_defects_by_class', {}), overlay=S[f].get('overlay'))
                   for f in folders if f in S},
        anomaly_frames=dict(
            rule='real only if matched (own NCC-registered persistence) in a neighbouring NORMAL frame; otherwise nuisance(noise, N_ANOM_UNVERIFIED), review_required=True',
            n=int(len(an)), real_confirmed=int((an.decision == 'real').sum()) if len(an) else 0,
            nuisance=int((an.decision == 'nuisance').sum()) if len(an) else 0,
            would_pass_strength_but_unverified=int((an.anom_would_real & (an.decision == 'nuisance')).sum()) if len(an) else 0,
            by_group={g: dict(n=int((an.grp == g).sum()) if len(an) else 0,
                              real=int(((an.grp == g) & (an.decision == 'real')).sum()) if len(an) else 0) for g in ['Edge', 'Side']}),
        straddlers_high_contrast=dict(n=int(len(st)), real=int((st.decision == 'real').sum()) if len(st) else 0,
                                      nuisance=int((st.decision == 'nuisance').sum()) if len(st) else 0,
                                      tiers=st.tier.value_counts().to_dict() if len(st) else {},
                                      note='tier capped at T2; every real one review_required'),
        regrown_merged_n_ge3=dict(n=int(len(rg)), real=int((rg.decision == 'real').sum()) if len(rg) else 0,
                                  by_reason=rg.reason_codes.str.split(';').str[0].value_counts().to_dict() if len(rg) else {},
                                  review=int(rg.review_required.sum()) if len(rg) else 0),
        stain_highlight=D[D.id == stain_id][['id', 'decision', 'fine_class', 'review_required', 'review_level', 'reason_codes', 'defect_id']].to_dict('records') if stain_id and len(D) else [],
        thresholds=TH)
    # keep the historical key name when the configured stain id is the reference one; always also under stain_highlight
    top['edge2_090_27_stain'] = top['stain_highlight']
    json.dump(top, open(f'{out}/summary.json', 'w'), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: top[k] for k in ['by_decision', 'real_by_class', 'nuisance_by_class', 'review_required', 'real_no_review', 'unique_defects_total', 'side_T2_conversion']},
                     ensure_ascii=False, default=str, indent=1))
    for f in folders:
        if f not in top['per_angle']:
            continue
        pa = top['per_angle'][f]
        print(f, pa['n'], pa['real'], pa['nuisance'], pa.get('review_required'), pa['unique_defects'], pa['real_by_class'], pa['nuisance_by_class'])
    return top
