"""Filter v4.2 decisions + defect consolidation.

Paths and the optional legacy rule (b) are arguments. Rule (b) is off unless rule_b=True,
in which case <work>/v31status.pkl must already exist.
"""
import json, os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from grok_aoi.filter.rules42 import run, ZH
from grok_aoi.filter.feats import folder_sort_key

TEXT={
'N_PREPROC_STRIP_ARTIFACT':'Side/270 frame 15 (anomaly frame) dark-band speck created by preprocess v3 (check e)',
'N_REGROWN_OVERMERGE':'regrown candidate with merged_n>=3 treated as over-merged texture',
'N_EDGE_ARTIFACT':'weak thin/elongated object on ROI boundary (<=6px) or low-overlap elongated straddler (strong ones are kept, see R_EDGE_STRADDLE_STRONG)',
'N_GLARE_LINE':'thin horizontal object on a Side glare/reflection line',
'N_SPECULAR_WEAK':'specular zone, not strong+compact enough (region-adapted threshold)',
'N_OIL_HAZE_LARGE':'>5000 px low-contrast hazy component in oil/water/ink/rough zone (texture haze; flagged for review, not an unflagged stain)',
'N_OIL_FILM':'oil-film zone, large faint region not locally distinct (oil film)',
'N_WATER_MARK':'water zone, large faint region not locally distinct (water mark)',
'N_UNEVEN_SHADING':'large faint region in uneven-illumination zone (shading)',
'N_CONTAM_TEXTURE':'contaminated zone: dense neighbourhood of similar blobs (texture)',
'N_ROUGH_TEXTURE':'rough zone: weak and not locally distinct (texture)',
'N_OIL_TEXTURE':'oil/ink zone: faint, not locally distinct (oil texture; review)',
'N_WATER_TEXTURE':'water zone: faint, not locally distinct (review)',
'N_CONTAM_BELOW_STRENGTH':'contaminated zone: below region-adapted strength (|c|<12 or tiny)',
'N_DENSE_TEXTURE':'dense neighbourhood of similar blobs, not distinct (texture)',
'N_FIXED_PATTERN':'same x,y (<=1.5px, similar area and |c|) repeats in >=3 other frames (>=5 if |c|>=20): fixed pattern',
'N_BELOW_STRENGTH':'below region-adapted strength thresholds (noise)',
'N_ANOM_UNVERIFIED':'anomaly frame: no confirmation in a normal frame',
'R_EDGE_STRADDLE_STRONG':'strong object on/straddling ROI edge kept as real (review)',
'R_STRADDLE_STRONG':'strong straddling ROI boundary object kept as real (review)',
'R_SPECULAR_STRONG':'strong compact object inside specular zone (kept, review)',
'R_CONTAM_VERYSTRONG':'very strong contrast object in contaminated zone',
'R_CONTAM_STRONG':'contaminated zone: |c|>=40 (never nuisance by zone; review unless locally distinct)',
'R_CONTAM_STAIN':'contaminated zone: large stain-like region (area>=400,|c|>=25,cnr>=5; always review L2)',
'N_SPECULAR_POLARITY':'specular zone: strong object but wrong polarity for the mode (Edge bright / Side dark): reflection, review L2',
'N_SIDE_T2_UNCONFIRMED':'Side tier-2 real converted to nuisance (noise) + review L2: (a) persistence measurable but not confirmed, or (b) new in v4 (not real in v3.1) and persistence not measurable',
'R_CONTAM_SCRATCH':'contaminated zone: long thin line-like object',
'R_CONTAM_MID':'contaminated zone: |c| 20-40 (rough zone >=25) and locally not buried in neighbours (nb_rank>=0.1) (review unless clearly distinct)',
'R_CONTAM_SPARSE':'contaminated zone, isolated (<=3 neighbours within 50px): absolute thresholds cnr>=8,|c|>=8,snr_raw>=20 (nb_rank unreliable), review L2',
'R_FAINT_BROAD':'clean-zone broad faint blob |c| 8-12 with cnr>=8 and raw peak depth>=18, review',
'R_CONTAM_FAINT':'contaminated zone: |c| 12-20, locally distinct from neighbours (kept for review)',
'R_EDGE_NORMAL_T1':'Edge normal/flat zone: tier-1 strength (snr_raw>=12,|c|>=10,area>=5,conf>=0.5) + cnr>=3.5',
'R_EDGE_NORMAL_T2':'Edge normal zone: tier-2 strength + cnr>=6,|c|>=10',
'R_EDGE_UNEVEN_T1':'Edge uneven zone: tier-1 with |c|>=12,cnr>=5',
'R_EDGE_UNEVEN_T2':'Edge uneven zone: tier-2 with cnr>=7,|c|>=10',
'R_SIDE_NORMAL_T1':'Side normal zone: tier-1 (snr_raw>=30,|c|>=10,area>=8) + cnr>=4',
'R_SIDE_NORMAL_T2':'Side normal zone: tier-2 (snr_raw>=16) + cnr>=6,|c|>=8,area>=8',
'R_SIDE_UNEVEN_T1':'Side uneven zone: tier-1 + cnr>=5 and (nb_rank>=0.5 or strong exemption cnr>=10 & snr_raw>=30)',
'R_SIDE_UNEVEN_T2':'Side uneven zone: tier-2 + cnr>=8 and (nb_rank>=0.5 or strong exemption)',
'R_SIDE_STRONG_EXEMPT':'Side isolated strong point (cnr>=10 & snr_raw>=30), kept as real (review)',
'R_FAINT_RELAXED':'clean-zone faint dot |c| 12-16, relaxed strength gate, review',
'+PERS_OK':'confirmed in neighbouring normal frame','+PERS_MISS':'not found in neighbouring frame (kept, review)',
'+ANOM_PERS_OK':'anomaly frame, confirmed in normal frame'}
def reason(r):
    base=r.split('+')[0]; ex=['+'+x for x in r.split('+')[1:]]
    return ';'.join([base]+[e[1:] for e in ex])
def rtext(r):
    parts=r.split('+'); t=TEXT.get(parts[0],'')
    for e in parts[1:]: t+='; '+TEXT.get('+'+e,e)
    return t

def build(work, db, width=2464, height=2056, rule_b=False, folders=None):
    work=str(work); db=str(db)
    m=pd.read_pickle(os.path.join(work,'M.pkl'))
    if len(m)==0:
        m.to_pickle(os.path.join(work,'D_final.pkl'))
        pd.DataFrame().to_pickle(os.path.join(work,'DEF_final.pkl'))
        print(0, 0, 0)
        return m, pd.DataFrame()
    if not folders:
        folders=sorted(m.folder.unique(), key=folder_sort_key)
    if rule_b:
        m['st31']=pd.read_pickle(os.path.join(work,'v31status.pkl')).st31
    else:
        m['st31']='real'
    pers=pd.concat([pd.read_pickle(os.path.join(work, f'persist_{f}.pkl')) for f in folders]).loc[m.index]
    glare=json.load(open(os.path.join(work,'glare_bands.json')))
    o=run(m,pers,glare,{'rule_b': 1 if rule_b else 0})
    import grok_aoi.filter.rules42 as rules42
    _a=rules42.SIDE_T2_CONV['mask_a'];_b=rules42.SIDE_T2_CONV['mask_b']
    o['t2_conv']=np.where(_a,'a',np.where(_b,'b',''))
    o['st31']=m.st31.values
    o['pframes']=pers.p_frames.values
    o['reason_codes']=[reason(r) for r in o.reason_code]
    o['reason_text']=[rtext(r) for r in o.reason_code]
    cand_cols=m[['id','folder','fr','frame','grp','bx','by','bw','bh','cx','cy','area_px','contrast','cnr','snr_raw','zone','anomaly_frame','path','merged_n','consolidation','contrast_tier','straddles_roi','cls','confidence','nb_rank','tier','roi_keep_reason','rim_dist','edge_completed','rep','solidity','nb50','nb_medc','nb_similar','major_len_px','elongation','thickness_px']].rename(columns={'tier':'tier_prior','cls':'det_class'})
    D=pd.concat([cand_cols.drop(columns=['rep']),o.drop(columns=['id','tier'])],axis=1); D['tier']=o.tier.values
    D['image']=[f'{db}/{f.split("_",1)[0]}/{f.split("_",1)[1]}/{int(k):02d}.png' for f,k in zip(D.folder,D.fr)]
    D['reason']=[f'{c} [{t}] |c|={abs(c_):.1f} cnr={cn:.1f} nb_rank={nr:.2f}' for c,t,c_,cn,nr in zip(D.reason_codes,D.reason_text,D.contrast,D.cnr,D.nb_rank)]
    D['persistence_frames']=D.pframes
    # ---- defects: union-find over real candidates
    real=D[D.decision=='real']
    defect_id={}
    summ={}
    alldef=[]
    for f in folders:
        R=real[real.folder==f]
        if len(R)==0: continue
        reg=json.load(open(os.path.join(work,f'reg_{f}.json')))
        idx={i:j for j,i in enumerate(R.index)}; par=list(range(len(R)))
        def find(a):
            while par[a]!=a: par[a]=par[par[a]]; a=par[a]
            return a
        def uni(a,b):
            a,b=find(a),find(b)
            if a!=b: par[b]=a
        # same-frame overlaps
        for fr,g in R.groupby('fr'):
            if len(g)<2: continue
            x0=g.bx.values;y0=g.by.values;x1=x0+g.bw.values;y1=y0+g.bh.values; ii=[idx[i] for i in g.index]
            t=cKDTree(np.c_[g.cx,g.cy]); 
            for a,b in t.query_pairs(r=max(30,0)):
                ix=min(x1[a],x1[b])-max(x0[a],x0[b]); iy=min(y1[a],y1[b])-max(y0[a],y0[b])
                if ix>0 and iy>0:
                    inter=ix*iy; aa=(x1[a]-x0[a])*(y1[a]-y0[a]); bb=(x1[b]-x0[b])*(y1[b]-y0[b])
                    if inter/min(aa,bb)>=0.5 or inter/(aa+bb-inter)>=0.3: uni(ii[a],ii[b])
        # cross frame
        by={k:g for k,g in R.groupby('fr')}
        for r in reg:
            if not r['trusted']: continue
            k=int(r['pair'][:2])
            if k not in by or k+1 not in by: continue
            ga,gb=by[k],by[k+1]; t=cKDTree(np.c_[gb.cx,gb.cy])
            p=np.c_[ga.cx,ga.cy]+np.array([r['ox'],r['oy']])
            ok=(p[:,0]>=0)&(p[:,0]<width)&(p[:,1]>=0)&(p[:,1]<height)
            for i in np.where(ok)[0]:
                rad=min(25,max(3,0.5*max(ga.bw.values[i],ga.bh.values[i])+2))
                for c in t.query_ball_point(p[i],rad):
                    ra=ga.area_px.values[i]; rb=gb.area_px.values[c]
                    if 0.2<=rb/max(ra,1)<=5: uni(idx[ga.index[i]],idx[gb.index[c]])
        grp=np.array([find(i) for i in range(len(R))])
        R=R.assign(root=grp)
        rows=[]
        for n,(root,g) in enumerate(sorted(R.groupby('root'),key=lambda kv:(kv[1].fr.min(),kv[1].cy.min())),1):
            did=f'{f}_D{n:05d}'; defect_id.update({i:did for i in g.index})
            top=g.loc[g.area_px.idxmax()]; frs=sorted(g.fr.unique())
            gf=g[g.fr==top.fr]
            x0=int(gf.bx.min());y0=int(gf.by.min());x1=int((gf.bx+gf.bw).max());y1=int((gf.by+gf.bh).max())
            cl=top.fine_class
            if (g.fine_class=='stain').any() and top.fine_class!='stain' and g.area_px.sum()>=300: cl='stain'
            rows.append(dict(defect_id=did,fine_class=cl,fine_class_zh=ZH[cl],rep_frame=int(top.fr),rep_image=top.image,rep_candidate_id=top.id,
                bbox_x=x0,bbox_y=y0,bbox_w=x1-x0,bbox_h=y1-y0,area_px=int(top.area_px),contrast=float(top.contrast),snr_raw=float(top.snr_raw),
                region=top.region,n_candidates=len(g),n_frames=len(frs),frames=';'.join(f'{x:02d}' for x in frs),candidate_ids=';'.join(g.id),
                review_required=bool(g.review_required.any()),review_level=int(g.review_level.max()),tier=min(g.tier),all_in_anomaly_frames=bool(g.anomaly_frame.all()),
                persistence_confirmed=bool(g.pers_confirmed.any())))
        alldef.append(pd.DataFrame(rows))
    D['defect_id']=D.index.map(lambda i:defect_id.get(i,''))
    DEF=pd.concat(alldef,ignore_index=True) if alldef else pd.DataFrame()
    D.to_pickle(os.path.join(work,'D_final.pkl')); DEF.to_pickle(os.path.join(work,'DEF_final.pkl'))
    print(len(D),D.id.nunique(),len(DEF))
    return D, DEF
