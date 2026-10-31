import os
"""Stage 2: decisions, unique-defect aggregation, outputs."""
import pandas as pd,numpy as np,json,os,cv2,sys
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
W,H=2464,2056
WORK=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_work'; OUT=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/filter'
ANGLES=[('edge','Edge1_000'),('edge','Edge1_090'),('edge','Edge2_000'),('edge','Edge2_090'),('side','Side_000'),('side','Side_090'),('side','Side_180'),('side','Side_270')]
P=dict(
 protect_contrast=60.0, protect_area=150,
 edge_band_px=12, edge_hard_px=4, edge_prox_px=6, edge_align_deg=25.0, edge_align_elong=2.0,
 line_art_w_frac=0.95, line_art_h=80,
 wm_contrast=30.0, wm_solidity=0.70, wm_sharp=0.10, wm_ring=0.70, wm_min_area=150,
 oil_contrast=30.0, oil_area=500, oil_coh=0.60, oil_sharp=0.10, oil_grad=0.20,
 illum_area=1000, illum_contrast=60.0, illum_dist=5, illum_sharp=0.10,
 rough_scratch_keep_elong=8.0, rough_scratch_keep_thick=6.0, rough_scratch_keep_contrast_sigma=15.0, rough_particle_contrast=30.0,
 fixed_frames=4,
 anomaly_strong_contrast=40.0,
 persist_target=0.90,
)
def calibrate_T(c):
    """smallest snr_int threshold t (grid) such that persistence among checkable normal-zone particles with snr_int>=t is >= target; bounded [10,40]"""
    g=c[(c.checkable_n>0)&(c['class']=='particle')&(c.zone=='normal')&(~c.anomaly_frame)]
    for t in np.arange(8,60,1.0):
        s=g[g.snr_int>=t]
        if len(s)<60: break
        if (s.persist_n>0).mean()>=P['persist_target']: return float(np.clip(t,10,40)),len(s),float((s.persist_n>0).mean())
    return 40.0,len(g),float((g.persist_n>0).mean()) if len(g) else np.nan

def decide(df,T):
    n=len(df)
    dec=np.array(['keep']*n,dtype=object); code=np.array(['KEEP_CONFIRMED']*n,dtype=object); txt=np.array(['']*n,dtype=object)
    done=np.zeros(n,bool)
    def setd(mask,d,c,t=''):
        m=mask&~done
        dec[m]=d; code[m]=c; txt[m]=t; done[m]=True
    ac=df.contrast.abs().values; sig=df.noise_floor_frame.values
    snr_px=ac/sig; snr_int=df.snr_int.values; area=df.area_px.values; cls=df['class'].values
    zone=df.zone.values; bits=df.nuisance_bits.values; sol=df.solidity.values; dist=df.dist_to_roi_edge_px.values
    el=df.elongation.values; al=df.edge_align_deg.values; anomaly=df.anomaly_frame.values
    sharp=df.tx_edge_sharp.values; ring=df.tx_ring.values; coh=df.tx_coh.values; grad=df.tx_grad_rel.values
    persist=df.persist_n.values>0; chk=df.checkable_n.values>0
    corrob=persist  # persistence found in neighbouring frame
    protected=(~anomaly)&(ac>=P['protect_contrast'])&(area>=P['protect_area'])&(dist>=15)&(df.fixed_nframes.values<P['fixed_frames'])
    # 0 anomaly frames
    strong=(ac>=P['anomaly_strong_contrast'])&(zone=='normal')&(snr_int>=2*T)
    setd(anomaly&corrob,'keep','ANOMALY_SUPPORT_ONLY','anomaly-frame candidate corroborated in neighbouring non-anomaly frame; used only as supporting evidence (merged, not counted as its own defect)')
    setd(anomaly&strong,'review','ANOMALY_FRAME_STRONG_UNCONFIRMED','anomaly frame, uncorroborated, but strong high-contrast normal-zone feature; not counted as confirmed')
    setd(anomaly,'reject','ANOMALY_FRAME_EXCLUDED','candidate in anomaly frame (Edge 00/45, Side 00/15) without corroboration; excluded')
    # fixed pattern (same pixel position in >=4 other frames of a scanning burst)
    setd((df.fixed_nframes.values>=P['fixed_frames'])&(area<50),'reject','FIXED_PATTERN_ARTIFACT','same image position repeated in >=4 other frames although the scene moves ~1940/2263 px per frame -> sensor/optics artifact')
    # scan line artefact: spans (almost) full frame width as thin line
    setd((df.bbox_w.values>=P['line_art_w_frac']*W)&(df.bbox_h.values<=P['line_art_h'])&(~protected),'reject','SCAN_LINE_ARTIFACT','thin line spanning whole frame width (stitching/line-scan artefact)')
    # edge artefacts
    thin=(el>=P['edge_align_elong'])&(al<=P['edge_align_deg'])
    setd((dist<=P['edge_band_px'])&thin&~((ac>=40)&(snr_int>=40)),'reject','EDGE_ARTIFACT_ALIGNED',f'within {P["edge_band_px"]}px of ROI boundary, elongated and aligned with boundary orientation (<=25deg)')
    setd((df.at_roi_mask.values==0),'reject','EDGE_ARTIFACT_OUTSIDE_ROI','centroid outside ROI mask')
    glare=((zone=='specular')|((bits&1)>0))&(dist<=P['illum_dist'])&(area>=50)&~protected
    setd(glare,'reject','EDGE_ARTIFACT_GLARE','specular glare region touching ROI boundary')
    illum=(cls=='stain')&(area>=P['illum_area'])&(dist<=P['illum_dist'])&(ac<P['illum_contrast'])&(sharp<P['illum_sharp'])&~protected
    setd(illum,'reject','EDGE_ARTIFACT_ILLUM_FALLOFF','large low-contrast smooth region touching ROI boundary (illumination falloff / boundary halo)')
    setd((dist<=P['edge_hard_px'])&(snr_int<2*T)&~protected,'reject','EDGE_ARTIFACT_HALO',f'within {P["edge_hard_px"]}px of ROI boundary and weak (SNR_int<2*T_conf): boundary halo')
    # water mark
    big=(area>=P['wm_min_area'])|(cls=='stain')
    wm_n=((ac<P['wm_contrast']).astype(int)+(sol<P['wm_solidity'])+(sharp<P['wm_sharp'])+(ring<P['wm_ring']))
    water_flag=(zone=='water')|((bits&8)>0)
    wm=big&((ac<P['wm_contrast'])|water_flag)&((wm_n>=3)|(water_flag&(wm_n>=2)))&~protected
    setd(wm,'reject','WIPE_WATERMARK','low contrast, large, low solidity / hollow ring or arc shape, diffuse smooth boundary'+' (water flag)' )
    # oil film
    oil_flag=(zone=='oil')|((bits&16)>0)
    fp=(area>=P['oil_area'])&(ac<P['oil_contrast'])&(coh>=P['oil_coh'])&(sharp<P['oil_sharp'])&(grad<P['oil_grad'])
    oil=((oil_flag&(ac<P['oil_contrast']))|fp)&~protected
    setd(oil,'reject','OIL_FILM_FINGERPRINT','low-contrast large smooth region with ridge-like coherent texture and/or oil flag')
    # rough band texture
    rough=(zone=='rough')|((bits&4)>0)
    lineok=persist&(el>=P['rough_scratch_keep_elong'])&(df.thickness_px.values<=P['rough_scratch_keep_thick'])&(ac>=P['rough_scratch_keep_contrast_sigma']*sig)
    setd(rough&(cls=='scratch')&lineok,'review','ROUGH_SCRATCH_STRONG_LINE','scratch in rough band that is persistent, thin and high-contrast: kept for review')
    setd(rough&(cls=='scratch'),'reject','ROUGH_TEXTURE_FRAGMENT','scratch-like fragment inside rough/machined texture band, not strongly persistent+line-like')
    # noise
    setd((snr_int<T)&chk&~persist&(area<=10),'reject','NOISE_NONPERSISTENT','small, low SNR, overlap region checked and no counterpart in neighbouring frame')
    setd((snr_int<0.6*T)&~persist,'review','LOW_SNR_UNCONFIRMED','SNR_int below 0.6*T_conf and no cross-frame confirmation')
    # review tiers
    setd(rough&(cls!='scratch')&(ac<P['rough_particle_contrast'])&~persist,'review','ROUGH_ZONE_WEAK','in rough band, contrast<30 and not persistent')
    setd(water_flag|oil_flag|((cls=='stain')&(ac<P['wm_contrast'])&((sol<P['wm_solidity'])|(sharp<P['wm_sharp'])|(ring<P['wm_ring']))&~protected),'review','NUISANCE_SUSPECT_STAIN','stain-like with some water/oil traits but not enough to reject')
    setd((df.bbox_w.values>=600)&(df.bbox_h.values<=60)&(ac<30)&(cls!='particle')&(sol<0.8),'review','STREAK_BAND','long thin low-contrast merged streak band; scratch-like but ambiguous')
    setd((dist<=P['edge_prox_px'])&(snr_int<3*T)&~persist&~protected,'review','EDGE_PROXIMITY_WEAK','within 6px of ROI boundary, moderate SNR, unconfirmed')
    setd(chk&~persist&(snr_int<3*T)&~protected,'review','NONPERSISTENT_MODERATE','overlap region checked but no counterpart found; moderate SNR')
    setd((snr_int<T)&~persist,'review','BELOW_CONF_THRESHOLD','SNR_int below confirmation threshold T_conf and not cross-frame confirmed')
    # keeps
    setd(persist,'keep','KEEP_PERSISTENT','counterpart found in neighbouring frame (cross-frame persistent)')
    setd(protected,'keep','KEEP_STRONG_PROTECTED','very strong high-contrast large feature, protected from stain/film/edge nuisance rules')
    setd(np.ones(n,bool),'keep','KEEP_STRONG_SINGLE_FRAME','SNR_int>=T_conf, no nuisance rule fired (single-frame evidence; overlap not available or not required)')
    return dec,code,txt

def main():
    os.makedirs(OUT,exist_ok=True)
    totals=dict(); allsum={}
    for mode,name in ANGLES:
        df,matches,info=pd.read_pickle(f'{WORK}/{name}.pkl')
        df['snr_px']=df.contrast.abs()/df.noise_floor_frame
        df['snr_int']=df.snr_px*np.sqrt(df.area_px)
        T,nT,rT=calibrate_T(df)
        dec,code,txt=decide(df,T)
        df['decision']=dec; df['reason_code']=code; df['reason']=txt
        # nuisance class for rejects
        def ncls(c):
            if c.startswith('WIPE'): return 'water_mark'
            if c.startswith('OIL'): return 'oil_film'
            if c.startswith('EDGE'): return 'edge_artifact'
            if c.startswith('NOISE'): return 'noise'
            if c.startswith('ROUGH'): return 'texture_fragment'
            if c.startswith('ANOMALY'): return 'anomaly_frame'
            return 'other'
        df['nuisance_class']=[ncls(c) if d=='reject' else '' for c,d in zip(code,dec)]
        # ---- unique defects: union-find over matches between surviving (keep/review) candidates
        alive=((df.decision!='reject')|(df.reason_code=='ANOMALY_SUPPORT_ONLY')).values
        # anomaly support-only candidates are merged but cannot seed a defect
        seed=(df.decision!='reject').values&~(df.reason_code=='ANOMALY_SUPPORT_ONLY').values
        n=len(df); idx=np.arange(n)
        m=np.array(matches) if matches else np.zeros((0,2),int)
        if len(m): m=m[alive[m[:,0]]&alive[m[:,1]]]
        g=coo_matrix((np.ones(len(m)),(m[:,0],m[:,1])),shape=(n,n)) if len(m) else coo_matrix((n,n))
        nc,lab=connected_components(g,directed=False)
        df['_grp']=lab
        sub=df[seed].copy()
        # defects with tier: keep if any member keep (non-support) else review
        ids=[];rows=[]
        # global scan coordinates
        frames=sorted(df.frame.unique()); sh=info['shifts']
        off={frames[0]:(0.0,0.0)}
        for k in range(len(frames)-1):
            ox,oy=off[frames[k]]; off[frames[k+1]]=(ox-sh[k]['dx'],oy-sh[k]['dy'])
        df['global_x']=df.centroid_x+df.frame.map(lambda f:off[f][0]); df['global_y']=df.centroid_y+df.frame.map(lambda f:off[f][1])
        sub=df[seed]
        dn=0
        defects=[]
        for gid,gg in sub.groupby('_grp'):
            members=df[(df._grp==gid)&alive] if False else None
            defects.append(gid)
        # faster: group alive members once
        al=df[alive]; gbs={k:v for k,v in al.groupby('_grp')}
        out=[]
        for gid in defects:
            mem=gbs[gid]; seeds=mem[mem.index.isin(sub.index)]
            tier='confirmed' if (seeds.decision=='keep').any() else 'review_required'
            best=seeds.sort_values(['decision','snr_int'],ascending=[True,False]).iloc[0]  # 'keep' < 'review'
            cls=best['class']
            nonan=mem[~mem.anomaly_frame]
            rep=best
            nfr=mem.frame.nunique()
            out.append(dict(
                gid=int(gid),tier=tier,final_class=cls,frame=rep.frame,x=float(rep.centroid_x),y=float(rep.centroid_y),
                bbox_x=int(rep.bbox_x),bbox_y=int(rep.bbox_y),bbox_w=int(rep.bbox_w),bbox_h=int(rep.bbox_h),
                global_x=float(rep.global_x),global_y=float(rep.global_y),
                n_candidates=int(len(mem)),n_frames=int(nfr),support_frames=','.join(sorted(mem.frame.unique())),
                n_anomaly_support=int(mem.anomaly_frame.sum()),
                cross_frame_persistent=bool((mem.persist_n>0).any() or len(mem)>1),
                median_area_px=float(mem.area_px.median()),median_contrast=float(mem.contrast.median()),median_snr_int=float(mem.snr_int.median()),
                median_cnr=float(mem.cnr.median()),median_confidence=float(mem.confidence.median()),median_solidity=float(mem.solidity.median()),
                median_elongation=float(mem.elongation.median()),median_feret_px=float(mem.feret_px.median()),
                zone=rep.zone,dist_to_roi_edge_px=float(rep.dist_to_roi_edge_px),
                reason_code=rep.reason_code,reason=rep.reason,
                candidate_ids=';'.join(mem.id.tolist())))
        D=pd.DataFrame(out)
        if len(D):
            D=D.sort_values(['frame','y']).reset_index(drop=True)
            pref='E' if mode=='edge' else 'S'
            D.insert(0,'defect_id',[f'{name}_D{i:05d}' for i in range(len(D))])
            gid2did=dict(zip(D.gid,D.defect_id)); 
            df['defect_id']=[gid2did.get(g,'') if a else '' for g,a in zip(df._grp,alive)]
            # candidates not seeds but alive (support-only)
            D['reason']=D.reason.fillna('')
            # reasons for defect
            D['reason']=[(f"{r.tier}: {r.reason_code}; {r.reason}; frames={r.n_frames}; SNR_int={r.median_snr_int:.1f}" ) for r in D.itertuples()]
        else:
            df['defect_id']=''
        # finalize outputs
        od=f'{OUT}/{mode}/{name}'; os.makedirs(od,exist_ok=True)
        # fine subtype for stains
        def sub_t(r):
            if r.final_class!='stain': return ''
            if abs(r.median_contrast)>=60: return 'debris_or_opaque_residue'
            if r.median_solidity<0.65: return 'speck_cluster_residue'
            return 'residue'
        if len(D): D['subtype']=[sub_t(r) for r in D.itertuples()]
        D.drop(columns=['gid']).to_csv(f'{od}/defects.csv',index=False)
        # defects.json (confirmed only per spec) + review list
        conf=D[D.tier=='confirmed'] if len(D) else D
        rev=D[D.tier=='review_required'] if len(D) else D
        json.dump(dict(angle=name,mode=mode,n_confirmed=int(len(conf)),defects=json.loads(conf.drop(columns=['gid']).to_json(orient='records'))),open(f'{od}/defects.json','w'),indent=1)
        json.dump(dict(angle=name,mode=mode,n_review_required=int(len(rev)),defects=json.loads(rev.drop(columns=['gid']).to_json(orient='records'))),open(f'{od}/review_required.json','w'),indent=1)
        # defects.csv per spec = confirmed real defects ; keep review in separate csv
        conf.drop(columns=['gid'],errors='ignore').to_csv(f'{od}/defects.csv',index=False)
        rev.drop(columns=['gid'],errors='ignore').to_csv(f'{od}/review_required.csv',index=False)
        dcols=['id','frame','centroid_x','centroid_y','bbox_x','bbox_y','bbox_w','bbox_h','class','area_px','contrast','cnr','snr_int','anomaly_frame','zone','dist_to_roi_edge_px','checkable_n','persist_n','decision','reason_code','nuisance_class','defect_id','reason']
        df[dcols].rename(columns={'class':'det_class','persist_n':'persist_frames','checkable_n':'overlap_checkable'}).to_csv(f'{od}/decisions.csv',index=False)
        # summary
        summ=dict(angle=name,mode=mode,total_candidates=int(n),
            decisions=df.decision.value_counts().to_dict(),
            reject_by_reason=df[df.decision=='reject'].reason_code.value_counts().to_dict(),
            review_by_reason=df[df.decision=='review'].reason_code.value_counts().to_dict(),
            keep_by_reason=df[df.decision=='keep'].reason_code.value_counts().to_dict(),
            reject_by_nuisance_class=df[df.decision=='reject'].nuisance_class.value_counts().to_dict(),
            candidates_by_detector_class=df['class'].value_counts().to_dict(),
            unique_confirmed=int(len(conf)),unique_review_required=int(len(rev)),
            confirmed_by_final_class=conf.final_class.value_counts().to_dict() if len(conf) else {},
            review_by_final_class=rev.final_class.value_counts().to_dict() if len(rev) else {},
            anomaly_frame_candidates=int(df.anomaly_frame.sum()),
            anomaly_excluded=int((df.reason_code=='ANOMALY_FRAME_EXCLUDED').sum()),
            anomaly_support_only=int((df.reason_code=='ANOMALY_SUPPORT_ONLY').sum()),
            anomaly_review=int((df.reason_code=='ANOMALY_FRAME_STRONG_UNCONFIRMED').sum()),
            overlap_checkable_candidates=int((df.checkable_n>0).sum()),
            overlap_persistent_candidates=int(((df.checkable_n>0)&(df.persist_n>0)).sum()),
            thresholds=dict(T_conf_snr_int=T,T_conf_calibration_n=nT,T_conf_calibration_persistence_rate=rT,T_low_snr_int=0.6*T,**P),
            noise_floor=dict(median_gray=info['noise_floor_median'],p10=info['noise_floor_p10'],p90=info['noise_floor_p90'],definition='1.4826*MAD of (img-gauss1.5) per 64px ROI block / 0.93, median over blocks then frames'),
            registration=dict(median_shift_dx_dy=info['median_shift'],methods=pd.Series([v['method'] for v in sh.values()]).value_counts().to_dict(),per_pair=sh),
        )
        json.dump(summ,open(f'{od}/summary.json','w'),indent=1,default=str)
        allsum[name]=summ
        pd.to_pickle((df,D),f'{WORK}/{name}_final.pkl')
        print(name,summ['decisions'],'confirmed',len(conf),'review',len(rev),'T',T,round(rT,3))
    # global
    tot=dict(total_candidates=sum(s['total_candidates'] for s in allsum.values()))
    def agg(key):
        out={}
        for s in allsum.values():
            for k,v in s[key].items(): out[k]=out.get(k,0)+v
        return out
    tot.update(decisions=agg('decisions'),reject_by_reason=agg('reject_by_reason'),review_by_reason=agg('review_by_reason'),keep_by_reason=agg('keep_by_reason'),
        reject_by_nuisance_class=agg('reject_by_nuisance_class'),unique_confirmed=sum(s['unique_confirmed'] for s in allsum.values()),
        unique_review_required=sum(s['unique_review_required'] for s in allsum.values()),confirmed_by_final_class=agg('confirmed_by_final_class'),review_by_final_class=agg('review_by_final_class'),
        anomaly_frame_candidates=sum(s['anomaly_frame_candidates'] for s in allsum.values()),anomaly_excluded=sum(s['anomaly_excluded'] for s in allsum.values()),
        anomaly_support_only=sum(s['anomaly_support_only'] for s in allsum.values()),anomaly_review=sum(s['anomaly_review'] for s in allsum.values()))
    tot['per_angle']={k:dict(total=s['total_candidates'],confirmed=s['unique_confirmed'],review_required=s['unique_review_required'],confirmed_by_class=s['confirmed_by_final_class'],decisions=s['decisions'],T_conf=s['thresholds']['T_conf_snr_int'],noise_floor=s['noise_floor']['median_gray']) for k,s in allsum.items()}
    json.dump(tot,open(f'{OUT}/summary.json','w'),indent=1)
main()
