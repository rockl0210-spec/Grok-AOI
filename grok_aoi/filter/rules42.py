"""Filter rules v4.2 (= v4.1 with: Side T2 conversion limited to (a) measurable-and-unconfirmed or (b) v4-new & not measurable; Edge1 path C very-strong gets L1 review). Base: v4.1 (v4 rules + changes 1-7 from Integration-Eval + v4.1 raw_rescue/haze/regrown/Edge1 rules; review_level L1/L2). Principle: flag, don't exclude; region-adapted thresholds; per-candidate distinctiveness."""
import pandas as pd, numpy as np
ANOM_EDGE={0,45}; ANOM_SIDE={0,15}
# Side_270 frame 15 dark-band specks (preprocess artifact). Overridden by the dataset profile.
E_STRIP=dict(enabled=True, folder='Side_270', frame=15, x0=1380, x1=1500, y0=780, y1=1460, max_area=200)

def apply_dataset(anom_edge, anom_side, e_strip=None):
    """Install dataset-specific anomaly frames and the optional e_strip window. Defaults match the reference profile."""
    global ANOM_EDGE, ANOM_SIDE, E_STRIP
    ANOM_EDGE=set(int(x) for x in anom_edge)
    ANOM_SIDE=set(int(x) for x in anom_side)
    if e_strip is not None:
        E_STRIP=dict(enabled=True, folder='Side_270', frame=15, x0=1380, x1=1500, y0=780, y1=1460, max_area=200)
        E_STRIP.update(e_strip)
        E_STRIP['frame']=int(E_STRIP['frame'])
        for k in ('x0','x1','y0','y1','max_area'):
            if k in E_STRIP and E_STRIP[k] is not None: E_STRIP[k]=float(E_STRIP[k])
TH=dict(
 vs=dict(ac=60,area=15,cnr=10),
 edge_art=dict(dist=6,elong=3.0,thick=2.5,major=12,straddle_overlap=0.5,straddle_elong=2.5,strong_ac=25,strong_cnr=6),
 normal_T1=dict(cnr=3.5), normal_T2=dict(cnr=6.0,ac=10,area=6),
 side_T2=dict(cnr=6.0,ac=8,area=8),
 uneven_edge=dict(ac=12,cnr=5.0,area=6,T2_cnr=7.0,T2_ac=10,shade_area=1500,shade_ac=25),
 side_uneven=dict(cnr=5.0,nb_rank=0.5,T2_cnr=8.0,exempt_cnr=10,exempt_snr=30),
 nb_abs=dict(nb50=3,cnr=5.0,ac=10),                       # fix 3: nb_rank unreliable if nb50<=3 -> absolute thresholds
 contam=dict(strong_ac=40,strong_pk=50,strong_pk_ac=22,mid_pk=35,mid_pk_ac=14,strong_cnr=2.5,strong_area=5,
             mid_ac=20,mid_cnr=3.0,mid_nbr=0.3,mid_dist_nbr=0.85,mid_dist_ratio=1.3,mid_dist_cnr=6,
             faint_ac=12,faint_cnr=4.5,faint_nbr=0.6,faint_simr=0.7,faint_nb50=35,faint_area=5,
             dense_simr=0.7,dense_nb50=15,min_area=4),
 haze=dict(area=5000,ac=25,solidity=0.7),
 oilfilm=dict(area=100,ac=25,dist_nbr=0.85,dist_ratio=1.3,review_ac=15),
 rough_stain=dict(area=400,ac=15,cnr=4),
 rough_scratch=dict(elong=4,major=40,ac=15,cnr=5),
 specular=dict(ac=30,elong=2.5,area=8,cnr=8),
 texture_dense=dict(nb50=45,nb50_side=25,nb_rank=0.5,cnr=7,cnr_side=8),
 fixed_pattern=dict(rep=3,area=40,rep_strong=5,strong_ac=20),      # fix 7: >=3 distinct frames same x,y (+similar area/|c|); strong needs 5
 faint=dict(lo=12,hi=16,cnr=3.0,area=4,snr_edge=7,snr_side=12,broad_ac=8,broad_cnr=8.0,broad_pk=18,broad_area=8,broad_snr=12),    # fix 4
 regrown=dict(merged_n=3,review_ac=25),
 glare=dict(pad=15,bh=25,ratio=1.8,near=80,near_elong=4,near_ratio=2.5,weak_ac=12,weak_cnr=8),
 rim=dict(dist=20,ac=15),                                          # fix 6
 big=dict(area=50000),                                             # fix 8
 persistence=dict(ctrl_max=0.5),
 sparse2=dict(ac=8,cnr=8,area=5,snr=20),      # change 2 (flip-null sweep, see README)
 stain2=dict(area=400,ac=25,cnr=5),
 rough_mid_ac=25,                              # change 6
 c7=dict(mid_nbr=0.1,dense_simr=0.85,dense_nb50=40),   # change 7
)
SIDE_T2_CONV={}
OPT0=dict(spec=1,sparse=1,ruleD=1,sideT2=2,rough25=1,c7=1,v41=1,pathC_vs=1,rule_b=0)   # sideT2: 0 off, 1 = v4.1 (all unconfirmed), 2 = v4.2. rule_b defaults OFF (persistence rule a only); set rule_b=1 to also apply legacy v3.1 rule (b).
ZH=dict(particle='落塵',stain='髒污',scratch='刮傷',water_mark='水痕',oil_film='油膜',edge_artifact='邊緣偽影',reflection='反光',texture='紋理',noise='雜訊')
def run(m,pers,glare,opt=None):
    O=dict(OPT0); O.update(opt or {})
    n=len(m); T=TH
    ac=m.contrast.abs().values; area=m.area_px.values; cnr=m.cnr.values; snr=m.snr_raw.values
    el=m.elongation.values; nbr0=m.nb_rank.values; nbm=m.nb_medc.values; nb50=m.nb50.values
    zone=m.zone.values; isE=(m.grp=='Edge').values; isS=~isE
    tier=m.tier.values.copy(); sol=m.solidity.values
    cand_path=m.path.values; ctier=m.contrast_tier.values
    simr=m.nb_similar.values/np.maximum(nb50,1)
    pk=np.abs(m.local_bg_gray.values-np.where(m.polarity.values=='dark',m.min_gray.values,m.max_gray.values))   # raw peak depth (cleaned contrast is stain-separation-reduced in contaminated zones)
    # --- fix 3: effective nb_rank
    na=T['nb_abs']; sparse=nb50<=na['nb50']
    nbr=np.where(sparse&(cnr>=na['cnr'])&(ac>=na['ac']),1.0,nbr0)
    Z=np.full(n,'normal',dtype=object)
    for z in ['uneven','specular','ink','oil','water','rough']: Z[zone==z]=z
    Z[isS&((m.nuisance_bits.values&1)>0)&(Z=='normal')]='specular'
    contam=np.isin(Z,['rough','oil','water','ink'])
    pt=pers.p_tested.values; pm=pers.p_match.values; pc=pers.p_ctrl.values
    anom=m.anomaly_frame.values
    pfn=np.array([any((f not in (ANOM_EDGE if e else ANOM_SIDE)) for f in fl) for fl,e in zip(pers.p_frames.values,isE)])
    pconf=pt&pm&(pc<T['persistence']['ctrl_max'])&pfn
    pfail=pt&~pm
    vs=(ac>=T['vs']['ac'])&(area>=T['vs']['area'])&(cnr>=T['vs']['cnr'])&(m.merged_n.values<3)&(~contam|(nbr>=0.85))
    dec=np.full(n,'',dtype=object); fc=np.full(n,'',dtype=object); rs=np.array(['']*n,dtype=object); rev=np.zeros(n,bool)
    und=np.ones(n,bool); fx=np.zeros((n,),dtype=object); fx[:]=''
    def setn(mask,cls,code,review=None):
        mk=mask&und; dec[mk]='nuisance'; fc[mk]=cls; rs[mk]=code; und[mk]=False
        if review is not None: rev[mk]=review[mk] if hasattr(review,'__len__') else review
    def setr(mask,code,review=False):
        mk=mask&und; dec[mk]='real'; rs[mk]=code; und[mk]=False
        rev[mk]=review[mk] if hasattr(review,'__len__') else review
    ci=dict(T['contam'])
    if O['c7']: ci['mid_nbr']=T['c7']['mid_nbr']; ci['dense_simr']=T['c7']['dense_simr']; ci['dense_nb50']=T['c7']['dense_nb50']
    pol=m.polarity.values; native=(isE&(pol=='dark'))|(isS&(pol=='bright'))
    rr=m.raw_rescue.values.astype(bool) if 'raw_rescue' in m else np.zeros(n,bool)
    htx=m.haze_texture.values.astype(bool) if 'haze_texture' in m else np.zeros(n,bool)
    mg=m.merged_n.values
    strongc=((ac>=ci['strong_ac'])|((pk>=ci['strong_pk'])&(ac>=ci['strong_pk_ac'])))&(cnr>=ci['strong_cnr'])&(area>=ci['strong_area'])
    # (e) dataset e_strip window (reference profile: Side_270 frame 15 dark-strip specks, preprocess artifact, anomaly frame)
    es=E_STRIP
    if es.get('enabled', True):
        e_strip=(m.folder.values==es['folder'])&(m.fr.values==int(es['frame']))&(m.cx.values>=es['x0'])&(m.cx.values<=es['x1'])&(m.cy.values>=es['y0'])&(m.cy.values<=es['y1'])&(area<es['max_area'])
    else:
        e_strip=np.zeros(n,bool)
    setn(e_strip,'noise','N_PREPROC_STRIP_ARTIFACT',review=False)
    # 1 regrown over-merge
    setn((m.merged_n.values>=T['regrown']['merged_n']),'texture','N_REGROWN_OVERMERGE',review=ac>=T['regrown']['review_ac'])
    # 2 edge artifact -- fix 5: strong straddlers/edge objects are never nuisance merely for straddling
    ea=T['edge_art']
    thin=(el>=ea['elong'])|((m.thickness_px.values<=ea['thick'])&(m.major_len_px.values>=ea['major']))
    A=(m.dist_to_roi_edge_px.values<=ea['dist'])&thin
    B=m.straddles_roi.values&(m.roi_overlap_frac.values<ea['straddle_overlap'])&(el>=ea['straddle_elong'])
    eastr=(ac>=ea['strong_ac'])&(cnr>=ea['strong_cnr'])&(area>=ea['strong_area'] if 'strong_area' in ea else area>=5)
    setr(~vs&(A|B)&eastr,'R_EDGE_STRADDLE_STRONG',review=True)
    setn(~vs&(A|B),'edge_artifact','N_EDGE_ARTIFACT',review=ac>=15)
    # 3 Side glare lines (fix 6: same-row consistency)
    g=T['glare']; refl=np.zeros(n,bool); inband=np.zeros(n,bool)
    if isS.any():
        by=m.by.values; bh=m.bh.values; bw=m.bw.values
        idx=np.where(isS)[0]; key=(m.folder.values[idx]+'_'+m.frame.values[idx])
        for i,k in zip(idx,key):
            for (y0,y1,e) in glare.get(k,[]):
                inter=(by[i]<=y1+g['pad'])&(by[i]+bh[i]>=y0-g['pad'])
                near=(by[i]<=y1+g['near'])&(by[i]+bh[i]>=y0-g['near'])
                if inter: inband[i]=True
                if (inter and bh[i]<=g['bh'] and bw[i]>=g['ratio']*bh[i] and el[i]>=2.0) or (near and el[i]>=g['near_elong'] and bw[i]>=g['near_ratio']*bh[i]):
                    refl[i]=True
    setn(~vs&refl,'reflection','N_GLARE_LINE',review=(ac>=20)|(area>=30))
    # 4 specular zone
    sp=T['specular']
    spok=(ac>=sp['ac'])&(el<sp['elong'])&(area>=sp['area'])&(cnr>=sp['cnr'])
    setn((Z=='specular')&~spok,'reflection','N_SPECULAR_WEAK',review=ac>=sp['ac'])
    if O['spec']: setn((Z=='specular')&spok&~native,'reflection','N_SPECULAR_POLARITY',review=True)   # change 1: strong specular needs native polarity (Edge dark / Side bright)
    setr((Z=='specular')&spok,'R_SPECULAR_STRONG',review=True)
    # 5 (a) large hazy low-contrast components in oil/water/ink zone -> oil_film/water_mark nuisance, review
    hz=T['haze']
    hazy=(area>=hz['area'])&(ac<hz['ac'])&((sol<hz['solidity'])|(ctier=='gray_low'))&np.isin(Z,['oil','water','ink','rough'])
    setn(~vs&hazy,'oil_film','N_OIL_HAZE_LARGE',review=True)
    # real oil_film / water_mark: large faint non-distinct regions in oil / water zones (fix 1)
    of=T['oilfilm']; dist_=(nbr>=of['dist_nbr'])&(ac>=of['dist_ratio']*np.maximum(nbm,1))
    big_faint=(area>=of['area'])&(ac<of['ac'])&~dist_&~strongc
    setn(~vs&(Z=='oil')&big_faint,'oil_film','N_OIL_FILM',review=ac>=of['review_ac'])
    setn(~vs&(Z=='ink')&big_faint,'oil_film','N_OIL_FILM',review=ac>=of['review_ac'])
    setn(~vs&(Z=='water')&big_faint,'water_mark','N_WATER_MARK',review=ac>=of['review_ac'])
    # 6 uneven shading
    ue=T['uneven_edge']
    setn(~vs&isE&(Z=='uneven')&(area>=ue['shade_area'])&(ac<ue['shade_ac']),'reflection','N_UNEVEN_SHADING',review=ac>=15)
    # 7 contaminated zones (rough/oil/water/ink): per-candidate distinctiveness (fix 2)
    rst=T['rough_stain']; rsc=T['rough_scratch']
    distinct=(nbr>=ci['mid_dist_nbr'])&(ac>=ci['mid_dist_ratio']*np.maximum(nbm,1))&(cnr>=ci['mid_dist_cnr'])
    notdense=~((simr>ci['dense_simr'])&(nb50>=ci['dense_nb50']))
    rD=contam&(ac>=30)&(cnr<4.5) if O['ruleD'] else np.zeros(n,bool)    # change 3 (rule D): contaminated real, cnr<4.5 & |c|>=30 -> no review from low cnr / non-distinct default
    setr(contam&vs,'R_CONTAM_VERYSTRONG',review=False)
    setr(contam&strongc,'R_CONTAM_STRONG',review=~(distinct&notdense)&~rD)
    if O['sparse']: rst=T['stain2']
    r_stain=(area>=rst['area'])&(ac>=rst['ac'])&(cnr>=rst['cnr'])
    r_scr=(el>=rsc['elong'])&(m.major_len_px.values>=rsc['major'])&(ac>=rsc['ac'])&(cnr>=rsc['cnr'])
    setr(contam&r_stain,'R_CONTAM_STAIN',review=True)
    setr(contam&r_scr,'R_CONTAM_SCRATCH',review=True)
    mid_ac=np.where(Z=='rough',T['rough_mid_ac'],ci['mid_ac']) if O['rough25'] else ci['mid_ac']
    mid=((ac>=mid_ac)|((pk>=ci['mid_pk'])&(ac>=ci['mid_pk_ac'])))&(cnr>=ci['mid_cnr'])&(nbr>=ci['mid_nbr'])&(area>=ci['min_area'])
    setr(contam&mid,'R_CONTAM_MID',review=~distinct&~rD)
    faint=(ac>=ci['faint_ac'])&(cnr>=ci['faint_cnr'])&(nbr>=ci['faint_nbr'])&(simr<=ci['faint_simr'])&(nb50<ci['faint_nb50'])&(area>=ci['faint_area'])
    setr(contam&faint,'R_CONTAM_FAINT',review=True)
    sp2=T['sparse2'] if O['sparse'] else dict(ac=8,cnr=6,area=5,snr=12)
    setr(contam&sparse&(ac>=sp2['ac'])&(cnr>=sp2['cnr'])&(area>=sp2['area'])&(snr>=sp2['snr']),'R_CONTAM_SPARSE',review=True)   # fix 3: nb_rank unreliable when nb50<=3 -> absolute thresholds
    setn(contam&(simr>ci['dense_simr'])&(nb50>=ci['dense_nb50']),'texture','N_CONTAM_TEXTURE',review=ac>=ci['faint_ac'])
    setn(contam&(Z=='rough'),'texture','N_ROUGH_TEXTURE',review=(ac>=15)&(cnr>=4))
    setn(contam&(Z=='oil')&(ac>=ci['faint_ac']),'oil_film','N_OIL_TEXTURE',review=True)
    setn(contam&(Z=='ink')&(ac>=ci['faint_ac']),'oil_film','N_OIL_TEXTURE',review=True)
    setn(contam&(Z=='water')&(ac>=ci['faint_ac']),'water_mark','N_WATER_TEXTURE',review=True)
    setn(contam,'noise','N_CONTAM_BELOW_STRENGTH',review=False)
    # 8 dense texture outside contaminated zones
    td=T['texture_dense']
    setn(~vs&(nbr<td['nb_rank'])&((isE&(nb50>=td['nb50'])&(cnr<td['cnr']))|(isS&(nb50>=td['nb50_side'])&(cnr<td['cnr_side']))),'texture','N_DENSE_TEXTURE',review=False)
    # 9 fixed pattern (fix 7): genuinely repeated position in >=3 distinct frames (+similar area/|c|)
    fp=T['fixed_pattern']; rep=m.rep.values
    setn(~vs&(rep>=np.where(ac>=fp['strong_ac'],fp['rep_strong'],fp['rep']))&(area<fp['area']),'noise','N_FIXED_PATTERN',review=ac>=15)
    # 10 strength gates per region
    tT1=(tier=='T1'); tT2=(tier=='T2')
    hs_cap=(ctier=='high')&m.straddles_roi.values            # fix 5 label: straddler cap T1->T2
    en=isE&np.isin(Z,['normal'])
    setr(en&(vs|(tT1&(cnr>=T['normal_T1']['cnr']))),'R_EDGE_NORMAL_T1',review=False)
    n2=T['normal_T2']
    setr(en&tT2&(cnr>=n2['cnr'])&(ac>=n2['ac'])&(area>=n2['area']),'R_EDGE_NORMAL_T2',review=True)
    eu=isE&(Z=='uneven')
    setr(eu&(vs|(tT1&(ac>=ue['ac'])&(cnr>=ue['cnr'])&(area>=ue['area']))),'R_EDGE_UNEVEN_T1',review=False)
    setr(eu&tT2&(cnr>=ue['T2_cnr'])&(ac>=ue['T2_ac']),'R_EDGE_UNEVEN_T2',review=True)
    sn=isS&np.isin(Z,['normal'])
    setr(sn&(vs|(tT1&(cnr>=T['normal_T1']['cnr']+0.5))),'R_SIDE_NORMAL_T1',review=False)
    s2=T['side_T2']
    setr(sn&tT2&(cnr>=s2['cnr'])&(ac>=s2['ac'])&(area>=s2['area']),'R_SIDE_NORMAL_T2',review=True)
    su=isS&(Z=='uneven'); sue=T['side_uneven']
    exempt=(cnr>=sue['exempt_cnr'])&(snr>=sue['exempt_snr'])                      # fix 3 exemption
    setr(su&(vs|(tT1&(cnr>=sue['cnr'])&((nbr>=sue['nb_rank'])|exempt))),'R_SIDE_UNEVEN_T1',review=False)
    setr(su&tT2&(cnr>=sue['T2_cnr'])&((nbr>=sue['nb_rank'])|exempt),'R_SIDE_UNEVEN_T2',review=True)
    # fix 3b: Side normal/uneven exempt strong isolated points that failed T-tier because of tier-cap
    setr(isS&np.isin(Z,['normal','uneven'])&exempt&(ac>=10)&(area>=8),'R_SIDE_STRONG_EXEMPT',review=True)
    # fix 5: strong straddler rescue (never nuisance merely for straddling)
    setr(m.straddles_roi.values&(ac>=20)&(cnr>=5)&(area>=5),'R_STRADDLE_STRONG',review=True)
    # fix 4: faint clean-zone dots, relaxed below-strength rejection, review
    fa=T['faint']; snr_min=np.where(isE,fa['snr_edge'],fa['snr_side'])
    faint_cl=np.isin(Z,['normal','uneven'])&(ac>=fa['lo'])&(ac<fa['hi'])&(cnr>=fa['cnr'])&(area>=fa['area'])&(snr>=snr_min)&(tier!='T3')
    setr(faint_cl,'R_FAINT_RELAXED',review=True)
    broad=np.isin(Z,['normal','uneven'])&(ac>=fa['broad_ac'])&(cnr>=fa['broad_cnr'])&(pk>=fa['broad_pk'])&(area>=fa['broad_area'])&(snr>=fa['broad_snr'])&isE
    setr(broad,'R_FAINT_BROAD',review=True)
    setn(np.ones(n,bool),'noise','N_BELOW_STRENGTH',review=(ac>=fa['lo'])&(cnr>=fa['cnr']))
    # change 5: Side T2 reals (new in v4) go back to nuisance+review unless persistence-confirmed
    if O['sideT2']:
        t2=(dec=='real')&np.isin(rs,['R_SIDE_NORMAL_T2','R_SIDE_UNEVEN_T2'])&~vs&~anom
        if O['sideT2']==1: sc=t2&~pconf; ca=sc; cb=np.zeros(n,bool)
        else:
            # v4.2: (a) persistence measurable (pers_tested) and NOT confirmed.
            # (b) is optional and OFF by default: v4-new (st31 not real/bigreal) and persistence not measurable.
            # Without a legacy v3.1 decisions file, rule (b) must not fire (historical full DB: 205 of 541 Side conversions).
            use_b=bool(O.get('rule_b', 0))
            v31r=m['st31'].isin(['real','bigreal']).values if (use_b and 'st31' in getattr(m,'columns',[]) ) else np.ones(n,bool)
            ca=t2&pt&~pconf; cb=(t2&~pt&~v31r) if use_b else np.zeros(n,bool); sc=ca|cb
        SIDE_T2_CONV['a']=int(ca.sum()); SIDE_T2_CONV['b']=int(cb.sum()); SIDE_T2_CONV['mask_a']=ca; SIDE_T2_CONV['mask_b']=cb
        dec[sc]='nuisance'; fc[sc]='noise'; rs[sc]='N_SIDE_T2_UNCONFIRMED'; rev[sc]=True
    # --- persistence adjust (no dropping: flag, don't exclude)
    R=(dec=='real'); code_add=np.array(['']*n,dtype=object)
    code_add[R&pconf]='+PERS_OK'
    clear=R&pconf&np.isin(rs,['R_EDGE_NORMAL_T2','R_SIDE_NORMAL_T2','R_EDGE_UNEVEN_T2','R_SIDE_UNEVEN_T2','R_CONTAM_MID','R_CONTAM_FAINT','R_FAINT_RELAXED'])
    rev[clear]=False
    flag=R&pfail&~anom&~vs
    rev[flag]=True; code_add[flag]='+PERS_MISS'
    # anomaly frames never confirm real on their own
    would=(dec=='real')&anom; keep=would&pconf; unver=would&~pconf
    dec[unver]='nuisance'; fc[unver]='noise'; rs[unver]='N_ANOM_UNVERIFIED'; rev[unver]=True
    code_add[keep]='+ANOM_PERS_OK'
    # review flags
    tier=np.where(hs_cap&(tier=='T1'),'T2',tier)
    R=(dec=='real'); lf=m.large_faint_region.values
    rev[R&m.straddles_roi.values]=True
    rev[R&lf]=True
    rev[R&(m.consolidation.values=='regrown')]=True
    if O['v41']:
        rev[rr]=True                                              # raw_rescue: always review (L2 when |c|<=30)
        rev[(mg>=T['regrown']['merged_n'])&(ctier=='high')]=True  # high-tier over-merged regrown: review
        e1=m.folder.str.startswith('Edge1').values
        rev[R&~vs&e1&(m.folder.values=='Edge1_090')&(Z=='oil')&(cand_path=='A')&(ctier=='gray_low')]=True
        rev[R&(~vs|bool(O['pathC_vs']))&e1&(cand_path=='C')&np.isin(ctier,['gray_mid','high'])]=True
    # (c) Edge1 path-B gray_low: polarity-flip false-point rate 6.6% (10.3% in rough) -> real only with review
    c_flag=R&(m.folder.str.startswith('Edge1').values)&(cand_path=='B')&(ctier=='gray_low')&~vs
    rev[c_flag]=True
    # fix 6: Side ROI top/bottom rim
    rim=m.rim_dist.values; rim_zone=isS&(rim<=T['rim']['dist'])
    rev[R&rim_zone&(ac<T['rim']['ac'])]=True
    # fix 6: glare-band rows: weak real candidates in a glare row are flagged consistently with the nuisance decision
    gw=T['glare']; rev[R&inband&((ac<20)|(cnr<gw['weak_cnr']))]=True
    # fix 8: area >50k always review; (a) any >5000 px real: review
    rev[area>=T['big']['area']]=True      # fix 8: any decision
    rev[area>=T['haze']['area']]=True     # (a) any >5000 px candidate is flagged, real or nuisance
    # fine class
    ml=m.major_len_px.values; th=m.thickness_px.values
    clsr=np.full(n,'particle',dtype=object)
    scr=((el>=3.5)&(ml>=25)&(th<=10))|((m.cls.values=='scratch')&(el>=3.0)&(ml>=20))
    stn=(area>=300)|((m.cls.values=='stain')&(area>=120))|lf
    clsr[stn]='stain'; clsr[scr&~stn]='scratch'; clsr[scr&(area>=300)&(el>=5)]='scratch'
    fc[R]=clsr[R]
    forced=R&(~vs)&(cnr<4.5)
    if O['ruleD']: forced=forced&~(contam&(ac>=30))               # change 3 (rule D)
    rev[forced]=True
    # change 4: review levels. L2 = priority triage, L1 = everything else under review
    lv2=np.zeros(n,bool)
    lv2|=(Z=='specular')|np.isin(rs,['R_CONTAM_SPARSE','R_CONTAM_STAIN','N_SIDE_T2_UNCONFIRMED','N_SPECULAR_POLARITY'])
    lv2|=flag|(area>=T['haze']['area'])
    lv2|=R&rim_zone&(ac<T['rim']['ac'])
    lv2|=R&inband&((ac<20)|(cnr<gw['weak_cnr']))
    if O['v41']: lv2|=(rr&(ac<=30))|htx
    rl=np.where(rev,np.where(lv2,2,1),0).astype(int)
    rs=np.array([a+b for a,b in zip(rs,code_add)],dtype=object)
    out=pd.DataFrame(dict(id=m.id.values,tier=tier,decision=dec,fine_class=fc,review_required=rev,review_level=rl,raw_rescue=rr,haze_texture=htx,reason_code=rs,region=Z,
        rim_zone=rim_zone,glare_row=inband,straddle_cap=hs_cap,
        pers_tested=pt,pers_match=pm,pers_confirmed=pconf,anom=anom,anom_would_real=would,vs=vs,rep=rep),index=m.index)
    out['fine_class_zh']=out.fine_class.map(ZH)
    return out
