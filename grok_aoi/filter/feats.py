"""Stage 1: load candidates (CSV), recompute/define zone, tier prior, neighbourhood + repeat features, rim distance."""
import numpy as np, pandas as pd, cv2, os, glob
from scipy.spatial import cKDTree
FOLDERS=['Edge1_000','Edge1_090','Edge2_000','Edge2_090','Side_000','Side_090','Side_180','Side_270']
def mode(f): return 'edge' if f.startswith('Edge') else 'side'
def folder_sort_key(f):
    src,ang=f.split('_',1)
    return ({'Edge1':0,'Edge2':1,'Side':2}.get(src,9), src, ang)
def discover_folders(detect_root):
    paths=sorted(glob.glob(os.path.join(detect_root,'*','*','candidates.csv')))
    found=[os.path.basename(os.path.dirname(p)) for p in paths]
    return sorted(found, key=folder_sort_key)
ZP=['rough','oil','ink','water','specular','uneven']   # precedence (detect v4); v3.1 snapshot re-labelled from v3 masks
ZBIT=dict(rough=4,oil=16,ink=32,water=8,specular=1,uneven=2)
def load(detect_root, folders=None):
    if folders is None: folders=discover_folders(detect_root)
    else: folders=sorted(folders, key=folder_sort_key)
    L=[]
    for f in folders:
        path=f'{detect_root}/{mode(f)}/{f}/candidates.csv'
        if not os.path.isfile(path): continue
        d=pd.read_csv(path,dtype={'frame':str,'angle':str})
        if len(d)==0: continue
        d['folder']=f; L.append(d)
    if not L: return pd.DataFrame()
    M=pd.concat(L,ignore_index=True)
    M=M.rename(columns={'bbox_x':'bx','bbox_y':'by','bbox_w':'bw','bbox_h':'bh','centroid_x':'cx','centroid_y':'cy','class':'cls'})
    M['frame']=M.frame.astype(str).str.zfill(2); M['fr']=M.frame.astype(int); M['grp']=np.where(M.folder.str.startswith('Edge'),'Edge','Side')
    M['id']=M['id'].astype(str)
    for c in ['contrast_gain','contrast_est_raw','edge_completed']:
        if c not in M: M[c]=np.nan if c!='edge_completed' else False
    M['edge_completed']=M.edge_completed.fillna(False).astype(bool)
    return M
def tier_prior(M):
    ac=M.contrast.abs(); E=(M.grp=='Edge').values
    T1=np.where(E,(M.snr_raw>=12)&(ac>=10)&(M.area_px>=5)&(M.confidence>=0.5),(M.snr_raw>=30)&(ac>=10)&(M.area_px>=8))
    T2=np.where(E,(M.snr_raw>=9)&(M.confidence>=0.5),(M.snr_raw>=16))
    cons=((M.consolidation!='none')&(M.contrast_tier!='high')).values
    return np.where(T1&~cons,'T1',np.where(T2,'T2','T3'))
def nb_features(M):
    n=len(M); nb50=np.zeros(n,int); medc=np.zeros(n,np.float32); sim=np.zeros(n,int); rank=np.ones(n,np.float32)
    pos={i:k for k,i in enumerate(M.index)}
    for (f,fr),g in M.groupby(['folder','fr']):
        ac=g.contrast.abs().values; P=np.c_[g.cx,g.cy]; t=cKDTree(P); ix=np.array([pos[i] for i in g.index])
        for k,l in enumerate(t.query_ball_point(P,50)):
            l=[j for j in l if j!=k]
            if not l: continue
            c=ac[l]; i=ix[k]; nb50[i]=len(l); medc[i]=np.median(c); sim[i]=int((c>=0.7*ac[k]).sum()); rank[i]=float((c<ac[k]).mean())
    return nb50,medc,sim,rank
def repeat_feature(M,r=1.5):
    """genuinely repeated fixed-pattern position: same x,y (<=r px) in OTHER frames of the folder, similar area (0.5-2x), same polarity, |c| ratio 0.67-1.5"""
    rep=np.zeros(len(M),int); pos={i:k for k,i in enumerate(M.index)}
    for f,g in M.groupby('folder'):
        P=np.c_[g.cx,g.cy]; fr=g.fr.values; ar=g.area_px.values.astype(float); ac=g.contrast.abs().values; pol=(g.polarity=='dark').values
        t=cKDTree(P); ix=np.array([pos[i] for i in g.index])
        for k,l in enumerate(t.query_ball_point(P,r)):
            s=set()
            for j in l:
                if fr[j]==fr[k] or pol[j]!=pol[k]: continue
                if 0.5<=ar[j]/max(ar[k],1)<=2 and 0.67<=ac[j]/max(ac[k],1e-3)<=1.5: s.add(fr[j])
            rep[ix[k]]=len(s)
    return rep
def mask_features(M,pre_root,recompute_zone):
    """per frame: v3 nuisance bits -> zone (if recompute_zone), rim distance to ROI top/bottom (Side)."""
    zone=M.zone.values.astype(object).copy(); rim=np.full(len(M),np.nan,np.float32); bits=M.nuisance_bits.values.copy()
    pos={i:k for k,i in enumerate(M.index)}
    for (f,fr),g in M.groupby(['folder','fr']):
        src,ang=f.split('_'); ix=np.array([pos[i] for i in g.index])
        if recompute_zone:
            nu=cv2.imread(f'{pre_root}/{src}/{ang}/{fr:02d}_nuisance.png',0); H,W=nu.shape
            x0=np.clip(g.bx.values,0,W);y0=np.clip(g.by.values,0,H);x1=np.clip(g.bx.values+g.bw.values,0,W);y1=np.clip(g.by.values+g.bh.values,0,H)
            ar=np.maximum((x1-x0)*(y1-y0),1); z=np.array(['normal']*len(g),dtype=object); bb=np.zeros(len(g),int)
            for name in ZP[::-1]:
                b=ZBIT[name]; I=np.zeros((H+1,W+1),np.int32); I[1:,1:]=((nu&b)>0).astype(np.int32).cumsum(0).cumsum(1)
                s=I[y1,x1]-I[y0,x1]-I[y1,x0]+I[y0,x0]; fr_=s/ar
                z[fr_>=0.25]=name; bb|=np.where(fr_>0,b,0)
            zone[ix]=z; bits[ix]=bb|0
        if g.grp.iloc[0]=='Side':
            mk=cv2.imread(f'{pre_root}/{src}/{ang}/{fr:02d}_mask.png',0)>0; H,W=mk.shape
            top=np.where(mk.any(0),mk.argmax(0),-1); bot=np.where(mk.any(0),H-1-mk[::-1].argmax(0),-1)
            cx=np.clip(g.cx.values.astype(int),0,W-1); cy=g.cy.values
            ok=top[cx]>=0; d=np.where(ok,np.minimum(cy-top[cx],bot[cx]-cy),np.nan); rim[ix]=d
    return zone,bits,rim
def enrich(M,pre_root,recompute_zone):
    if len(M)==0: return M
    M=M.copy()
    M['tier']=tier_prior(M)
    M['nb50'],M['nb_medc'],M['nb_similar'],M['nb_rank']=nb_features(M)
    M['rep']=repeat_feature(M)
    M['zone_src']=M.zone
    M['zone'],M['nbits'],M['rim_dist']=mask_features(M,pre_root,recompute_zone)
    return M
def build(detect_root,pre_root,recompute_zone,folders=None):
    return enrich(load(detect_root, folders), pre_root, recompute_zone)
