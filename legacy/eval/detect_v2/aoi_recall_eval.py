#!/usr/bin/env python3
import os
"""Independent raw-image dot recall scan (no ground truth).  Usage:
  python3 aoi_recall_eval.py --detect-dir <legacy-root>/aoi_out/detect --out OUTDIR [--baseline DIR|pathA] [--thr 12]
Reference dots: local background = large median (100 px), residual = bg-raw (Edge, dark dots) or raw-bg (Side, bright dots),
connected residual>=thr, area>=3, <=1500, compact (fill>=0.6, L/W<=2.5), inside ROI (eroded 3 px). A reference dot is covered
if a candidate bbox (+2 px) contains its centroid and candidate area <= 30x dot area (strict). Recall is reported by peak-residual bin.
Baseline: another detect dir, or 'pathA' = path=='A' rows of the same dir (v2 path A is identical to v1)."""
import argparse,glob,os,sys,json
import cv2,numpy as np,pandas as pd
from multiprocessing import Pool
PRE=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/preprocess';DB=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_db'
BINS=[12,20,40,1e9];BL=['12-20','20-40','40+']
def bgmed(a,k=25,ds=4):
    s=cv2.resize(a,None,fx=1/ds,fy=1/ds,interpolation=cv2.INTER_AREA);m=cv2.medianBlur(s,k)
    return cv2.resize(m,(a.shape[1],a.shape[0]),interpolation=cv2.INTER_LINEAR).astype(np.float32)
def scan(args):
    src,ang,fr,pol,thr=args
    o=cv2.imread(f'{DB}/{src}/{ang}/{fr:02d}.png',0);roi=cv2.imread(f'{PRE}/{src}/{ang}/{fr:02d}_mask.png',0)>0
    nu=cv2.imread(f'{PRE}/{src}/{ang}/{fr:02d}_nuisance.png',-1)
    a=o.copy();a[~roi]=int(np.median(o[roi]))
    g=cv2.GaussianBlur(a.astype(np.float32),(0,0),0.8);bg=bgmed(a)
    r=(bg-g) if pol=='dark' else (g-bg)
    n,l,st,ce=cv2.connectedComponentsWithStats(((r>=thr)&roi).astype(np.uint8),connectivity=8)
    er=cv2.erode(roi.astype(np.uint8),np.ones((7,7),np.uint8))>0
    rows=[]
    for i in range(1,n):
        x,y,w,h,A=st[i]
        if A<3 or A>1500: continue
        cx,cy=ce[i]
        if not er[int(cy),int(cx)]: continue
        sub=l[y:y+h,x:x+w]==i;ys,xs=np.nonzero(sub)
        (_,_),(rw,rh),_=cv2.minAreaRect(np.c_[xs,ys].astype(np.float32))
        L,W=max(rw,rh)+1,min(rw,rh)+1;fill=A/(np.pi/4*L*W)
        if fill<0.6 or L/W>2.5: continue
        rr=r[y:y+h,x:x+w][sub]
        rows.append(dict(src=src,ang=ang,fr=fr,cx=cx,cy=cy,area=A,L=L,W=W,fill=fill,peak=float(rr.max()),mean=float(rr.mean()),rough=bool(nu[int(cy),int(cx)]&4)))
    return rows
def load_cands(d):
    fs=sorted(glob.glob(d+'/*/*_*/candidates.csv'))
    c=pd.concat([pd.read_csv(f,dtype={'frame':str,'angle':str}) for f in fs],ignore_index=True)
    c['folder']=c.source+'/'+c.angle;c['fr']=c.frame.astype(int)
    if 'path' not in c: c['path']='A'
    return c
def cover(b,c,pad=2,ratio=30):
    out=np.zeros(len(b),bool)
    if len(c)==0 or len(b)==0: return out
    x0=c.bbox_x.values-pad;y0=c.bbox_y.values-pad;x1=c.bbox_x.values+c.bbox_w.values+pad;y1=c.bbox_y.values+c.bbox_h.values+pad;ca=c.area_px.values
    for i,(cx,cy,ar) in enumerate(zip(b.cx.values,b.cy.values,b.area.values)):
        out[i]=((cx>=x0)&(cx<=x1)&(cy>=y0)&(cy<=y1)&(ca<=ratio*ar)).any()
    return out
def jobs(thr):
    J=[]
    for s in ['Edge1','Edge2']:
        for a in ['000','090']:
            fr=set(range(5,41,5))
            if a=='090': fr|=set(range(1,45))   # all normal frames for the 090 strips (where the reported misses were)
            J+=[(s,a,f,'dark',thr) for f in sorted(fr)]
    J+=[('Side',a,f,'bright',thr) for a in ['000','090','180','270'] for f in [3,6,9,12]]
    return J
def summ(g,cols):
    d={'n':len(g)}
    for c in cols: d[c]=round(g[c].mean()*100,1) if len(g) else np.nan
    return pd.Series(d)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--detect-dir',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--baseline',default='pathA');ap.add_argument('--thr',type=float,default=12);ap.add_argument('--procs',type=int,default=6)
    A=ap.parse_args();os.makedirs(A.out,exist_ok=True)
    with Pool(A.procs) as p: res=p.map(scan,jobs(A.thr))
    B=pd.DataFrame([r for rr in res for r in rr]);B['folder']=B.src+'/'+B.ang
    B=B[~B.fr.isin([0,45])&~((B.src=='Side')&B.fr.isin([0,15]))]
    new=load_cands(A.detect_dir)
    if A.baseline=='pathA': base=new[new.path=='A']
    elif A.baseline.endswith('.pkl'): base=pd.read_pickle(A.baseline);base['folder']=base.source+'/'+base.angle;base['fr']=base.frame.astype(int)
    else: base=load_cands(A.baseline)
    B['bin']=pd.cut(B.peak,BINS,right=False,labels=BL).astype(str)
    cn=[];co=[];keys=[]
    for (f,fr),idx in B.groupby(['folder','fr']).groups.items():
        b=B.loc[idx];cn.append(pd.Series(cover(b,new[(new.folder==f)&(new.fr==fr)]),index=idx));co.append(pd.Series(cover(b,base[(base.folder==f)&(base.fr==fr)]),index=idx))
    B['new']=pd.concat(cn);B['base']=pd.concat(co)
    B['dotsize']=np.where(B.area>=12,'area>=12','area3-11')
    B.to_pickle(A.out+'/ref_blobs.pkl')
    pd.set_option('display.width',200)
    outs={}
    outs['by_folder_bin']=B.groupby(['folder','bin']).apply(lambda g:summ(g,['base','new'])).unstack('bin')
    outs['by_folder']=B.groupby('folder').apply(lambda g:summ(g,['base','new']))
    outs['by_folder_zone']=B.groupby(['folder','rough']).apply(lambda g:summ(g,['base','new']))
    outs['all_bins']=B.groupby('bin').apply(lambda g:summ(g,['base','new']))
    outs['edge_bins_by_size']=B[B.src!='Side'].groupby(['bin','dotsize']).apply(lambda g:summ(g,['base','new']))
    pf=B.groupby(['folder','fr']).apply(lambda g:pd.Series(dict(n=len(g),base=g.base.mean()*100,new=g.new.mean()*100,miss=int((~g.new).sum()),n40=int((g.peak>=40).sum()),miss40=int(((g.peak>=40)&~g.new).sum()),miss20_40=int(((g.peak>=20)&(g.peak<40)&~g.new).sum()),miss12_20=int(((g.peak<20)&~g.new).sum())))).reset_index()
    pf.round(1).to_csv(A.out+'/recall_per_frame.csv',index=False)
    for k,v in outs.items(): v.round(1).to_csv(f'{A.out}/recall_{k}.csv');print('==',k);print(v.round(1).to_string())
    print('== worst frames (n>=30, by miss count)');print(pf.sort_values('miss',ascending=False).head(15).round(1).to_string())
    print('== worst frames by recall of peak>=20 dots (n20>=20)')
    B20=B[B.peak>=20];w=B20.groupby(['folder','fr']).apply(lambda g:pd.Series(dict(n=len(g),new=g.new.mean()*100,base=g.base.mean()*100))).reset_index();print(w[w.n>=20].sort_values('new').head(10).round(1).to_string())
    json.dump(dict(detect_dir=A.detect_dir,baseline=A.baseline,thr=A.thr,n_ref=len(B)),open(A.out+'/recall_meta.json','w'))
if __name__=='__main__': main()
