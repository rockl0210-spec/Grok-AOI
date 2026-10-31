import os
import sys;sys.path.insert(0,os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/detect_v1')
import cv2,numpy as np,pandas as pd,glob
from multiprocessing import Pool
PRE=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/preprocess';DB=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_db'
def bgmed(a,k=25,ds=4):
    s=cv2.resize(a,None,fx=1/ds,fy=1/ds,interpolation=cv2.INTER_AREA)
    m=cv2.medianBlur(s,k)
    return cv2.resize(m,(a.shape[1],a.shape[0]),interpolation=cv2.INTER_LINEAR).astype(np.float32)
def scan(args):
    src,ang,fr,pol,thr=args
    o=cv2.imread(f'{DB}/{src}/{ang}/{fr:02d}.png',0)
    roi=cv2.imread(f'{PRE}/{src}/{ang}/{fr:02d}_mask.png',0)>0
    # fill outside ROI with local value to keep bg median sane: use image as is but replace outside with ROI median
    a=o.copy();a[~roi]=int(np.median(o[roi]))
    g=cv2.GaussianBlur(a.astype(np.float32),(0,0),0.8)
    bg=bgmed(a)
    r=(bg-g) if pol=='dark' else (g-bg)
    m=((r>=thr)&roi).astype(np.uint8)
    n,l,st,ce=cv2.connectedComponentsWithStats(m,connectivity=8)
    roi_er=cv2.erode(roi.astype(np.uint8),np.ones((7,7),np.uint8))>0
    rows=[]
    for i in range(1,n):
        x,y,w,h,A=st[i]
        if A<3 or A>1500: continue
        cx,cy=ce[i]
        if not roi_er[int(cy),int(cx)]: continue
        ys,xs=np.nonzero(l[y:y+h,x:x+w]==i)
        pts=np.c_[xs,ys].astype(np.float32)
        (_,_),(rw,rh),_=cv2.minAreaRect(pts)
        L,W=max(rw,rh)+1,min(rw,rh)+1
        sol=A/max(w*h,1)
        if (L/W>3.5 and L>=12) or sol<0.3: continue
        rr=r[y:y+h,x:x+w][l[y:y+h,x:x+w]==i]
        rows.append(dict(src=src,ang=ang,fr=fr,cx=cx,cy=cy,x=x,y=y,w=w,h=h,area=A,L=L,W=W,peak=float(rr.max()),mean=float(rr.mean())))
    return rows
def load_old():
    d=pd.read_pickle(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/detect_v2/old/old_master_proxy.pkl')
    return d[['id','folder','fr','bbox_x','bbox_y','bbox_w','bbox_h','centroid_x','centroid_y','area_px']]
def load_new():
    fs=sorted(glob.glob(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/detect/*/*_*/candidates.csv'))
    d=pd.concat([pd.read_csv(f,dtype={'frame':str,'angle':str}) for f in fs],ignore_index=True)
    d['folder']=d.source+'/'+d.angle;d['fr']=d.frame.astype(int)
    return d
def cover(B,C,pad=2,maxratio=30):
    """B blobs df for one folder-frame; C candidates df. returns loose, strict flags"""
    lo=np.zeros(len(B),bool);st=np.zeros(len(B),bool);cid=np.empty(len(B),object)
    if len(C)==0: return lo,st,cid
    bx0=C.bbox_x.values-pad;by0=C.bbox_y.values-pad;bx1=C.bbox_x.values+C.bbox_w.values+pad;by1=C.bbox_y.values+C.bbox_h.values+pad;ca=C.area_px.values;ids=C.id.values
    for i,(cx,cy,ar) in enumerate(zip(B.cx.values,B.cy.values,B.area.values)):
        ins=(cx>=bx0)&(cx<=bx1)&(cy>=by0)&(cy<=by1)
        if ins.any():
            lo[i]=True
            ok=ins&(ca<=maxratio*ar)
            if ok.any(): st[i]=True;cid[i]=ids[ok][0]
    return lo,st,cid
