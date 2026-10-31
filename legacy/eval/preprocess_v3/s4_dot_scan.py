import os
"""Independent THR=12 dark-dot scan on RAW, v2 cleaned, v3 cleaned (Edge, dark dots). Same definition as detect_v2/aoi_recall_eval.py
(bg = 100px-ish median via 4x downsample/k25, gaussian 0.8, residual>=THR, area 3-1500, fill>=0.6, L/W<=2.5, centroid in ROI eroded 7x7).
For every blob detected in any of the 3 images, the peak residual of the SAME pixel region (blob dilated 3x3) is measured in all 3 images.
Usage: python3 s4_dot_scan.py [thr=12]  -> tables/dots_all.pkl, tables/zones.csv"""
import sys,cv2,numpy as np,pandas as pd
from multiprocessing import Pool
P3=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/preprocess';P2=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/preprocess_v2';DB=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_db'
THR=float(sys.argv[1]) if len(sys.argv)>1 else 12
def bgmed(a,k=25,ds=4):
    s=cv2.resize(a,None,fx=1/ds,fy=1/ds,interpolation=cv2.INTER_AREA);m=cv2.medianBlur(s,k)
    return cv2.resize(m,(a.shape[1],a.shape[0]),interpolation=cv2.INTER_LINEAR).astype(np.float32)
def resid(img,roi,pol='dark'):
    a=img.copy();a[~roi]=int(np.median(img[roi]))
    g=cv2.GaussianBlur(a.astype(np.float32),(0,0),0.8);bg=bgmed(a)
    return (bg-g) if pol=='dark' else (g-bg)
def blobs(r,roi,er,thr):
    n,l,st,ce=cv2.connectedComponentsWithStats(((r>=thr)&roi).astype(np.uint8),connectivity=8);out=[]
    for i in range(1,n):
        x,y,w,h,A=st[i]
        if A<3 or A>1500: continue
        cx,cy=ce[i]
        if not er[int(cy),int(cx)]: continue
        sub=l[y:y+h,x:x+w]==i;ys,xs=np.nonzero(sub)
        (_,_),(rw,rh),_=cv2.minAreaRect(np.c_[xs,ys].astype(np.float32))
        L,W=max(rw,rh)+1,min(rw,rh)+1;fill=A/(np.pi/4*L*W)
        if fill<0.6 or L/W>2.5: continue
        out.append((x,y,w,h,A,cx,cy,sub))
    return out
def job(a):
    f,n=a;fr=int(n);src,ang=f.split('/')
    raw=cv2.imread(f'{DB}/{f}/{n}.png',0);roi=cv2.imread(f'{P3}/{f}/{n}_mask.png',0)>0
    nu=cv2.imread(f'{P3}/{f}/{n}_nuisance.png',0);st=cv2.imread(f'{P3}/{f}/{n}_stain.png',0)
    c2=cv2.imread(f'{P2}/{f}/{n}_cleaned.png',0);c3=cv2.imread(f'{P3}/{f}/{n}_cleaned.png',0)
    er=cv2.erode(roi.astype(np.uint8),np.ones((7,7),np.uint8))>0
    R={'raw':resid(raw,roi),'v2':resid(c2,roi),'v3':resid(c3,roi)}
    contam=((nu&(8|16|32))>0);rough=((nu&4)>0)&~contam
    stz=cv2.dilate(((st!=128)&roi).astype(np.uint8),np.ones((15,15),np.uint8))>0
    zone=np.full(roi.shape,3,np.uint8)  # 3 clean
    zone[rough]=2;zone[stz&~contam&~rough]=1;zone[contam]=0   # 0 contam(oil/water/ink) 1 stain-zone only 2 rough only 3 clean
    zone[~er]=255
    stl=(st.astype(np.float32)-128)/8
    # noise sigma per zone per image (robust, on residual)
    sig={};zrows=[]
    for z,nm in [(0,'contam'),(1,'stainzone'),(2,'rough'),(3,'clean')]:
        zm=zone==z;rw=dict(folder=f,frame=n,zone=nm,area=int(zm.sum()))
        for k in R:
            v=R[k][zm][::7] if zm.any() else np.array([0.])
            rw['sig_'+k]=float(1.4826*np.median(np.abs(v-np.median(v)))) if len(v)>50 else np.nan
            rw['nblob_'+k]=0
        sig[nm]=rw;zrows.append(rw)
    rows=[]
    k3=np.ones((3,3),np.uint8)
    for k in R:
        for (x,y,w,h,A,cx,cy,sub) in blobs(R[k],roi,er,THR):
            pad=3;x0=max(x-pad,0);y0=max(y-pad,0);x1=min(x+w+pad,roi.shape[1]);y1=min(y+h+pad,roi.shape[0])
            m=np.zeros((y1-y0,x1-x0),np.uint8);m[y-y0:y-y0+h,x-x0:x-x0+w]=sub;m=cv2.dilate(m,k3)>0
            rec=dict(folder=f,src=src,ang=ang,fr=fr,det=k,cx=cx,cy=cy,area=int(A),zone=int(zone[int(cy),int(cx)]),stain=float(stl[int(cy),int(cx)]),
                     nu=int(nu[int(cy),int(cx)]))
            for kk in R:
                rr=R[kk][y0:y1,x0:x1][m];rec['peak_'+kk]=float(rr.max());rec['n12_'+kk]=int((rr>=THR).sum())
            rows.append(rec);nm=['contam','stainzone','rough','clean'][rec['zone']] if rec['zone']<4 else None
            if nm: sig[nm]['nblob_'+k]+=1
    return rows,zrows
if __name__=='__main__':
    J=[(f,f'{n:02d}') for f in ['Edge1/000','Edge1/090','Edge2/000','Edge2/090'] for n in range(1,45)]
    with Pool(8) as p: res=p.map(job,J,chunksize=2)
    d=pd.DataFrame([r for rr,_ in res for r in rr]);z=pd.DataFrame([r for _,zz in res for r in zz])
    d.to_pickle('tables/dots_all.pkl');z.to_csv('tables/zones.csv',index=False);print(len(d),d.det.value_counts().to_dict())
