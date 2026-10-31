import os
"""Stage 1: registration, cross-frame persistence, fixed-pattern test, noise floor, geometry & texture features."""
import cv2,numpy as np,pandas as pd,json,sys,os
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
W,H=2464,2056
DB=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_db'; PRE=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/preprocess'; DET=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/detect'
WORK=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_work'
def rd(p,flag=0): return cv2.imread(p,flag)
def hp3(im):
    f=im.astype(np.float32); return f-cv2.GaussianBlur(f,(0,0),3)

def ncc_shift(mode,a,b):
    A=hp3(a);B=hp3(b)
    if mode=='edge':
        t=100; tpl=A[H-t:H,200:W-200]; res=cv2.matchTemplate(B[:t+100,:],tpl,cv2.TM_CCOEFF_NORMED)
        _,mx,_,loc=cv2.minMaxLoc(res); x,y=loc; r2=res.copy(); r2[max(0,y-6):y+7,max(0,x-6):x+7]=-1
        return mx,float(r2.max()),(loc[0]-200,loc[1]-(H-t))
    else:
        t=150; tpl=A[300:H-300,W-t:W]; res=cv2.matchTemplate(B[:,:t+100],tpl,cv2.TM_CCOEFF_NORMED)
        _,mx,_,loc=cv2.minMaxLoc(res); x,y=loc; r2=res.copy(); r2[max(0,y-6):y+7,max(0,x-6):x+7]=-1
        return mx,float(r2.max()),(loc[0]-(W-t),loc[1]-300)

def vote_shift(mode,A,B):
    if len(A)==0 or len(B)==0: return 0,None
    d=(B[None,:,:]-A[:,None,:]).reshape(-1,2)
    ax=1 if mode=='edge' else 0
    lo,hi=(-2100,-1750) if mode=='edge' else (-2400,-2150)
    m=(d[:,ax]>=lo)&(d[:,ax]<=hi)&(np.abs(d[:,1-ax])<=30)
    dd=d[m]
    if len(dd)==0: return 0,None
    best=(0,None)
    for c in dd[::max(1,len(dd)//400)]:
        n=((np.abs(dd[:,0]-c[0])<=3)&(np.abs(dd[:,1]-c[1])<=3)).sum()
        if n>best[0]: best=(n,c)
    c=best[1]; s=dd[(np.abs(dd[:,0]-c[0])<=3)&(np.abs(dd[:,1]-c[1])<=3)]
    return best[0],(float(np.median(s[:,0])),float(np.median(s[:,1])))

def noise_floor(img,mask):
    f=img.astype(np.float32); hp=f-cv2.GaussianBlur(f,(0,0),1.5)
    # block-wise MAD in ROI (64px blocks), median over valid blocks; correct for hp attenuation of white noise
    bs=64; vals=[]
    for y in range(0,H-bs+1,bs):
        for x in range(0,W-bs+1,bs):
            if mask[y:y+bs,x:x+bs].mean()<250: continue
            b=hp[y:y+bs,x:x+bs]; vals.append(1.4826*np.median(np.abs(b-np.median(b))))
    if not vals: return np.nan
    return float(np.median(vals))/0.93   # 0.93 ~ sqrt(1-1/(4*pi*1.5^2)) attenuation of gaussian(1.5) subtraction

def run(key):
    mode,name=key.split('/'); src,ang=name.split('_')
    df=pd.read_csv(f'{DET}/{mode}/{name}/candidates.csv',dtype={'frame':str,'angle':str})
    frames=sorted(df.frame.unique()); nF=len(frames)
    fidx={f:i for i,f in enumerate(frames)}
    pol_dark=(mode=='edge')
    # ---- registration
    pairs={}; log=[]
    for k in range(nF-1):
        fa,fb=frames[k],frames[k+1]
        a=rd(f'{DB}/{src}/{ang}/{fa}.png'); b=rd(f'{DB}/{src}/{ang}/{fb}.png')
        mx,sec,s=ncc_shift(mode,a,b)
        A=df[(df.frame==fa)&(df.contrast.abs()>=12)][['centroid_x','centroid_y']].values
        B=df[(df.frame==fb)&(df.contrast.abs()>=12)][['centroid_x','centroid_y']].values
        nv,vs=vote_shift(mode,A,B)
        pairs[k]=dict(ncc=mx,ncc2=sec,ncc_shift=s,votes=nv,vote_shift=vs)
    good=[p['ncc_shift'] for p in pairs.values() if p['ncc']>=0.08 and p['ncc']>=2.5*p['ncc2']]
    med=np.median(np.array(good),axis=0) if good else (np.array([-1.,-1940.]) if mode=='edge' else np.array([-2262.,0.5]))
    shifts={}
    for k,p in pairs.items():
        if p['ncc']>=0.08 and p['ncc']>=2.5*p['ncc2']: s=p['ncc_shift'];m='ncc'
        elif p['votes']>=4: s=p['vote_shift'];m='vote'
        else: s=tuple(med);m='median'
        # sanity vs median
        if abs(s[0]-med[0])>20 or abs(s[1]-med[1])>20: s=tuple(med);m='median(out-of-range)'
        shifts[k]=dict(dx=float(s[0]),dy=float(s[1]),method=m,ncc=float(p['ncc']),votes=int(p['votes']))
    # ---- noise floor + per-frame arrays
    nf=[];  per_frame_noise={}
    cache={}
    def getimg(f):
        if f not in cache:
            cache.clear()  # keep only a few
            cache[f]=(rd(f'{DB}/{src}/{ang}/{f}.png'),rd(f'{PRE}/{src}/{ang}/{f}_mask.png'))
        return cache[f]
    n=len(df)
    df['persist_n']=0; df['checkable_n']=0; df['persist_det']=0; df['persist_img']=0; df['obs_ratio']=np.nan
    df['match_prev']='' ; df['match_next']=''
    cx=df.centroid_x.values; cy=df.centroid_y.values; fr=df.frame.map(fidx).values
    feret=df.feret_px.values; contrast=df.contrast.values; sig_loc=df.noise_sigma.values
    idx_by_f={i:np.where(fr==i)[0] for i in range(nF)}
    ids=df.id.values
    matches=[]  # (idx_a, idx_b)
    imgs={}
    def load(i):
        f=frames[i]
        if f not in imgs:
            if len(imgs)>3: imgs.pop(next(iter(imgs)))
            imgs[f]=(rd(f'{DB}/{src}/{ang}/{f}.png').astype(np.float32),rd(f'{PRE}/{src}/{ang}/{f}_mask.png'))
        return imgs[f]
    def eval_dir(i,j,sgn):
        # candidates of frame i evaluated in neighbour frame j.  shift s: pos_in_(k+1)=pos_in_k+s
        k=min(i,j); s=shifts[k]; dx,dy=(s['dx'],s['dy']) if j>i else (-s['dx'],-s['dy'])
        ia=idx_by_f[i]; ib=idx_by_f[j]
        if len(ia)==0: return
        img_b,mask_b=load(j)
        mx=cx[ia]+dx; my=cy[ia]+dy
        marg=12
        ins=(mx>=marg)&(mx<=W-1-marg)&(my>=marg)&(my<=H-1-marg)
        okm=np.zeros(len(ia),bool)
        xi=np.clip(np.round(mx[ins]).astype(int),0,W-1); yi=np.clip(np.round(my[ins]).astype(int),0,H-1)
        okm[ins]=mask_b[yi,xi]>0
        # ROI must hold around (3px) too -> use erosion approx: check 4 neighbours
        for ddx,ddy in ((6,0),(-6,0),(0,6),(0,-6)):
            xx=np.clip(np.round(mx+ddx).astype(int),0,W-1); yy=np.clip(np.round(my+ddy).astype(int),0,H-1)
            okm&=mask_b[yy,xx]>0
        chk=ia[okm]
        if len(chk)==0: return
        df.loc[df.index[chk],'checkable_n']+=1
        # detection match
        tb=cKDTree(np.c_[cx[ib],cy[ib]]) if len(ib) else None
        img=img_b
        for q,ai in zip(np.where(okm)[0],chk):
            px,py=mx[q],my[q]
            tol=max(5.0,0.3*feret[ai]) if feret[ai]<300 else 0.15*feret[ai]
            det=False
            if tb is not None:
                for bj in tb.query_ball_point([px,py],tol):
                    b=ib[bj]
                    if np.sign(contrast[b])!=np.sign(contrast[ai]): continue
                    r=df.area_px.values[b]/df.area_px.values[ai]
                    if feret[ai]<300 and (r>6 or r<1/6): continue
                    det=True; matches.append((ai,b)); break
            ratio=np.nan; pim=False
            if feret[ai]<=60:
                r=int(4+0.5*feret[ai]+3); R=r+14
                x0,x1=int(max(0,px-R)),int(min(W,px+R+1)); y0,y1=int(max(0,py-R)),int(min(H,py+R+1))
                crop=img[y0:y1,x0:x1]
                if crop.size>0:
                    sm=cv2.blur(crop,(3,3))
                    bg=np.median(crop)
                    xa,xb=int(max(0,px-r-x0)),int(min(x1-x0,px+r+1-x0)); ya,yb=int(max(0,py-r-y0)),int(min(y1-y0,py+r+1-y0))
                    win=sm[ya:yb,xa:xb]
                    if win.size>0:
                        sig=(bg-win.min()) if pol_dark else (win.max()-bg)
                        ratio=sig/abs(contrast[ai])
                        pim=(sig>=0.6*abs(contrast[ai])) and (sig>=4*max(sig_loc[ai],0.6))
            else: pim=det
            if det: df.iat[ai,df.columns.get_loc('persist_det')]=1
            if pim or det:
                df.iat[ai,df.columns.get_loc('persist_n')]+=1
            if pim: df.iat[ai,df.columns.get_loc('persist_img')]=1
            df.iat[ai,df.columns.get_loc('obs_ratio')]=ratio
    for i in range(nF):
        for j in (i-1,i+1):
            if 0<=j<nF: eval_dir(i,j,0)
    # noise floors
    for i,f in enumerate(frames):
        img,mask=load(i); per_frame_noise[f]=noise_floor(img,mask)
    # ---- fixed pattern
    nonan=~df.anomaly_frame.values
    P=np.c_[cx,cy]; t=cKDTree(P[nonan]); sub=np.where(nonan)[0]
    nb=t.query_ball_point(P,2.5)
    fixed=np.zeros(n,int)
    for q in range(n):
        fs={fr[sub[m]] for m in nb[q]}-{fr[q]}
        fixed[q]=len(fs)
    df['fixed_nframes']=fixed
    # ---- geometry: dt-based edge orientation for near-edge candidates, using ROI mask
    df['edge_align_deg']=np.nan; df['at_roi_mask']=1
    near=np.where(df.dist_to_roi_edge_px.values<=40)[0]
    for i in range(nF):
        sel=[q for q in near if fr[q]==i]
        if not sel: continue
        _,mask=load(i)
        dt=cv2.distanceTransform((mask>0).astype(np.uint8),cv2.DIST_L2,5)
        dts=cv2.GaussianBlur(dt,(0,0),3)
        gx=cv2.Sobel(dts,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(dts,cv2.CV_32F,0,1,ksize=3)
        for q in sel:
            xi=int(np.clip(round(cx[q]),0,W-1)); yi=int(np.clip(round(cy[q]),0,H-1))
            df.iat[q,df.columns.get_loc('at_roi_mask')]=int(mask[yi,xi]>0)
            g=np.hypot(gx[yi,xi],gy[yi,xi])
            if g<1e-3: continue
            tang=np.degrees(np.arctan2(gx[yi,xi],-gy[yi,xi]))%180  # tangent direction of boundary
            # candidate axis from mask patch
            pp=rd(df.mask_patch.values[q],0)
            if pp is None: continue
            ys,xs=np.nonzero(pp)
            if len(xs)<3: continue
            c=np.cov(np.c_[xs,ys].T); w,v=np.linalg.eigh(c); ax=np.degrees(np.arctan2(v[1,1],v[0,1]))%180
            d=abs(ax-tang); d=min(d,180-d)
            df.iat[q,df.columns.get_loc('edge_align_deg')]=d
    # ---- texture features for large / non-particle candidates
    big=np.where((df.area_px.values>=60)|(df['class'].values!='particle'))[0]
    cols=['tx_std','tx_edge_sharp','tx_grad_rel','tx_coh','tx_ring','tx_fill']
    for c in cols: df[c]=np.nan
    for i in range(nF):
        sel=[q for q in big if fr[q]==i]
        if not sel: continue
        img,_=load(i)
        for q in sel:
            x,y,w,h=int(df.bbox_x.values[q]),int(df.bbox_y.values[q]),int(df.bbox_w.values[q]),int(df.bbox_h.values[q])
            if w*h>4_000_000 or (w>2500): pass
            pp=rd(df.mask_patch.values[q],0)
            if pp is None: continue
            pad=6; x0,y0=max(0,x-pad),max(0,y-pad); x1,y1=min(W,x+w+pad),min(H,y+h+pad)
            crop=img[y0:y1,x0:x1]
            sc=1.0
            if max(crop.shape)>600:
                sc=600/max(crop.shape); crop=cv2.resize(crop,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA)
            m=np.zeros((y1-y0,x1-x0),np.uint8); m[y-y0:y-y0+pp.shape[0],x-x0:x-x0+pp.shape[1]]=(pp>0)
            if sc!=1.0: m=cv2.resize(m,(crop.shape[1],crop.shape[0]),interpolation=cv2.INTER_NEAREST)
            mb=m>0
            if mb.sum()<5: continue
            g1=cv2.GaussianBlur(crop,(0,0),1.0)
            gx=cv2.Sobel(g1,cv2.CV_32F,1,0); gy=cv2.Sobel(g1,cv2.CV_32F,0,1); gm=np.hypot(gx,gy)/8.0
            dil=cv2.dilate(m,np.ones((3,3),np.uint8))>0; ero=cv2.erode(m,np.ones((3,3),np.uint8))>0
            ring=dil&~ero
            con=abs(df.contrast.values[q])
            df.iat[q,df.columns.get_loc('tx_std')]=float(crop[mb].std())
            df.iat[q,df.columns.get_loc('tx_edge_sharp')]=float(gm[ring].mean()/max(con,1)) if ring.any() else np.nan
            df.iat[q,df.columns.get_loc('tx_grad_rel')]=float(gm[ero].mean()/max(con,1)) if ero.any() else np.nan
            # structure-tensor coherence inside mask
            gxx=cv2.GaussianBlur(gx*gx,(0,0),3);gyy=cv2.GaussianBlur(gy*gy,(0,0),3);gxy=cv2.GaussianBlur(gx*gy,(0,0),3)
            lam=np.sqrt((gxx-gyy)**2+4*gxy**2); coh=lam/(gxx+gyy+1e-6)
            df.iat[q,df.columns.get_loc('tx_coh')]=float(np.median(coh[ero])) if ero.sum()>10 else np.nan
            # ring-ness: inner core vs periphery darkness (dark polarity: depth) 
            dtm=cv2.distanceTransform(m,cv2.DIST_L2,3)
            mx=dtm.max()
            if mx>=6:
                core=dtm>=0.6*mx; per=(dtm>0)&(dtm<=0.3*mx)
                bgv=df.local_bg_gray.values[q]
                dc=abs(bgv-crop[core].mean()); dp=abs(bgv-crop[per].mean())
                df.iat[q,df.columns.get_loc('tx_ring')]=float(dc/max(dp,1e-3))
            df.iat[q,df.columns.get_loc('tx_fill')]=float(mb.sum()/(m.shape[0]*m.shape[1]))
    nfl=[v for v in per_frame_noise.values() if np.isfinite(v)]
    df['noise_floor_frame']=df.frame.map(per_frame_noise)
    info=dict(key=key,shifts=shifts,median_shift=list(map(float,med)),noise_floor_median=float(np.median(nfl)),
              noise_floor_p10=float(np.percentile(nfl,10)),noise_floor_p90=float(np.percentile(nfl,90)),per_frame_noise=per_frame_noise,
              n_matches=len(matches))
    # unique ids via union find on matches saved separately
    pd.to_pickle((df,matches,info),f'{WORK}/{name}.pkl')
    return name
if __name__=='__main__':
    from multiprocessing import Pool
    keys=[f'edge/{a}' for a in ['Edge1_000','Edge1_090','Edge2_000','Edge2_090']]+[f'side/Side_{a}' for a in ['000','090','180','270']]
    if len(sys.argv)>1: keys=sys.argv[1:]
    with Pool(min(8,len(keys))) as p:
        for r in p.imap_unordered(run,keys): print('done',r,flush=True)
