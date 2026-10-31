import cv2,numpy as np,pandas as pd,json,sys,glob,os
from scipy.spatial import cKDTree
REF_H, REF_W = 2056, 2464
DB = os.environ.get('GROK_AOI_DB') or os.environ.get('AOI_DB')
PRE = os.environ.get('GROK_AOI_PRE') or os.environ.get('PRE_ROOT') or os.environ.get('AOI_PRE')
def configure(db, pre):
    global DB, PRE
    DB, PRE = str(db), str(pre)
def _need():
    if not DB or not PRE:
        raise RuntimeError('persist.configure(db, pre) was not called (or set GROK_AOI_DB and GROK_AOI_PRE)')
def ld(src,ang,k):
    _need()
    p=f'{DB}/{src}/{ang}/{k:02d}.png'
    im=cv2.imread(p,0)
    if im is None: raise FileNotFoundError(p)
    return im
def frames_on_disk(src, ang):
    _need()
    out=[]
    for p in glob.glob(f'{DB}/{src}/{ang}/*.png'):
        stem=os.path.splitext(os.path.basename(p))[0]
        if stem.isdigit(): out.append(int(stem))
    return sorted(set(out))
def hp(im,s): f=im.astype(np.float32); return f-cv2.GaussianBlur(f,(0,0),s)
def ncc_edge(a,b):
    A=hp(a,5);B=hp(b,5); t=B[10:90,200:2200]
    r=cv2.matchTemplate(A[1840:,:],t,cv2.TM_CCOEFF_NORMED); mv=float(r.max()); iy,ix=np.unravel_index(r.argmax(),r.shape)
    return mv,(200-ix, -(iy+1840-10))
def ncc_side(a,b):
    A=hp(a,4);B=hp(b,4); t=B[400:1600,10:110]
    r=cv2.matchTemplate(A[360:1640,1800:],t,cv2.TM_CCOEFF_NORMED)
    mv=float(r.max()); iy,ix=np.unravel_index(r.argmax(),r.shape)
    return mv,(-(ix+1800-10), 40-iy)
def _scaled_ncc(a, b, edge):
    """Same NCC windows as ncc_edge / ncc_side, scaled when the frame is not 2464x2056."""
    H, W = a.shape[:2]
    sy, sx = H / float(REF_H), W / float(REF_W)
    def box(y0, y1, x0, x1):
        Y0=max(0, min(H-1, int(round(y0*sy)))); Y1=max(Y0+1, min(H, int(round(y1*sy))))
        X0=max(0, min(W-1, int(round(x0*sx)))); X1=max(X0+1, min(W, int(round(x1*sx))))
        return Y0, Y1, X0, X1
    A, B = hp(a, 5 if edge else 4), hp(b, 5 if edge else 4)
    if edge:
        ty0,ty1,tx0,tx1 = box(10, 90, 200, 2200)
        sy0 = max(0, min(H-1, int(round(1840*sy))))
        t, search = B[ty0:ty1, tx0:tx1], A[sy0:, :]
        if t.size==0 or search.shape[0] < t.shape[0] or search.shape[1] < t.shape[1]:
            return 0.0, (0.0, -1940.0*sy)
        r=cv2.matchTemplate(search, t, cv2.TM_CCOEFF_NORMED)
        mv=float(r.max()); iy,ix=np.unravel_index(r.argmax(), r.shape)
        return mv, (float(tx0 - ix), float(ty0 - (sy0+iy)))
    ty0,ty1,tx0,tx1 = box(400, 1600, 10, 110)
    sy0,sy1,sx0,sx1 = box(360, 1640, 1800, REF_W)
    t, search = B[ty0:ty1, tx0:tx1], A[sy0:sy1, sx0:sx1]
    if t.size==0 or search.shape[0] < t.shape[0] or search.shape[1] < t.shape[1]:
        return 0.0, (-2251.0*sx, 0.0)
    r=cv2.matchTemplate(search, t, cv2.TM_CCOEFF_NORMED)
    mv=float(r.max()); iy,ix=np.unravel_index(r.argmax(), r.shape)
    return mv, (float(tx0 - (sx0+ix)), float(ty0 - (sy0+iy)))
def ncc_pair(a, b, edge):
    H, W = a.shape[:2]
    if H==REF_H and W==REF_W:
        return (ncc_edge if edge else ncc_side)(a, b)
    return _scaled_ncc(a, b, edge)
def vote(pa,pb,pred,rad=7):
    if len(pa)==0 or len(pb)==0: return pred,0,0.0
    tb=cKDTree(pb); lst=tb.query_ball_point(pa+np.array(pred),r=rad)
    offs=[pb[j]-pa[i] for i,l in enumerate(lst) for j in l]
    if not offs: return pred,0,0.0
    o=np.array(offs)
    H,xe,ye=np.histogram2d(o[:,0],o[:,1],bins=[np.arange(pred[0]-rad,pred[0]+rad+1.01,1.0),np.arange(pred[1]-rad,pred[1]+rad+1.01,1.0)])
    ix,iy=np.unravel_index(H.argmax(),H.shape)
    c=np.array([(xe[ix]+xe[ix+1])/2,(ye[iy]+ye[iy+1])/2])
    m=np.hypot(o[:,0]-c[0],o[:,1]-c[1])<1.6
    return tuple(o[m].mean(0)),int(H.max()),float(np.mean(H))
def run_folder(folder,M,log):
    src,ang=folder.split('_',1); edge=src.startswith('Edge')
    G=M[M.folder==folder] if len(M) else M
    by={k:g for k,g in G.groupby('fr')} if len(G) else {}
    disk=frames_on_disk(src, ang)
    if not disk and len(G): disk=sorted(int(x) for x in G.fr.unique())
    cache={}
    def I(k):
        if k not in cache:
            if len(cache)>3: cache.pop(next(iter(cache)))
            cache[k]=ld(src,ang,k)
        return cache[k]
    if not disk: return [], by
    H,W=I(disk[0]).shape[:2]
    have=set(disk)
    pairs=[]
    for k in disk:
        if k+1 not in have: continue
        mv,off=ncc_pair(I(k),I(k+1),edge); pairs.append(dict(k=k,ncc=mv,off=off))
    good=[p for p in pairs if p['ncc']>=0.2]
    if good: med=np.median([p['off'] for p in good],axis=0)
    else: med=np.array([0.0, -1940.0*(H/float(REF_H))] if edge else [-2251.0*(W/float(REF_W)), 0.0])
    # per-pair refine by candidate votes (same thresholds as the reference-size implementation)
    res=[]
    for p in pairs:
        k=p['k']; ga,gb=by.get(k),by.get(k+1)
        pred=p['off'] if p['ncc']>=0.2 else tuple(med)
        if ga is None or gb is None or len(ga)==0 or len(gb)==0:
            use,votes,method=pred,0,('ncc' if p['ncc']>=0.2 else 'fallback')
        else:
            pa=np.c_[ga.cx,ga.cy]; pb=np.c_[gb.cx,gb.cy]
            sa=(ga.cy.values>=-pred[1]-20) if edge else (ga.cx.values>=-pred[0]-20)
            sb=(gb.cy.values<=H+pred[1]+20) if edge else (gb.cx.values<=W+pred[0]+20)
            if not np.any(sa) or not np.any(sb):
                off,votes,mean_bin=pred,0,0.0
            else:
                off,votes,mean_bin=vote(pa[sa],pb[sb],pred)
            vote_ok=(votes>=5 and votes>=3*max(mean_bin,0.3))
            use=off if vote_ok else pred
            method='vote' if vote_ok else ('ncc' if p['ncc']>=0.2 else 'fallback')
        res.append(dict(pair=f'{k:02d}->{k+1:02d}',ncc=round(float(p['ncc']),3),ox=float(use[0]),oy=float(use[1]),votes=int(votes),method=method,trusted=bool(method!='fallback')))
        log(f'{folder} {k:02d}->{k+1:02d} ncc={p["ncc"]:.2f} off=({float(use[0]):.1f},{float(use[1]):.1f}) votes={votes} {method}')
    return res,by
def match_all(folder,M,res,by,rng=np.random.RandomState(0)):
    """for each candidate returns tested(bool), match(bool), ctrl(float), partner ids. uses fwd & backward"""
    src,ang=folder.split('_',1)
    G=M[M.folder==folder] if len(M) else M.iloc[0:0]
    if len(G)==0:
        return pd.DataFrame(dict(id=[],p_tested=[],p_match=[],p_ctrl=[],p_frames=[]))
    H,W=ld(src,ang,int(G.fr.iloc[0])).shape[:2]
    n=len(G); tested=np.zeros(n,bool); match=np.zeros(n,bool); ctrl=np.zeros(n,np.float32); pf=[[] for _ in range(n)]
    pos={i:j for j,i in enumerate(G.index)}
    masks={}
    def roi(k):
        if k not in masks:
            if len(masks)>3: masks.pop(next(iter(masks)))
            masks[k]=cv2.imread(f'{PRE}/{src}/{ang}/{k:02d}_mask.png',0)>0
        return masks[k]
    tr={k:(cKDTree(np.c_[g.cx,g.cy]),g) for k,g in by.items()}
    for r in res:
        k=int(r['pair'][:2]); off=np.array([r['ox'],r['oy']])
        if not r['trusted']: continue
        for (s,d,sign) in ((k,k+1,1),(k+1,k,-1)):
            if s not in by or d not in tr or len(by[s])==0: continue
            g=by[s]; t,gd=tr[d]
            o=off*sign
            p=np.c_[g.cx,g.cy]+o
            inside=(p[:,0]>=8)&(p[:,0]<W-8)&(p[:,1]>=8)&(p[:,1]<H-8)
            if not inside.any(): continue
            mk=roi(d); pi=np.clip(p[inside].astype(int),0,[W-1,H-1]); inroi=mk[pi[:,1],pi[:,0]]
            idx=np.where(inside)[0][inroi]
            if len(idx)==0: continue
            rad=np.maximum(3.0,0.5*np.maximum(g.bw.values[idx],g.bh.values[idx])+2.0)
            rad=np.minimum(rad,25)
            for ii,rr in zip(idx,rad):
                gi=pos[g.index[ii]]
                cand=t.query_ball_point(p[ii],rr)
                tested[gi]=True
                # polarity same always within mode; require area ratio sanity
                a0=g.area_px.values[ii]
                ok=[c for c in cand if 0.2<=gd.area_px.values[c]/max(a0,1)<=5.0]
                if ok:
                    match[gi]=True; pf[gi].append(int(d))
                # chance control: 4 random displaced probes
                hits=0
                for _ in range(4):
                    ang_=rng.uniform(0,2*np.pi); dist=rng.uniform(25,60)
                    q=p[ii]+dist*np.array([np.cos(ang_),np.sin(ang_)])
                    c2=t.query_ball_point(q,rr)
                    if any(0.2<=gd.area_px.values[c]/max(a0,1)<=5.0 for c in c2): hits+=1
                ctrl[gi]=max(ctrl[gi],hits/4.0) if tested[gi] else hits/4.0
    return pd.DataFrame(dict(id=G.id.values,p_tested=tested,p_match=match,p_ctrl=ctrl,p_frames=[sorted(set(x)) for x in pf]),index=G.index)
if __name__=='__main__':
    work=sys.argv[1] if len(sys.argv)>1 else '.'
    M=pd.read_pickle(os.path.join(work,'M.pkl') if os.path.isfile(os.path.join(work,'M.pkl')) else os.path.join(work,'master.pkl'))
    folders=sys.argv[2:] or sorted(M.folder.unique())
    for f in folders:
        L=open(os.path.join(work,f'persist_{f}.log'),'w')
        def log(s, L=L): L.write(s+'\n'); L.flush()
        res,by=run_folder(f,M,log)
        json.dump(res,open(os.path.join(work,f'reg_{f}.json'),'w'),indent=0)
        out=match_all(f,M,res,by)
        out.to_pickle(os.path.join(work,f'persist_{f}.pkl'))
        print(f,'tested',out.p_tested.mean().round(3) if len(out) else None,'match|tested',out[out.p_tested].p_match.mean().round(3) if len(out) and out.p_tested.any() else None,flush=True)
