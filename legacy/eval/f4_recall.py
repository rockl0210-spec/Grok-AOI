import os
"""(4) recall retention: THR=12 raw dark-dot reference (Edge normal frames) -> covered by detection -> covered by filter 'real'."""
from common import *
import cv2
m=pd.read_pickle('master3.pkl') if False else pd.read_pickle('master2.pkl')
m['fr']=m.frame.astype(int);m['fold']=m.source+'/'+m.angle
R=pd.read_pickle(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/preprocess_v3/tables/dots_all.pkl');R=R[R.det=='raw'].reset_index(drop=True)
P2=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/preprocess_v2'
# v2 rough bit at blob
nu2=np.zeros(len(R),int)
for (f,fr),g in R.groupby(['folder','fr']):
    a=cv2.imread(f'{P2}/{f}/{fr:02d}_nuisance.png',0);nu2[g.index]=a[g.cy.astype(int).clip(0,2055),g.cx.astype(int).clip(0,2463)]
R['nu2']=nu2;R['peak']=R.peak_raw
R['bin']=pd.cut(R.peak,[12,20,40,1e9],right=False,labels=['12-20','20-40','40+'])
R['zone3']=R.zone.map({0:'contam(v3 oil/water)',1:'stain-zone',2:'rough',3:'clean'})
R['rough_v2']=(R.nu2&4)>0
cov=np.zeros(len(R),bool);covr=np.zeros(len(R),bool);covn=np.zeros(len(R),bool);rc=['']*len(R);rid=['']*len(R);nreal=np.zeros(len(R),int)
cg={k:g for k,g in m.groupby(['fold','fr'])}
for (f,fr),g in R.groupby(['folder','fr']):
    c=cg.get((f,fr))
    if c is None: continue
    bx,by,bw,bh=[c[k].values for k in ['bbox_x','bbox_y','bbox_w','bbox_h']];ca=c.area_px.values;rl=(c.decision=='real').values;rr=c.reason_code.values;ids=c.candidate_id.values
    cx=g.cx.values[:,None];cy=g.cy.values[:,None];ar=g.area.values[:,None]
    H=(cx>=bx-2)&(cx<=bx+bw+2)&(cy>=by-2)&(cy<=by+bh+2)&(ca<=30*ar)
    for j,i in enumerate(g.index):
        h=np.nonzero(H[j])[0]
        if len(h)==0: continue
        cov[i]=True;hr=h[rl[h]]
        if len(hr): covr[i]=True;nreal[i]=len(hr);rid[i]=ids[hr[0]]
        else:
            covn[i]=True;rc[i]=rr[h[np.argmin(ca[h])]].split(';')[0]
R['det_cov']=cov;R['real_cov']=covr;R['nuis_only']=covn;R['nuis_reason']=rc;R['real_id']=rid
R.to_pickle('recall_blobs.pkl')
out=open(T+'recall.txt','w')
def pr(*a): print(*a);print(*a,file=out)
def agg(g):
    c=g.det_cov.sum();return pd.Series(dict(n=len(g),det_cov=c,det_cov_pct=100*c/len(g),real_cov=g.real_cov.sum(),real_pct_of_all=100*g.real_cov.mean(),real_retention_pct=100*g.real_cov.sum()/max(c,1)))
for name,keys in [('bin',['bin']),('bin_x_area12',['bin',R.area>=12]),('zone3_x_bin',['zone3','bin']),('rough_v2_x_bin',['rough_v2','bin']),('folder_x_bin',['folder','bin']),('folder_x_zone3',['folder','zone3'])]:
    t=R.groupby(keys,observed=True).apply(agg).round(1);t.to_csv(T+f'recall_{name}.csv');pr('\n##',name);pr(t)
# fine bins for faint
R['fine']=pd.cut(R.peak,[12,14,16,20,25,30,40,60,1e9],right=False)
t=R.groupby(['rough_v2','fine'],observed=True).apply(agg).round(1);t.to_csv(T+'recall_rough_x_finebin.csv');pr('\n## rough(v2) x fine bin');pr(t)
# what are the covered-but-not-real?
pr('\n## reasons for covered-but-nuisance blobs (all)');pr(R[R.nuis_only].nuis_reason.value_counts().head(12))
pr('by bin');pr(pd.crosstab(R[R.nuis_only].nuis_reason,R[R.nuis_only].bin).sort_values('12-20',ascending=False).head(10))
# faint gray dots: peak 12-20, non-rough clean zone
fg=R[(R.peak<20)];pr('\nfaint (<20) blobs: n',len(fg),'det cov %.1f real cov %.1f retention %.1f'%(100*fg.det_cov.mean(),100*fg.real_cov.mean(),100*fg.real_cov.sum()/fg.det_cov.sum()))
# Edge1/090/20 left edge dots
L=pd.read_csv(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/detect_v31/tables/t3_leftrow_blobs.csv');pr('\nleftrow cols',L.columns.tolist())
e=R[(R.folder=='Edge1/090')&(R.fr==20)];pr('Edge1/090/20 all blobs',len(e),'det',int(e.det_cov.sum()),'real',int(e.real_cov.sum()));
for nm,g in [('x<780',e[e.cx<780]),('x<260',e[e.cx<260])]: pr(nm,agg(g).round(1).to_dict());pr(g.groupby('bin',observed=True).apply(agg).round(1))
pr('\nnamed dots (x~116-128; y 771,1011,1090,1932):')
nd=e[(e.cx>=100)&(e.cx<=145)&(e.cy.sub(771).abs()<25)|(e.cx>=100)&(e.cx<=145)&(e.cy.sub(1011).abs()<25)|(e.cx>=100)&(e.cx<=145)&(e.cy.sub(1090).abs()<25)|(e.cx>=100)&(e.cx<=145)&(e.cy.sub(1932).abs()<25)]
pr(nd[['cx','cy','area','peak','zone3','det_cov','real_cov','nuis_reason','real_id']].round(1).to_string())
# leftmost 20 columns in x<260 by frame 18-23
pr('\nEdge1/090 frames 18-23 left x<260');lg=R[(R.folder=='Edge1/090')&R.fr.between(18,23)&(R.cx<260)]
pr(lg.groupby('fr').apply(agg).round(1))
pr('reasons (nuisance-only) among those:',lg[lg.nuis_only].nuis_reason.value_counts().to_dict())
R.to_pickle('recall_blobs.pkl')
