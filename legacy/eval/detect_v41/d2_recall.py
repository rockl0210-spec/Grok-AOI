import os
"""Full-frame THR=12 dark-dot Edge reference (185,608 blobs, 176 normal frames; preprocess_v3 dots_all.pkl det=raw) vs v3.1 / v4 / v4.1 candidates.
strict cover = centroid in bbox+-2 and cand area <=30x blob area ('s'); any = no area limit ('a'). Reuses the zone masks / v3.1+v4 covers computed in detect_v4/recall_full.pkl."""
from common import *
R=pd.read_pickle(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/detect_v4/recall_full.pkl').rename(columns={'v31':'s31','v4':'s4','a31':'a31','a4':'a4'})
c=pd.read_pickle('v41.pkl')
def cover(R,c,strict):
    cov=np.zeros(len(R),bool);cg={k:g for k,g in c.groupby(['folder','frame'])}
    for (f,fr),g in R.groupby(['folder','fr']):
        cc=cg.get((f.replace('/','_'),f'{fr:02d}'))
        if cc is None: continue
        bx,by,bw,bh=[cc[x].values for x in ['bbox_x','bbox_y','bbox_w','bbox_h']];ca=cc.area_px.values
        cx=g.cx.values[:,None];cy=g.cy.values[:,None];ar=g.area.values[:,None]
        m=(cx>=bx-2)&(cx<=bx+bw+2)&(cy>=by-2)&(cy<=by+bh+2)
        if strict: m&=(ca<=30*ar)
        cov[g.index.values]=m.any(1)
    return cov
R['s41']=cover(R,c,True);R['a41']=cover(R,c,False)
# coverage by raw_rescue candidates only (does the rescue itself create the gain?)
R['s41_norr']=cover(R,c[~c.raw_rescue],True);R['a41_norr']=cover(R,c[~c.raw_rescue],False)
R.to_pickle('recall_full.pkl')
o=open(T+'recall_full.txt','w')
def pr(*x): print(*x);print(*x,file=o)
def agg(g): return pd.Series(dict(n=len(g),v31=100*g.s31.mean(),v4=100*g.s4.mean(),v41=100*g.s41.mean(),v41_norr=100*g.s41_norr.mean(),d41_4=100*(g.s41.mean()-g.s4.mean()),lost=int((g.s4&~g.s41).sum()),gain=int((~g.s4&g.s41).sum())))
def show(nm,keys,df=R):
    t=df.groupby(keys,observed=True).apply(agg).round(1);t.to_csv(T+'recall_full_'+nm+'.csv');pr('\n##',nm);pr(t.to_string())
show('bin','bin');show('zone5_x_bin',['zone5','bin']);show('folder_x_bin',['folder','bin']);show('folder_x_zone5_x_bin',['folder','zone5','bin']);show('zone3stain_x_bin',['zone','bin'])
R['fine']=pd.cut(R.peak,[12,14,16,20,25,30,40,60,1e9],right=False);show('finebin','fine');show('oil_finebin_by_folder',['folder','fine'],R[R.zone5=='oil/ink'])
R['f1216']=pd.cut(R.peak,[12,16,20,1e9],right=False,labels=['12-16','16-20','20+']);show('folder_x_zone5_x_1216',['folder','zone5','f1216'],R[R.peak<20])
W=R.groupby(['folder','fr']).apply(agg).reset_index();W.to_csv(T+'recall_full_frames.csv',index=False)
pr('\n## worst frames by v4.1 (n>=500)');pr(W[W.n>=500].sort_values('v41').head(12).round(1).to_string())
pr('\n## biggest per-frame drops v4->v4.1 (n>=100)');pr(W[W.n>=100].sort_values('d41_4').head(10).round(1).to_string())
pr('\n## biggest gains');pr(W[W.n>=100].sort_values('d41_4',ascending=False).head(8).round(1).to_string())
pr('\n## churn v4->v4.1: lost',int((R.s4&~R.s41).sum()),'gained',int((~R.s4&R.s41).sum()))
pr(R[R.s4&~R.s41].groupby(['folder','bin'],observed=True).size().unstack(fill_value=0).to_string())
# strong
for nm,z in [('contaminated (oil/ink/water)',R.zone5.isin(['oil/ink','water'])),('ALL zones',R.zone5.notna())]:
    s=R[(R.peak>=40)&z];pr('\n##',nm,'>=40 n',len(s),'| any-size uncovered v31/v4/v41:',int((~s.a31).sum()),int((~s.a4).sum()),int((~s.a41).sum()),'| strict (<=30x):',int((~s.s31).sum()),int((~s.s4).sum()),int((~s.s41).sum()))
    pr(' v41 uncovered any by folder/zone5:',s[~s.a41].groupby(['folder','zone5']).size().to_dict())
s=R[(R.zone==0)&(R.peak>=40)];pr('\n## v3 zone==0 (the 19,843 population) n',len(s),'any v31/v4/v41',int((~s.a31).sum()),int((~s.a4).sum()),int((~s.a41).sum()),'strict',int((~s.s31).sum()),int((~s.s4).sum()),int((~s.s41).sum()))
pr('   without raw_rescue candidates: any',int((~s.a41_norr).sum()),'strict',int((~s.s41_norr).sum()))
