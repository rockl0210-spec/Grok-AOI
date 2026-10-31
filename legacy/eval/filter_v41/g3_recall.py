import os
"""Retention reproduction: THR=12 reference blobs (Edge: preprocess_v3 dots_all raw; Side: detect_v3 ref_blobs) -> covered -> real / real-nonreview; v3.1, v4, v4.1."""
from common import *
m=pd.read_pickle('master41.pkl');m4=pd.read_pickle(E4+'master4.pkl');mm=pd.read_pickle(E4+'master31.pkl')
if 'base' not in m4: m4['base']=m4.reason_code.str.split('[+;]').str[0]
R=pd.read_pickle(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/preprocess_v3/tables/dots_all.pkl');R=R[R.det=='raw'].reset_index(drop=True)
R['bin']=pd.cut(R.peak_raw,[12,20,40,1e9],right=False,labels=['12-20','20-40','40+'])
R['zone3']=R.zone.map({0:'contam',1:'stain-zone',2:'rough',3:'clean'});R['fkey']=R.folder.str.replace('/','_');R['area_']=R.area;R['cx_']=R.cx;R['cy_']=R.cy
S=pd.read_pickle(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/detect_v3/recall/ref_blobs.pkl');S=S[S.src=='Side'].reset_index(drop=True)
S['fkey']='Side_'+S.ang;S['bin']=pd.cut(S.peak,[12,20,40,1e9],right=False,labels=['12-20','20-40','40+']);S['zone3']=np.where(S.rough,'rough','other');S['area_']=S.area;S['cx_']=S.cx;S['cy_']=S.cy
def cover(R,M,tag):
    cov=np.zeros(len(R),bool);rl=np.zeros(len(R),bool);rn=np.zeros(len(R),bool);base=['']*len(R)
    cg={k:g for k,g in M.groupby(['folder','fr'])}
    for (f,fr),g in R.groupby(['fkey','fr']):
        c=cg.get((f,fr))
        if c is None: continue
        bx,by,bw,bh=[c[k].values for k in ['bbox_x','bbox_y','bbox_w','bbox_h']];ca=c.area_px.values;real=c.real.values;rev=c.review_required.values;bs=c.base.values
        H=(g.cx_.values[:,None]>=bx-2)&(g.cx_.values[:,None]<=bx+bw+2)&(g.cy_.values[:,None]>=by-2)&(g.cy_.values[:,None]<=by+bh+2)&(ca<=30*g.area_.values[:,None])
        cov[g.index]=H.any(1);R_=H&real;rl[g.index]=R_.any(1);rn[g.index]=(R_&~rev).any(1)
        for j,i in enumerate(g.index):
            h=np.nonzero(H[j])[0]
            if len(h) and not R_[j].any(): base[i]=bs[h[np.argmin(ca[h])]]
    R['cov_'+tag]=cov;R['real_'+tag]=rl;R['realnr_'+tag]=rn;R['nubase_'+tag]=base
for X in (R,S):
    cover(X,mm,'v31');cover(X,m4,'v4');cover(X,m,'v41')
    X['realL1nr_v41']=False
R.to_pickle('recall_blobs.pkl');S.to_pickle('recall_blobs_side.pkl')
out=open(T+'recall.txt','w')
def pr(*a): print(*a);print(*a,file=out)
def agg(g):
    r=dict(n=len(g))
    for t in ['v31','v4','v41']:
        c=g['cov_'+t].sum();r[f'cov_{t}']=100*c/len(g);r[f'ret_{t}']=100*g['real_'+t].sum()/max(c,1)
    for t in ['v4','v41']: r[f'ret_nr_{t}']=100*g['realnr_'+t].sum()/max(g['cov_'+t].sum(),1)
    return pd.Series(r)
for name,keys in [('bin',['bin']),('zone3_x_bin',['zone3','bin']),('folder_x_bin',['folder','bin'])]:
    t=R.groupby(keys,observed=True).apply(agg).round(1);t.to_csv(T+f'recall_{name}.csv');pr('\n## EDGE',name);pr(t)
t=S.groupby(['bin'],observed=True).apply(agg).round(1);pr('\n## SIDE ref blobs by bin');pr(t);t.to_csv(T+'recall_side_bin.csv')
t=S.groupby(['fkey','bin'],observed=True).apply(agg).round(1);pr('\n## SIDE by folder x bin');pr(t);t.to_csv(T+'recall_side_folder_bin.csv')
# claims
cl=pd.DataFrame({'claim_v4':{('contam',b):v for b,v in zip(['12-20','20-40','40+'],[28.9,65.7,92.4])},'claim_v41':{('contam',b):v for b,v in zip(['12-20','20-40','40+'],[38.6,78.3,96.0])}})
rep=R.groupby(['zone3','bin'],observed=True).apply(agg)
cl['repro_v4']=[rep.loc[('contam',b),'ret_v4'] for z,b in cl.index];cl['repro_v41']=[rep.loc[('contam',b),'ret_v41'] for z,b in cl.index]
cl['claim_v41_nonreview']=[7.2,20.5,45.6]; cl['claim_v4_nonreview']=[3.7,9.9,41.7];cl['repro_v4_nr']=[rep.loc[('contam',b),'ret_nr_v4'] for z,b in cl.index];cl['repro_v41_nr']=[rep.loc[('contam',b),'ret_nr_v41'] for z,b in cl.index]
pr('\n## contam retention claim vs repro');pr(cl.round(1))
rr=pd.DataFrame({'claim_v41_nr':[5.2,29.7,74.5],'repro_v4_nr':[rep.loc[('rough',b),'ret_nr_v4'] for b in ['12-20','20-40','40+']],'repro_v41_nr':[rep.loc[('rough',b),'ret_nr_v41'] for b in ['12-20','20-40','40+']],'repro_v41_incl':[rep.loc[('rough',b),'ret_v41'] for b in ['12-20','20-40','40+']],'repro_v4_incl':[rep.loc[('rough',b),'ret_v4'] for b in ['12-20','20-40','40+']]},index=['12-20','20-40','40+'])
pr('\n## rough');pr(rr.round(1))
cl.to_csv(T+'recall_claim_vs_repro.csv')
pr('\n## Side blobs real in v4 but lost (nuisance) in v4.1:',int((S.real_v4&~S.real_v41).sum()),'of',int(S.real_v4.sum()));pr(S[S.real_v4&~S.real_v41].groupby('bin',observed=True).size().to_dict())
lost=S[S.real_v4&~S.real_v41];pr('  lost blobs peak quantiles',lost.peak.describe().round(1).to_dict())
out.close()
