import os
from common import *
out=open(T+'ret42.txt','w')
def P(*a):
    s=' '.join(str(x) for x in a);print(s);out.write(s+'\n')
d=load_dec(F42);d['real']=d.decision=='real'
S=pd.read_pickle(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/detect_v3/recall/ref_blobs.pkl');S=S[S.src=='Side'].reset_index(drop=True)
S['fkey']='Side_'+S.ang;S['bin']=pd.cut(S.peak,[12,20,40,1e9],right=False,labels=['12-20','20-40','40+'])
R=pd.read_pickle(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/preprocess_v3/tables/dots_all.pkl');R=R[R.det=='raw'].reset_index(drop=True)
R['bin']=pd.cut(R.peak_raw,[12,20,40,1e9],right=False,labels=['12-20','20-40','40+']);R['zone3']=R.zone.map({0:'contam',1:'stain-zone',2:'rough',3:'clean'});R['fkey']=R.folder.str.replace('/','_')
for X in (R,S): X['fr_']=X.fr.astype(int)
d['fr_']=d.frame.astype(int)
def cover(X,tag):
    cov=np.zeros(len(X),bool);rl=np.zeros(len(X),bool);rn=np.zeros(len(X),bool)
    cg={k:g for k,g in d.groupby(['folder','fr_'])}
    for (f,fr),g in X.groupby(['fkey','fr_']):
        c=cg.get((f,fr))
        if c is None: continue
        bx,by,bw,bh=[c[k].values for k in ['bbox_x','bbox_y','bbox_w','bbox_h']];ca=c.area_px.values;real=c.real.values;rev=c.review_required.values
        H=(g.cx.values[:,None]>=bx-2)&(g.cx.values[:,None]<=bx+bw+2)&(g.cy.values[:,None]>=by-2)&(g.cy.values[:,None]<=by+bh+2)&(ca<=30*g.area.values[:,None])
        cov[g.index]=H.any(1);rl[g.index]=(H&real).any(1);rn[g.index]=(H&real&~rev).any(1)
    X['cov']=cov;X['real']=rl;X['nr']=rn
cover(S,'s');cover(R,'e')
def agg(g): 
    c=g['cov'].sum();return pd.Series(dict(n=len(g),cov=100*c/len(g),ret=100*g.real.sum()/max(c,1),ret_nr=100*g.nr.sum()/max(c,1)))
t=S.groupby('bin',observed=True).apply(agg).round(1);P('## Side ref retention v4.2 (blobs, coverage, retention incl review, non-review)');P(t);t.to_csv(T+'ret_side_bin.csv')
t2=S.groupby(['fkey','bin'],observed=True).apply(agg).round(1);P(t2);t2.to_csv(T+'ret_side_folder_bin.csv')
t=R.groupby('bin',observed=True).apply(agg).round(1);P('\n## Edge overall');P(t);t.to_csv(T+'ret_edge_bin.csv')
t=R.groupby(['zone3','bin'],observed=True).apply(agg).round(1);P(t);t.to_csv(T+'ret_edge_zone_bin.csv')
t=R.groupby(['fkey','bin'],observed=True).apply(agg).round(1);P(t);t.to_csv(T+'ret_edge_folder_bin.csv')
out.close()
