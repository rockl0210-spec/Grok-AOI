import os
"""Persistence proxy (a4.py logic, detect_v4/run/pipe) for v3.1/v4/v4.1; v4.1 subsets: raw_rescue, haze_texture, new vs in-v4."""
from common import *
a=pd.read_pickle('v31.pkl');b=pd.read_pickle('v4.pkl');c=pd.read_pickle('v41.pkl')
P4=pd.read_csv(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/detect_v4/run/pipe/tables/t4_pairs_candidates.csv');P3=pd.read_csv(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/detect_v31/run/pipe/tables/t4_pairs_candidates.csv');P41=pd.read_csv('run/pipe/tables/t4_pairs_candidates.csv')
def prep(c,P):
    c=c[~c.anomaly_frame.astype(bool)].copy();cols=['id','kind','path','contrast_tier','zone','class','area_px','consolidation','confidence','contrast']+[k for k in ['raw_rescue','haze_texture'] if k in c]
    m=P.merge(c[cols],on='id',how='inner');return m[m.tested]
c['inprev']=False;bg={k:g for k,g in b.groupby(['folder','frame'])}
for k,g in c.groupby(['folder','frame']):
    q=bg.get(k)
    if q is None: continue
    x=g.centroid_x.values[:,None];y=g.centroid_y.values[:,None]
    c.loc[g.index,'inprev']=((x>=q.bbox_x.values)&(x<=q.bbox_x.values+q.bbox_w.values)&(y>=q.bbox_y.values)&(y<=q.bbox_y.values+q.bbox_h.values)).any(1)
c.to_pickle('v41_inprev.pkl')
T3=prep(a,P3);T4=prep(b,P4);T41=prep(c,P41).merge(c[['id','inprev']],on='id')
def agg(g):
    m=g.match.mean();ch=g.ctrl.mean();return pd.Series(dict(tested=len(g),match_pct=100*m,chance_pct=100*ch,excess=100*(m-ch),struct_frac=100*max(0,(m-ch)/(1-ch))))
o=open(T+'persist_v41.txt','w')
def pr(*x): print(*x);print(*x,file=o)
pr('persistence proxy (overlap strips, mapped position inside other frame ROI; chance=8 shifted offset controls). Edge normal frames only.')
nr=T41[~T41.raw_rescue];rr=T41[T41.raw_rescue]
pr('\n== all');x=pd.concat([agg(T3).rename('v3.1'),agg(T4).rename('v4'),agg(T41).rename('v4.1'),agg(nr).rename('v4.1 w/o rescue'),agg(rr).rename('v4.1 raw_rescue'),agg(nr[nr.inprev]).rename('v4.1 old (in v4)'),agg(nr[~nr.inprev]).rename('v4.1 new non-rescue'),agg(T41[T41.haze_texture]).rename('v4.1 haze_texture')],axis=1).round(1);pr(x);x.to_csv(T+'persist_all.csv')
for k in (['zone'],['path'],['contrast_tier'],['consolidation']):
    x=pd.concat([T4.groupby(k).apply(agg).add_prefix('v4_'),nr.groupby(k).apply(agg).add_prefix('v41_norr_'),rr.groupby(k).apply(agg).add_prefix('rr_')],axis=1).round(1);x=x[[q for q in x.columns if 'tested' in q or 'match' in q or 'chance' in q or 'struct' in q]];pr('\n==',k);pr(x);x.to_csv(T+f'persist_by_{k[0]}.csv')
pr('\n== raw_rescue by folder');pr(rr.groupby('folder').apply(agg).round(1))
pr('\n== raw_rescue by peak |contrast| bin');pr(rr.assign(b=pd.cut(rr.contrast.abs(),[0,30,40,60,1e9])).groupby('b',observed=True).apply(agg).round(1))
pr('\n== raw_rescue by area');pr(rr.assign(b=pd.cut(rr.area_px,[0,5,8,15,30,1e9])).groupby('b',observed=True).apply(agg).round(1))
pr('\n== non-rescue tiny (<=15 px) vs rescue, same zone oil');o_=nr[(nr.zone=='oil')&(nr.area_px<=15)];pr(agg(o_).round(1).to_dict())
pr('\n== folder');pr(pd.concat([T4.groupby('folder').apply(agg).add_prefix('v4_'),nr.groupby('folder').apply(agg).add_prefix('v41_norr_')],axis=1).round(1))
# detector's own persistence field
for n,x in [('v31',a),('v4',b),('v41',c)]:
    e=x[(x.kind=='Edge')&~x.anomaly_frame.astype(bool)];pr(n,'persistence_checked>0',int((e.persistence_checked>0).sum()),'persistent',int((e.persistence>0).sum()),'rate',round((e.persistence>0).sum()/max(1,(e.persistence_checked>0).sum()),3),'by folder',e.groupby('folder').apply(lambda g:int((g.persistence_checked>0).sum())).to_dict())
e=c[(c.kind=='Edge')&~c.anomaly_frame.astype(bool)];pr('v4.1 rescue: detector persistence_checked>0',int(((e.persistence_checked>0)&e.raw_rescue).sum()),'of',int(e.raw_rescue.sum()),'persistent',int(((e.persistence>0)&e.raw_rescue).sum()))
