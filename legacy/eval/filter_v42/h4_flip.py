import os
from common import *
out=open(T+'flip42.txt','w')
def P(*a):
    s=' '.join(str(x) for x in a);print(s);out.write(s+'\n')
EV=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/filter_v41/'
U={'v41':pd.read_pickle(EV+'union_dec_full.pkl'),'v41_noSideT2':pd.read_pickle(EV+'union_dec_noSideT2.pkl'),'v42':pd.read_pickle('union_dec_v42.pkl')}
M=pd.read_pickle(EV+'M_union41.pkl')
for k,o in U.items():
    o['kind']=np.where(o.folder.str.startswith('Edge'),'Edge','Side');o['real']=o.decision=='real';o['base']=o.reason_code.str.split('[+;]').str[0];o['zone']=M.zone.values
rows=[]
for k,o in U.items():
    f=o[o.is_flip];r=o[~o.is_flip]
    rows.append(dict(ver=k,flips=len(f),flip_real=int(f.real.sum()),pass_pct=round(100*f.real.mean(),2),flip_real_nr=int((f.real&~f.review_required).sum()),pass_nr_pct=round(100*(f.real&~f.review_required).mean(),2),real=int(r.real.sum()),false_share=round(100*f.real.sum()/r.real.sum(),3),
      Edge_fs=round(100*f[f.kind=='Edge'].real.sum()/r[(r.kind=='Edge')].real.sum(),3),Side_fs=round(100*f[f.kind=='Side'].real.sum()/r[(r.kind=='Side')].real.sum(),3),Edge_pass=round(100*f[f.kind=='Edge'].real.mean(),1),Side_pass=round(100*f[f.kind=='Side'].real.mean(),1)))
t=pd.DataFrame(rows);P('## flip null (union run, dummy persistence)');P(t.to_string());t.to_csv(T+'flip42_variants.csv')
P('\nNOTE: union run has no persistence info -> v4.2 plan D conversions (a: tested&unconfirmed, b: v4-new&untestable) cannot be exercised; the union v42 run therefore differs from v41 mainly by Edge1 path C review (+68 L1) and review levels.')
for k in ['v41','v42']:
    o=U[k];f=o[o.is_flip&o.real];r=o[~o.is_flip&o.real]
    lv=pd.DataFrame({'flip_real':f.groupby('review_level').size(),'real':r.groupby('review_level').size()}).fillna(0).astype(int);lv['false_share_pct']=(100*lv.flip_real/lv.real).round(2)
    P(f'\n## review-level false share among real [{k}] (0=no review)');P(lv.to_string())
    lk=pd.DataFrame({'flip_real':f.groupby(['kind','review_level']).size(),'real':r.groupby(['kind','review_level']).size()}).fillna(0).astype(int);lk['false_share_pct']=(100*lk.flip_real/lk.real).round(2);P(lk.to_string())
    P('L2 flips by base:',f[f.review_level==2].base.value_counts().to_dict())
    if k=='v42': lv.to_csv(T+'flip42_level.csv');lk.to_csv(T+'flip42_kind_level.csv')
# actual v42 decisions: level split
d=load_dec(F42);d['real']=d.decision=='real'
P('\n## actual v4.2 L2 composition: total',int((d.review_level==2).sum()),'real',int((d.real&(d.review_level==2)).sum()),'nuisance',int((~d.real&(d.review_level==2)).sum()))
P(' nuisance L2 by reason',d[~d.real&(d.review_level==2)].reason_code.str.split('[+;]').str[0].str.split(' ').str[0].value_counts().head(8).to_dict())
P(' real L2 by reason',d[d.real&(d.review_level==2)].reason_code.str.split('[+;]').str[0].str.split(' ').str[0].value_counts().head(10).to_dict())
P(' L2 by kind',d[d.review_level==2].groupby(d.folder.str[:4]).size().to_dict())
# triage lift persistence (non-circular)
# stratified estimate
F=U['v42'][U['v42'].is_flip].copy()
zbf=lambda z:np.select([np.isin(z,['oil','water']),z=='rough',z=='uneven',z=='specular'],['contam','rough','uneven','specular'],'normal')
F['zb']=zbf(F.zone.values);F['ab']=pd.cut(F.ac,[0,12,20,40,1e9],right=False,labels=['<12','12-20','20-40','40+']).astype(str);F['tr']=F.tierp
def prep(dd):
    dd=dd.copy();dd['kind']=dd.folder.str[:4];dd['zb']=zbf(dd.zone.values);dd['ab']=pd.cut(dd.contrast.abs(),[0,12,20,40,1e9],right=False,labels=['<12','12-20','20-40','40+']).astype(str);dd['tr']=dd.tier_prior if 'tier_prior' in dd else dd.tier;dd['nr']=dd.real&~dd.review_required;return dd
d['kind']=d.folder.str[:4];d41=load_dec(F41);d41['real']=d41.decision=='real'
K=['kind','zb','ab','tr']
res=[]
for nm,dd in [('v41',d41),('v42',d)]:
    dd=prep(dd);dd=dd[~dd.anomaly_frame.astype(bool)]
    for c in ['real','nr']:
        p=dd.groupby(K)[c].mean().rename('p');n=F.groupby(K).size().rename('n');t=pd.concat([n,p],axis=1).dropna();t['e']=t.n*t.p
        e=t.e.sum();res.append(dict(ver=nm,col=c,est_false=round(e),Edge=round(t.e.get('Edge',0).sum()) if False else round(t.groupby(level=0).e.sum().get('Edge',0)),Side=round(t.groupby(level=0).e.sum().get('Side',0)),real_n=int(dd[c].sum()),pct=round(100*e/dd[c].sum(),3)))
t=pd.DataFrame(res);P('\n## stratified false-real estimate (same 5,022 flips; strata kind x zone x |c| bin x tier_prior)');P(t.to_string());t.to_csv(T+'flip42_strat.csv')
out.close()
