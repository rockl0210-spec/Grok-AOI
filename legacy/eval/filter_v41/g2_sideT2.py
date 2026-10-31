import os
"""Task (2): who are the 7,755 Side reals turned into nuisance N_SIDE_T2_UNCONFIRMED."""
from common import *
m=pd.read_pickle('master41.pkl');R=pd.read_pickle(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/filter_v4/R_real.pkl')
S=m[(m.kind=='Side')].copy();S['rc4b']=S.rc4.str.split('[+;]').str[0]
X=S[(S.dec4=='real')&(S.decision=='nuisance')].copy()
out=open(T+'sideT2.txt','w')
def pr(*a): print(*a);print(*a,file=out)
pr('n converted',len(X),'; reason in v4.1',X.reason_code.value_counts().to_dict(),'; v4 reason',X.rc4b.value_counts().to_dict())
pr('v4 review of these:',X.rev4.value_counts().to_dict(),'| v4.1 review level',X.review_level.value_counts().to_dict())
X['status31']=np.where(X.st31.isin(['real','bigreal']),'real in v3.1',np.where(X.st31.isin(['nuis','bignuis']),'nuisance in v3.1','not detected in v3.1'))
pr('\n## v3.1 status of the 7,755');pr(X.status31.value_counts().to_string())
pr(pd.crosstab(X.folder,X.status31,margins=True).to_string())
pr('\n## zone x tier');pr(pd.crosstab(X.zb,X.mytier,margins=True).to_string())
pr('\n## contrast |c| bins');pr(pd.crosstab(pd.cut(X.ac,[0,8,10,12,15,20,30,50,1e9],right=False),X.status31,margins=True).to_string())
pr('\n## snr / cnr / area quantiles');pr(X[['ac','cnr','snr_raw','area_px']].describe(percentiles=[.1,.25,.5,.75,.9]).round(1).to_string())
pr('\n## persistence-testable?')
X['t']=np.where(X.tested,np.where(X.good_reg.fillna(True),'tested','tested(no good reg)'),'not testable (outside overlap strip / mask)')
pr(X.t.value_counts().to_string());pr(pd.crosstab(X.folder,X.t).to_string())
pr('filter pers_tested/confirmed (own):',X.pers_tested.value_counts().to_dict(),X.pers_confirmed.value_counts().to_dict())
tt=X[X.tested&X.good_reg.fillna(True).astype(bool)]
mt=tt.match_any.mean();ch=tt.ctrl_mean.mean();pr('tested subset: n',len(tt),'match',round(100*mt,1),'chance',round(100*ch,1),'struct',round(100*(mt-ch)/(1-ch),1))
for nm,g in [('v3.1-real',tt[tt.status31=='real in v3.1']),('not v3.1-real',tt[tt.status31!='real in v3.1'])]:
    if len(g): mt=g.match_any.mean();ch=g.ctrl_mean.mean();pr('  ',nm,len(g),'match',round(100*mt,1),'chance',round(100*ch,1),'struct',round(100*(mt-ch)/(1-ch),1))
# all v4 Side T2 reals (still real or converted), how many remain real in v4.1
A=S[S.rc4b.isin(['R_SIDE_NORMAL_T2','R_SIDE_UNEVEN_T2'])]
pr('\nv4 Side *_T2 reals',len(A),'-> v4.1 decision',A.decision.value_counts().to_dict(),'| kept real & reason',A[A.real].reason_code.value_counts().head(6).to_dict())
# among v4 T2 reals: persistence-confirmed (own filter pconf) kept
pr('kept real PERS_OK share', A[A.real].reason_code.str.contains('PERS_OK').mean().round(3))
# new-in-v4 vs old
A=A.assign(status31=np.where(A.st31.isin(['real','bigreal']),'real in v3.1',np.where(A.st31.isin(['nuis','bignuis']),'nuisance in v3.1','not detected in v3.1')))
pr('\n## v4 Side *_T2 reals by v3.1 status x outcome');pr(pd.crosstab(A.status31,A.decision,margins=True).to_string())
# ----- reference-dot retention in Side
# Side dots: preprocess_v3 dots_all? check zones
D=pd.read_pickle(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/preprocess_v3/tables/dots_all.pkl');pr('\ndots_all folders',D.folder.unique().tolist()[:12],'det',D.det.unique().tolist())
out.close();X.drop(columns=['t']).to_pickle('X7755.pkl')
