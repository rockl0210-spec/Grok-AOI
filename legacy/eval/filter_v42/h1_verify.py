from common import *
out=open(T+'verify42.txt','w')
def P(*a):
    s=' '.join(str(x) for x in a);print(s);out.write(s+'\n')
d=load_dec(F42);o=load_dec(F41)
d['real']=d.decision=='real';o['real']=o.decision=='real';d['kind']=d.folder.str[:4];o['kind']=o.folder.str[:4]
P('## integrity');P('rows',len(d),'unique',d.candidate_id.nunique(),'v41 rows',len(o),'ids equal',set(d.candidate_id)==set(o.candidate_id))
P('real',int(d.real.sum()),'nuis',int((~d.real).sum()),'review',int(d.review_required.sum()),'L1',int((d.review_level==1).sum()),'L2',int((d.review_level==2).sum()),'real L1',int((d.real&(d.review_level==1)).sum()),'real L2',int((d.real&(d.review_level==2)).sum()),'real non-review',int((d.real&~d.review_required).sum()))
P('review_level>0 <=> review_required:',int(((d.review_level>0)!=d.review_required).sum()),'mismatches; NaN in key cols:',int(d[['decision','fine_class','review_level','reason_code','same_object_group','same_object_n','same_object_n_real']].isna().sum().sum()))
s=d[d.kind=='Side'];P('Side real total',int(s.real.sum()),s[s.real].groupby('folder').size().to_dict())
for k in ['Edge','Side']:
    P(k,'L2 v41',int(((o.kind==k)&(o.review_level==2)).sum()),'-> v42',int(((d.kind==k)&(d.review_level==2)).sum()),'| real v41',int(((o.kind==k)&o.real).sum()),'-> v42',int(((d.kind==k)&d.real).sum()))
# diff vs v41
x=d.merge(o,on='candidate_id',suffixes=('','_o'))
cols=['decision','fine_class','review_required','review_level','reason_code','tier']
ch={c:(x[c]!=x[c+'_o']) for c in cols};anyc=np.zeros(len(x),bool)
for c in ch: anyc|=ch[c].values
x['chg']=anyc
P('\n## diff vs v41 (all columns',cols,')');P(x.groupby('kind').chg.agg(['sum','size']))
E_=x[(x.kind=='Edge')&x.chg];P('Edge changed rows',len(E_),'; columns changed:',{c:int(ch[c][(x.kind=='Edge').values].sum()) for c in cols})
P('Edge changed: path/tier',E_.groupby(['folder','decision','real','review_level_o','review_level']).size().to_string())
P('Edge changed det_path/tier_prior:',E_.det_path.value_counts().to_dict() if 'det_path' in E_ else '', E_.tier_prior.value_counts().to_dict() if 'tier_prior' in E_ else '')
P('Edge changed rows contrast |c| range',E_.contrast.abs().min(),E_.contrast.abs().max(),'folders',E_.folder.value_counts().to_dict(), 'fine_class',E_.fine_class.value_counts().to_dict())
P('Edge1 path C gray_mid/high real:',int((d.folder.str.startswith('Edge1')&(d.det_path=='C')&d.real).sum()))
# side diff
S_=x[(x.kind=='Side')&x.chg];P('\n## Side changed rows',len(S_));P(pd.crosstab([S_.decision_o,S_.review_level_o],[S_.decision,S_.review_level]).to_string())
P('Side changed reason_code (v41 -> v42) top:');P(S_.groupby(['reason_code_o','reason_code']).size().sort_values(ascending=False).head(8).to_string())
# plan D
cv=d[(d.reason_code.str.startswith('N_SIDE_T2_UNCONFIRMED'))];P('\n## N_SIDE_T2_UNCONFIRMED in v42:',len(cv),'all nuisance',bool((~cv.real).all()),'all L2',bool((cv.review_level==2).all()),'fine_class',cv.fine_class.value_counts().to_dict(),'anomaly',int(cv.anomaly_frame.sum()))
P('  by folder',cv.folder.value_counts().sort_index().to_dict());P('  rule a (pers_tested & not confirmed):',int((cv.pers_tested&~cv.pers_confirmed).sum()),' rule b (not tested):',int((~cv.pers_tested).sum()),' tested&confirmed',int((cv.pers_tested&cv.pers_confirmed).sum()))
c41=o[o.reason_code.str.startswith('N_SIDE_T2_UNCONFIRMED')];P('v41 N_SIDE_T2_UNCONFIRMED',len(c41),'of which anomaly',int(c41.anomaly_frame.sum()))
rs=x[(x.kind=='Side')&x.reason_code_o.str.startswith('N_SIDE_T2_UNCONFIRMED')&x.real];P('restored real (v41 T2-unconfirmed nuisance -> v42 real):',len(rs),'L1',int((rs.review_level==1).sum()),'L2',int((rs.review_level==2).sum()),'| restored reasons',rs.reason_code.str.split('[+;]').str[0].str.split(' ').str[0].value_counts().to_dict())
P('converted (v41 real->v42 nuis N_SIDE_T2) none expected? v41 real & v42 nuisance (Side):',int((x[x.kind=="Side"].real_o&~x[x.kind=="Side"].real).sum()),' v41 nuis & v42 real:',int((~x[x.kind=="Side"].real_o&x[x.kind=="Side"].real).sum()))
# other changed rows besides restored
oth=S_[~(S_.reason_code_o.str.startswith('N_SIDE_T2_UNCONFIRMED'))];P('Side changed not from T2-unconfirmed:',len(oth));P(oth.groupby(['decision_o','decision','review_level_o','review_level']).size().to_string() if len(oth) else '')
# special
for i in ['Side_000_05_c0736','Edge2_090_27_c1848']:
    r=d[d.candidate_id==i].iloc[0];P(i,r.decision,r.fine_class,'L',r.review_level,r.reason_code.split(' ')[0],'area',r.area_px)
P('\n## defects');dd=load_def(F42);P('unique defects',len(dd),'by class',dd.fine_class.value_counts().to_dict(),'review L1/L2',int((dd.review_level==1).sum()),int((dd.review_level==2).sum()))
# same_object
P('\n## same_object_group sanity')
g=d.groupby('same_object_group').agg(n=('candidate_id','size'),n_decl=('same_object_n','first'),nr_decl=('same_object_n_real','first'),nr=('real','sum'),nuniq_n=('same_object_n','nunique'),nfr=('frame','nunique'),nfold=('folder','nunique'),
  x0=('bbox_x','min'),y0=('bbox_y','min'),x1=('bbox_x',lambda s:0),area=('area_px','sum'))
d['x1']=d.bbox_x+d.bbox_w;d['y1']=d.bbox_y+d.bbox_h;gg=d.groupby('same_object_group').agg(x0=('bbox_x','min'),y0=('bbox_y','min'),x1=('x1','max'),y1=('y1','max'),npos=('contrast',lambda s:int((s>0).sum())),nneg=('contrast',lambda s:int((s<0).sum())),area=('area_px','sum'),nreal=('real','sum'),n=('real','size'),
  ncls=('fine_class',lambda s:s[s.isin(['particle','stain','scratch'])].nunique()),nfr=('frame','nunique'),nfold=('folder','nunique'),zone=('zone',lambda s:s.value_counts().index[0]),kind=('kind','first'),folder=('folder','first'))
gg['bw']=gg.x1-gg.x0;gg['bh']=gg.y1-gg.y0;gg['bbox_area']=gg.bw*gg.bh;gg['fill']=gg.area/gg.bbox_area.clip(lower=1)
P('groups',len(gg),'>=2:',int((gg.n>=2).sum()),'members in >=2',int(gg[gg.n>=2].n.sum()),'real members in >=2 groups',int(gg[gg.n>=2].nreal.sum()),'max',int(gg.n.max()),'groups with real',int((gg.nreal>0).sum()),'multi-real',int((gg.nreal>1).sum()))
P('declared n==actual',bool((g.n==g.n_decl).all()),'declared n_real==actual',bool((g.nr==g.nr_decl).all()),'n consistent within group',bool((g.nuniq_n==1).all()),'groups spanning >1 frame',int((gg.nfr>1).sum()),'>1 folder',int((gg.nfold>1).sum()))
P('size distribution n:',pd.cut(gg.n,[0,1,2,5,10,25,50,100,300]).value_counts().sort_index().to_dict())
P('cross-polarity groups (both + and - contrast):',int(((gg.npos>0)&(gg.nneg>0)).sum()),' zero-contrast members',int((d.contrast==0).sum()))
big=gg.sort_values('n',ascending=False).head(15);P('top15 groups by n:');P(big[['kind','folder','n','nreal','npos','nneg','bw','bh','fill','ncls','zone']].to_string())
P('groups n>=50:',int((gg.n>=50).sum()),'; their members',int(gg[gg.n>=50].n.sum()),'by zone',gg[gg.n>=50].zone.value_counts().to_dict(),'by folder',gg[gg.n>=50].folder.value_counts().to_dict())
P('groups n>=20:',int((gg.n>=20).sum()),'by zone',gg[gg.n>=20].zone.value_counts().to_dict())
P('large-span groups (bw or bh > 600px):',int(((gg.bw>600)|(gg.bh>600)).sum()),'(>1000):',int(((gg.bw>1000)|(gg.bh>1000)).sum()),' with n>=2 only:',int((((gg.bw>600)|(gg.bh>600))&(gg.n>=2)).sum()))
P('real groups mixing >1 subclass among members (n>=2):',int((gg[gg.n>=2].ncls>1).sum()),'of',int((gg.n>=2).sum()),'; real members',gg[gg.n>=2].groupby('ncls').size().to_dict())
big_area=d[d.area_px>250000];P('boxes area>250k',len(big_area),'in groups n>1:',int((big_area.same_object_n>1).sum()))
P('mixed real/nuisance groups (n>=2):',int(((gg.n>=2)&(gg.nreal>0)&(gg.nreal<gg.n)).sum()),' all-real:',int(((gg.n>=2)&(gg.nreal==gg.n)).sum()),' all-nuis:',int(((gg.n>=2)&(gg.nreal==0)).sum()))
P('same_object_n_real among REAL candidates: median',float(d[d.real].same_object_n_real.median()),'real cands in a group with >1 real:',int((d.real&(d.same_object_n_real>1)).sum()),'of',int(d.real.sum()))
big.drop(columns=['x0','y0','x1','y1']).to_csv(T+'same_object_top15.csv')
gg.groupby(['kind','zone']).n.agg(['size','sum','max']).to_csv(T+'same_object_by_kind_zone.csv')
out.close()
