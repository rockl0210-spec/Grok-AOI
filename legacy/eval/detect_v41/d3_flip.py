import os
"""Flip-null (polarity swapped, persistence off) for v3.1/v4/v4.1; ratio flip/real is an upper bound on false share."""
from common import *
import glob
def ld(D):
    c=pd.concat([pd.read_csv(f,dtype={'frame':str,'angle':str}) for f in sorted(glob.glob(D+'/*/*_*/candidates.csv'))],ignore_index=True)
    c['grp']=np.where(c.source=='Side','Side',c.source);return c[~c.anomaly_frame.astype(bool)]
R=ld(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/detect_v41');F=ld(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/detect_v41/null_flip/out')
R4=ld(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/detect_v4');F4=ld(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/detect_v4/null_flip/out')
R3=ld(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/detect');F3=ld(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/detect_v31/null_flip/out')
print(len(R),len(F),len(R4),len(F4),len(R3),len(F3))
o=open(T+'flip_v41.txt','w')
def pr(*x): print(*x);print(*x,file=o)
def tab(R,F,by):
    r=R.groupby(by).size();f=F.groupby(by).size();t=pd.concat([r,f],axis=1,keys=['real','flip']).fillna(0).astype(int);t['null_pct']=(100*t.flip/t.real).round(2);return t
for nm,by in [('grp','grp'),('folder',['source','angle']),('path',['grp','path']),('tier',['grp','contrast_tier']),('path_tier',['grp','path','contrast_tier']),('zone',['grp','zone'])]:
    ts=[tab(R3,F3,by).add_prefix('v31_') if nm!='zone' else None,tab(R4,F4,by).add_prefix('v4_'),tab(R,F,by).add_prefix('v41_')]
    x=pd.concat([t for t in ts if t is not None],axis=1);pr('\n==',nm);pr(x.to_string());x.to_csv(T+f'flip_{nm}.csv')
for n,(r,f) in {'v31':(R3,F3),'v4':(R4,F4),'v41':(R,F)}.items():
    e=r.source.str.startswith('Edge');ef=f.source.str.startswith('Edge');pr(n,'Edge real',e.sum(),'flip',ef.sum(),round(100*ef.sum()/e.sum(),2),'| Side real',(~e).sum(),'flip',(~ef).sum(),round(100*(~ef).sum()/(~e).sum(),2),'| Edge1',round(100*(f.source=='Edge1').sum()/(r.source=='Edge1').sum(),2),'Edge2',round(100*(f.source=='Edge2').sum()/(r.source=='Edge2').sum(),2))
# raw_rescue subset (v4.1)
rr=R[R.raw_rescue];fr=F[F.raw_rescue]
pr('\nraw_rescue: real',len(rr),'flip',len(fr),'null %',round(100*len(fr)/len(rr),2));pr(pd.concat([rr.groupby(['source','angle']).size(),fr.groupby(['source','angle']).size()],axis=1,keys=['real','flip']).fillna(0).astype(int))
nr=R[(R.source!='Side')&~R.raw_rescue];nf=F[(F.source!='Side')&~F.raw_rescue]
pr('Edge w/o raw_rescue: real',len(nr),'flip',len(nf),round(100*len(nf)/len(nr),2));pr('Edge1 w/o rescue',round(100*(nf.source=='Edge1').sum()/(nr.source=='Edge1').sum(),2),'Edge2 w/o rescue',round(100*(nf.source=='Edge2').sum()/(nr.source=='Edge2').sum(),2))
pr('\nEdge folders w/o rescue');pr(pd.concat([nr.groupby(['source','angle']).size(),nf.groupby(['source','angle']).size()],axis=1,keys=['real','flip']).assign(pct=lambda d:(100*d.flip/d.real).round(2)))
pr('\nEdge2/090 oil path C: real',len(R[(R.source=='Edge2')&(R.angle=='090')&(R.path=='C')&(R.zone=='oil')]),'(rescue',len(rr[(rr.source=='Edge2')&(rr.angle=='090')]),') flip',len(F[(F.source=='Edge2')&(F.angle=='090')&(F.path=='C')&(F.zone=='oil')]))
# haze_texture
ht=R[R.haze_texture];pr('\nhaze_texture real',len(ht),'flip haze_texture',int(F.haze_texture.sum()),'flip gray_low area>=150 in Edge',int(((F.source!='Side')&(F.contrast_tier=='gray_low')&(F.area_px>=150)).sum()),'real',int(((R.source!='Side')&(R.contrast_tier=='gray_low')&(R.area_px>=150)).sum()))
for n,f in (('v4.1',F),('v4',F4),('v3.1',F3)):
    e=f[f.source!='Side'];pr(n,'flip Edge area>=150 gray_low:',int(((e.contrast_tier=='gray_low')&(e.area_px>=150)).sum()))
e=R[R.source!='Side']
for n,(r,f) in {'v31':(R3,F3),'v4':(R4,F4),'v41':(R,F)}.items():
    r=r[r.source!='Side'];f=f[f.source!='Side'];pr('\n',n,'Edge >5000 px: real',(r.area_px>5000).sum(),'oil',((r.area_px>5000)&(r.zone=='oil')).sum(),'| flip',(f.area_px>5000).sum(),'oil',((f.area_px>5000)&(f.zone=='oil')).sum(),'| >2000 real',(r.area_px>2000).sum(),'flip',(f.area_px>2000).sum())
# path A gray_low Edge1 and path B by folder
for n,(r,f) in {'v31':(R3,F3),'v4':(R4,F4),'v41':(R,F)}.items():
    for q,qq in (('A','gray_low'),('B','gray_low'),('A','gray_mid'),('C','gray_mid')):
        for s in ('Edge1','Edge2'):
            a=((r.source==s)&(r.path==q)&(r.contrast_tier==qq)).sum();b=((f.source==s)&(f.path==q)&(f.contrast_tier==qq)).sum();pr(n,s,q,qq,a,b,round(100*b/max(a,1),2))
