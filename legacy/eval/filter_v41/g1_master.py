import os
"""master41.pkl: v4.1 decisions + features + persistence + status in v4 (id for Side, geometry for Edge) and v3.1 (geometry)."""
from common import *
d=load_dec();s=load_snap()
cols=['id','source','angle','centroid_x','centroid_y','polarity','confidence','merged_n','contrast_tier','roi_overlap_frac','dist_to_roi_edge_px','elongation','major_len_px','thickness_px','nuisance_bits','path','solidity','min_gray','max_gray','local_bg_gray','large_faint_region']
m=d.merge(s[cols],left_on='candidate_id',right_on='id').drop(columns='id')
m['kind']=np.where(m.folder.str.startswith('Edge'),'Edge','Side');m['fr']=m.frame.astype(int);m['real']=m.decision=='real'
m['ac']=m.contrast.abs();m['base']=m.reason_code.str.split('+').str[0].str.split(';').str[0]
m['mytier']=tiers(m);dem=(m.mytier=='T1')&(m.consolidation!='none')&(m.contrast_tier!='high');m.loc[dem,'mytier']='T2'
m['zb']=zb(m.nuisance_bits.values)
P4=pd.read_csv(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/detect_v41/run/pipe/tables/t4_pairs_candidates.csv');P5=pd.read_csv(E4+'tables/side_pairs_v4.csv')
P=pd.concat([P4[['id','tested','match','ctrl','good_reg','pair']],P5[['id','tested','match','ctrl','good_reg','pair']]])
ct=P[P.tested].groupby('id').agg(ctrl_mean=('ctrl','mean'),match_any=('match','max'),pair=('pair','first'),good_reg=('good_reg','min'))
m=m.join(ct,on='candidate_id');m['tested']=m.match_any.notna();m['match_any']=m.match_any.fillna(False).astype(bool)
m['real']=m.real.astype(bool)
# v4 status
m4=pd.read_pickle(E4+'master4.pkl');mm=pd.read_pickle(E4+'master31.pkl')
def status(M,Q,tag):
    st=np.array(['none']*len(M),dtype=object);rv=np.zeros(len(M),bool);lk=np.array(['']*len(M),dtype=object)
    gq={k:g for k,g in Q.groupby(['folder','fr'])}
    for k,g in M.groupby(['folder','fr']):
        q=gq.get(k)
        if q is None: continue
        bx,by,bw,bh=[q[c].values for c in ['bbox_x','bbox_y','bbox_w','bbox_h']];qa=q.area_px.values;qr=q.real.values;qv=q.review_required.values;bs=q.base.values
        x=g.centroid_x.values[:,None];y=g.centroid_y.values[:,None];a=g.area_px.values[:,None]
        H=(x>=bx-2)&(x<=bx+bw+2)&(y>=by-2)&(y<=by+bh+2);Hs=H&(qa<=30*a)
        anyreal=(Hs&qr).any(1);anynu=(Hs&~qr).any(1);bigreal=(H&~Hs&qr).any(1);anybig=(H&~Hs).any(1)
        s_=np.where(anyreal,'real',np.where(anynu,'nuis',np.where(bigreal,'bigreal',np.where(anybig,'bignuis','none'))))
        ii=M.index.get_indexer(g.index);st[ii]=s_
        rv[ii]=(Hs&qr&qv).any(1)&~(Hs&qr&~qv).any(1)
        # base reason of the smallest containing candidate
        for j,i in enumerate(ii):
            h=np.nonzero(Hs[j])[0]
            if len(h): lk[i]=bs[h[np.argmin(qa[h])]]
    M['st'+tag]=st;M['rv'+tag]=rv;M['lk'+tag]=lk
status(m,mm,'31');status(m,m4,'4')
# exact id match for Side (identical ids to v4)
v4=m4.set_index('candidate_id')
sd=(m.kind=='Side')
for c_,n_ in [('decision','dec4'),('review_required','rev4'),('reason_code','rc4'),('mytier','tier4')]:
    m[n_]=np.where(sd,m.candidate_id.map(v4[c_]) ,None)
m.to_pickle('master41.pkl')
print(m.shape);print(pd.crosstab(m.real,m.st4));print(pd.crosstab(m.real,m.st31))
print(pd.crosstab(m[sd].dec4,m[sd].decision))
