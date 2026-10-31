import os
"""Direct false-real measurement for v4.1: inject polarity-flip null (detect_v41/null_flip, normal frames) into the real v4.1 candidate set, run rules41.run
(dummy persistence for everything) for variants. Reads <legacy-root>/aoi_out/filter_v41/_code read-only; output in this folder only."""
import sys,os,json,time
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
sys.path.insert(0,os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/filter_v41/_code')
import numpy as np,pandas as pd
import feats,rules41
FL=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/detect_v41/null_flip/out';SN=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/filter_v41/_input_snapshot_v41';PRE=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/preprocess'
t0=time.time()
R=feats.load(SN);R['is_flip']=False
F=feats.load(FL);F['is_flip']=True;F['id']='F_'+F['id'];F=F[~F.anomaly_frame.astype(bool)].copy()
print('real',len(R),'flip normal',len(F),flush=True)
M=pd.concat([R,F],ignore_index=True);M['tier']=feats.tier_prior(M)
M['nb50'],M['nb_medc'],M['nb_similar'],M['nb_rank']=feats.nb_features(M);M['rep']=feats.repeat_feature(M);M['zone_src']=M.zone
M['zone'],M['nbits'],M['rim_dist']=feats.mask_features(M,PRE,True)
for c in ['raw_rescue','haze_texture']: M[c]=M[c].fillna(False).astype(bool)
M.to_pickle('M_union41.pkl');print('feat',round(time.time()-t0),flush=True)
pers=pd.DataFrame(dict(p_tested=False,p_match=False,p_ctrl=0.0),index=M.index);pers['p_frames']=[[] for _ in range(len(M))]
glare=json.load(open(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/filter_v41/_code/glare_bands.json'))
for nm,opt in [('full',{}),('noSideT2',{'sideT2':0}),('v4rules',dict(spec=0,sparse=0,ruleD=0,sideT2=0,rough25=0,c7=0,v41=0))]:
    o=rules41.run(M,pers,glare,opt);o['is_flip']=M.is_flip.values;o['folder']=M.folder.values;o['fr']=M.fr.values
    o.to_pickle(f'union_dec_{nm}.pkl');print(nm,'done',round(time.time()-t0),flush=True)
