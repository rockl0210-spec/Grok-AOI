import os
import sys,os,json,time
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
sys.path.insert(0,os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/filter_v42/_code')
import numpy as np,pandas as pd
import rules42
EV=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/filter_v41/'
M=pd.read_pickle(EV+'M_union41.pkl');n=len(M)
pers=pd.DataFrame(dict(p_tested=False,p_match=False,p_ctrl=0.0),index=M.index);pers['p_frames']=[[] for _ in range(n)]
glare=json.load(open(os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/filter_v42/_code/glare_bands.json'))
o=rules42.run(M,pers,glare,{});o['is_flip']=M.is_flip.values;o['folder']=M.folder.values;o['fr']=M.fr.values
o['zone']=M.zone.values;o['ac']=M.contrast.abs().values;o['snr']=M.snr_raw.values;o['conf']=M.confidence.values;o['area']=M.area_px.values;o['tierp']=M.tier.values
o.to_pickle('union_dec_v42.pkl');print('done',len(o),o.is_flip.sum())
