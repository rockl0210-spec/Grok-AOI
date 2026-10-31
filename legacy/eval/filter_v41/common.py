import os
"""Shared loaders for filter_v41 verification."""
import pandas as pd,numpy as np,glob,os,sys
sys.dont_write_bytecode=True
pd.set_option('display.width',250,'display.max_columns',60,'display.max_rows',300)
FD=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/filter_v41';SN=FD+'/_input_snapshot_v41';FD4=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/filter_v4';SN4=FD4+'/_input_snapshot_v4';FD31=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/filter';SN31=FD31+'/_input_snapshot_v31'
E=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/filter_v41/';T=E+'tables/';FG=E+'fig/';E4=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/filter_v4/'
FOLD=['Edge1_000','Edge1_090','Edge2_000','Edge2_090','Side_000','Side_090','Side_180','Side_270']
def mode(f): return 'edge' if f.startswith('Edge') else 'side'
def load_dec(fd=FD):
    return pd.concat([pd.read_csv(f'{fd}/{mode(f)}/{f}/decisions.csv',dtype={'frame':str}).assign(folder=f) for f in FOLD],ignore_index=True)
def load_snap(sn=SN):
    return pd.concat([pd.read_csv(f'{sn}/{mode(f)}/{f}/candidates.csv',dtype={'frame':str,'angle':str}).assign(folder=f) for f in FOLD],ignore_index=True)
def load_def(fd=FD):
    return pd.concat([pd.read_csv(f'{fd}/{mode(f)}/{f}/defects.csv').assign(folder=f) for f in FOLD],ignore_index=True)
def tiers(d):
    e=(d.kind=='Edge').values;ac=d.contrast.abs()
    t1=np.where(e,(d.snr_raw>=12)&(ac>=10)&(d.area_px>=5)&(d.confidence>=0.5),(d.snr_raw>=30)&(ac>=10)&(d.area_px>=8))
    t2=np.where(e,(d.snr_raw>=9)&(d.confidence>=0.5),(d.snr_raw>=16))
    return np.where(t1,'T1',np.where(t2,'T2','drop'))
def zb(b): return np.select([(b&4)>0,(b&(16|8))>0,(b&2)>0,(b&1)>0],['rough','contam','uneven','specular'],'normal')
