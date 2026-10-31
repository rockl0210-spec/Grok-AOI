import os
import pandas as pd,numpy as np,glob,os,json
pd.set_option('display.width',250);pd.set_option('display.max_rows',500);pd.set_option('display.max_columns',60)
V41=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/detect_v41';V4=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/detect_v4';V31=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/detect'
D=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/detect_v41/';T=D+'tables/';FG=D+'fig/'
FOLD=['Edge1_000','Edge1_090','Edge2_000','Edge2_090','Side_000','Side_090','Side_180','Side_270']
def load(root):
    out=[]
    for f in FOLD:
        k='edge' if f.startswith('Edge') else 'side'
        d=pd.read_csv(f'{root}/{k}/{f}/candidates.csv',dtype={'frame':str,'angle':str});d['folder']=f;d['kind']=k.capitalize();out.append(d)
    return pd.concat(out,ignore_index=True)
