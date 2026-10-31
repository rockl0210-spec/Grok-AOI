import os
import sys;sys.dont_write_bytecode=True
import pandas as pd,numpy as np,os
pd.set_option('display.width',250,'display.max_columns',60,'display.max_rows',300)
F42=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/filter_v42';F41=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_out/filter_v41'
E=os.environ['GROK_AOI_LEGACY_ROOT'] + '/aoi_eval/filter_v42/';T=E+'tables/'
FOLD=['Edge1_000','Edge1_090','Edge2_000','Edge2_090','Side_000','Side_090','Side_180','Side_270']
def mode(f): return 'edge' if f.startswith('Edge') else 'side'
def load_dec(fd):
    return pd.concat([pd.read_csv(f'{fd}/{mode(f)}/{f}/decisions.csv',dtype={'frame':str}).assign(folder=f) for f in FOLD],ignore_index=True)
def load_def(fd):
    return pd.concat([pd.read_csv(f'{fd}/{mode(f)}/{f}/defects.csv').assign(folder=f) for f in FOLD],ignore_index=True)
