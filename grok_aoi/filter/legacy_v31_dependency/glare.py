"""Regenerate Side glare-band rows from raw frames (median profile, window 81, thr max(5, 6*MAD)).

Not called by grok_aoi.py. The packaged filter uses grok_aoi/filter/glare_bands.json.
  python glare.py <db> <out.json>
"""
import json, glob, os, sys
import cv2
import numpy as np
from scipy.ndimage import median_filter

def build(db, angles=('000', '090', '180', '270')):
    out = {}
    for ang in angles:
        for f in sorted(glob.glob(os.path.join(db, 'Side', ang, '*.png'))):
            stem = os.path.splitext(os.path.basename(f))[0]
            if not stem.isdigit():
                continue
            k = int(stem)
            im = cv2.imread(f, 0)
            if im is None:
                continue
            im = im.astype(np.float32)
            rp = np.median(im, axis=1)
            base = median_filter(rp, size=81, mode='nearest')
            ex = rp - base
            mad = np.median(np.abs(ex - np.median(ex))) * 1.4826 + 1e-3
            thr = max(5.0, 6 * mad)
            rows = ex > thr
            bands = []
            i = 0
            while i < len(rows):
                if rows[i]:
                    j = i
                    while j + 1 < len(rows) and rows[j + 1]:
                        j += 1
                    if j - i + 1 <= 60:
                        bands.append([max(0, i - 4), min(len(rows) - 1, j + 4), round(float(ex[i:j + 1].max()), 1)])
                    i = j + 1
                else:
                    i += 1
            out[f'Side_{ang}_{k:02d}'] = bands
    return out

if __name__ == '__main__':
    db = sys.argv[1]
    dest = sys.argv[2] if len(sys.argv) > 2 else 'glare_bands.json'
    out = build(db)
    json.dump(out, open(dest, 'w'))
    print(sum(1 for v in out.values() if v), len(out))
