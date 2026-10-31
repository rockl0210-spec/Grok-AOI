"""Historical helper that built a real+flip union feature table. Not used by grok_aoi.py.

The packaged proxy evaluation lives in grok_aoi.evalproxy and takes explicit paths.
"""
import os, sys

def main():
    raise SystemExit('mk_union.py is not part of the packaged pipeline. Use: python grok_aoi.py --db ... --out ... --with-eval')

if __name__ == '__main__':
    main()
