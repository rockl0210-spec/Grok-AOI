#!/usr/bin/env python3
"""Entry point: python grok_aoi.py --db <image database> --out <output dir>."""
from grok_aoi.cli import main
import sys
if __name__ == '__main__':
    raise SystemExit(main())
