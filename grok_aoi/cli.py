"""Command line for the full AOI chain."""
from __future__ import annotations

import argparse
import os
import sys
import traceback


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog='grok-aoi',
        description='Surface optical-cleanliness AOI: preprocess v3 → detection v4.1 → filter v4.2 → CSV, overlays, report.',
    )
    ap.add_argument('--db', required=True, help='image database root: <DB>/<Mode>/<angle>/NN.png')
    ap.add_argument('--out', required=True, help='output directory (created if missing)')
    ap.add_argument('--folders', nargs='*', default=None,
                    help='subset, e.g. Edge1/000 Side/000 (default: every mode/angle found under --db)')
    ap.add_argument('--nproc', type=int, default=max(1, min(4, os.cpu_count() or 2)),
                    help='worker processes for detection and overlays (default: min(4, CPU))')
    ap.add_argument('--config', default=None, help='dataset profile YAML (default: packaged willson profile)')
    ap.add_argument('--preprocess-config', default=None, help='preprocess v3 YAML (default: packaged config)')
    ap.add_argument('--with-eval', action='store_true',
                    help='also run proxy eval: THR=12 retention and a polarity-flip false-real estimate')
    ap.add_argument('--skip-eval', action='store_true', help='do not run proxy eval (this is the default)')
    ap.add_argument('--resume', action='store_true',
                    help='skip a stage whose outputs were already written for the same db, folders and config')
    ap.add_argument('--legacy-v31-decisions', default=None,
                    help='optional filter-v3.1 decisions root (side/<Folder>/decisions.csv) to enable Side T2 rule (b)')
    ap.add_argument('--glare-bands', default=None, help='glare_bands.json (default: packaged reference bands)')
    args = ap.parse_args(argv)
    if args.with_eval and args.skip_eval:
        ap.error('pass only one of --with-eval and --skip-eval')
    try:
        from grok_aoi.pipeline import run_all
        run_all(
            db=args.db, out=args.out, folders=args.folders, nproc=args.nproc,
            config=args.config, preprocess_config=args.preprocess_config,
            with_eval=bool(args.with_eval), resume=args.resume,
            legacy_v31=args.legacy_v31_decisions, glare_bands=args.glare_bands,
        )
    except Exception as e:
        traceback.print_exc()
        print(f'error: {type(e).__name__}: {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
