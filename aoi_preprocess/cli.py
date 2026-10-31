from __future__ import annotations

import argparse
import logging
import sys

from .config import ConfigError, load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aoi_preprocess", description="AOI preprocessing + ROI masking")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process a file or a directory of images")
    r.add_argument("--input", required=True, help="grayscale PNG file or directory (BMP/TIFF/color = fallback)")
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--out", required=True)
    r.add_argument("--mode", help="edge | side (default: from the folder name, see mode_rules in config.yaml)")
    r.add_argument("--frames", help="comma list of frame stems to process, e.g. 00,01,22,45 (safe sampling)")
    r.add_argument("--all", action="store_true", help="confirm processing a whole big folder")
    r.add_argument("--allow-protected-out", action="store_true", help="allow writing under config safety.protected_out_roots (explicitly approved full run)")
    r.add_argument("--debug-dir", help="write debug files (corrected, classes, overlay) here instead of <out>/_debug")
    r.add_argument("--root", help="input root; output sub-folders mirror the path below it (e.g. the image database root)")
    r.add_argument("--flat", help="flat-field frame (overrides config)")
    r.add_argument("--dark", help="dark frame (overrides config)")
    r.add_argument("-v", "--verbose", action="store_true")
    s = sub.add_parser("synth", help="generate the synthetic demo input (+ ground truth, flat, dark)")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--kind", choices=("panel", "wafer"), default="panel", help="panel = square panel (default)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if getattr(a, "verbose", False) else logging.INFO,
                        format="%(levelname)s %(message)s")
    try:
        if a.cmd == "synth":
            from .synthetic import make_synthetic
            for k, v in make_synthetic(a.out, a.seed, a.kind).items():
                print(f"{k}: {v}")
            return 0
        cfg = load_config(a.config)
        if "modes" in cfg and cfg.to_dict().get("pipeline") == "v3":
            from .pipeline_v3 import run_v3
            reports, failed = run_v3(a.input, cfg, a.out, a.mode, a.frames.split(",") if a.frames else None,
                                     a.all, a.root, a.allow_protected_out)
            print(f"done: {len(reports)} ok, {len(failed)} failed -> {a.out}")
            return 1 if failed else 0
        if "modes" in cfg and cfg.to_dict().get("pipeline") == "v2":
            from .pipeline_v2 import run_v2
            reports, failed = run_v2(a.input, cfg, a.out, a.mode, a.frames.split(",") if a.frames else None,
                                     a.all, a.root, a.allow_protected_out)
            print(f"done: {len(reports)} ok, {len(failed)} failed -> {a.out}")
            return 1 if failed else 0
        if "modes" in cfg:
            from .real_pipeline import run_real
            reports, failed = run_real(a.input, cfg, a.out, a.mode, a.frames.split(",") if a.frames else None,
                                       a.all, a.root, a.allow_protected_out, a.debug_dir)
            print(f"done: {len(reports)} ok, {len(failed)} failed -> {a.out}")
            return 1 if failed else 0
        from .pipeline import run
        reports, failed = run(a.input, cfg, a.out, a.flat, a.dark)
        print(f"done: {len(reports)} ok, {len(failed)} failed -> {a.out}")
        return 1 if failed else 0
    except (ConfigError, FileNotFoundError, ValueError) as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
