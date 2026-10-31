"""End-to-end driver: preprocess v3 → detection v4.1 → filter v4.2 → CSV, overlays, report."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path

import yaml

from grok_aoi import __version__

PKG = Path(__file__).resolve().parent
DEFAULT_PROFILE = PKG / 'configs' / 'dataset.yaml'
DEFAULT_PREPROCESS = PKG / 'configs' / 'preprocess_v3.yaml'
DEFAULT_GLARE = PKG / 'filter' / 'glare_bands.json'

log = logging.getLogger('grok_aoi')


class _Dumper(yaml.SafeDumper):
    pass


def _repr_str(dumper, data):
    if data.isdigit():
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


_Dumper.add_representer(str, _repr_str)


def _setup_log(out: Path):
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(out / 'logs' / 'run.log', encoding='utf-8')
    fh.setFormatter(fmt)
    log.addHandler(sh)
    log.addHandler(fh)
    log.propagate = False


def load_profile(path):
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'dataset profile must be a mapping: {path}')
    return data


def discover(db: Path, only=None):
    """Return [(mode, angle, [frame stems])] for every Mode/angle that has NN.png."""
    if not db.is_dir():
        raise FileNotFoundError(db)
    want = None
    if only:
        want = set()
        for item in only:
            item = item.strip().strip('/')
            if not item:
                continue
            want.add(item)
            if '/' in item:
                a, b = item.split('/', 1)
                want.add(f'{a}_{b}')
            elif '_' in item:
                a, b = item.split('_', 1)
                want.add(f'{a}/{b}')
    found = []
    for mode in sorted(p for p in db.iterdir() if p.is_dir() and not p.name.startswith('.')):
        for ang in sorted(p for p in mode.iterdir() if p.is_dir() and not p.name.startswith('.')):
            frames = sorted(p.stem for p in ang.glob('*.png') if p.stem.isdigit())
            if not frames:
                continue
            key = f'{mode.name}/{ang.name}'
            fid = f'{mode.name}_{ang.name}'
            if want and key not in want and fid not in want and mode.name not in want:
                continue
            found.append((mode.name, ang.name, frames))
    if not found:
        raise FileNotFoundError(f'no <Mode>/<angle>/NN.png frames under {db}' + (f' matching {sorted(want)}' if want else ''))
    return found


def _materialize_preprocess(profile, pp_path: Path, dest: Path):
    data = yaml.safe_load(pp_path.read_text(encoding='utf-8'))
    data.setdefault('safety', {})
    data['safety']['protected_out_roots'] = []
    if 'preprocess_labels' in profile and profile['preprocess_labels'] is not None:
        data.setdefault('common', {}).setdefault('v3', {})['labels'] = profile['preprocess_labels']
    af = profile.get('anomaly_frames') or {}
    modes = data.setdefault('modes', {})
    if 'edge' in modes and 'edge' in af:
        modes['edge'].setdefault('abnormal', {})['expected_frames'] = [f'{int(x):02d}' for x in af['edge']]
    if 'side' in modes and 'side' in af:
        modes['side'].setdefault('abnormal', {})['expected_frames'] = [f'{int(x):02d}' for x in af['side']]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.dump(data, Dumper=_Dumper, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return dest


def _fp(*parts):
    raw = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def _marker(out: Path, stage):
    return out / '_status' / f'{stage}.json'


def _done(out: Path, stage, fp, resume):
    if not resume:
        return False
    p = _marker(out, stage)
    if not p.is_file():
        return False
    try:
        meta = json.loads(p.read_text())
    except json.JSONDecodeError:
        return False
    return meta.get('fingerprint') == fp and meta.get('ok') is True


def _mark(out: Path, stage, fp, extra=None):
    p = _marker(out, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = {'fingerprint': fp, 'ok': True, 'stage': stage}
    if extra:
        body.update(extra)
    p.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=str))


def run_all(db, out, folders=None, nproc=2, config=None, preprocess_config=None,
            with_eval=False, resume=False, legacy_v31=None, glare_bands=None):
    db, out = Path(db).resolve(), Path(out).resolve()
    if db == out or out in db.parents or db in out.parents:
        raise ValueError(f'refusing to write inside the input tree: db={db} out={out}')
    out.mkdir(parents=True, exist_ok=True)
    _setup_log(out)
    t_all = time.perf_counter()
    profile_path = Path(config) if config else DEFAULT_PROFILE
    pp_src = Path(preprocess_config) if preprocess_config else DEFAULT_PREPROCESS
    profile = load_profile(profile_path)
    glare_src = Path(glare_bands) if glare_bands else DEFAULT_GLARE
    if not glare_src.is_file():
        raise FileNotFoundError(glare_src)
    selected = discover(db, folders)
    log.info('Grok-AOI %s  db=%s  folders=%s  nproc=%s  eval=%s  rule_b=%s',
             __version__, db, [f'{a}/{b}' for a, b, _ in selected], nproc, with_eval, bool(legacy_v31))
    cfg_dir = out / 'config'
    pp_copy = _materialize_preprocess(profile, pp_src, cfg_dir / 'preprocess_v3.yaml')
    shutil.copy(profile_path, cfg_dir / 'dataset.yaml')
    timings = {}
    meta = {
        'version': __version__,
        'db': str(db),
        'out': str(out),
        'folders': [f'{a}_{b}' for a, b, _ in selected],
        'folder_frames': {f'{a}/{b}': fr for a, b, fr in selected},
        'config': str(profile_path),
        'preprocess_config': str(pp_copy),
        'profile': profile,
        'nproc': nproc,
        'with_eval': with_eval,
        'rule_b': bool(legacy_v31),
        'legacy_v31': str(legacy_v31) if legacy_v31 else None,
        'glare_bands': str(glare_src),
    }
    (cfg_dir / 'run_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

    base_fp = _fp(str(db), meta['folders'], pp_copy.read_text(encoding='utf-8'), profile_path.read_text(encoding='utf-8'))
    fil_fp = _fp(base_fp, str(legacy_v31 or ''), glare_src.read_bytes().hex() if glare_src.stat().st_size < 5_000_000 else glare_src.stat().st_size)

    # ---- preprocess ----
    pre = out / 'preprocess'
    t = time.perf_counter()
    if _done(out, 'preprocess', base_fp, resume):
        log.info('[preprocess] resume: skip')
    else:
        from aoi_preprocess.config import load_config
        from aoi_preprocess.pipeline_v3 import run_v3
        cfg = load_config(pp_copy)
        failed = []
        for mode, ang, frames in selected:
            src = db / mode / ang
            log.info('[preprocess] %s/%s  %d frames', mode, ang, len(frames))
            reports, bad = run_v3(src, cfg, pre, run_all=True, root=db, allow_protected=True)
            log.info('[preprocess] %s/%s done ok=%d failed=%d', mode, ang, len(reports), len(bad))
            failed.extend(bad)
        if failed:
            raise RuntimeError(f'preprocess failed on {len(failed)} file(s): {failed[:8]}')
        _mark(out, 'preprocess', base_fp)
    timings['preprocess'] = round(time.perf_counter() - t, 2)

    # ---- detect ----
    det = out / 'detect'
    t = time.perf_counter()
    if _done(out, 'detect', base_fp, resume):
        log.info('[detect] resume: skip')
    else:
        from grok_aoi.detect import aoi_detect
        only = {(a, b) for a, b, _ in selected}
        log.info('[detect] v4.1  %d folders  nproc=%s', len(only), nproc)
        aoi_detect.run(only=only, nproc=nproc, pre=str(pre), db=str(db), out=str(det),
                       anom=profile.get('anomaly_frames'), flip=False)
        _mark(out, 'detect', base_fp)
    timings['detect'] = round(time.perf_counter() - t, 2)

    # ---- filter ----
    work = out / 'filter' / 'work'
    froot = out / 'filter'
    t = time.perf_counter()
    if _done(out, 'filter', fil_fp, resume):
        log.info('[filter] resume: skip')
    else:
        from grok_aoi.filter import feats
        from grok_aoi.filter.stage_feats import build_features
        from grok_aoi.filter.stage_persist import run_persistence
        from grok_aoi.filter.v31 import build_st31
        from grok_aoi.filter.build42 import build as build_decisions
        from grok_aoi.filter.group42 import apply as apply_groups
        from grok_aoi.filter.out42 import write_outputs
        from grok_aoi.filter.top42 import write_summary
        from grok_aoi.filter.oil42 import write_oil_list
        from grok_aoi.filter.rules42 import apply_dataset
        work.mkdir(parents=True, exist_ok=True)
        shutil.copy(glare_src, work / 'glare_bands.json')
        af = profile.get('anomaly_frames') or {}
        apply_dataset(af.get('edge', []), af.get('side', []), profile.get('e_strip'))
        folder_ids = [f'{a}_{b}' for a, b, _ in selected]
        log.info('[filter] features')
        M = build_features(str(det), str(pre), str(work), True, folder_ids)
        if len(M) == 0:
            log.info('[filter] no candidates; writing empty decisions')
            for a, b, _ in selected:
                mode = 'edge' if a.startswith('Edge') else 'side'
                d = froot / mode / f'{a}_{b}'
                d.mkdir(parents=True, exist_ok=True)
                (d / 'decisions.csv').write_text('')
                (d / 'decisions.json').write_text('[]')
                (d / 'defects.csv').write_text('')
                (d / 'defects.json').write_text('[]')
            json.dump({'n_candidates': 0, 'by_decision': {}, 'side_T2_conversion': {'total': 0, 'rule_a_measurable_unconfirmed': 0, 'rule_b_v4new_not_measurable': 0, 'note': 'no candidates'}},
                      open(froot / 'summary.json', 'w'), indent=1)
            json.dump({}, open(work / 'summ_all.json', 'w'))
            pd_empty(work)
        else:
            present = sorted(M.folder.unique(), key=feats.folder_sort_key)
            for f in present:
                log.info('[filter] persistence %s', f)
                run_persistence(f, str(work), str(db), str(pre))
            rule_b = bool(legacy_v31)
            if rule_b:
                log.info('[filter] legacy v3.1 spatial status from %s', legacy_v31)
                build_st31(str(work), str(legacy_v31))
            else:
                log.info('[filter] rule (b) off; Side T2 uses persistence rule (a) only')
            im = _image_size(db, selected, profile)
            log.info('[filter] decisions  image=%sx%s  rule_b=%s', im[0], im[1], rule_b)
            build_decisions(str(work), str(db), width=im[0], height=im[1], rule_b=rule_b, folders=present)
            log.info('[filter] same-object groups')
            apply_groups(str(work / 'D_final.pkl'))
            log.info('[filter] write decisions')
            write_outputs(str(work), str(froot), str(db), folders=present, exemplar=profile.get('overlay_exemplar') or {})
            log.info('[filter] summary')
            write_summary(str(work), str(froot), inputs={
                'detection': 'v4.1', 'preprocess': 'v3', 'filter': 'v4.2',
                'db': str(db), 'rule_b': rule_b, 'legacy_v31': str(legacy_v31) if legacy_v31 else None,
            }, folders=present, highlight_ids=profile.get('highlight_ids') or [],
               stain_id=profile.get('stain_id'), rule_b=rule_b, legacy_path=str(legacy_v31) if legacy_v31 else None)
            write_oil_list(str(work), str(froot / 'oil_film_c20_review_list.csv'))
        _mark(out, 'filter', fil_fp, {'n_candidates': int(len(M))})
    timings['filter'] = round(time.perf_counter() - t, 2)

    # ---- deliverables ----
    final = out / 'final'
    t = time.perf_counter()
    if _done(out, 'deliver', fil_fp, resume):
        log.info('[deliver] resume: skip')
    else:
        from grok_aoi.report.make_csv import write_csvs
        from grok_aoi.report.make_overlays import write_overlays
        log.info('[deliver] CSV')
        write_csvs(str(froot), str(final), meta['folders'])
        log.info('[deliver] overlays')
        write_overlays(str(froot), str(pre), str(db), str(final), nproc=nproc, folders=meta['folders'])
        _mark(out, 'deliver', fil_fp)
    timings['deliver'] = round(time.perf_counter() - t, 2)

    # ---- eval (optional) ----
    t = time.perf_counter()
    if with_eval:
        eval_fp = _fp(fil_fp, 'eval')
        if _done(out, 'eval', eval_fp, resume):
            log.info('[eval] resume: skip')
        else:
            from grok_aoi.evalproxy import run_eval
            glare = json.loads(glare_src.read_text())
            log.info('[eval] proxy metrics')
            run_eval(str(db), str(pre), str(det), str(froot), str(work), str(out / 'eval'),
                     profile.get('anomaly_frames') or {}, nproc, meta['folders'], glare)
            _mark(out, 'eval', eval_fp)
    else:
        log.info('[eval] skipped (pass --with-eval to compute retention and flip estimates)')
    timings['eval'] = round(time.perf_counter() - t, 2)

    # ---- report (always, cheap) ----
    t = time.perf_counter()
    meta['timings_s'] = timings
    meta['timings_s']['total'] = round(time.perf_counter() - t_all, 2)
    (cfg_dir / 'run_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    from grok_aoi.report.generate import render
    html_path = render(out)
    timings['report'] = round(time.perf_counter() - t, 2)
    meta['timings_s'] = timings
    meta['timings_s']['total'] = round(time.perf_counter() - t_all, 2)
    (cfg_dir / 'run_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    render(out)
    log.info('[report] %s', html_path)
    log.info('done in %.1fs  %s', time.perf_counter() - t_all, {k: timings[k] for k in timings})
    return html_path


def pd_empty(work: Path):
    import pandas as pd
    pd.DataFrame().to_pickle(work / 'M.pkl')
    pd.DataFrame().to_pickle(work / 'D_final.pkl')
    pd.DataFrame().to_pickle(work / 'DEF_final.pkl')


def _image_size(db: Path, selected, profile):
    import cv2
    a, b, frames = selected[0]
    im = cv2.imread(str(db / a / b / f'{frames[0]}.png'), cv2.IMREAD_UNCHANGED)
    if im is None:
        spec = profile.get('image') or {}
        return int(spec.get('width') or 2464), int(spec.get('height') or 2056)
    h, w = im.shape[:2]
    spec = profile.get('image') or {}
    sw, sh = spec.get('width'), spec.get('height')
    if sw and sh and (int(sw), int(sh)) != (w, h):
        log.info('frame size %dx%d differs from profile %sx%s; bounds use the frame', w, h, sw, sh)
    return w, h
