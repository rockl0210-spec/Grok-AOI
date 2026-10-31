# Grok-AOI source bundle — MANIFEST

Prepared 2026-10-31 (Asia/Taipei) from the box (<research-tree>). Originals untouched. No images, outputs, pickles, .venv or secrets are included (secret grep was clean).
Final versions bundled: preprocess **v3**, detection **v4.1**, filter **v4.2**, final overlays/CSV (`aoi_final/_code`).

## 0. Verdict
There is **no top-level driver** anywhere on the box. Every stage was run by hand, in a dev-style working directory, with env vars and pickles passed implicitly between stages. All stages have scripts (except the written report, see gap G1). A one-command `main` still has to be written (wrapper) and the hard-coded paths in section 4 parameterized. The result is **not bit-for-bit reproducible from these scripts alone** because of the v3.1 filter dependency (G2), reused v4.1 persistence pickles (G3), and the data-specific glare bands / labels (G4, G5).

## 1. Data layout the code assumes
`DB/<Mode>/<angle>/NN.png`, Mode in {Edge1, Edge2 (bright-field, angles 000/090), Side (dark-field, 000/090/180/270)}, 2464x2056 8-bit gray. NN = 00..45 (Edge, 46 frames) or 00..15 (Side, 16). 8 "folders" total: Edge1_000, Edge1_090, Edge2_000, Edge2_090, Side_000, Side_090, Side_180, Side_270 (248 frames).

## 2. Run order (as actually used) and what each step needs/produces
Names: DB=<DB> (input DB), PRE=<PRE>, DET=<DET>, F42=<FILTER>, FIN=<FINAL>.

### Step 1 — preprocess v3 (`preprocess/`)
- Run from `preprocess/` (package `aoi_preprocess`), once per folder (8x):
  `python -m aoi_preprocess run --root $DB --config config_v3.yaml --out $PRE --input <Mode>/<angle> --all --allow-protected-out`
  (cli args: --input --config --out --mode --frames --all --allow-protected-out --root --debug-dir -v). `--all` is required (safety limit max_files_without_flag=24 < 46 frames). `config_v3.yaml` has `safety.protected_out_roots: [<OUT>]` so `--allow-protected-out` is needed when out is under that path.
  Exact flags used in the original runs were reconstructed from cli.py/README — verify with `python -m aoi_preprocess run -h`.
- Output per frame `$PRE/<Mode>/<angle>/NN_{cleaned,mask,noise,nuisance,stain}.png` + `report.json`. Nuisance bits: 1 specular, 2 uneven, 4 rough, 8 water, 16 oil, 32 ink.
- Optional summaries: `tools/summarize_v3.py` (summary_v3.json, suspected_patterns.csv), `tools/make_figs_v3.py`. Test: `tests/test_v3.py`.
- Files: aoi_preprocess/{cli,config,pipeline_v3,pipeline_v2,real_pipeline,roi_modes,roi_v2,roi,sequence,illumination,io_utils,denoise,noise_map,nuisance,outputs,patterns,registration,stain,synthetic,pipeline,__init__,__main__}.py. pipeline_v3 imports pipeline_v2/real_pipeline/roi*/sequence/illumination, so the "older" modules are REQUIRED; only `pipeline.py` and `synthetic.py` are lazy-imported legacy (v1/synthetic path) and kept so cli.py imports cleanly. `config_v2.yaml` kept (pp v2 is needed only for the legacy v3.1 filter dependency, G2). README.md (Chinese, 30 KB) documents v1–v3.

### Step 2 — detection v4.1 (`detect/aoi_detect.py`, single script)
- `python aoi_detect.py [Edge1/000 Side/090 ...]` (no args = all 8 folders; 8 procs ~219 s). Needs PRE (`*_cleaned.png`, `report.json`, masks) + DB.
- Env vars (all optional): AOI_OUT (output dir), AOI_NPROC (default 6), AOI_FRAMES ("Edge1/000:10,20;..."), AOI_NOMASK, AOI_FLIP (polarity-flip null model), AOI_RR, AOI_HZ*, AOI_PEXCL, AOI_CG, AOI_SPLIT3N ... (see script).
- Output: `$DET/{edge|side}/<Src>_<ang>/{NN_candidates.json,candidates.csv,summary.json,masks/,suppressed_bands.json,suppressed_lines.json}` (394,375 candidates in the final run).
- Deps: numpy, cv2, scipy.ndimage, skimage (h_maxima, watershed), multiprocessing.
- `make_overlays_candidates_optional.py` = cosmetic candidate overlays (not needed). Superseded & NOT included: v1/v2/v3/v3.1 detectors (v3.1 detector is included only under filter/legacy_v31_dependency, see G2).

### Step 3 — filter v4.2 (`filter/`)  — dev-style, run inside ONE working directory (cwd holds M.pkl etc.)
Order:
1. `python stage_feats.py $DET $PRE 1`  → `M.pkl` (feats.build: loads all detect JSON/CSV, tier prior, neighbourhood feats, mask/zone features; arg3=1 recompute zone from PRE masks).
2. `python v31.py` → `v31status.pkl` (st31 = spatial status of each Side candidate vs. the v3.1 filter's decisions). **Reads `<FILTER_V31>/side/<folder>/decisions.csv` (v3.1 filter output) and an eval pickle `<EVAL>/filter_v41/master41.pkl` (sanity print only — delete those last 2 lines).** See G2.
3. For each of the 8 folders: `python stage_persist.py <folder>` → `persist_<folder>.pkl`, `reg_<folder>.json`, `persist_<folder>.log` (persist.py: NCC registration between consecutive frames + vote matching; env PRE_ROOT, default $PRE). NOTE: the v4.2 run did NOT recompute these — the v4.2 dev dir symlinks M.pkl/persist_*.pkl/reg_*.json/glare_bands.json to filter_v41_dev (G3). Step 1 + 3 would regenerate them (same detection v4.1, so expected identical; not re-verified).
4. `python build42.py` → `D_final.pkl`, `DEF_final.pkl` (needs M.pkl, v31status.pkl, persist_*.pkl, glare_bands.json; imports rules42; decisions + union-find defect consolidation; writes `image` path using <DB>).
5. `python group42.py D_final.pkl` → adds `same_object_group/_n/_n_real` columns **in place** (3rd arg → writes D_final_g.pkl instead). MUST run before out42 (build42 does not add them; out42 reads them). env GD=1 and optional BIGMAX arg (default 250000) per README/earlier use.
6. `OUT=$F42 DET_LABEL=v4.1 PRE_LABEL=v3 python out42.py` → `$F42/{edge|side}/<Src>_<ang>/{decisions.csv,decisions.json,defects.csv,defects.json,summary.json,overlay_*.png}` + `summ_all.json` (cwd). Reads raw images from <DB> for overlays.
7. `OUT=$F42 NEXP=394375 TOPIN='<json of input paths/versions>' [STAIN_ID=...] python top42.py` → `$F42/summary.json` + ID verification (every candidate id appears exactly once). NEXP = expected candidate count (must be changed per dataset!).
8. optional `python oil42.py` → oil_film_c20_review_list.csv (writes to hard-coded $F42).
- Config/thresholds: `rules42.py` — `TH` dict (all thresholds), `OPT0` switches, `ZH` (zh names), `run(m, pers, glare, opt)`; `glare_bands.json` (static per-Side-frame row bands). Rule changes vs v4.1: Side T2 conversion limited to (a) persistence-measurable & unconfirmed or (b) v4-new (st31 not real) & not measurable; Edge1 path C very-strong gets L1 review.
- Filter helper `mk_union.py` is eval-only (flip-null union set; hard-coded eval paths). Dev-only scripts NOT copied: fe.py fe2.py t42.py effects.py eff42.py sw.py cover.py retention.py miss.py side_ret42.py edge_ret42.py final_ret.py quick.py an.py.

### Step 4 — final deliverables (`report/`)
- `python make_csv.py` → `$FIN/final_defects.csv` (real candidates, columns incl. subclass, review_level L1/L2 label), `final_defects_unique.csv` (defects.csv merged), `defects_per_frame.csv`.
- `python make_overlays.py` → `$FIN/overlays/<folder>/<NN>.jpg` (0.5x raw image + real boxes; colors particle green/stain orange/scratch blue; L1 dashed, L2 thick+halo; ROI contour from PRE `*_mask.png`) + `contact_sheet.jpg` per folder (Pool(8)). Reads `$F42/*/decisions.csv`, DB, PRE.
- `reference_report_page_source.tsx.txt` = the source of the Canvas "report Page" (WriteCanvas). It has **hard-coded numbers** from the final run, so it is a layout/wording reference only, NOT a generator. It is an output sample; delete if not wanted.

### Step 5 — evaluation (`eval/`, proxy evaluation; no ground truth exists)
Used to produce the numbers in the report; NOT part of a normal run. Chain:
- Reference scan (THR=12 "ref blobs"): `eval/detect_v2/aoi_recall_eval.py --detect-dir <det> --out <dir> [--thr 12 --procs 6]` → `ref_blobs.pkl` (+ refscan.py; `eval/detect_eval_v41/recall_v4.py --dirs ... --out ...` is the v4 variant). Edge raw-dot reference: `eval/preprocess_v3/s4_dot_scan.py` → `tables/dots_all.pkl`. v3.1 era: `eval/f4_recall.py`. The scripts expect specific cwd/dirs.
- Polarity-flip null: run the SAME detector with env `AOI_FLIP=1 AOI_OUT=<dir>` (the box used a copy `aoi_eval/detect_v41/null_flip/aoi_detect_flip.py` = aoi_detect.py with OUT hard-coded; not bundled, identical logic) → `$FLIP/{edge,side}`.
- `eval/filter_v41/g5_union.py` (real + flip candidates → M_union41.pkl, union_dec_*.pkl; imports filter_v41's rules41/feats — **rules41.py is not bundled**), g1_master.py (master41.pkl), g2_sideT2.py (X7755.pkl), g3_recall.py (recall_blobs*.pkl).
- v4.2 checks: `eval/filter_v42/h1_verify.py` (integrity, diff v4.1→v4.2, same_object sanity), `h2_ret.py` (retention vs ref blobs), `h3_flip.py` (rules42 on the union set → union_dec_v42.pkl), `h4_flip.py` (false-share tables); results/discussion in `eval.md`. Run with cwd = eval/filter_v42 (`from common import *`, writes `tables/`).
- Eval needs M_union41.pkl, which the user's cleanup deleted (see eval/filter_v42/README.md): must be rebuilt via g5_union.py (needs filter_v41 code, not bundled) or ported to rules42/feats.

## 3. Dependencies (versions on the box, Python 3.13.5) — see requirements.txt
numpy 2.5.4, scipy 1.18.1, pandas 3.0.6, opencv-python-headless 5.0.0.93, scikit-image 0.26.0, PyYAML 6.0.3, matplotlib 3.11.2, pillow 13.0.0, tifffile 2026.10.15, imageio 2.38.1. (Preprocess venv at aoi_preprocess/.venv; detect/filter used system python3 — PyYAML only needed by preprocess; pandas only by filter/report/eval. pandas is not in the preprocess venv.) 8 cores, 15 GB RAM were available; feats/persist/build42 on ~394k candidates are memory-heavy.

## 4. Hard-coded paths / values to parameterize
- `<DB>` (input DB): filter/{persist,build42,out42}.py, report/make_overlays.py, eval/*, detect/aoi_detect.py (`DB`), legacy glare.py/stage1.py.
- `<PRE>` (PRE): detect/aoi_detect.py (`PRE`), report/make_overlays.py, filter/persist.py (env PRE_ROOT default), eval/*.
- `<DET>`: aoi_detect.py default OUT (AOI_OUT env exists). `<FILTER>`: report/make_csv.py + make_overlays.py (F42), filter/oil42.py, eval/filter_v42/common.py. `<FINAL>`: report/*.py (OUT).
- filter/v31.py: `<FILTER_V31>/side/{f}/decisions.csv`, `<EVAL>/filter_v41/master41.pkl`.
- filter/mk_union.py: `<EVAL>/detect_v41/null_flip/out`, `<FILTER>/_input_snapshot_v41`.
- eval/*: many `<EVAL>/**`, `<FILTER_V31>_v41/_code`, `<FILTER>/_code` paths (eval-only).
- filter/*.py rely on cwd for M.pkl, v31status.pkl, persist_*.pkl, reg_*.json, D_final.pkl, DEF_final.pkl, glare_bands.json, summ_all.json (cwd-relative).
- preprocess/config_v3.yaml: `safety.protected_out_roots: [<OUT>]`; **dataset-specific hard-coded labels** (Willson labels for Edge1/000, Edge1/090, Side/000, fixture frames 0 and 15) — fine for the current DB, must be generic/optional for another DB.
- Dataset constants: image size 2464x2056, frame counts, anomaly frames (Edge {0,45}, Side {0,15}) in rules42.py (ANOM_EDGE/ANOM_SIDE), special-case rule `e_strip` (Side_270 frame 15 x 1380-1500, y 780-1460) in rules42.py, FOLDERS lists, `NEXP=394375`, `STAIN_ID` default `Edge2_090_27_c1848` (top42.py), special IDs in eval/h1_verify.py.

## 5. Gaps
- G1 **Report**: no script creates the final report. The Page was authored by hand (WriteCanvas) with hard-coded numbers. A report generator (e.g. Markdown/HTML from summary.json + defects_per_frame.csv + final_defects*.csv + overlays) must be written. The eval retention/flip tables (aoi_eval/filter_v42/tables) are not regenerated by the main flow.
- G2 **v3.1 filter dependency**: `rules42` rule (b) uses `st31`, produced by v31.py from the **v3.1 filter's Side decisions.csv** (aoi_out/filter/side). v3.1 = detector v3.1 (`legacy_v31_dependency/aoi_detect_v31.py`) on **preprocess v2** (config_v2.yaml, pipeline: v2) + filter stage1.py/stage2.py (+ glare.py). Those are included under filter/legacy_v31_dependency/ but they were not re-run/verified here, and reproducing the exact v3.1 decisions requires the pp v2 outputs (the v3.1 filter README says it used `<PRE_V2>`). If st31 is absent, rules42 treats all as "real" so rule (b) converts nothing (changes Side results: 205 candidates) → not identical to the delivered v4.2. Choice: ship v3.1 chain, or drop rule (b)/make it optional.
- G3 Filter v4.2 reused v4.1's M.pkl/persist_*.pkl/reg_*.json via symlinks (not recomputed). Regenerating via steps 1+3 is expected to match (same detection snapshot, identical aoi_detect.py sha256 d2574a8f…) but untested.
- G4 `glare_bands.json` is a static file generated by legacy `glare.py` (raw row-profile excess; median filter 81, thr max(5, 6*MAD)). Bundled both; the generator is not in any pipeline step and was not re-run.
- G5 config_v3.yaml has dataset-specific hard-coded labels (above).
- G6 No driver/CLI/orchestration; filter stages are cwd+pickle+env-var based; no logging/progress/resume; `NEXP`/`TOPIN` must be supplied by hand.
- G7 Eval reference sets (ref_blobs.pkl, recall_blobs*.pkl, X7755.pkl, M_union41.pkl, flip-null outputs) came from ad hoc runs; M_union41.pkl was deleted; filter_v41 code (rules41.py, feats of v41) not bundled. Eval is documentation-only for now.
- G8 Exact original CLI invocations/logs for step 1 and steps 3–7 were not recorded in a script; reconstructed from sources (stage*.py argv, env var reads). Overlay/review-level semantics (L1/L2) are defined in rules42.py/out42.py/make_csv.py.
- G9 Tests: only preprocess tests/test_v3.py; none for detect/filter.

## 6. Suggested main-program shape (not implemented)
`python grok_aoi.py --db <DB> --out <OUT> [--procs N]`: (1) pp v3 for each Mode/angle found, (2) aoi_detect with AOI_OUT, (3) filter steps 1,3,4,5,6,7 in a work dir with st31 optional (G2), (4) make_csv + make_overlays reading <OUT>, (5) generate report.md/html + CSVs. All paths from CLI; discover folders from the DB rather than the FOLDERS list.
