# AOI filter v4.2 (detection v4.1 + preprocess v3 masks)

Created 2026-10-30 (Asia/Taipei). Built on filter_v41 with the same v4.1 detection snapshot (`_input_snapshot_v42/` is an identical copy of `filter_v41/_input_snapshot_v41/`, snapshot time 19:47:14+08:00, sha256 lists verified; 394,375 candidates) and preprocess v3 (<PRE>). `filter_v41/`, `filter_v4/` and `filter/` were not modified (`filter/` is only read for the v3.1 decisions). Code: `_code/` (rules42.py is the rule set), results: `_dev_results/`.

Every candidate ID appears exactly once (394,375 unique). Per angle: `edge|side/<Src>_<angle>/{decisions.csv,decisions.json,defects.csv,defects.json,summary.json,overlay_*.png}`. New columns in decisions.csv/json: `same_object_group`, `same_object_n`, `same_object_n_real`. Overlay marks: `?` = review L1, `!` = review L2.

## Totals
real 213,680 / nuisance 180,695; review 145,090 (L1 138,104, L2 6,986; real+review 101,569, real L1 96,653, real L2 4,916; nuisance review 43,521, L2 2,070); real without review 112,111; unique defects 196,282.
(v4.1: real 206,466, review 145,022, L1 131,275, L2 13,747, real without review 112,179, defects 189,292.)

## Changes vs v4.1
1. Rough-zone mid_ac = 25: unchanged (rough zone only).
2. Side T2 conversion (the v4.1 `N_SIDE_T2_UNCONFIRMED` rule) now only for `R_SIDE_NORMAL_T2` / `R_SIDE_UNEVEN_T2` reals that are not anomaly-frame and are one of:
   * (a) persistence measurable (`pers_tested`) and NOT confirmed: 336 candidates;
   * (b) a v4-new real (not real in v3.1) and persistence not measurable: 205 candidates.
   Total 541 (Side_000 95, Side_090 172, Side_180 127, Side_270 147), all nuisance/noise `N_SIDE_T2_UNCONFIRMED`, review L2. All the other former conversions (7,214 candidates) are real again and get their review_level from the normal rules (not forced into L2: 6,590 are L1, 624 L2 for the usual L2 reasons). Anomaly-frame candidates are excluded from the conversion (they stay `N_ANOM_UNVERIFIED`; in v4.1 their label was overwritten, which is why a raw rerun of the v4.1 rule gives 7,928 instead of 7,755).
   "Real in v3.1" = the candidate centroid lies in a v3.1 candidate bbox (+-2 px) of the same frame (<FILTER_V31>/side/*/decisions.csv, match since IDs differ), v3.1 box area <= 30x the candidate area, and that v3.1 decision was real (or a large, >30x area, real box). This reproduces Eval's `st31` for all 60,527 Side candidates.
   Count note: Eval estimated about 475 conversions (C = 272 measurable-unconfirmed). The filter's own persistence confirm (`pers_confirmed` = tested & matched & control & fn check) gives 336 for (a); (b) 205 vs the estimate. Total 541.
3. Edge1 path C gray_mid/high reals (901, including the 68 that had no review in v4.1) stay real with review (660 L1, 241 L2 for the usual reasons). Edge1 L1 +68.
4. `same_object_group` (all 394,375 candidates). Method: inside one (folder, frame), union-find on candidate boxes; two candidates are linked if their bboxes, each dilated by 1 px, intersect (overlap, containment or a gap of up to 2 px) AND they have the same contrast polarity. Boxes of area > 250,000 px2 (21) never link. Singletons get their own group. Id `<folder>_<frame>_g<NNNNN>`. `same_object_n` = group size, `same_object_n_real` = real members. Dilation 2 px was rejected (chains to groups of 6,525); 0 px gives max group 151, 1 px gives max group 241 (oil-zone fibers/patches). Stats: 255,142 groups, 39,896 with >=2 members (179,129 candidates; 109,021 of them real), largest group 241, 135,352 groups contain a real, 19,475 of those with more than one real member. Defect consolidation (`defect_id`) is a separate, earlier mechanism and is unchanged.

## Side reference retention (Eval reference set, 10,731 blobs, with review / real without review)
bin 12-20 / 20-40 / 40+: v3.1 63.0/88.1/98.5 (nr 36.6/80.2/94.2); v4 68.6/89.7/99.2; v4.1 43.9/83.5/95.9 (nr 35.6/79.1/94.5); v4.2 67.6/89.4/99.1 (nr 35.6/79.1/94.5). Tables: `_dev_results/side_ret_bin_v42.csv`, `side_ret_folder_bin_v42.csv`. Edge retention is unchanged from v4.1.

## Extra lists
`oil_film_c20_review_list.csv` (nuisance, fine_class oil_film, |c|>=20; 6,258 rows, identical to v4.1 since none of the changes touch it) and `oil_film_c20_review_list_v4.csv` (from filter_v4 outputs, 1,704 rows).

## Special cases
Side_000_05_c0736: real particle (R_SIDE_UNEVEN_T1), review L1. Edge2_090_27_c1848: real stain, review L2.
