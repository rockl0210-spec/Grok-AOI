# Independent verification: detection v4.1 (<DET>, on preprocess v3)
Date: Fri 2026-10-30 (Asia/Taipei, UTC+8). Read-only on detect/, detect_v4/, detect_v41/, filter*/, detect_eval_v41/. Every number below was recomputed by me from candidates.csv + raw/cleaned images (bot numbers are quoted only for comparison). Scripts in this folder (common.py, t1_schema.py, d2_recall.py, d3_flip.py, d3_persist.py, d4_*.py); tables in `tables/`, figures in `fig/`. Compared: v3.1 / v4 / v4.1.

## Verdict
**GO: switch the filter from v4 to v4.1** (v4.1 is better than v4 on every recall bin, repairs v4's regressions, and removes v4's haze blow-up). All of the bot's headline numbers reproduce. Conditions / things the filter side must know (details in "Remaining issues"):
(a) the 4,094 raw_rescue candidates are real raw-image dots (persistence 77.8% vs 36.2% chance) but 71% of them are one dense oil dot-lattice region (Edge2/090 frames 25+26) and they are tiny (median 7 px): the filter should treat them as low-priority/review, not as 4k confident "particles". The bot's reason ("v3 cleaned erases them") is NOT supported by pixel contrast (cleaned residual -49 vs raw -44, see sec. 3);
(b) the null is slightly worse where haze is now split into cores: Edge1/090 oil path A gray_low null 2.0% -> 4.0% (Edge1 path A gray_low 2.16 -> 2.67%), with only +2.8 pts oil 12-16 recall there;
(c) Edge1/090 remains the weakest folder (12-20: 68.0%, oil 12-16: 49.7%) and the uneven zone is still ~39% at 12-16;
(d) the filter_v4 run is already finished (summary.json written 19:24 UTC+8, 378,226 candidates) so a v4.1 run is a re-run, not a replacement mid-run.

## 1. Integrity / schema; Side identity  (t1_schema.py; tables/v41_schema.*, v41_overlap_check.csv)
* 394,375 rows (v3.1 376,277; v4 378,226), 8 folders, IDs unique, no NaN fields (except nullable nuisance_labels and Side persistence), bbox/centroid inside image, area >=3, confidence 0.023-1.0, contrast_tier vs |contrast| 0 mismatches, straddles_roi vs roi_overlap_frac 0 mismatches. New columns raw_rescue (4,094 True, as claimed) and haze_texture (2,538 True, as claimed).
* keep reasons: inside 389,186, frac 4,489, high_contrast 700; overlap recomputed on 7,075 sampled rows: median error 2e-5, p99 5e-4, 0 errors >0.05.
* Counts v3.1 / v4 / v4.1: path A 316,677 / 317,050 / 343,535; B 50,041 / 41,081 / 37,763; C 9,559 / 20,095 / 13,077. Tiers gray_low 258,679 / 268,178 / 264,209; gray_mid 85,070 / 83,304 / 100,906; high 32,528 / 26,744 / 29,260. Zones oil 14 / 155,497 / 170,887; rough 182,628 / 25,693 / 26,103. Consolidation split 21,002 / 28,716 / 13,292; regrown 3,955 / 3,176 / 3,940. Per folder v4.1: Edge1/000 39,051; Edge1/090 83,890; Edge2/000 103,345; Edge2/090 107,562 (v4: 37,859 / 80,076 / 99,151 / 100,613).
* **Side identical to v4: confirmed.** All four Side folders: same rows, same IDs, same values in every field except the path/version strings (mask_patch, sensitivity_version). 60,527 rows.
* Anomaly-frame candidates 3,715 / 3,665 / 3,665.

## 2. Recall reproduction (THR=12 full-frame Edge reference, d2_recall.py; tables/recall_full_*)
Reference: my 185,608 raw-image dark blobs (dots_all.pkl det=raw, 176 normal frames), strict cover = centroid in bbox+-2 and cand area <=30x blob. **Caveat: the reference is built from the raw image, raw_rescue also detects on the raw image, so recall gains attributable to raw_rescue are partly circular; the column "w/o rescue" removes raw_rescue candidates from the cover, and sec. 3 tests reality independently.**
| bin (peak) | n | v3.1 | v4 | v4.1 | v4.1 w/o rescue |
|---|---|---|---|---|---|
| Edge 12-20 | 71,107 | 72.3 | 74.5 | **79.0** | 78.4 |
| Edge 20-40 | 84,118 | 91.7 | 94.5 | **96.6** | 95.2 |
| Edge 40+ | 30,383 | 99.8 | 99.8 | **100.0** | 99.8 |
* By zone 12-20 (v3.1 / v4 / v4.1): oil/ink 62.2 / 65.5 / **72.0** (w/o rescue 71.2); rough 80.3 / 81.4 / 83.6; normal 95.8 / 95.7 / 96.3; uneven 53.3 / 54.5 / 55.7. 20-40 oil: 88.3 / 92.2 / 95.3 (93.2 w/o rescue).
* By folder 12-20 / 20-40 (v3.1 / v4 / v4.1): Edge1/000 81.8/76.6/**84.8** and 98.1/98.1/99.5; Edge1/090 60.1/66.8/**68.0** and 87.0/93.0/93.1; Edge2/000 83.3/82.0/**87.8** and 96.8/96.8/98.9; Edge2/090 73.0/75.9/**81.7** and 91.1/93.3/97.9 (w/o rescue 79.9 / 94.1: Edge2/090 is where rescue matters).
* Contaminated 12-20 (bot's list, all reproduced): **Edge1/000 oil 72.9/58.5/78.2; Edge2/000 oil 74.0/70.8/82.6; stain-only zone (v3 zone 1) 89.2/82.1/89.3; Edge1/000 oil 12-16 63.2/44.7/66.8**. The v4 Edge1/000 regression is repaired (now above v3.1). Others: Edge1/090 oil 12-20 53.9/62.1/63.2, Edge2/090 oil 64.6/68.9/77.0.
* Contaminated (oil/ink/water) >=40, n=19,843: uncovered any-size **22 / 37 / 12**; strict (<=30x) **35 / 50 / 13** (bot 22/37/12 and 35/50/13 reproduced). Without raw_rescue candidates v4.1 is 53 any / 54 strict, i.e. v4.1's own detector is worse than v4 on strong contaminated dots (37 -> 53) and the rescue is what brings it to 12. Remaining 12: Edge1/090 oil 9, Edge2/090 oil 3.
* Churn v4 -> v4.1: 2,826 blobs lost, 7,878 gained. Lost 12-20 / 20-40 by folder: Edge1/090 1,350 / 715, Edge2/090 298 / 70, Edge2/000 257 / 43, Edge1/000 77 / 9. Edge1/090 gets essentially nothing from v4.1 (+1.2 at 12-20, 0.0 at 20-40) and its frames 19-23 are slightly worse than v4 (-0.2 to -1.0 per frame). Worst v4.1 frames (n>=500): Edge1/090/18 79.5, /22 80.0, /17 80.5, /16 81.5, /28 82.4. Biggest gains: Edge2/000/33 80.1 -> 92.9, Edge1/000/21 +11.6, /20 +9.7.
* Unresolved items reproduced: Edge1/090 oil 12-16 44.0 / 46.9 / **49.7**; Edge1/090 oil 16-20 60.8/72.6/72.7; **Edge uneven 12-16 pooled 38.3 / 37.9 / 38.9** (n=1,239; bot ~38% OK): Edge1/000 49.4, Edge1/090 30.6, Edge2/000 45.5, **Edge2/090 19.0** (flat since v3.1). Uneven 20-40: 95.0/99.5/99.7.

## 3. raw_rescue: are the 4,094 dots real?  (d3_persist.py, d3_flip.py, d4_rescue_sheet.py; fig/sheet_rescue_*.png, tables/persist_*, rescue_*.txt)
Facts: 4,094 rows, all path C, 4,088 oil zone (4 uneven, 1 rough, 1 water), class particle 4,058 / scratch 27 / stain 9; area median 7 px (p75 12, max 207), contrast -24..-70 (median -31.5), tier gray_mid 3,685 / high 409. Concentration: **Edge2/090 3,620 (88%), Edge1/090 346, Edge2/000 127, Edge1/000 1**; within Edge2/090 frames **25: 1,900 (12.0% of that frame's candidates) and 26: 1,015 (8.9%)** = 2,915 = 71% of all rescue. All of this is the dense oil dot-lattice region (13k-15k candidates per frame); concentration is plausible (that is where the raw blobs are), not an artefact of one bad frame.
| test | raw_rescue | other tiny oil (<=15 px) | note |
|---|---|---|---|
| persistence proxy match / chance / structural | 77.8% / 36.2% / 65.2% (n=419) | 71.5% / 21.8% / 63.6% (n=8,516) | rescued dots persist as well as ordinary tiny oil dots; chance is high because the lattice is dense (excess +41.6 pts) |
| by peak |c| <=30 / 30-40 / 40-60 | struct 48.5% (n=147) / 76.4% (226) / 68.1% (46) | | the weakest third is only ~half-real |
| flip null | 101/4,094 = **2.47%** (bot OK); Edge2/090 1.3%, Edge2/000 3.1%, **Edge1/090 10.4%** (36/346) | | Edge1/090 rescue is the least trustworthy part; null counts bright specks, which a dark-dot lattice has few of, so for Edge2/090 it is a weak test |
| detector persistence field | 219 of 4,094 checked, 192 persistent (88%) | | |
Only 10% of rescue is testable by persistence (overlap strips), 396 of those 419 in Edge2/090. Crop sheets (raw | cleaned, red box on raw): the rescued items are 4-7 px dark specks in rows of the same dot type as their neighbours (oil film dot lattice), real in the raw image. **The cleaned image does not erase them**: median residual min vs 31 px median bg is -44 (raw) vs -49 (cleaned) for 350 rescued dots (ordinary tiny oil candidates: -32 vs -34), 0% of rescued dots have cleaned residual > -12. So the stated reason for raw_rescue is not what the pixels show; they were missed by the cleaned-image detector for other reasons (thresholds/claims). Reality as image features: yes. Defect relevance: unclear (same family as the lattice texture). With the raw-based reference, rescue explains 1,622 of the covered reference blobs (Edge2/090 1,435).
Effect on totals: rescue-only gain in recall 12-20 +0.6, 20-40 +1.4, 40+ +0.2 pts; contaminated >=40 uncovered 53 -> 12.

## 4. Flip null and persistence proxy (d3_flip.py, d3_persist.py; tables/flip_*, persist_*)
Own flip run: v4.1 code copy, polarities swapped, persistence off (null_flip/aoi_detect_flip.py), normal frames only.
| | v3.1 | v4 | v4.1 |
|---|---|---|---|
| Edge total | 0.76% (2,370/313,205) | 1.10% (3,464/315,239) | **1.07%** (3,552/331,388) |
| Edge1 / Edge2 | 1.50 / 0.34 | 2.25 / 0.42 | 2.20 / 0.41 |
| Edge1/000, /090 | 2.70, 0.88 | 3.16, 1.82 | 3.22, **1.72** |
| Edge2/000, /090 | 0.44, 0.24 | 0.56, 0.28 | 0.53, 0.30 |
| Edge1 path B | 0.99 | 3.45 | **1.61** |
| Edge1 path A gray_low | 1.66 | 2.16 | **2.67** |
| Edge2 path B | 0.09 | 0.46 | 0.33 |
| Side | 2.48% | 2.48% | 2.48% (identical) |
(Bot values 0.99/3.45/1.61, 0.88/1.82/1.72, 0.76/1.10/1.07, 2.16->2.67 reproduced.) Without raw_rescue Edge is 1.05% (Edge1 2.17, Edge2 0.40), so rescue barely moves the null (101 flip hits).
* Edge1 path A gray_low null rise localized: oil zone 2.16 -> 3.71% (Edge1/090 oil 2.00 -> **4.01%**, flip 309 -> 571; frames 18/20/17 hold +96/+60/+34 of it; Edge1/000 oil 2.72 -> 2.92); rough zone 10.3% (unchanged), normal 0.69% (unchanged). It is the price of cutting haze into cores: real count in that cell fell (15,485 -> 14,240) while flip count rose. v4's Edge1 path B problem (gray_low 6.57%) is mostly fixed (2.58%, still above v3.1 1.96%).
* Path C small cells got worse: Edge1 path C gray_mid 0.20 -> **5.67%** (43/759), Edge1 path C high 5.80 -> **32.3%** (52/161), Edge2 path C high 9.64 -> 12.8% (79/616), Edge2 path C gray_mid 0.89 -> 1.65% (62/3,752). Edge1 path C shrank 8,890 -> 3,486 rows while flip hits rose 106 -> 151; path C high is only ~0.8k rows in total (Edge1+Edge2 777) and it was already 112%/220% in v3.1.
* Flip candidates with area >=150 gray_low (haze-like): v3.1 108, v4 697, **v4.1 209**. >5000 px flip: 4 / 27 / 10 (oil 0 / 20 / 5).
* Persistence proxy (my a4.py, independent of the detector field; normal Edge frames): tested v3.1 31,145 / v4 30,842 / v4.1 32,735; match 79.3 / 79.5 / 79.6%; chance 11.7 / 11.6 / 13.1% (12.8% without rescue); structural fraction 76.5 / 76.8 / 76.6%. v4.1 candidates already present in v4 (n=32,183): 79.8 / 12.8 / 76.8%. **New non-rescue v4.1 candidates not inside any v4 box (n=133 tested only): 51.1% vs 19.8% chance, structural 39%.** haze_texture (n=218): 67.4 / 13.5 / 62.3%. By zone v4.1 w/o rescue: normal 85.7, oil 76.1 (chance 20.8), rough 73.1, uneven 76.4. Path B 64.6 (chance 23.9), C 63.5 w/o rescue. Folder v4 -> v4.1 (w/o rescue) match: Edge1/090 69.2 -> 68.4, Edge2/090 84.1 -> 85.1, Edge2/000 84.2 -> 84.4, Edge1/000 73.3 -> 73.1.
* Detector `persistence_checked>0` (normal Edge frames): **24,646 / 18,968 / 25,655** reproduced; by folder Edge1/090 4,285 / 2,272 / 4,436, Edge2/090 7,765 / 5,784 / 8,462: the vote fix restores the v3.1 level. Persistent share among checked 84.7 / 85.0 / 84.3%.

## 5. haze_texture flag and the >5000 px candidates (d4_haze_sheet.py; fig/sheet_haze_texture_random24.png; tables/big5000_v41.txt)
* haze_texture 2,538 (all normal frames, none on Side): zone oil 2,168 / rough 290 / normal 59 / uneven 21; tier gray_low 2,524; path A 2,413; area median 208 (p90 399); median |contrast| 12.3; solidity median 0.86; confidence median 0.29 after x0.6. Flags 61% of all Edge gray_low >=150 px candidates (2,538 of 4,167; 1,629 stay unflagged).
* Crop sheet (24 random): faint low-contrast haze patches, dim vignette strips next to the dark ROI edge, faint stipple; a minority (2-3 of 24) are boxes around a row of small dots. Flag is appropriate for most of them. The flag does not separate on persistence (88.3% of 196 checked flagged vs 88.5% of unflagged gray_low >=150): haze is physically persistent, so the flag is a texture/priority label, not a "not real" label. Flip hits among flagged-type candidates 187 (7.4% of flagged real count) vs 1.07% overall: the flag concentrates the null risk about 7x.
* **>5000 px Edge candidates (normal frames), v3.1 / v4 / v4.1: 49 / 158 / 51; oil 0 / 112 / 6** (reproduced). v4.1 ones: median |contrast| 118.7, solidity 0.89, 38 of 51 tier high, 50 stain — i.e. the genuine big stains of v3.1 again. Of the 108 v4 new-big haze regions, 106 are no longer >5000 px (largest piece <=150 px in 10, <=600 px in 91). Cost: each such region now becomes ~100 compact pieces (median 99 candidates per v4 box, median largest piece 275 px), which is where the +26k path A rows and the Edge1/090 oil null rise come from.

## 6. merged_n>=3 regression 85 -> 173 (tables/big5000_v41.txt; inline)
Confirmed 168 / 85 / 173; **not a quality regression, v4.1 returns to the v3.1 level**. All 173 are consolidation=regrown (clusters merging >=3 neighbouring fragments), 130 oil / 38 rough / 5 uneven, path A 136, median area 175 px (v4's: 116), 0 on anomaly frames, class stain 94 / particle 70 / scratch 9. Every one has a v4 candidate at its location (74 of them were already merged_n>=3 in v4; 84 were a single piece, 15 merged_n=2); 95 of the 173 sat in a v4 `split` piece; their v4.1 area is on median 0.60x the v4 counterpart. Mechanism: v4's oversized haze blobs were being split (split 28,716 -> 13,292 in v4.1) so fewer fragments existed to be regrown; with compact cores the regrow step re-merges fragment clusters (as in v3.1, 168). Downstream: filter rule N_REGROWN_OVERMERGE sends merged_n>=3 to nuisance/texture, so these 173 (17 tier high) go to texture; 0.04% of rows, harmless but worth knowing.

## 7. Left-edge 185, c0736, big stain (tables/leftedge_v41.*, c0736_stain27_v41.txt)
* Left-edge 185 Edge1/090 frames 18-23 (my blob table, strict cover): **175 / 181 / 180** (frames 18-23: v3.1 24/25/16/31/48/29; v4 24/26/20/31/48/32; v4.1 24/25/20/31/48/32). Lost vs v4: frame 19 (111,414), 21% in ROI, peak 97: in v4 covered only by a 184 px gray_low piece (contrast -11.5), in v4.1 the haze-split removed that piece. Still missed since v3.1/v4: (112,1417), (113,1552) on frame 19; (122,1346), (113,1596) on frame 22. 0 newly gained vs v4. Candidate area / blob area for ROI-touching blobs: median 0.25 (unchanged).
* c0736 (Side_000_05, centroid (294.1,1731.7), 44 px, contrast 20.3, overlap 0.295, keep reason high_contrast): present and identical in v3.1 (c0737), v4, v4.1 (Side unchanged).
* Edge2/090/27 big stain: still one candidate, 99,404 px, contrast -148.3, normal zone, class stain, tier high, conf 1.0 (v4.1 id c1848; v4 c1574; v3.1 c1408 99,397 px).

## Remaining issues
1. **raw_rescue is a risk for downstream precision**: 4,094 tiny (median 7 px) path C dots, 88% Edge2/090, 71% in two frames (25, 26) of an oil dot lattice. They are real raw features but lattice-like and the weakest third (|c|<=30) is only ~half-real; Edge1/090 rescue flip null is 10%. Filter should down-rank or review them, and the bot's explanation ("cleaned erases real dots") should be corrected: pixels in cleaned show them at equal contrast.
2. Raw-based reference circularity: recall gains from rescue (oil 12-20 +0.8, Edge2/090 20-40 +3.8) cannot be fully independently verified; only 10% of rescue is testable by persistence.
3. Edge1/090 still the weakest: 12-20 68.0%, oil 12-16 49.7%, no gain from v4.1; its null (oil path A gray_low 4.0%) went up. Edge uneven 12-16 ~39% (Edge2/090 19%) unchanged; Edge1 rough zone null 10.3%.
4. Strong contaminated misses: v4.1's own detector (without rescue) 53 vs v4 37; rescue masks that, 12 left (9 Edge1/090 oil).
5. Haze split cost: ~26k more path A rows; each former haze blob becomes ~100 cores; null in that region rose. Path C small-cell nulls rose (Edge1 gray_mid 0.2 -> 5.7%, high 5.8 -> 32%).
6. One strong left-edge blob lost vs v4 (frame 19 (111,414)); 4 left-edge blobs missed since v3.1.
7. Detection >40 contaminated and the Side population are unchanged from v4 (Side flip null 2.48%, Side/090 4.06%).
