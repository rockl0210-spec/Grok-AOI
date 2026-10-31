# AOI false-positive filtering + fine classification (rerun from scratch)

Snapshot time: 2026-10-30T17:48:05+08:00 (Asia/Taipei). Input copies: `_input_snapshot_v31/`.

## Input versions
- Detection: **v3.1** (`<OUT>/detect/`, 376,277 candidates).
- Nuisance flags / masks: **preprocess v2 only** (`<PRE_V2>`). preprocess v3 was NOT used.
  Caveat: `cleaned_image` in the detect JSON and `preprocess_root` in the manifest still point at v3 paths; they were ignored.
- Raw images: `<DB>`. Grayscale single-channel features only. All coordinates are original-image pixels.

## Outputs
Per angle `<edge|side>/<Src>_<angle>/`: decisions.csv/json, defects.csv/json, summary.json, overlay_*.png.
Top level: summary.json, this README, DONE (written after verifying every v3.1 ID appears exactly once: 376,277 unique IDs).
Overlay: real boxes green=particle, orange=stain, blue=scratch (thick + "?" = review_required); nuisance drawn as corner ticks (max 40 per type).

## Classes
real: particle (落塵), stain (髒污), scratch (刮傷). nuisance: water_mark, oil_film, edge_artifact, reflection, texture, noise.
Where truly uncertain: real with review_required=true. Reflective/uneven/rough regions are judged region-by-region with region-adapted thresholds, never excluded wholesale.
Zone precedence (from detector): rough > oil > water > specular > uneven > normal. cnr=|contrast|/local sigma.

## Rule order
1. Very-strong bypass: |c|>=60, area>=15, cnr>=10, merged_n<3 (and nb_rank>=0.85 if rough) -> real.
2. Regrown over-merge: merged_n>=3 -> nuisance texture (review if |c|>=25).
3. Edge artifact: dist to ROI edge<=6 and (elongation>=3, or thickness<=2.5 and major>=12); or straddler overlap<0.5 and elongation>=2.5.
4. Side reflection (glare bands from raw row-profile excess): bbox within +-15 px of band, bh<=25, width>=1.8*height, elong>=2; or within 80 px with elong>=4 and width>=2.5*height.
5. Fixed pattern: same pixel position recurring in >=3 frames and area<40 -> texture.
6. Region rules (tier T1/T2 is a prior only):
   - Edge normal: T1 prior + cnr>=3.5; T2 + cnr>=6, |c|>=10, area>=6 (review).
   - Edge uneven: T1 |c|>=12, cnr>=5, area>=6; T2 cnr>=7, |c|>=10 (review); shading nuisance if area>=1500 and |c|<25.
   - Side normal: T1 + cnr>=4; T2 + cnr>=6, |c|>=8, area>=8 (review).
   - Side uneven: T1 cnr>=5 & nb_rank>=0.5; T2 cnr>=8 & nb_rank>=0.5.
   - Rough: strong dot |c|>=18, cnr>=6, nb_rank>=0.85, |c|>=1.3*nb_medc, area>=8; marginal (review) |c|>=16, cnr>=5, nb_rank>=0.75, area>=6, nb_similar/nb50<=0.5; stain area>=400,|c|>=15,cnr>=4; scratch elong>=4, major>=40, |c|>=15, cnr>=5.
   - Specular keep: |c|>=30, elong<2.5, area>=8, cnr>=8 (always review).
   - Dense texture: nb_rank<0.5 with nb50>=45 and cnr<7 (Edge); nb50>=25 and cnr<8 (Side).
   - Water flag |c|<25; oil flag |c|<30 and area>=100 (nuisance only if feature-weak).
7. Anomaly frames (Edge 00/45, Side 00/15): never real alone; need a match (p_ctrl<0.5) in a normal frame. Strong but unverified -> nuisance/noise N_ANOM_UNVERIFIED with review=True.
8. High-contrast straddlers: capped at T2, always review if kept real.

## Persistence limits
Burst frames are tiles (Edge step ~-1940 px y, ~116 px overlap; Side step ~2251 px x, ~213 px overlap), not repeated views, so cross-frame persistence is testable only in overlap strips (~8-14% of candidates). Registration: NCC refined by candidate-offset voting, trusted only; chance control from 4 random probes. Decisions therefore rest mainly on region-adapted features and local distinctiveness (nb50, nb_medc, nb_rank, nb_similar).
Defects merge across frames via overlap-strip matches into unique physical defects (defect_id).

## Known gaps
- water_mark (1) and oil_film (0) are nearly empty: preprocess v2 flags rarely fired on these; large faint appearance-only regions are kept as stain with review.
- Counts are large because the tier prior is dense with small dark specks; the no-review real subset is the high-confidence set.
- Side real candidates are dense along the ROI top/bottom rim; worth a manual look.
