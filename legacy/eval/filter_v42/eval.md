# filter v4.2 快速驗證（2026-10-30，Asia/Taipei）
方法同 filter_v41 eval（腳本 h1–h4；tables/ 內為彙總表，無逐候選檔）。**結論：無阻擋問題，可進最終交付。**

## 1. 聲明重現（全部 ✓）
| 聲明 | 重現 |
|---|---|
| 394,375 ID 各出現一次 | ✓（唯一 394,375；與 v4.1 ID 集合相同；無 NaN；review_level>0 ⇔ review_required） |
| Side real 25,730（000 3,521／090 11,569／180 5,068／270 5,572） | ✓ 完全一致（v4.1 18,516） |
| L2 6,986（Side 9,373→2,612、Edge 不變） | ✓（Edge 4,374＝4,374）；real L2 4,916、nuisance L2 2,070；L1 138,104；real 非 review 112,111 |
| Side 參考保留（含 review）67.6/89.4/99.1；非 review 35.6/79.1/94.5 | ✓ 完全一致（覆蓋率 86.2/99.3/100） |
| plan D 轉換 541（a 336、b 205），全 nuisance＋L2 | ✓：N_SIDE_T2_UNCONFIRMED 541，全 noise、全 L2、無 anomaly 幀；a＝已測且未確認 336、b＝不可測 205；各資料夾 95/172/127/147 |
| 還原 real 7,214（L1 6,590、L2 624） | ✓；v4.1 的 7,928＝7,214＋541＋173（anomaly 幀候選回復為 N_ANOM_UNVERIFIED，L1 171/L2 2）；除此之外 Side 無其他改動 |
| c0736 real L1；c1848 real stain L2 | ✓ |
| Edge 與 v4.1 相同，除 68 筆 | ✓ 333,848 筆 Edge：decision／fine_class／reason_code／tier 全同；僅 68 筆 review_level 0→1（Edge1_000 38、Edge1_090 30；Edge1 path C、tier T1、|c| 60–128；stain 37／particle 24／scratch 7） |
| 唯一缺陷 196,282 | ✓（particle 183,525／stain 9,804／scratch 2,953） |

Edge 因決策不變，保留率與 v4.1 完全一致（contam 38.6/78.3/96.0；非 review 7.2/20.5/45.6；Edge1_090 42.0/77.4/96.3）。Side 參考保留：Side_000 80.8/92.3/98.5、Side_090 63.7/87.4/99.0、Side_180 84.1/96.9/99.1、Side_270 68.5/91.9/99.8（12-20/20-40/40+）。

## 2. flip null（5,022 flip，與 v4.1 eval 同法）
| | flip 通過 | 非 review 通過 | 假 real／real | Edge | Side |
|---|---|---|---|---|---|
| v4.1 | 1,422（28.32%） | 866（17.24%） | 0.69% | 0.59% | 1.79% |
| v4.2（無 persistence 的 union run） | 1,532（30.51%） | 849（16.91%） | 0.72% | 0.59% | 1.64% |
- union run 的 flip 沒有 persistence，plan D 的 (a)(b) 兩條都不會觸發，因此這是 T2 部分的上界（等同「關掉 Side T2 轉換」＝v4.1 的 noSideT2 變體）；真實 v4.2 因 541 筆被轉成 nuisance 會略低。這是已知限制，不是退步：v4.1 的整批降級本來就只省約 110 個 flip。
- 分層估計（kind×zone×|c| bin×tier_prior，同一組 flip）：假 real 估計 v4.1 1,794（0.87%）→ v4.2 2,019（0.95%）；增加全在 Side（336→561，+225），Edge 1,458 不變。非 review 1,069→1,068（0.95%）。換言之 v4.2 以約 +225 個估計假 real（約 +0.08 個百分點）換回 7,214 筆 real 與 Side 參考 40+ 99.1%（v4.1 95.9%）。
  （此估計用 tier_prior，與 v4.1 eval 的 1,848 因分層法略不同；兩版本在此以同法重算。）

## 3. L2 分診提升
- 實際 L2 6,986＝real 4,916＋nuisance 2,070。nuisance L2：N_CONTAM_TEXTURE 668、N_OIL_FILM 661、N_SIDE_T2_UNCONFIRMED 541、N_EDGE_ARTIFACT 95…；real L2：R_CONTAM_MID 2,055、R_SIDE_NORMAL_T1 742、R_CONTAM_STRONG 503、R_SIDE_NORMAL_T2 399、R_SIDE_UNEVEN_T2 295…。L2 不再被 T2 副產物主導（v4.1 為 58%，v4.2 為 7.7%）。
- flip 假比例（real 內）：**L2 6.49%（181/2,790）vs L1 0.49% vs 非 review 0.78%**＝約 13×（L1）／8×（非 review）。Edge L2 6.48%、Side L2 6.49%（v4.1 8.17%）。Side L1 1.61%、Side 非 review 1.14%。
- L2 flip 主因：R_CONTAM_SPARSE 53、R_SIDE_UNEVEN_T2 33、R_SIDE_NORMAL_T1 29、R_FAINT_RELAXED 25、R_SIDE_UNEVEN_T1 21。L1 仍不比非 review 更銳利（0.49% vs 0.78%），與 v4.1 結論相同。

## 4. same_object_group 合理性
- 一致性：255,142 組；n 與 n_real 與實際成員數完全相符；無跨幀、無跨資料夾組；**無跨極性組**（所有成員 contrast 同號）；面積 >250,000 px² 的 16 個框都不入群。多成員組 39,896（179,129 候選；real 成員 109,021）；含 real 的組 135,352、其中 19,475 組含 >1 real。
- 規模分布（n）：1：215,246；2：21,977；3–5：11,999；6–10：3,463；11–25：1,756；26–50：434；51–100：197；>100：70；最大 241。
- **超大組＝油紋鏈結**：n≥50 有 272 組（24,328 成員），其中 252 組在 oil zone、16 rough、4 normal；資料夾 Edge2_090 167／Edge1_090 52／Edge2_000 40／Edge1_000 12／Side_090 1。前 15 大全是 oil、全負極性、成員 real 約 50–90%（例如 Edge2_090_21_g02600：241 成員、215 real；bbox 270×308，填充率 0.15）。這些組是纖維/紋路上的碎點串成一串，**不代表一個物體**；有 44 組外接框 >600 px（26 組 >1000 px，如 Edge1_000_27_g00000：224 成員，160×1,244）。
- 混類：多成員組中 4,787 組含 ≥2 種 real 子類（particle/stain/scratch），單一組內 real 與 nuisance 混合 21,004 組（全 real 9,689、全 nuisance 9,203）。
- 判讀：same_object_n 在 normal/clean 區可用來合併碎片（n 小）；在 oil/rough 區 n≥20（1,062 組，945 在 oil）宜視為「油紋/纖維叢集」而非單一缺陷，報告註明、不用於計缺陷數。非阻擋。

## 5. 其他
- 97,803／213,680 real 候選同組內另有 real，說明碎片化仍普遍（用於「一物多框」議題；缺陷 defect_id 合併機制不變）。
- 限制同 v4.1 eval：無 ground truth；flip null 在油紋區為下限；union run 無 persistence。
