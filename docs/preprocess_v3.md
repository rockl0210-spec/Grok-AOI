# aoi_preprocess — 真實 AOI 資料（Edge1/Edge2/Side）前處理 + ROI 遮罩

> 本檔分三部分：**★ v2＝現行真實資料流程（`config_v2.yaml`）**；第一部分＝v1 真實資料多模式流程（`config.yaml`，保留）；第二部分＝原本的合成面板／晶圓流程（`config_panel.yaml`，供 demo 與 `tests/test_smoke.py`）。

流程：**載入 → 結構偵測＋ROI（依模式）→ 照明校正 → 降噪 → 夾具類別 → 輸出**。模式（`edge` / `side`）由資料夾名稱決定（`config.yaml: mode_rules`：路徑中含 `edge` → edge、含 `side` → side），或用 `--mode edge|side` 強制。所有參數都在 `config.yaml`（`common` + `modes.edge` / `modes.side`，模式設定會覆蓋 common）。

---

# ★ v2（現行真實資料流程，`config_v2.yaml`，`pipeline: v2`）

> v1（`config.yaml`，下面「第一部分」）保留不動；v1 輸出已搬到 `<PRE>_v1`。v2 輸出在 `<PRE>/<Edge1|Edge2|Side>/<角度>/`。
> v2 的程式：`aoi_preprocess/{pipeline_v2,roi_v2,nuisance,noise_map,registration}.py`；工具：`tools/{make_overview_v2,summarize_v2,spot_survival_v2,show_dev}.py`；測試：`tests/test_v2.py`。

## v2 規則（Willson）

**只有「真實物理邊界、外部背景、夾具」被排除（mask=0）。** 反光、不均勻／偏暗照明、粗糙紋理帶、水痕、油膜一律留在 ROI 內、照常檢測，只用 `NN_nuisance.png` **標記**，不排除、不擦除。
Edge 的 ROI 與 v1 的差別：不再排除「粗糙帶」「暗區」「角落以外的整塊」；Side 的 ROI 不再排除斜角（bevel）、暗區、反光、端帽。

## 執行

```bash
# 取樣（開發）
.venv/bin/python -m aoi_preprocess run --root <DB> --config config_v2.yaml --out dev_real_v2 \
    --input <DB>/Edge1/090 --frames 17,18,19
# 整個角度資料夾（>24 檔要 --all；輸出在 <OUT> 要 --allow-protected-out）
.venv/bin/python -m aoi_preprocess run --root <DB> --config config_v2.yaml \
    --out <PRE> --input <DB>/Edge1/090 --all --allow-protected-out
# 圖：raw | cleaned | mask | nuisance 疊色 | noise（加 --compare 為 raw | cleaned | ROI+nuisance 疊色）
.venv/bin/python tools/make_overview_v2.py --raw <DB>/Edge1/090 --out <v2輸出>/Edge1/090 --frames 17,22 --png fig.png [--compare]
# 統計（每個資料夾 ROI %、各 bit 面積 %、對位位移、執行時間；--v1 對照 v1 ROI）
.venv/bin/python tools/summarize_v2.py <PRE> --v1 <PRE>_v1 --json summary.json
# 3 px 小點存活（正常區 vs 各 nuisance 區）
.venv/bin/python tools/spot_survival_v2.py Edge1/090:25,40 Side/090:07 ...
.venv/bin/python tests/test_v2.py          # v2 單元測試（合成資料）；tests/test_real_modes.py、tests/test_smoke.py 為 v1／合成流程測試
```

## 每張圖的輸出（與原圖同尺寸，不裁切、不縮放，座標＝原始像素座標）

| 檔案 | 格式 | 內容 |
|---|---|---|
| `NN_mask.png` | 8-bit，0/255 | **255＝要檢測的 ROI**；0＝外部背景／物理邊界（邊緣線＋`edge_margin_px` 小邊界）／夾具／Side 上下黑色背景／Side 端框（零件端點以外的霧／端帽背景） |
| `NN_nuisance.png` | 8-bit bit-flags | 只在 ROI 內有意義，ROI 外為 0；各 bit 可重疊（見下表） |
| `NN_noise.png` | 16-bit | 局部穩健 σ（MAD）× 100（8-bit 灰階單位）；`σ = 值 / 100`；ROI 外為 0（比例寫在每個資料夾的 `report.json: noise_map`） |
| `NN_cleaned.png` | 8-bit | 背景／照明校正後（Edge：序列背景×自身增益，除法；Side：自身背景相減；再做 σ=60 px 的局部穩健歸一化）＋**保守 NLM**（patch 3、distance 4、h=0.5σ）；不做會抹掉 ≥3 px 小點的平滑 |

**nuisance bit-flags**（`report.json: nuisance_bit_encoding`）

| bit | 值 | 意義 | 方法（程式 `nuisance.py`） |
|---|---|---|---|
| 0 | 1 | 反光／飽和 | 灰階 ≥250（膨脹 3 px）；Side 另加大面積亮區（≥200，opening 後，外擴 40 px 的光暈） |
| 1 | 2 | 不均勻照明／偏暗或偏亮區 | 平滑後的亮度圖相對參考（Edge 75 百分位；Side 中位數）過低／過高，並保留與邊緣連通者 |
| 2 | 4 | 粗糙紋理帶（Edge） | v1 的 `edge_roughness`（高頻能量的窗口統計＋遲滯）＋ 10 px 外擴 |
| 3 | 8 | **疑似**水痕 | 低頻（gauss 12 px − gauss 80 px）、低梯度、中等面積、輪廓偏模糊的區域 |
| 4 | 16 | **疑似**油膜 | 同上但面積更大（≥150k px）或輪廓更不銳利 |

水痕／油膜只是**低階區域標記**（`疑似`），沒有真值可驗證，從不擦除、不排除；先抑制 100 px 以下的緊緻缺陷（避免缺陷周圍被當成水痕）、避開飽和／粗糙／不均勻區、避開細長條（厚度 <120 px，例如 Side 上下邊的光暈）。

## 每個角度資料夾的 `report.json`

`nuisance_bit_encoding`、`noise_map`（scale=100）、`mask_encoding`、`abnormal_frames`（Edge 00/45、Side 00/15，附 `abnormal_frames_detail`：flags、ROI %、各 bit %、統計）、`frames.<NN>`（`roi_percent`、`class_percent`、`nuisance_percent_of_roi`／`_of_frame`、`noise_sigma_levels8`、`stats_roi_levels8`、`runtime_s`、結構／照明參數）、`registration`（僅 Edge）與 `parameters`（完整參數）。

**對位（Edge，`registration.py`）**：對相鄰兩張算 `dx, dy, method, confidence, trusted`（`dx,dy`＝第 k+1 張內容相對第 k 張的位移，+x 向右、+y 向下，已用合成位移驗證正負號）。方法 `phase_correlation`（1/4 尺度高通＋Hann 視窗，response ≥0.2 才候選）；**交叉驗證**：若邊緣線可用（信心 ≥0.5），phase correlation 在邊緣法線方向的分量必須與邊緣線位移相差 ≤20 px，否則視為假峰值、記在 `phase_correlation_rejected`，改用 `edge_line_normal_only`（只量得垂直邊緣的分量，另一分量記 0＝未知，信心＝兩張邊緣線 inlier 比例）；無邊緣線可驗證時（角落張／線斷裂）phase correlation 僅在 response ≥0.5 且位移 ≤`max_line_shift_px`(300 px) 才標 `trusted`（備註 `not cross-checked`；超過 300 px 視為不合理而不信任，但數值仍保留供檢視）；兩者皆無則 `none`。相鄰張內容幾乎不重疊時，phase correlation 通常信心很低，這是資料特性。

## 理由與取捨

- **斜角（bevel）保留在 ROI 內**：它在零件表面上、屬於可檢測面（Side 的列範圍取資料夾共識 `[上緣+8, 下緣−8]`，只切掉上下黑色背景與 8 px 邊界）。
- **Side 端框（00／15）**：零件端點以外是霧／端帽背景，不是零件表面，故從端點（欄方向最外側的陡峭灰階梯度）截掉；端點以內的反光／偏暗全部保留並標 bit0／bit1。
- **Edge 邊界**：基板（距外部背景 >30 px）；不再有影像邊框邊界。碰到影像邊但不是真邊緣的暗物（例如 Edge2/090/27 的黑色異物）**不是外部背景**：外部背景必須橫跨影像寬／高的 ≥50%（`bg_min_extent_frac`），其餘暗塊填回基板、留在 ROI。
- **夾具**：資料中沒有確認的夾具；自動夾具規則（`fixture_auto`）在 Edge 會把異物誤判成夾具，故 v2 預設關閉，改用手動矩形 `roi.fixture_rects: [[x0,y0,x1,y1], …]`（原始像素座標）。
- **ROI 外的 cleaned 值沒有意義**（背景底限除法的結果），請以 mask 為準，不要在 mask=0 的區域做判讀。
- 對 ROI 起始處（距邊緣 30 px）cleaned 值約為平台值的 +10%／−5% 內。

## 3 px 小點存活（`tools/spot_survival_v2.py`，結果在 `dev_real_v2/spot_survival_v2.json`）

把 3 px 圓點（模糊 0.5 px，Edge ±10/20/40、Side +6/12/24 灰階）注入原圖，整條 v2 流程重跑（照明→局部歸一化→水痕/油膜→NLM），量 cleaned 的峰值反應 ÷（注入振幅×局部增益）＝保留率，並以雜訊圖算 CNR。正常區與各 nuisance 區（不均勻、粗糙帶、油膜、反光）的保留率中位數在振幅 ≥20 時 ≥0.89（Edge）／≥0.95（Side），沒有被局部歸一化或 NLM 抹平；粗糙帶／不均勻區的 CNR 較低是因為本身雜訊高（物理），不是演算法壓平。

## 已知限制／疑慮（v2）

1. 水痕／油膜（bit3/4）沒有真值，只是保守的低階標記，樣本上很少觸發；需要整合端提供標註或判斷範例才能調。
2. Side bit1（偏暗／不均勻）的門檻（`dim_frac 0.65`、`hot_frac 1.6`）為經驗值；端框附近常佔 35–52 %。
3. 沒有夾具標註，`fixture_rects` 預設為空。
4. Edge 相鄰張位移：只量得垂直邊緣的分量，平行邊緣的位移在內容不重疊時無法估計。
5. Edge 粗糙帶（bit2）在 Edge1/090 17–27、Edge2/090 20–27 可佔 ROI 的 15–57 %；cleaned 在該帶內只做照明校正＋保守 NLM，不做紋理抑制（會傷 3 px 點），因此只標記。


---

# 第一部分：v1 真實資料流程（`config.yaml`，保留）

## 資料事實（來自整合端，已用樣本驗證）
- `<DB>`：248 張 8-bit 灰階 PNG，2464×2056。`Edge1|Edge2/{000,090}` 各 46 張（00–45），`Side/{000,090,180,270}` 各 16 張（00–15）。沒有 ground truth、flat/dark、基板尺寸或 um/px。
- **Edge**：亮場基板邊緣特寫（一側是邊緣外的暗背景，緊鄰一條粗糙紋理帶，其內側是平滑表面）。連續掃描，相鄰張相關性 ≈ 0.97；00 / 45 為角落／暈影不完整畫面。
- **Side**：暗場側面條帶：上下為黑背景，中間條帶含上、下倒角亮帶與鏡面反射亮帶。相鄰張相關性低（0.1–0.4）；00 / 15 是條帶末端（端蓋）畫面，背景變亮。

## 執行（安全防護）
```bash
cd <research-tree>/aoi_preprocess
# 取樣（開發驗證用）：--frames 指定張號；輸出鏡像 <root> 下的路徑
.venv/bin/python -m aoi_preprocess run --input <DB>/Edge1/000 --root <DB> --config config.yaml \
    --out dev_real --frames 00,01,22,45 -v
# 整個資料夾（>24 檔必須明確加 --all）；每個角度資料夾各跑一次
.venv/bin/python -m aoi_preprocess run --input <DB>/Side/000 --root <DB> --config config.yaml --out OUT --all
# 其他：--mode edge|side 強制模式；--debug-dir DIR 把除錯檔（corrected / classes / overlay_preview）放別處
.venv/bin/python tests/test_real_modes.py ; .venv/bin/python tests/test_smoke.py
```
防護：輸出目錄不得在輸入樹內；`config.yaml: safety.protected_out_roots`（預設 `<OUT>`）需加 `--allow-protected-out` 才能寫入（正式全量執行才使用）；超過 `safety.max_files_without_flag`（24）個檔案時必須給 `--frames` 或 `--all`。輸入一律唯讀。

## 輸出
`<out>/<Edge1|Edge2|Side>/<角度>/NN_cleaned.png`（單通道、輸入位元深度 = 8-bit、尺寸與原圖相同、不裁切不縮放；ROI 外的像素保留，由 mask 決定檢測範圍）、`NN_mask.png`（單通道 8-bit，**ROI = 255 / 排除 = 0**）、`report.json`（每個角度資料夾一份）。除錯檔（`NN_corrected.png`、`NN_classes.png` = 類別圖、`NN_overlay_preview.png` = 彩色疊圖 **僅供預覽**）只放在 `_debug/`（或 `--debug-dir`），不在正式輸出夾內。

類別碼（`NN_classes.png`）：0 基板外／背景、**1 有效 ROI**、2 邊緣帶、3 暗區（陰影／暈影）、4 粗糙帶、5 夾具、6 鏡面反射、7 倒角帶、8 端蓋區、9 影像邊框。

`report.json`：`abnormal_frames`（含 `abnormal_frames_detail`：原因、是否由統計偵測、是否在 config 的 `expected_frames` 名單）、`sequence_background`、每張的 `flags / notes / valid_roi_percent / class_percent / structure（偵測到的邊線、列邊界、信心…）/ illumination / noise_sigma_levels8 / stats / runtime_s`，以及整份 `parameters`。

## Edge 模式 ROI（`modes.edge.roi`，`roi_modes.edge_*`）
1. **基板／背景分割**（`structure`）：高斯 σ3 → 閾值 = 背景 + 0.25·(平台 − 背景)（平台 = 第 90 百分位、背景 = 第 1 百分位；差 < 40 階視為找不到邊）→ 15 px 開閉運算 → 最大連通塊 → 填洞。
2. **邊線**：邊界點做穩健 TLS 直線擬合（`line_inlier_frac` = 落在線上的邊界點比例，只作 notes）。
3. **邊緣帶**：距邊線 `edge_margin_px = 30` px 排除（異常畫面再加 `abnormal_extra_margin_px = 30`）。
4. **粗糙帶**（`rough_band`）：紋理密度 T = 81 px 方窗平均(|影像 − 高斯 σ6| > 6 階)；種子 T > 0.10 且距邊 ≤ 200 px，遲滯成長 T > 0.06，閉運算 61 px，外擴 30 px。**粗糙帶排除**。
5. **暗區**（陰影／暈影）：緩變亮度圖 L（1/4 縮圖 median5 + 高斯 σ16 px）< 0.90×基板 75 百分位 → 排除；但**孤立暗塊**（不在邊緣 60 px 內且面積 < 60000 px²，例如大的暗點、纖維環）釋放為可檢區，避免把真缺陷當暗區吃掉。
6. **影像邊框** 6 px。夾具自動偵測（`fixture_auto`，亮 > 1.5×／暗 < 0.45× 中位數且面積 ≥ 20000 px²）目前樣本中未偵測到夾具。
7. **異常畫面（自動，依統計）**：`edge_not_found`、`large_background_area`（背景面積 > 12%）、`corner_frame`（背景同時碰到 ≥ 2 個影像邊）。`large_dim_zone`、`edge_not_straight` 只寫入 notes，不判異常（波浪狀粗糙邊緣的正常畫面也會觸發）。

## Side 模式 ROI（`modes.side.roi`，`roi_modes.side_*`）
1. **條帶上下邊界**：32 個欄區塊各取中位數列剖面 → 平滑 → 階梯梯度；在固定機構的先驗窗內（上緣 [0.11, 0.18]·H、下緣 [0.81, 0.88]·H）投票，並回報信心。信心不足時退回 `fallback_rows_frac`（[0.141, 0.846]，旗標 `rows_low_confidence`）。
2. **倒角帶**：邊界內側固定偏移（上 115 px、下 125 px）標為類別 7 並排除（可在 `strip` 調整，見下方疑慮）。
3. **鏡面反射**：8-bit ≥ 200 → 開運算 9 px → 面積 ≥ 3000 px² 的連通塊 → 外擴 40 px，**獨立類別 6**（不併入一般無效區）。
4. **條帶末端（端蓋）畫面**：鏡面帶若佔條帶 ≥ 60% 高度且寬 ≤ 25% 影像寬 → 判 `strip_end_frame`；帶的兩側中位數較亮者為端蓋區（類別 8，整塊排除），另一側才是可檢區。
5. 影像邊框 6 px。

## 照明校正（`illumination.source`：self / sequence / hybrid）
- 全部由影像本身估計（沒有 flat/dark）。**self**：基板 ROI 遮罩加權的大核高斯背景（σ150 px，離群剔除），Edge 用 `divide`、Side 用 `subtract`（`target_level 0.08`，暗場資料不適合除法）。**sequence**：同一角度資料夾**最多 9 張等距、排除異常畫面**的逐像素中位數（再 σ4 平滑），不會讀整個資料夾。**hybrid**：sequence 中位數 × 以 self 估計的殘差增益（σ150）——掃描平移很小，序列中位數能抓到固定的光學／照明圖樣，殘差增益吸收平移造成的差異。
- **量測**（`tools/illum_compare.py`，Edge 四個資料夾各 4 張非異常畫面共 16 張；ROI 內校正後影像經 σ40 平滑的 std，佔 ROI 中位數的 %）：原圖平均 1.63%；self 0.80%；sequence 1.28%；**hybrid 0.70%**；中位數 self 0.34% → **hybrid 0.22%**（典型畫面 hybrid 0.06–0.33% 優於 self 0.25–0.46%）。高頻雜訊 MAD 幾乎不變（1.255% → 1.254%），代表沒有放大雜訊或抹掉細節。剩餘高值（3–6%）出自畫面本身的大型暗斑（缺陷，不是照明）。→ **Edge 預設 hybrid；異常畫面（00/45）改用 self；Side 只用 self（各張相關性低，序列中位數無意義）**。

## 降噪與 3 px 小點存活（`tools/spot_survival.py`）
在真實畫面的有效 ROI 內注入合成圓點（直徑 2/3/4/5 px、光學模糊 0.5 px、對比 ±10/20/40 階（Edge）或 +6/12/24 階（Side）、重新量化成 8-bit），經過完整的照明校正＋降噪後量測：對比保留率 =（5×5 視窗極值 − 周圍 8–13 px 環中位數）降噪後／降噪前；CNR = 對比／環標準差。3 px 點的結果（Edge 暗點，A = 10 / 20 / 40）：

| 降噪 | 保留率 | CNR | 備註 |
|---|---|---|---|
| none | 1.00 | 5.0 / 8.8 / 17.5 | 雜訊 σ 2.28 階 |
| median 3×3 | 0.59 / 0.71 / 0.76 | 6.9 / 14.6 / 31.2 | 小點被削弱，不採用 |
| median 5×5 | 0.24 / 0.15 / 0.11 | — | 小點被抹掉 |
| bilateral (輕) | 0.62 / 0.78 / 0.87 | 9.6 / 21.1 / 47.5 | |
| **NLM h=0.5σ** | **0.89 / 0.99 / 1.00** | **15.7 / 30.6 / 61.2** | 雜訊 σ 2.28 → 0.51；**預設** |
| NLM h=0.8σ | 0.74 / 0.97 / 1.00 | 18.9 / 42.8 / 87.6 | 對最弱點保留不足 |

Side 亮點 3 px（A = 6 / 12 / 24）：NLM 0.5 → 保留 0.85 / 0.97 / 1.00、CNR 11.9 / 25.4 / 48.6（none：7.4 / 14.3 / 27.5；median3 只有 0.70 / 0.76 / 0.79）。2 px 點在極弱對比（A=10 / 6）仍會被削弱（NLM 0.62 / 0.72），≥ 3 px 且對比 ≥ 約 2× 雜訊 σ 的缺陷基本不受影響。完整數字在 `dev_real/_spot_survival_{edge,side}.{txt,json}`。→ **兩個模式預設 `nlm`，`h_factor 0.5`**（`patch 3`、`distance 4`；約 2 s／張）。

## 已知限制／疑慮
- Edge 粗糙帶與暗區是依紋理密度／亮度的**判斷式排除**；Edge1/090、Edge2/090 的中段畫面（約 #17–26）粗糙帶很深（> 1700 px），有效 ROI 只剩 38–60%，其中可能含真實缺陷（刮痕／殘膠），需要現場確認是否要縮減排除。
- 角落畫面（00/45）的陰影區被排除；可能有一部分可回收。
- Side 倒角帶偏移為固定值（115／125 px），未必代表真實幾何；偏亮的條帶末端畫面使列邊界信心較低（00 退回先驗列）；少數畫面（如 Side 000/03）下緣被偵測在較內側（約 −60 px，較保守）。
- Side 自估背景為線性相減，遇到大面積漫射亮區的陡峭邊緣（如 Side/180/01）會在邊緣附近留下暗帶假影。
- 沒有 um/px 與夾具座標，所以所有尺寸都是 px，夾具只靠自動偵測（目前樣本未觸發）。

---

# 第二部分：原合成面板／晶圓流程（`config_panel.yaml`）

流程：**載入 → 照明校正 → 降噪 → ROI 遮罩 → 輸出**。每個階段是獨立函式，所有可調參數都在一個 `config_panel.yaml`。

## I/O 約定（重要）

| 項目 | 規格 |
|---|---|
| **主要輸入** | **單通道灰階 PNG**（8-bit 或 16-bit） |
| 備援輸入 | BMP／TIFF（含 float TIFF）／彩色影像 → 會轉成單通道灰階並在 JSON 的 `input.channels_note` 註記（`color_to_gray` 可選 luma/mean/r/g/b） |
| `*_cleaned.png` | **單通道灰階 PNG**；輸入 16-bit → 輸出 16-bit，輸入 8-bit → 輸出 8-bit |
| `*_mask.png` | **單通道 8-bit PNG，255 = 有效 ROI，0 = 排除** |
| `*_preview.png` | 單通道 8-bit、對比拉伸，僅供目視 |
| `*_overlay_preview.png` | 彩色（BGR/RGB）疊圖，**僅供預覽（preview only）**，不要餵給後續演算法 |
| `*_report.json` | 參數、位元深度中繼資料、ROI 統計、雜訊估計 |
| `*_corrected.png` / `*_denoised.png` | 中間結果（`save_intermediates`），同樣是單通道灰階 |
| 16-bit TIFF | 預設**不輸出**；需要時設 `output.write_tiff16: true` |

內部一律以 float32、範圍 [0,1] 運算（1.0 = 輸入位元深度的滿刻度；12-bit 資料存在 16-bit PNG 時請設 `input.full_scale: 4095`）。因此 `sigma_color`、`target_level`、閾值等都是 [0,1] 單位。

## 安裝與執行

```bash
cd <research-tree>/aoi_preprocess
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt      # 已建好

# 處理單檔或整個資料夾（資料夾只吃 png/bmp/tif/tiff，不遞迴，除非 input.recursive: true）
.venv/bin/python -m aoi_preprocess run --input <file|dir> --config config_panel.yaml --out <dir>
# 有實測 flat / dark 時（也可寫在 config 的 illumination.flat_path / dark_path）
.venv/bin/python -m aoi_preprocess run --input img.png --config config_panel.yaml --out out --flat flat.png --dark dark.png
# 產生合成 demo 資料 / 重做 demo 圖與缺陷檢查
.venv/bin/python -m aoi_preprocess synth --out demo                       # 預設：旋轉 1.4° 的方形面板
.venv/bin/python -m aoi_preprocess synth --kind wafer --out demo_circle   # 圓形晶圓（circle 模式用）
.venv/bin/python tools/make_summary.py                                   # demo/summary.png + defect_check.json（含擬合矩形 vs 真值）
.venv/bin/python tools/make_summary.py --demo demo_circle --config demo_circle/config_circle.yaml
.venv/bin/python tests/test_smoke.py
```
> 注意：`--input` 給資料夾時，資料夾內的所有影像都會被當作待檢影像，請把 flat／dark 放在別的資料夾。

## 各階段與參數（對應 `config_panel.yaml`）

### 1. 載入 `io_utils.load_image`
轉 float32 [0,1]，保留 `bit_depth`、`full_scale`、`src_dtype`、`observed_min/max` 等中繼資料。

### 2. 照明校正 `illumination.correct_illumination`
- **有 flat**：`(img − dark) / (flat − dark)`，再縮放使 ROI 中位數 = `target_level`。尺寸須與影像相同。
- **沒有 flat**：由影像自行估計背景（`illumination.background.method`）：
  - `gaussian`：大 σ 高斯（`gaussian_sigma_px`，null = 0.10×短邊）；
  - `morphological`：大結構元素開/閉運算（`morph_kernel_px`）；
  - `polynomial`：多項式曲面擬合（`poly_degree`，預設 4，適合暗角 vignetting）。
  - 三者都在縮小影像上估計（`work_max_side`），並用**基板遮罩（預設為擬合出的矩形）**加權（`use_wafer_weight`；遮罩內縮 `weight_erode_px`，並扣除 config 中的夾具多邊形 `exclude_fixture_polygons`，所以夾具／板外不會拉偏背景）與離群值剔除（`robust_iters`、`robust_clip_sigma`）避免缺陷／夾具拉偏背景。
  - **背景尺度必須遠大於最大缺陷**，否則大缺陷會被當成背景吃掉。
- `mode`：`divide`（乘性照明，預設）或 `subtract`；`target_level`：校正後平均灰階（預設 0.5，保留亮顆粒不飽和的空間）；`bg_floor_frac`：避免除以接近 0 的背景。

### 3. 降噪 `denoise.denoise`（`denoise.method`）
- `none` / `median`（`ksize`）/ `bilateral`（`diameter`、`sigma_color`、`sigma_space`）/ `nlm`（`patch_size`、`patch_distance`、`h_factor`）。
- **取捨**：任何降噪都是平均，而小顆粒缺陷正是「小、孤立」的結構。median 核越大越容易刪掉小顆粒（3×3 會削弱 1 px 缺陷）；bilateral 的 `sigma_color` 要「高於雜訊、低於要檢出的最弱缺陷對比」；NLM 保細節最好但最慢，`h_factor` 越小越安全。**務必用真實樣本確認最弱缺陷的對比／CNR 仍在**（見下方 demo 檢查）。

### 4. ROI 遮罩 `roi.*`
**基板邊界** `roi.wafer`（key 名稱沿用 `wafer`，實際意義是「基板／面板邊界」）：`shape: rect | circle`，**預設 `rect`**。

**矩形模式（預設）— `detect_wafer` → `fit_rect_robust`：**
1. 在 ≤ `work_max_side` 的影像上做高斯模糊 + Otsu → 閉/開運算 → 取最大輪廓；`minAreaRect` 給出初始方位與四個邊的歸屬。
2. 每條邊只取中央 `rect.span_frac`（預設 80%）的輪廓點（**跳過角落**，圓角／倒角不影響直線擬合）。
3. 每條邊先用「到初始方向的有號距離直方圖眾數」定出邊的位置，只保留 `mode_tol_px`（3 px）內的點——夾具突起（亮、向外）或缺口（暗、向內）只占少數點，被視為離群值；再做 TLS 直線擬合 + MAD 迭代剔除（`outlier_sigma`）。
4. 四條直線兩兩求交 → 四個角點。每邊角度獨立，所以**小角度旋轉自動處理**（`side_angles_deg` 會一併回報；旋轉超過 `warn_rotation_deg` 會警告）。`rect.allow_rotation: false` 則強制邊與影像軸平行。
5. 可選角落形狀 `rect.corner.type: none|round|chamfer` + `size_px`（圓角半徑／45° 倒角腳長），該區域直接從基板遮罩移除。
- 報告 `roi.wafer` 中有：`angle_deg`、`size_wh`、`center_xy`、`corners`、每邊擬合點數與 RMS、以及純 `minAreaRect` 的尺寸（對照用）。**純 minAreaRect 會被夾具撐大**（demo：高度 1111/1179 vs 真值 1040/1180），穩健邊線擬合後誤差 ≈ 2 px。
- **邊緣排除帶** `edge_exclusion_px`：從**每一條邊向內**量測（距離轉換），圓角／倒角處也沿實際輪廓。可選 `edge_exclusion_per_side_px: {top,right,bottom,left}` 逐邊加寬（實際寬度 = max(全域, 該邊)）。
- **角落額外排除** `corner_exclusion_radius_px`：以四個標稱角點為圓心的圓盤（0 = 關閉），用於角落應力／暗角／夾持區。

**圓形模式（`shape: circle`，保留）**：預設 `circle.method: contour`（Otsu → 最大輪廓 → MAD 剔除離群點的最小平方圓擬合），可選 `hough`；`fallback_to_other` 允許一種失敗時改用另一種。圓形模式忽略 `rect.*`、逐邊寬度與角落排除。範例設定：`demo_circle/config_circle.yaml`。

**共同**：`manual.enabled: true` 可直接指定位置；`object_brighter_than_background` 設基板比背景亮或暗；`detect_on: raw|corrected`；`boundary_shrink_px` 額外內縮。

**夾具排除** `fixture`：`polygons`（像素或 0–1 座標，`polygon_units`）+ `margin_px` 外擴 + 可選 `mask_png`（白 = 排除）+ 可選自動偵測 `auto`：在邊界內側 `edge_zone_px` 的帶狀區域搜尋——因為是到最近邊界的距離，**所以沿四條邊與四個角都會搜尋**——找「比基板中位數亮 `bright_ratio` 倍／暗 `dark_ratio` 倍」且面積 ≥ `min_area_px` 的大塊；小顆粒不會被當夾具。報告 `fixture_auto_components` 列出每個偵測到的夾具所在邊（top/left/…／corner:…）。

**最終有效 ROI** = 基板內 ∧ 非邊緣帶（每邊） ∧ 非角落圓盤 ∧ 非夾具。

### 5. 輸出 `outputs.write_outputs`
見上方 I/O 表。`zero_outside_valid: true` 可讓 cleaned 在 ROI 外補 0；預設保留像素，改由 mask 決定檢測範圍。

## 預設值選擇理由
- 多項式 degree 4 + 遮罩加權 + 離群剔除：暗角是平滑徑向衰減，多項式最穩、最快；不依賴影像中的缺陷大小。
- 降噪預設 `bilateral`（`sigma_color 0.04` ≈ 3–4× 雜訊 σ）：在合成資料上缺陷對比保留 ≈ 86%、CNR 約 3 倍，速度快。`nlm` 保留更好（≈ 94%）但約慢 100 倍；`median 3` 對 1 px 缺陷傷害明顯。
- `target_level 0.5`：亮、暗缺陷都不會在 16/8-bit 裁切。
- `shape: rect`：目標主要是方／矩形基板。擬合邊界落在斜邊半高處，約比真實外緣內縮 1–2 px，已被邊緣帶涵蓋。
- `span_frac 0.8`、`mode_tol_px 3`：只用每邊中段、以眾數定位，夾具（占邊長 10–25%）與圓角都不會影響擬合。
- `edge_exclusion_px 20`：合成斜邊約 8–10 px，加上擬合誤差（約 2 px）的餘裕。`corner_exclusion_radius_px 45`：圓角半徑 30 + 餘裕。**兩者實機請依實際斜邊／角落形狀調整。**
- `weight_erode_px 10`：背景擬合遠離邊緣衰減區。

## Demo
**`demo/`（預設：旋轉方形面板）** — `synthetic_panel_u16.png`（16-bit 灰階輸入；1180×1040 面板、旋轉 1.4°、圓角 30 px、暗角、雜訊；8 個真實顆粒，其中 3 個靠近邊緣但在邊緣帶之外（距邊 28 / 35 / 69 px）；4 個應被遮掉的 decoy：邊緣帶 1、角落 1、夾具 A（亮，上緣，config 多邊形）上 1、夾具 B（暗，左緣，自動偵測）上 1）、`synthetic_panel_u8.png`、`flat_u16.png`／`dark_u16.png`、`ground_truth.json`（含面板中心／尺寸／角度／角點／缺陷）、`out/`、`out_u8/`、`out_flat/`（flat+dark）、`summary.png`（原圖／校正後／降噪後／ROI 疊圖，疊圖上洋紅虛線 = 真值矩形）、`defect_check.json`（擬合矩形 vs 真值、缺陷對比前後、各降噪方法比較）。

**`demo_circle/`（circle 模式範例）** — 同樣內容但為圓形晶圓，使用 `config_circle.yaml`。

## v3:Willson 標註 → 油膜 / 水痕 / 治具偵測 + 污染層分離

**CLI**:`python -m aoi_preprocess run --root <DB> --config config_v3.yaml --out OUT --input <資料夾/角度> --all`
(輸出契約同 v2,新增 `NN_stain.png`)。

- **nuisance bit flags**(可重疊,不互斥):1 鏡面、2 不均勻、4 粗糙(v3 已縮小,不再吞掉油膜/水痕)、8 水痕、16 油膜、32 疑似載片本體油墨(低信心,≥150 px 才標,多半已被 16/8 涵蓋)。
- **偵測**(`patterns.py`,模板式、手調門檻,非訓練):油膜 = 高頻局部 RMS 能量 E 的定向羽狀紋;水痕 = 密集小暗點/環狀乾燥水珠/亮色水窪;油墨 = 大面積暗色不規則斑塊。油膜/水痕只加旗標,**不縮小 ROI**。
- **治具**(Side 端點影格 00 / 15,所有角度):失焦肋條 holder 殘影帶 → `class 3`,自 `NN_mask.png` 排除;bbox/信心記錄於 `report.json → frames[NN].fixtures`。
- **污染層分離**(`stain.py`,`tv_gated`):`R = corrected − TV(corrected)`,僅衰減 |R| ≤ 12 levels 的成分(c1=6、c2=12 平滑閘),且只作用於污染區膨脹 25 px 並羽化 15 px 的範圍。|R|>12 的缺陷完全不動。`NN_stain.png` = 128 + 8×被移除層(8-bit,ROI 外為 0)。
- **overlap 指標**:`frames[NN].overlap`(≥2 旗標同時覆蓋的 ROI 面積 %)。
- **labels**:`frames[NN].labels` 區分 `willson_confirmed` 與 `auto_detected`;彙整 CSV:`OUT/suspected_patterns.csv`;驗證彙整:`tools/summarize_v3.py`;圖:`tools/make_figs_v3.py`;注入驗證:`tools/eval_separation_v3.py`。
- **測試**:`python tests/test_v3.py`(合成影像)。
