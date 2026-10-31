"""Traditional-Chinese run report: HTML, Markdown, and JSON. Every count comes from this run."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from grok_aoi.report.make_csv import discover_folders, _mode

NUISANCE_BITS = [
    (1, 'specular', '鏡面反光'),
    (2, 'uneven', '照明不均'),
    (4, 'rough', '粗糙'),
    (8, 'water', '水痕'),
    (16, 'oil', '油膜'),
    (32, 'ink', '載體墨跡 slide ink'),
]


def _esc(v):
    return html.escape('' if v is None else str(v))


def _num(v):
    if v is None:
        return ''
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, float):
        if pd.isna(v):
            return ''
        return f'{v:.4g}'
    return str(v)


def _table(headers, rows):
    th = ''.join(f'<th>{_esc(h)}</th>' for h in headers)
    body = []
    for r in rows:
        body.append('<tr>' + ''.join(f'<td>{_esc(_num(c))}</td>' for c in r) + '</tr>')
    return f'<table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def _md_table(headers, rows):
    lines = ['| ' + ' | '.join(str(h) for h in headers) + ' |',
             '| ' + ' | '.join('---' for _ in headers) + ' |']
    for r in rows:
        lines.append('| ' + ' | '.join(_num(c).replace('|', '/') for c in r) + ' |')
    return '\n'.join(lines)


def _load_decisions(filter_out, folders):
    parts = []
    for f in folders:
        p = Path(filter_out) / _mode(f) / f / 'decisions.csv'
        if p.is_file() and p.stat().st_size:
            d = pd.read_csv(p, dtype={'frame': str})
            if len(d):
                parts.append(d.assign(folder=f))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _json_load(path):
    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding='utf-8'))


def build_summary(out_dir):
    out = Path(out_dir)
    meta = _json_load(out / 'config' / 'run_meta.json')
    filt = _json_load(out / 'filter' / 'summary.json')
    folders = meta.get('folders') or discover_folders(out / 'filter')
    dec = _load_decisions(out / 'filter', folders)
    per = out / 'final' / 'defects_per_frame.csv'
    pf = pd.read_csv(per, dtype={'frame': str}) if per.is_file() and per.stat().st_size else pd.DataFrame()
    uniq = out / 'final' / 'final_defects_unique.csv'
    ud = pd.read_csv(uniq) if uniq.is_file() and uniq.stat().st_size else pd.DataFrame()
    metrics = _json_load(out / 'eval' / 'metrics.json')
    return meta, filt, folders, dec, pf, ud, metrics


def render(out_dir):
    out = Path(out_dir)
    meta, filt, folders, dec, pf, ud, metrics = build_summary(out)
    report = out / 'report'
    report.mkdir(parents=True, exist_ok=True)

    n_cand = int(len(dec)) if len(dec) else int(filt.get('n_candidates') or 0)
    n_real = int((dec.decision == 'real').sum()) if len(dec) else 0
    n_nuis = n_cand - n_real
    n_l1 = int((dec.review_level == 1).sum()) if len(dec) else 0
    n_l2 = int((dec.review_level == 2).sum()) if len(dec) else 0
    real = dec[dec.decision == 'real'] if len(dec) else dec
    n_real_l0 = int((real.review_level == 0).sum()) if len(real) else 0
    n_real_l1 = int((real.review_level == 1).sum()) if len(real) else 0
    n_real_l2 = int((real.review_level == 2).sum()) if len(real) else 0
    n_def = int(len(ud)) if len(ud) else int(filt.get('unique_defects_total') or 0)

    by_class = []
    for name in ('particle', 'stain', 'scratch'):
        sub = real[real.fine_class == name] if len(real) else real
        by_class.append([name, {'particle': '落塵', 'stain': '髒污', 'scratch': '刮傷'}[name],
                         len(sub),
                         int((sub.review_level == 0).sum()) if len(sub) else 0,
                         int((sub.review_level == 1).sum()) if len(sub) else 0,
                         int((sub.review_level == 2).sum()) if len(sub) else 0])

    folder_rows = []
    if len(pf):
        for f, g in pf.groupby('folder', sort=False):
            folder_rows.append([
                f, int(g.candidates.sum()), int(g.real.sum()), int(g.nuisance.sum()),
                int(g.particle.sum()), int(g.stain.sum()), int(g.scratch.sum()),
                int(g.real_no_review.sum()), int(g.real_L1.sum()), int(g.real_L2.sum()),
                int(g.nuisance_review_L1.sum()), int(g.nuisance_review_L2.sum()),
            ])
    folder_headers = ['資料夾', '候選', 'real', 'nuisance', '落塵', '髒污', '刮傷',
                      'real 不複檢', 'real L1', 'real L2', 'nuisance L1', 'nuisance L2']

    frame_headers = list(pf.columns) if len(pf) else []
    frame_rows = pf.values.tolist() if len(pf) else []

    zone_rows = []
    if len(dec) and 'zone' in dec.columns:
        for (zone, decision), g in dec.groupby(['zone', 'decision']):
            zone_rows.append([zone, decision, len(g), int((g.review_level == 0).sum()),
                              int((g.review_level == 1).sum()), int((g.review_level == 2).sum())])
    zone_headers = ['區域 zone', '決策', '數量', 'review 0', 'L1', 'L2']

    l2 = dec[dec.review_level == 2].copy() if len(dec) else dec
    if len(l2):
        l2['_abs'] = l2.contrast.abs()
        l2['_real'] = (l2.decision == 'real').astype(int)
        l2 = l2.sort_values(['_real', '_abs'], ascending=[False, False])
        keep = [c for c in ['candidate_id', 'folder', 'frame', 'decision', 'fine_class', 'contrast', 'cnr',
                            'zone', 'review_level', 'reason_code', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'image'] if c in l2.columns]
        l2[keep].to_csv(report / 'l2_priority.csv', index=False)
        show = l2.head(200)
        l2_rows = show[keep].values.tolist()
        l2_headers = keep
    else:
        pd.DataFrame().to_csv(report / 'l2_priority.csv', index=False)
        l2_rows, l2_headers = [], ['candidate_id']

    # overlays
    shots = []
    for f in folders:
        csheet = out / 'final' / 'overlays' / f / 'contact_sheet.jpg'
        if csheet.is_file():
            shots.append((f, Path('..') / 'final' / 'overlays' / f / 'contact_sheet.jpg'))

    t2 = filt.get('side_T2_conversion') or {}
    profile = meta.get('profile') or {}
    timings = meta.get('timings_s') or {}

    summary = {
        'n_candidates': n_cand,
        'n_real': n_real,
        'n_nuisance': n_nuis,
        'review_L1': n_l1,
        'review_L2': n_l2,
        'real_no_review': n_real_l0,
        'real_L1': n_real_l1,
        'real_L2': n_real_l2,
        'unique_defects': n_def,
        'by_class': [{'class': r[0], 'zh': r[1], 'n': r[2], 'L0': r[3], 'L1': r[4], 'L2': r[5]} for r in by_class],
        'folders': folder_rows,
        'side_T2': t2,
        'eval': metrics or None,
        'timings_s': timings,
    }
    (report / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

    def section_eval():
        if not metrics:
            return ('<h2>代理評測</h2><p>本次未啟用評測（預設略過；加上 <code>--with-eval</code> 會重跑 polarity-flip 偵測並計算參考點保留率）。</p>',
                    '## 代理評測\n\n本次未啟用評測。加上 `--with-eval` 會計算參考點保留率與 polarity-flip 假 real 估計。\n')
        ret = metrics.get('retention') or {}
        flip = metrics.get('flip') or {}
        defin = ret.get('definition') or {}
        html_parts = ['<h2>代理評測</h2>']
        md = ['## 代理評測', '']
        html_parts.append('<p>沒有 ground truth。保留率是對獨立 THR=12 參考點的覆蓋，不是召回率；假 real 是 polarity-flip 代理，不是精度。</p>')
        md.append('沒有 ground truth。保留率是對獨立 THR=12 參考點的覆蓋，不是召回率；假 real 是 polarity-flip 代理，不是精度。')
        if defin:
            html_parts.append('<ul>' + ''.join(f'<li><b>{_esc(k)}</b>：{_esc(v)}</li>' for k, v in defin.items()) + '</ul>')
        html_parts.append(f'<p>參考點數 {ret.get("n_ref", 0)}。對比 bin 為 12–20／20–40／40+。</p>')
        md.append(f'參考點數 {ret.get("n_ref", 0)}。')
        for key, title in (('by_bin', '對比 bin（含 Edge／Side）'), ('by_zone_bin', '區域 × 對比 bin'), ('by_folder_bin', '資料夾 × 對比 bin')):
            rows = ret.get(key) or []
            if not rows:
                continue
            headers = list(rows[0].keys())
            recs = [[r.get(h) for h in headers] for r in rows]
            html_parts.append(f'<h3>{title}</h3>' + _table(headers, recs))
            md.append(f'### {title}')
            md.append(_md_table(headers, recs))
            md.append('')
        html_parts.append('<h3>Polarity-flip 假 real 估計</h3>')
        md.append('### Polarity-flip 假 real 估計')
        if flip:
            items = [(k, flip[k]) for k in flip if k != 'stratified']
            html_parts.append('<ul>' + ''.join(f'<li><b>{_esc(k)}</b>：{_esc(v)}</li>' for k, v in items) + '</ul>')
            for k, v in items:
                md.append(f'- **{k}**：{v}')
            st = flip.get('stratified') or []
            if st:
                headers = list(st[0].keys())
                # by_kind is a dict; stringify
                recs = []
                for r in st:
                    recs.append([json.dumps(r[h], ensure_ascii=False) if isinstance(r[h], dict) else r[h] for h in headers])
                html_parts.append(_table(headers, recs))
                md.append(_md_table(headers, recs))
        html_parts.append('<p>union／flip 路徑沒有 persistence，Side T2 的規則 (a)(b) 不會觸發，因此 flip 通過率是該項的上界。</p>')
        md.append('')
        md.append('flip 路徑沒有 persistence，Side T2 規則 (a)(b) 不會觸發，通過率是該項的上界。')
        return '\n'.join(html_parts), '\n'.join(md)

    eval_html, eval_md = section_eval()
    img_html = []
    img_md = []
    for f, rel in shots:
        img_html.append(f'<figure><figcaption>{_esc(f)} 接觸表</figcaption><img src="{_esc(rel.as_posix())}" alt="{_esc(f)} contact sheet"></figure>')
        img_md.append(f'![{f}]({rel.as_posix()})')
    # per-frame links
    links = []
    for f in folders:
        d = out / 'final' / 'overlays' / f
        if not d.is_dir():
            continue
        jpgs = sorted(p for p in d.glob('*.jpg') if p.name != 'contact_sheet.jpg')
        if not jpgs:
            continue
        bits = ' '.join(f'<a href="../final/overlays/{f}/{p.name}">{p.stem}</a>' for p in jpgs)
        links.append(f'<p><b>{_esc(f)}</b> {bits}</p>')

    limits = f'''<h2>已知限制</h2>
<ul>
<li>預設不使用 filter v3.1 的規則 (b)。本次 Side T2 轉換：規則 (a) {t2.get("rule_a_measurable_unconfirmed", 0)} 筆、規則 (b) {t2.get("rule_b_v4new_not_measurable", 0)} 筆（合計 {t2.get("total", 0)}）。原始全庫交付裡約 205／541 筆 Side 轉換來自規則 (b)；那是歷史數字，不是本次輸出。要重現該路徑請加 <code>--legacy-v31-decisions</code>。</li>
<li>在原始全庫評測中 Edge1/090 是最弱資料夾；油膜／粗糙區的淡灰點大多在 filter 被標成 nuisance（複檢），而不是在偵測階段被丟掉。</li>
<li>油紋上的纖維會被切成許多框。<code>same_object_n</code> 在 oil／rough 區是叢集大小，不是缺陷個數，不要拿來當計數。</li>
<li>干擾區只標註、不排除。跨 ROI 的點保留。靈敏度包含淡灰點。</li>
<li>2 幀 persistence 只在相鄰幀的重疊帶內有效；重疊帶外的候選 <code>pers_tested</code> 為 false。</li>
</ul>'''

    css = '''body{font-family:"Noto Sans CJK TC","PingFang TC","Microsoft JhengHei",sans-serif;margin:2rem;line-height:1.45;color:#1a1a1a;background:#fafafa}
h1,h2,h3{color:#14213d} table{border-collapse:collapse;margin:0.6rem 0 1.2rem;font-size:14px;background:#fff}
th,td{border:1px solid #ccc;padding:4px 8px;text-align:right} th{background:#e9ecef;text-align:center}
td:first-child,th:first-child{text-align:left} code{background:#eee;padding:0 4px}
.metrics{display:flex;flex-wrap:wrap;gap:12px} .metrics div{background:#fff;border:1px solid #ddd;padding:10px 14px;min-width:140px}
.metrics b{display:block;font-size:1.4rem} figure{margin:1rem 0} img{max-width:100%;height:auto;border:1px solid #ddd}
.legend i{display:inline-block;width:14px;height:14px;margin-right:4px;vertical-align:-2px}
'''
    legend = '''<p class="legend">
<i style="background:#3cdc3c"></i>落塵 particle（綠）
<i style="background:#ff9600"></i>髒污 stain（橙）
<i style="background:#1e8cff"></i>刮傷 scratch（藍）
　實線＝不複檢；虛線＝L1；粗框加白暈＝L2；洋紅＝ROI。小於 6 px 的框以 6 px 繪製。
</p>'''
    bits = _table(['bit', '名稱', '說明'], [[b, n, z] for b, n, z in NUISANCE_BITS])
    meta_rows = [
        ['資料庫', meta.get('db')],
        ['輸出', meta.get('out')],
        ['資料夾', ', '.join(folders)],
        ['影像', f"{(profile.get('image') or {}).get('width')}×{(profile.get('image') or {}).get('height')}"],
        ['異常幀 edge', (profile.get('anomaly_frames') or {}).get('edge')],
        ['異常幀 side', (profile.get('anomaly_frames') or {}).get('side')],
        ['e_strip', json.dumps(profile.get('e_strip'), ensure_ascii=False)],
        ['規則 (b)', '啟用 ' + str(meta.get('legacy_v31')) if meta.get('rule_b') else '關閉（只使用 persistence 規則 a）'],
        ['設定檔', meta.get('config')],
        ['各階段秒數', json.dumps(timings, ensure_ascii=False)],
    ]
    page = f'''<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8"><title>Grok-AOI 檢測報告</title><style>{css}</style></head>
<body>
<h1>Grok-AOI 表面潔淨度檢測報告</h1>
<p>preprocess v3 → detection v4.1 → filter v4.2。以下數字全部來自本次執行，不是歷史全庫交付的數字。</p>
<div class="metrics">
<div><b>{n_cand}</b>候選</div>
<div><b>{n_real}</b>real（落塵／髒污／刮傷）</div>
<div><b>{n_nuis}</b>nuisance</div>
<div><b>{n_real_l0} / {n_real_l1} / {n_real_l2}</b>real 的 review 0 / L1 / L2</div>
<div><b>{n_l1} / {n_l2}</b>全體 L1 / L2</div>
<div><b>{n_def}</b>合併後缺陷</div>
</div>
<h2>怎麼讀複檢等級</h2>
<ul>
<li><b>0</b>：real，不需複檢（實線框）。</li>
<li><b>L1</b>：需要看過（虛線）。多數中等疑點在這裡。</li>
<li><b>L2</b>：優先複檢（粗框＋白暈）。包含未確認的 Side T2、鏡面區、大面積、低對比污染區等。</li>
</ul>
<p>nuisance 不是「刪除」。干擾區只被標成複檢，ROI 邊界上的點仍保留。</p>
<h2>執行與設定</h2>
{_table(['項目','值'], meta_rows)}
<h2>流程</h2>
<ol>
<li>Preprocess v3：ROI 遮罩、噪聲圖、污漬分離、nuisance 位元。</li>
<li>Detection v4.1：path A／B／C、raw rescue、haze 切分。座標是原圖像素。</li>
<li>Filter v4.2：real／nuisance、子類、review 0／L1／L2、2 幀 persistence、缺陷合併。</li>
<li>CSV、標註圖、本報告。</li>
</ol>
<h2>Nuisance 位元</h2>
<p>像素值是下列位元的和，可重疊。0 表示 ROI 外或沒有旗標。</p>
{bits}
<h2>各資料夾</h2>
{_table(folder_headers, folder_rows) if folder_rows else '<p>沒有逐幀統計。</p>'}
<h2>子類 × 複檢（只計 real）</h2>
{_table(['class','中文','數量','review 0','L1','L2'], by_class)}
<h2>區域分解</h2>
{_table(zone_headers, zone_rows) if zone_rows else '<p>沒有 zone 欄位。</p>'}
<h2>L2 優先複檢清單</h2>
<p>共 {len(l2) if len(dec) else 0} 筆（real 優先，再依 |contrast|）。下表最多 200 列，完整清單見 <a href="l2_priority.csv">l2_priority.csv</a>。</p>
{_table(l2_headers, l2_rows) if l2_rows else '<p>沒有 L2。</p>'}
<h2>逐幀統計</h2>
<details><summary>展開 {len(frame_rows)} 列</summary>
{_table(frame_headers, frame_rows) if frame_rows else '<p>空</p>'}
</details>
<h2>標註圖</h2>
{legend}
{''.join(img_html) if img_html else '<p>沒有接觸表。</p>'}
<details><summary>逐幀 JPG</summary>{''.join(links)}</details>
{eval_html}
{limits}
</body></html>'''
    (report / 'report.html').write_text(page, encoding='utf-8')

    md = []
    md.append('# Grok-AOI 表面潔淨度檢測報告')
    md.append('')
    md.append('preprocess v3 → detection v4.1 → filter v4.2。數字來自本次執行。')
    md.append('')
    md.append(f'- 候選 {n_cand}，real {n_real}，nuisance {n_nuis}')
    md.append(f'- real review 0／L1／L2 = {n_real_l0}／{n_real_l1}／{n_real_l2}')
    md.append(f'- 全體 L1／L2 = {n_l1}／{n_l2}，合併缺陷 {n_def}')
    md.append('')
    md.append('## 執行與設定')
    md.append(_md_table(['項目', '值'], meta_rows))
    md.append('')
    md.append('## 各資料夾')
    md.append(_md_table(folder_headers, folder_rows) if folder_rows else '（無）')
    md.append('')
    md.append('## 子類 × 複檢（real）')
    md.append(_md_table(['class', '中文', '數量', 'review 0', 'L1', 'L2'], by_class))
    md.append('')
    md.append('## 區域')
    md.append(_md_table(zone_headers, zone_rows) if zone_rows else '（無）')
    md.append('')
    md.append(f'## L2 優先複檢（{len(l2) if len(dec) else 0} 筆，完整檔 l2_priority.csv）')
    md.append(_md_table(l2_headers, l2_rows[:50]) if l2_rows else '（無）')
    md.append('')
    md.append('## 標註圖')
    md.append('綠＝落塵、橙＝髒污、藍＝刮傷；實線＝不複檢、虛線＝L1、粗框白暈＝L2；洋紅＝ROI。')
    md.extend(img_md)
    md.append('')
    md.append(eval_md)
    md.append('')
    md.append('## 已知限制')
    md.append(f'- 規則 (b) 預設關閉。本次 (a)={t2.get("rule_a_measurable_unconfirmed", 0)} (b)={t2.get("rule_b_v4new_not_measurable", 0)}。歷史全庫約 205／541 來自規則 (b)，那不是本次數字。')
    md.append('- 油紋纖維會碎成很多框，same_object_n 不是缺陷數。')
    md.append('- 淡灰點在 oil／rough 區多半在 filter 被標成 nuisance。')
    md.append('- 原始全庫評測中 Edge1/090 最弱。')
    (report / 'report.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    return report / 'report.html'
