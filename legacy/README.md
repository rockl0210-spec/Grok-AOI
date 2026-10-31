# 歷史評測腳本

這些腳本是研究階段的代理評測，依賴當時的中間檔（多數已不在這個套件裡）。它們**不是** `grok_aoi.py` 的一部分。

正式評測請用：

```bash
python grok_aoi.py --db <DB> --out <OUT> --with-eval
```

若仍要執行這裡的腳本，先設定研究樹根目錄：

```bash
export GROK_AOI_LEGACY_ROOT=/path/to/research/tree
```

路徑會接在這個根目錄後面（例如 `$GROK_AOI_LEGACY_ROOT/aoi_db`）。沒有該環境變數時，腳本在讀取路徑時會失敗。
