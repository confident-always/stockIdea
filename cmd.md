# 更新到最新交易日数据
```bash
python update_stocklist.py
```

# 股票数据处理流水线使用说明
```bash
python run_pipeline.py

# 画线流水线
```bash
python run_draw_lines_pipeline.py --date 2025-10-22 --workers 6
# 输出目录: {date}-drawLine/
```

# 获取所有股票完整历史数据
### 运行fetch_kline_akshare

```bash
# akshare
# 增量更新模式（补充到最新日期）
python fetch_kline_akshare.py --mode update --stocklist ./stocklist.csv --exclude-boards gem star bj --out ./data --workers 6

# 全量抓取模式（原有功能）
python fetch_kline_akshare.py --start 0 --end today --stocklist ./stocklist.csv --exclude-boards gem star bj --out ./data --workers 6

# tushare
# 增量更新模式（补充到最新日期）
python fetch_kline.py --mode update --stocklist ./stocklist.csv --exclude-boards gem star bj --out ./data --workers 6
```
### 运行选股
```bash
python select_stock.py --data-dir ./data --config ./configs.json --meta-workers 6
```
# 涨跌幅过滤
```bash
python adx_filter.py --workers 6

python adx_filter.py --input-dir ./20251111-res --output-dir ./20251111-resByFilter --workers 6

```
### 画线
```bash
python draw_lines_base.py  --workers 6
```
```bash
python draw_lines_mid.py --date 2025-10-18 --workers 6
```
```bash
python draw_lines_back.py --date 2025-10-18 --workers 6
```
