# 获取所有股票完整历史数据
### 运行fetch_kline_akshare

```bash
python fetch_kline_akshare.py --start 0 --end today --stocklist ./stocklist.csv --exclude-boards gem star bj --out ./data --workers 12
```
### 运行选股
```bash
python select_stock.py --data-dir ./data --config ./configs.json --meta-workers 6
```
# 涨跌幅过滤
```bash
python adx_filter.py --workers 6

python adx_filter.py --input-dir ./20251018-res --output-dir ./20251018-resByFilter --workers 6

```
# 股票数据处理流水线使用说明

### 方法一：使用Python脚本 (推荐)
```bash
python run_pipeline.py
```

### 画线
```bash
python draw_lines_unified.py  --workers 6
```
```bash
python draw_lines_unified.py --date 2025-10-18 --workers 6
```
