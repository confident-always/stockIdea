# 获取所有股票完整历史数据
```bash
python fetch_kline.py --start 0 --end today --stocklist ./stocklist.csv --exclude-boards gem star bj --out ./data --workers 12
```
### 运行fetch_kline_akshare

```bash
python fetch_kline_akshare.py --start 0 --end today --stocklist ./stocklist.csv --exclude-boards gem star bj --out ./data --workers 12

python fetch_kline_akshare.py --start 0 --end today --stocklist ./stocklist.csv --out ./data --workers 12
```
### 运行选股
```bash
python select_stock.py --data-dir ./data --config ./configs.json --meta-workers 8
```
# 涨跌幅过滤
```bash
python adx_filter.py --input-dir res --output-dir resByFilter --workers 8
```
