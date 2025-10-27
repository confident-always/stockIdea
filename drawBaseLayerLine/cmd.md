# 画线流水线
```bash
python run_draw_lines_pipeline.py --date 2025-10-27 --workers 4
# 输出目录: {date}-drawLine/
```
# 处理指定股票
```bash
# 处理多个股票
python draw_lines_mid.py --codes 000001 600000 002603
python draw_lines_back.py --codes 000001 600000 002603
# 指定线程数
python run_draw_lines_pipeline.py --codes 600118 --workers 2
```
### 组合画线
```bash
# 处理指定日期（会自动生成mid图片）
cd drawBaseLayerLine
python draw_lines_all.py --date 2025-10-25

# 处理指定股票代码
python draw_lines_all.py --codes 000001 600000 --workers 2
```

### 基础层使用

```bash
# 处理指定日期的股票（只生成基础图表）
cd drawBaseLayerLine
python draw_lines_base.py --date 2025-10-20 --workers 4

# 输出目录: {date}-drawLineRes/
```

### 中间层使用

```bash
# 处理指定日期的股票（生成基础图表 + AnchorM线）
cd drawBaseLayerLine
python draw_lines_mid.py --date 2025-10-20 --workers 4

# 输出目录: {date}-drawLineMid/
```

### 背景格子
# 指定线程数（加快处理速度）
```bash
python draw_lines_back.py --date 2025-10-22 --workers 4
# 输出目录: {date}-drawLineBack/
```
