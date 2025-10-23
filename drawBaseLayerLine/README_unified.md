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
python draw_line_back.py --date 2025-10-22 --workers 4
```
