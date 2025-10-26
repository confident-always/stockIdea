# draw_lines_all.py - AnchorM和AnchorBack合并画线脚本

## 功能说明

`draw_lines_all.py` 在 **Mid图片基础上直接绘制AnchorBack线条**，将 **AnchorM** 和 **AnchorBack** 两种算法合并到一张图中。

### 核心特性

1. **基于Mid图片绘制**
   - 复制mid图片作为基础（不存在则自动生成）
   - 调用BackLineDrawer计算AnchorBack线的数据
   - 使用matplotlib在mid图片上直接绘制蓝色线条和标注

2. **智能数据计算**
   - 调用`BackLineDrawer.compute_anchor_back_lines()`方法
   - 获取最佳N值、B值序列、K值序列、匹配得分等数据
   - 基于真实计算结果绘制，保证准确性

3. **双算法完整展示**
   - **AnchorM算法**（紫色线）：来自mid图片的原始内容
   - **AnchorBack算法**（蓝色线）：通过matplotlib叠加绘制
   - 两种算法的线条和标注同时显示，便于对比

4. **信息框对称布局**
   - 左上角：AnchorM 信息框（紫色边框，M值系统）
   - 右上角：AnchorBack 信息框（蓝色边框，N值系统）**新绘制**
   - 两个信息框完全对称分布，清晰展示两种算法参数

## 使用方法

### 推荐使用流水线脚本

```bash
cd drawBaseLayerLine
python run_draw_lines_pipeline.py --date 2025-10-25 --workers 4
```

流水线会自动：
1. 运行 draw_lines_mid.py 生成紫色 AnchorM 图片
2. 运行 draw_lines_back.py 生成蓝色 AnchorBack 图片  
3. 运行 draw_lines_all.py 在mid基础上添加back线

### 单独运行

```bash
# 处理指定日期（会自动生成mid图片）
cd drawBaseLayerLine
python draw_lines_all.py --date 2025-10-25

# 处理指定股票代码
python draw_lines_all.py --codes 000001 600000
```

**特点**：
- 如果mid图片不存在，脚本会自动调用 `draw_lines_mid.py` 生成
- 自动清除旧的输出目录，确保结果最新
- 如果back数据计算失败，直接使用mid图片

**输出**：
- 输出目录：`{日期}-drawLineAll/`
- 文件命名：`{前缀}_{代码}_{股票名}_3all.png` 或 `{代码}_{股票名}_3all.png`
- 例如：`ADX105_000155_川能动力_3all.png` 或 `000155_川能动力_3all.png`

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--date` | 指定日期（格式：YYYY-MM-DD） | 当前日期 |
| `--workers` | 并发线程数（用于生成mid图片） | 4 |
| `--codes` | 股票代码列表，多个代码用空格分隔 | 无 |

## 技术实现

### 绘制流程

1. **加载Mid图片**
   ```python
   img = Image.open(mid_path)
   # 创建matplotlib figure，尺寸与mid图片完全一致
   fig = plt.figure(figsize=(img_width/dpi, img_height/dpi), dpi=dpi)
   ax = fig.add_axes([0, 0, 1, 1])  # 无边距
   ax.imshow(img)  # 显示mid图片为背景
   ```

2. **计算AnchorBack数据**
   ```python
   # 调用BackLineDrawer的计算方法
   back_data = back_drawer.compute_anchor_back_lines(df, anchor_idx, anchor_date, code)
   # 获取：best_N, B_values, K_values, avg_score, matches_count等
   ```

3. **绘制蓝色线条**
   ```python
   for k_val, B_k_price in zip(K_values, B_values):
       # 价格转换为像素坐标
       y_px = chart_top + (price_max - B_k_price) / (price_max - price_min) * (chart_bottom - chart_top)
       # 绘制横线
       ax.plot([chart_left, chart_right], [y_px, y_px], color='#1E90FF', ...)
       # 绘制左侧标注
       ax.text(chart_left - 10, y_px, f"K={k_val} 价格={B_k_price:.2f}", ...)
   ```

4. **绘制右上角N信息框**
   ```python
   # 计算对称位置（与左上角对称）
   box_right = img_width - info_box_left - info_box_width
   # 绘制信息框背景
   box = mpatches.FancyBboxPatch((box_right, box_top), width, height, ...)
   # 绘制文本
   ax.text(text_x, text_y, f"N={best_N:.2f}", ...)
   ax.text(text_x, text_y + line_height, f"Match_B: [{k_values}]", ...)
   ax.text(text_x, text_y + line_height*2, f"AvgScore: {avg_score:.1f}%", ...)
   ```

5. **保存图片**
   ```python
   plt.savefig(image_path, dpi=dpi, pad_inches=0)  # 保持原始尺寸
   ```

### 关键技术点

1. **坐标系转换**
   - Mid图片使用像素坐标系（原点在左上角）
   - 图表区域：左100px，右100px，顶135px，底100px（基于mplfinance标准布局）
   - 价格到Y轴像素的转换：`y_px = chart_top + (price_max - price) / (price_max - price_min) * chart_height`

2. **数据计算**
   - 调用 `BackLineDrawer.find_stage_lows_unified(df)` 检测阶段低点
   - 调用 `BackLineDrawer.compute_anchor_back_lines(df, anchor_idx, anchor_date, code)` 计算N值数据
   - 返回的数据包含：best_N, B_values, K_values, anchor_A, avg_score, matches_count, per_k_matches等

3. **图片尺寸保持**
   - 使用 `fig.add_axes([0, 0, 1, 1])` 创建无边距的axes
   - 不使用 `bbox_inches='tight'`，避免图片尺寸改变
   - 直接指定figsize和dpi，保证输出尺寸与mid图片一致

### 优势

- ✅ **数据准确**：调用back的计算方法，保证数据准确性
- ✅ **保持一致性**：基于mid图片绘制，保持整体风格一致
- ✅ **尺寸不变**：输出图片尺寸与mid图片完全一致，不变形
- ✅ **中文显示正常**：继承mid图片的字体渲染，中文显示完美
- ✅ **处理速度快**：只计算back数据，不重新绘制整个图表
- ✅ **自动化**：自动检查和生成依赖图片，无需手动操作

## 输出示例

### 图表特点

**_3all.png 图表包含**：
- ✅ 紫色AnchorM线（左侧标注，来自mid图片）
- ✅ 蓝色AnchorBack线（左侧标注，新绘制）
- ✅ 左上角信息框：AnchorM参数（M值、MatchBars、得分等，来自mid图片）
- ✅ 右上角信息框：AnchorBack参数（N值、Match_B、得分等，新绘制）**完全对称**
- ✅ 标题、K线图、百分比线等（来自mid图片）

**信息框内容示例**：
```
右上角（蓝色边框）：
N=0.43
Match_B: [3, 5, 7, 9, ...]
AvgScore: 16.3%
Matches: 4/10
```

## 配置文件

算法参数在 `lineConfig.json` 中配置：

```json
{
  "anchorMLines": {
    "enabled": true,
    "zigzag_percent": 10,
    "m_range": {"start": 13.0, "end": 9.0, "step": -0.1},
    "max_k": 20,
    "line_style": {"color": "#8A2BE2", "linewidth": 3.0, "alpha": 0.9}
  },
  "anchorBackLines": {
    "enabled": true,
    "zigzag_percent": 15,
    "n_range": {"start": 0.23, "end": 0.68, "step": 0.01},
    "k_list": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    "line_style": {"color": "#1E90FF", "linewidth": 3.0, "alpha": 0.9}
  }
}
```

## 注意事项

1. **自动清除旧文件**：
   - 每次运行会自动清除 `{日期}-drawLineAll/` 目录
   - 确保结果是最新的

2. **依赖要求**：
   - 需要mid图片（不存在会自动生成）
   - 需要CSV数据文件在 `../data/` 目录

3. **容错处理**：
   - 如果CSV文件不存在，直接复制mid图片
   - 如果阶段低点检测失败，直接复制mid图片
   - 如果back数据计算失败，直接复制mid图片

## 更新日志

### v3.0 (2025-10-25)
- ✅ **重新设计**：改用在mid图片基础上直接绘制的方案
- ✅ **调用back计算方法**：使用 `BackLineDrawer.compute_anchor_back_lines()` 获取真实数据
- ✅ **matplotlib叠加绘制**：在mid图片上直接绘制蓝色线条和信息框
- ✅ **保持图片尺寸**：输出图片与mid图片尺寸完全一致
- ✅ **数据准确性**：基于真实计算结果，保证准确性
- ✅ **右上角对称信息框**：N信息框与左上角M信息框完全对称

### v2.2 (2025-10-25)
- ✅ 自动依赖管理：自动检查并生成mid和back图片
- ✅ 自动清除旧文件：运行前自动清除旧的输出目录
- ✅ 简化实现：完全基于图片叠加
- ✅ 更好的容错：如果back图片不存在，直接使用mid图片

### v2.1 (2025-10-25)
- ✅ 智能过滤：完全排除左上角信息框区域
- ✅ 精确提取：从back图片精确提取信息框并放置到右上角
- ✅ 优化底部过滤：排除底部区域的蓝色元素
- ✅ 无重复叠加：百分比线、锚定低点线只显示一次

### v2.0 (2025-10-25)
- ✅ 改用图片叠加方案
- ✅ 解决中文乱码问题
- ✅ 增强蓝色线条显示
- ✅ 添加右上角信息框

### v1.0 (2025-10-25)
- ✅ 创建 `draw_lines_all.py` 脚本
- ✅ 实现AnchorM和AnchorBack双算法合并显示
- ✅ 集成到流水线

---

**提示**：如需单独运行某一种算法，请使用：
- `draw_lines_mid.py`（仅AnchorM，紫色线）
- `draw_lines_back.py`（仅AnchorBack，蓝色线）
- `draw_lines_all.py`（两种算法合并，紫色+蓝色）
