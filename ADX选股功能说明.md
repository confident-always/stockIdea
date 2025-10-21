# ADX选股功能说明

## 功能概述

ADX选股系统是基于DMI（Directional Movement Index）指标的趋势跟踪选股工具。系统包含两个核心选股器：**ADX Selector**（趋势确认）和 **PDI Selector**（多头力量），通过检测特定的上穿信号来识别潜在的趋势启动机会。

### 核心特点
- ✅ **交易日计数**: 所有日期参数（lookback_days、pre_high_days等）均按**交易日**计数，经过严格测试验证
- 🎯 **多周期支持**: 支持39日、105日、243日三个不同周期的DMI指标
- 🔄 **两种策略**: ADX上穿MDI（趋势确认）和PDI上穿MDI（多头启动）
- 📊 **丰富过滤**: 支持涨跌幅、前高价格、历史低点等多维度过滤

---

## DMI指标说明

### 指标组成
DMI（方向动量指标）包含三条线：
- **PDI** (Positive Directional Indicator): 正向方向线，衡量多头力量
- **MDI** (Minus Directional Indicator): 负向方向线，衡量空头力量
- **ADX** (Average Directional Index): 平均方向指数，衡量趋势强度

### 计算周期
系统预设三个DMI周期：
- **39日**: 短期趋势，灵敏度高
- **105日**: 中期趋势，平衡型
- **243日**: 长期趋势，稳定性强

---

## 两种选股策略

### 1. ADX Selector（趋势确认策略）

#### 信号定义
在最近 **lookback_days** 个交易日内，寻找 **ADX 上穿 MDI** 的信号：
- **前一交易日**: ADX < MDI
- **当前交易日**: ADX > MDI **且** PDI > MDI

#### 技术含义
- ADX上穿MDI表示趋势强度开始增强
- 同时要求PDI > MDI，确保是多头趋势
- 适合捕捉趋势确认后的跟随机会

#### 配置示例
```json
{
  "class": "ADXSelector",
  "alias": "ADX105",
  "activate": true,
  "params": {
    "dmi_period": 105,
    "lookback_days": 30
  }
}
```

#### 使用场景
- ✅ 趋势跟随：在趋势确认后介入
- ✅ 中长线布局：适合持仓时间较长的投资者
- ✅ 强势股筛选：ADX值越高表示趋势越强

### 2. PDI Selector（多头启动策略）

#### 信号定义
在最近 **lookback_days** 个交易日内，寻找 **PDI 上穿 MDI** 的信号：
- **前一交易日**: PDI < MDI
- **当前交易日**: PDI > MDI

#### 技术含义
- PDI上穿MDI表示多头力量超越空头力量
- 趋势可能从下跌或震荡转为上涨
- 适合捕捉趋势启动的早期机会

#### 配置示例
```json
{
  "class": "PDISelector",
  "alias": "PDI39",
  "activate": true,
  "params": {
    "dmi_period": 39,
    "lookback_days": 30
  }
}
```

#### 使用场景
- ✅ 趋势启动：在多头力量刚刚占优时介入
- ✅ 短线操作：适合灵活进出的投资者
- ✅ 底部反转：捕捉空头转多头的转折点

---

## 配置参数详解

### configs.json 结构

```json
{
  "selectors": [
    {
      "class": "ADXSelector",      // 选择器类型
      "alias": "ADX105",            // 别名（用于文件命名）
      "activate": true,             // 是否启用
      "params": {
        "dmi_period": 105,          // DMI计算周期（交易日）
        "lookback_days": 30         // 回看天数（交易日）
      }
    }
  ],
  "ADXfilteringconfig": {
    "active": true,                 // 是否启用过滤
    "minRange": "0%",               // 最小涨幅/最大跌幅
    "maxRange": "25%",              // 最大涨幅限制
    "lookback_days": 3,             // 上穿前后天数（交易日）
    "maxRangeUp": "none",           // 期间最大涨幅要求
    "preHighDays": 60,              // 前高天数（交易日）
    "preHighMinRange": "-2%",       // 前高最小涨幅
    "preHighMaxRange": "15%",       // 前高最大涨幅
    "lookback_months": 120,         // 历史低点回看月数
    "HisLowPointRange": "none"      // 历史低点涨幅区间
  },
  "PDIfilteringconfig": {
    // 同ADXfilteringconfig结构
  }
}
```

### 核心参数说明

#### 1. 选股参数
| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `dmi_period` | int | DMI指标周期（交易日） | 39, 105, 243 |
| `lookback_days` | int | 回看天数（交易日），从最后一个交易日往前数 | 30 |

#### 2. 过滤参数

**基础涨跌幅过滤**
| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `minRange` | string | 最小涨幅/最大跌幅，负值表示允许的跌幅 | "0%", "-5%" |
| `maxRange` | string | 最大涨幅限制 | "25%", "none" |

**上穿期间涨幅**
| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `lookback_days` | int | 上穿前后查看的交易日数 | 3 |
| `maxRangeUp` | string | 期间最大涨幅要求 | "5%", "none" |

**前高价格过滤**
| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `preHighDays` | int | 前高天数（交易日） | 60 |
| `preHighMinRange` | string | 前高最小涨幅，可为负 | "-2%" |
| `preHighMaxRange` | string | 前高最大涨幅 | "15%" |

**历史低点过滤**
| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `lookback_months` | int | 回看月数 | 120 |
| `HisLowPointRange` | string | 涨幅区间 | "16-600%", "none" |

---

## 使用流程

### 1. 数据准备
确保 `data/` 目录下有K线数据文件（CSV格式）：
```
data/
  ├── 000001.csv
  ├── 600000.csv
  └── ...
```

CSV文件格式：
```csv
date,open,high,low,close,volume
2025-01-02,10.00,10.50,9.80,10.20,1000000
...
```

### 2. 选股执行

#### 基本用法
```bash
# 使用当前日期选股
python select_stock.py

# 指定日期选股
python select_stock.py --date 2025-10-21
```

#### 高级选项
```bash
# 指定配置文件
python select_stock.py --config custom_configs.json

# 指定输出目录
python select_stock.py --output-dir myresults

# 指定并发数
python select_stock.py --workers 8
```

#### 输出结果
选股结果保存在 `{date}-res/` 目录：
```
20251021-res/
  ├── ADX39_2025-10-21.csv
  ├── ADX105_2025-10-21.csv
  ├── ADX243_2025-10-21.csv
  ├── PDI39_2025-10-21.csv
  ├── PDI105_2025-10-21.csv
  └── PDI243_2025-10-21.csv
```

CSV格式：
```csv
code,name,industry,PDI,MDI,ADX,cross_date
000001,平安银行,银行,25.3,18.7,22.5,2025-10-15
...
```

### 3. 结果过滤

#### 基本用法
```bash
# 使用当前日期过滤
python adx_filter.py

# 指定日期过滤
python adx_filter.py --date 2025-10-21
```

#### 高级选项
```bash
# 指定输入/输出目录
python adx_filter.py --input-dir 20251021-res --output-dir 20251021-filtered

# 指定并发数
python adx_filter.py --workers 12
```

#### 输出结果
过滤结果保存在 `{date}-resByFilter/` 目录，包含额外字段：
- `历史低点涨幅(120月)`: 当前价相对历史低点的涨幅
- `枢轴点价格`: 上穿日的基准价格
- `当前涨幅`: 枢轴点到当前的涨幅
- `最小涨幅`: 枢轴点之后的最小涨幅（可能为负）
- `最大涨幅`: 枢轴点之后的最大涨幅
- `上穿期间最大涨幅`: 上穿前后N天的最大涨幅
- `前高价格`: 上穿前N天的最高价
- `前高当前涨幅`: 前高价到当前的涨幅
- `前高最小涨幅`: 前高价到后续最低价的涨幅
- `前高最大涨幅`: 前高价到后续最高价的涨幅

---

## 交易日计数验证

### 测试结论
经过全面测试（`test_trading_days.py`），验证以下结论：

✅ **Selector.py**
- `dmi_period`: 按交易日计数（使用 `rolling`）
- `lookback_days`: 按交易日计数（使用 `iloc` 切片）
- 所有均线计算按交易日计数

✅ **adx_filter.py**
- `pre_high_days`: 按交易日计数（使用 `iloc` 切片）
- `lookback_days`: 按交易日计数（使用 `iloc` 切片）
- 日期过滤按实际交易日过滤

✅ **关键机制**
- `df.iloc[start:end]`: 按行索引切片，天然就是交易日
- `df.rolling(window=N)`: 计算最近N行，即最近N个交易日
- `df[df['date'] > signal_date]`: 过滤出所有大于信号日期的交易日

### 示例验证
```python
# 假设有10个交易日：2025-01-02, 01-03, 01-06, 01-07, ..., 01-15
# （注意：01-04、01-05是周末，不是交易日）

lookback_days = 3
signal_idx = 7  # 第8个交易日

# 获取前3个交易日
start_idx = signal_idx - lookback_days  # = 4
pre_data = df.iloc[4:7]  # 获取索引4、5、6三行

# 结果：得到第5、6、7个交易日（正好3个交易日）
```

---

## 过滤逻辑详解

### 1. 枢轴点价格计算
```python
# 信号日当天的收盘价作为基准
base_price = signal_day['close']
```

### 2. 涨跌幅过滤
```python
# 计算信号日之后的最高价和最低价（收盘价）
max_up_pct = (max_close - base_price) / base_price
max_down_pct = (min_close - base_price) / base_price

# 过滤条件
up_ok = max_up_pct <= maxRange      # 涨幅不超过上限
down_ok = max_down_pct >= minRange   # 跌幅不超过下限（负值）
```

### 3. 上穿期间最大涨幅
```python
# 上穿前后N个交易日的最大涨幅
start_idx = signal_idx - lookback_days
end_idx = signal_idx + lookback_days
period_data = df.iloc[start_idx:end_idx+1]

max_range_up = (max_close - min_open) / min_open

# 过滤条件
crossover_ok = max_range_up >= maxRangeUp  # 期间涨幅达到要求
```

### 4. 前高价格过滤
```python
# 前高价格M：上穿前N个交易日的最高价（开盘或收盘）
start_idx = signal_idx - preHighDays
pre_data = df.iloc[start_idx:signal_idx]
pre_high_price = max(pre_data['open'].max(), pre_data['close'].max())

# 前高最小涨幅K：最高价H之后的最低价J相对于M的涨幅
post_high_date = df[df['close'] == post_max_close]['date'][0]
post_min_close = df[df['date'] >= post_high_date]['close'].min()
pre_high_min_range = (post_min_close - pre_high_price) / pre_high_price

# 前高最大涨幅B：最高价H相对于M的涨幅
pre_high_max_range = (post_max_close - pre_high_price) / pre_high_price

# 过滤条件
pre_high_ok = pre_high_min_range >= preHighMinRange
pre_high_max_ok = pre_high_max_range <= preHighMaxRange
```

### 5. 历史低点涨幅过滤
```python
# 历史低点：最近N个月内的最低价
lookback_date = last_date - pd.DateOffset(months=lookback_months)
window_df = df[df['date'] >= lookback_date]
hist_low = window_df['low'].min()

# 历史低点涨幅
hislow_pct = (current_close - hist_low) / hist_low

# 过滤条件（区间）
hislow_ok = low_bound <= hislow_pct <= high_bound
```

---

## 实战案例

### 案例1：保守型配置
适合稳健投资者，严格控制风险

```json
{
  "selectors": [
    {
      "class": "ADXSelector",
      "alias": "ADX243",
      "params": {
        "dmi_period": 243,
        "lookback_days": 30
      }
    }
  ],
  "ADXfilteringconfig": {
    "active": true,
    "minRange": "0%",          // 不允许跌破枢轴点
    "maxRange": "15%",         // 涨幅不超过15%
    "maxRangeUp": "5%",        // 上穿期间至少涨5%
    "preHighDays": 60,
    "preHighMinRange": "0%",   // 不允许跌破前高
    "preHighMaxRange": "10%"   // 前高之后涨幅不超过10%
  }
}
```

### 案例2：激进型配置
适合短线投资者，追求高收益

```json
{
  "selectors": [
    {
      "class": "PDISelector",
      "alias": "PDI39",
      "params": {
        "dmi_period": 39,
        "lookback_days": 10
      }
    }
  ],
  "PDIfilteringconfig": {
    "active": true,
    "minRange": "-5%",         // 允许5%回调
    "maxRange": "50%",         // 涨幅可达50%
    "maxRangeUp": "none",      // 不限制上穿期间涨幅
    "preHighMinRange": "-10%", // 允许跌破前高10%
    "preHighMaxRange": "none"  // 不限制前高之后涨幅
  }
}
```

### 案例3：平衡型配置
兼顾收益和风险

```json
{
  "selectors": [
    {
      "class": "ADXSelector",
      "alias": "ADX105",
      "params": {
        "dmi_period": 105,
        "lookback_days": 30
      }
    }
  ],
  "ADXfilteringconfig": {
    "active": true,
    "minRange": "-2%",         // 允许小幅回调
    "maxRange": "25%",         // 涨幅控制在25%
    "maxRangeUp": "3%",        // 上穿期间至少涨3%
    "preHighDays": 60,
    "preHighMinRange": "-2%",  // 允许小幅跌破前高
    "preHighMaxRange": "15%",  // 前高之后涨幅15%
    "HisLowPointRange": "16-600%"  // 历史低点涨幅区间
  }
}
```

---

## 常见问题

### Q1: lookback_days 是交易日还是自然日？
**A**: 交易日。所有天数参数都按交易日计数，已通过测试验证。

### Q2: 为什么有的股票没有被选出？
**A**: 可能的原因：
1. 在 `lookback_days` 期间内没有发生上穿信号
2. DMI周期太长，历史数据不足
3. 被后续的过滤条件筛掉

### Q3: 如何调整选股灵敏度？
**A**: 
- 增加灵敏度：减小 `dmi_period`，增大 `lookback_days`
- 降低灵敏度：增大 `dmi_period`，减小 `lookback_days`

### Q4: 多周期如何选择？
**A**:
- **39日**: 适合短线，信号多但噪音也多
- **105日**: 适合中线，信号质量较高
- **243日**: 适合长线，信号少但可靠性高

### Q5: 过滤条件设置为 "none" 表示什么？
**A**: 表示不进行该项过滤，该条件永远为真。

### Q6: 前高价格和枢轴点价格有什么区别？
**A**:
- **前高价格**: 上穿前N个交易日的历史最高价
- **枢轴点价格**: 上穿日当天的收盘价（基准价）

### Q7: 如何理解负值的涨跌幅参数？
**A**:
- `minRange = "0%"`: 不允许跌破枢轴点
- `minRange = "-5%"`: 允许最多跌5%
- `preHighMinRange = "-2%"`: 允许跌破前高最多2%

---

## 性能优化建议

### 1. 并发处理
```bash
# 根据CPU核心数调整workers
python select_stock.py --workers 8
python adx_filter.py --workers 12
```

### 2. 数据管理
- 定期清理旧的结果目录
- K线数据文件不要过大（建议<10MB）
- 使用SSD存储数据文件

### 3. 配置优化
- 只启用需要的selector（设置 `activate: false`）
- 合理设置 `lookback_days`（不要过大）
- 过滤条件不要设置得过于严格（否则可能选不出股票）

---

## 注意事项

1. **数据质量**: 确保K线数据完整、准确，缺失数据会影响DMI计算
2. **回看天数**: `lookback_days` 设置过小可能漏掉信号，过大会增加计算量
3. **参数调优**: 不同市场环境需要不同参数，建议回测验证
4. **过滤逻辑**: 理解每个过滤条件的含义，避免设置矛盾的条件
5. **交易日**: 系统自动处理交易日，不需要手动排除周末节假日
6. **信号日期**: 过滤器中的 `--date` 参数用于修正CSV中的日期，实际计算仍基于K线数据

---

## 已知问题及修复

### Bug #1: 空DataFrame处理 (已修复)

**问题描述**:
在`Selector.py`的`ADXSelector`和`PDISelector`类中，`select()`和`select_with_details()`方法在处理空DataFrame或缺少必要列的DataFrame时，未进行有效性检查就直接访问`date`列，导致`KeyError`异常。

**影响范围**:
- 传入空DataFrame时程序崩溃
- 传入缺少必要列的DataFrame时程序崩溃
- 影响系统鲁棒性

**修复方案**:
在`ADXSelector`和`PDISelector`类中添加`_is_valid_dataframe()`方法：
```python
def _is_valid_dataframe(self, df: pd.DataFrame) -> bool:
    """检查DataFrame是否有效（非空且包含必要列）"""
    if df.empty:
        return False
    required_columns = {'date', 'open', 'high', 'low', 'close'}
    return required_columns.issubset(df.columns)
```

在`select()`和`select_with_details()`方法开头添加检查：
```python
for code, df in data.items():
    # 先检查DataFrame是否有效
    if not self._is_valid_dataframe(df):
        continue
    # ... 其余逻辑
```

**修复验证**: ✅ 已通过全面测试

---

## 更新日志

### v1.1 (2025-10-21)
- 🐛 修复空DataFrame处理Bug
- ✅ 添加DataFrame有效性检查
- ✅ 完成8项全面功能测试（100%通过）
- ✅ 提升系统鲁棒性

### v1.0 (2025-10-21)
- ✅ 完整的ADX和PDI选股功能
- ✅ 多维度过滤系统
- ✅ 交易日计数验证
- ✅ 并发处理支持
- ✅ 详细的日志输出
- ✅ 配置化参数管理

---

**版本**: 1.1  
**日期**: 2025-10-21  
**状态**: ✅ 已实现并测试通过（含交易日计数验证、Bug修复）

