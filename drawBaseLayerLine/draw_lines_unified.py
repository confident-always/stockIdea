#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一版基础层画线脚本
整合所有drawBaseLayerLine功能为单一可执行脚本

功能特性：
1. 从resByFilter中提取所有股票数据
2. 智能数据验证和清洗
3. 多算法阶段低点检测（全局最低点、滑动窗口、价格分位数、技术指标）
4. 高质量图表绘制（K线图、阶段低点线、百分比涨幅线）
5. 多线程批量处理
6. 完整的错误处理和日志记录
7. 命令行参数支持

使用方法：
    # 处理单只股票
    python draw_lines_unified.py --stock 002895
    
    # 批量处理所有股票
    python draw_lines_unified.py --all
    
    # 指定输出目录和线程数
    python draw_lines_unified.py --all --output drawLineRes --workers 4
    
    # 从指定数据目录读取
    python draw_lines_unified.py --all --data-dir ../data
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging
from datetime import datetime
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from scipy.signal import argrelextrema

# 设置matplotlib后端和字体
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，支持多线程
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 字体配置 - 使用macOS系统支持的中文字体
plt.rcParams['font.family'] = ['Heiti TC', 'PingFang HK', 'Arial Unicode MS', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('draw_lines_unified.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 线程锁
progress_lock = threading.Lock()
matplotlib_lock = threading.Lock()  # 用于matplotlib操作的线程安全


class UnifiedLineDrawer:
    """统一版画线器 - 整合所有功能"""
    
    def __init__(self, config_file: str = "lineConfig.json"):
        """初始化统一画线器"""
        self.config_file = config_file
        self.percent_list = self._load_config()
        self.stock_info = self._load_stock_info()
        self.processed_count = 0
        self.total_count = 0
        logger.info(f"✅ 统一画线器初始化完成")
        logger.info(f"📊 加载{len(self.percent_list)}个百分比配置: {self.percent_list}")
        logger.info(f"📈 加载{len(self.stock_info)}只股票信息")
    
    def _load_config(self) -> List[str]:
        """加载配置文件中的百分比数据和ZigZag参数"""
        try:
            config_path = Path(self.config_file)
            if not config_path.exists():
                # 尝试在上级目录查找
                config_path = Path("..") / self.config_file
                if not config_path.exists():
                    logger.warning(f"⚠️ 配置文件 {self.config_file} 不存在，使用默认配置")
                    default_percents = ["3%", "16%", "25%", "34%", "50%", "67%", "128%", "228%", "247%", "323%", "457%", "589%", "636%", "770%", "823%", "935%"]
                    self.zigzag_period = 20
                    self.zigzag_threshold = 0.05
                    return default_percents
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                percent_dic = config.get('percent_dic', [])
                self.zigzag_period = config.get('zigzag_period', 20)
                self.zigzag_threshold = config.get('zigzag_threshold', 0.05)
                logger.info(f"✅ 成功加载配置文件: {config_path}")
                logger.info(f"🔧 ZigZag周期: {self.zigzag_period}, 阈值: {self.zigzag_threshold}")
                return percent_dic
        except Exception as e:
            logger.error(f"❌ 加载配置文件失败: {e}")
            # 使用默认配置
            default_percents = ["3%", "16%", "25%", "34%", "50%", "67%", "128%", "228%", "247%", "323%", "457%", "589%", "636%", "770%", "823%", "935%"]
            self.zigzag_period = 20
            self.zigzag_threshold = 0.05
            logger.info(f"使用默认配置")
            return default_percents
    
    def _load_stock_info(self) -> Dict[str, Dict[str, str]]:
        """从stocklist.csv加载股票信息（名称和行业）"""
        stock_info = {}
        try:
            # 尝试多个可能的stocklist.csv路径
            possible_paths = ["../stocklist.csv", "stocklist.csv", "./stocklist.csv"]
            stocklist_file = None
            
            for path in possible_paths:
                if os.path.exists(path):
                    stocklist_file = path
                    break
            
            if stocklist_file:
                # 从stocklist.csv加载股票信息
                try:
                    df = pd.read_csv(stocklist_file)
                    logger.info(f"📁 从{stocklist_file}加载股票信息")
                    
                    if 'symbol' in df.columns and 'name' in df.columns and 'industry' in df.columns:
                        for _, row in df.iterrows():
                            code = str(row['symbol']).zfill(6)
                            name = str(row['name'])
                            industry = str(row['industry']) if pd.notna(row['industry']) else "未知行业"
                            stock_info[code] = {'name': name, 'industry': industry}
                        
                        logger.info(f"✅ 从stocklist.csv加载股票信息完成，共{len(stock_info)}只股票")
                        return stock_info
                    else:
                        logger.warning(f"⚠️ stocklist.csv缺少必要字段: symbol, name, industry")
                except Exception as e:
                    logger.warning(f"⚠️ 读取stocklist.csv失败: {e}")
            
            # 如果stocklist.csv不可用，回退到从resByFilter目录加载
            logger.info("📁 stocklist.csv不可用，回退到从resByFilter目录加载股票信息")
            possible_paths = ["../resByFilter", "resByFilter", "./resByFilter"]
            res_dir = None
            
            for path in possible_paths:
                if os.path.exists(path):
                    res_dir = path
                    break
            
            if not res_dir:
                logger.warning(f"⚠️ 未找到resByFilter目录，尝试的路径: {possible_paths}")
                return stock_info
            
            csv_files = glob.glob(os.path.join(res_dir, "*.csv"))
            logger.info(f"📁 在{res_dir}中找到{len(csv_files)}个CSV文件")
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if 'code' in df.columns and 'name' in df.columns:
                        for _, row in df.iterrows():
                            code = str(row['code']).zfill(6)
                            name = str(row['name'])
                            stock_info[code] = {'name': name, 'industry': "未知行业"}
                except Exception as e:
                    logger.warning(f"⚠️ 读取文件失败 {csv_file}: {e}")
            
            logger.info(f"✅ 加载股票信息完成，共{len(stock_info)}只股票")
            return stock_info
            
        except Exception as e:
            logger.error(f"❌ 加载股票信息失败: {e}")
            return stock_info
    
    def get_stock_list(self, data_dir: str = "../data") -> List[Tuple[str, str, str]]:
        """获取股票列表，返回(code, name, industry)"""
        stock_list = []
        
        # 如果有股票信息，优先使用
        if self.stock_info:
            for code, info in self.stock_info.items():
                name = info.get('name', code)
                industry = info.get('industry', '未知行业')
                stock_list.append((code, name, industry))
            logger.info(f"📋 从股票信息获取股票列表: {len(stock_list)}只")
            return stock_list
        
        # 否则从数据目录获取
        try:
            data_path = Path(data_dir)
            if not data_path.exists():
                logger.error(f"❌ 数据目录不存在: {data_dir}")
                return stock_list
            
            csv_files = list(data_path.glob("*.csv"))
            for csv_file in csv_files:
                code = csv_file.stem
                # 尝试从文件名推断股票名称，或使用代码作为名称
                name = code
                industry = "未知行业"
                stock_list.append((code, name, industry))
            
            logger.info(f"📋 从数据目录获取股票列表: {len(stock_list)}只")
            return stock_list
            
        except Exception as e:
            logger.error(f"❌ 获取股票列表失败: {e}")
            return stock_list
    
    def validate_and_load_data(self, stock_code: str, data_dir: str) -> Optional[pd.DataFrame]:
        """验证并加载股票数据"""
        try:
            # 标准化股票代码（补零到6位）
            normalized_code = str(stock_code).zfill(6)
            
            # 构建文件路径
            data_path = Path(data_dir)
            csv_file = data_path / f"{normalized_code}.csv"
            
            if not csv_file.exists():
                logger.warning(f"⚠️ 数据文件不存在: {csv_file}")
                return None
            
            # 检查文件大小
            file_size = csv_file.stat().st_size
            if file_size < 1000:  # 小于1KB的文件可能有问题
                logger.warning(f"⚠️ 数据文件过小: {csv_file} ({file_size} bytes)")
                return None
            
            # 读取数据
            df = pd.read_csv(csv_file)
            
            # 验证必要列
            required_columns = ['date', 'open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"⚠️ 缺少必要列 {missing_columns}: {stock_code}")
                return None
            
            # 检查数据行数
            if len(df) < 100:  # 至少需要100天的数据
                logger.warning(f"⚠️ 数据行数不足: {stock_code} ({len(df)} rows)")
                return None
            
            # 数据清洗
            df = df.dropna(subset=required_columns)
            
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # 验证价格数据合理性
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if (df[col] <= 0).any():
                    logger.warning(f"⚠️ 发现非正价格数据: {stock_code}")
                    df = df[df[col] > 0]
            
            # 验证高低价关系
            invalid_rows = (df['high'] < df['low']) | (df['high'] < df['open']) | (df['high'] < df['close']) | \
                          (df['low'] > df['open']) | (df['low'] > df['close'])
            if invalid_rows.any():
                logger.warning(f"⚠️ 发现价格逻辑错误: {stock_code}")
                df = df[~invalid_rows]
            
            if len(df) < 50:  # 清洗后数据太少
                logger.warning(f"⚠️ 清洗后数据不足: {stock_code} ({len(df)} rows)")
                return None
            
            logger.debug(f"✅ 数据验证通过: {stock_code} ({len(df)} rows)")
            return df
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败 {stock_code}: {e}")
            return None
    
    def find_stage_lows_unified(self, df: pd.DataFrame) -> List[Tuple[int, float, str]]:
        """统一版阶段低点检测 - 基于通达信TROUGHBARS算法"""
        try:
            # 从配置文件读取zigzag参数
            zigzag_period = getattr(self, 'zigzag_period', 20)
            zigzag_threshold = getattr(self, 'zigzag_threshold', 0.05)
            
            if len(df) < zigzag_period:
                logger.warning(f"⚠️ 数据不足，需要至少{zigzag_period}个数据点")
                return []
            
            logger.debug(f"🔍 开始TROUGHBARS阶段低点检测: 周期={zigzag_period}, 阈值={zigzag_threshold}")
            
            # 实现通达信TROUGHBARS算法
            def troughbars(low_prices: np.ndarray, period: int) -> np.ndarray:
                """通达信TROUGHBARS函数实现"""
                result = np.zeros(len(low_prices), dtype=int)
                
                for i in range(len(low_prices)):
                    start_idx = max(0, i - period + 1)
                    end_idx = i + 1
                    
                    if end_idx - start_idx < period:
                        result[i] = -1
                        continue
                    
                    window_lows = low_prices[start_idx:end_idx]
                    min_idx_in_window = np.argmin(window_lows)
                    actual_min_idx = start_idx + min_idx_in_window
                    distance = i - actual_min_idx
                    result[i] = distance
                
                return result
            
            # 实现通达信REF和BARSLAST算法
            def find_lowest_price_with_barslast(low_prices: np.ndarray, trough_distances: np.ndarray) -> Tuple[int, float]:
                """实现: 低价1:=REF(L,BARSLAST(最低APP=0))"""
                zero_distance_indices = np.where(trough_distances == 0)[0]
                
                if len(zero_distance_indices) == 0:
                    min_idx = np.argmin(low_prices)
                    logger.debug(f"⚠️ 未找到距离为0的点，使用全局最低点: 索引={min_idx}, 价格={low_prices[min_idx]:.2f}")
                    return min_idx, low_prices[min_idx]
                
                # 找到历史最高价作为参考
                max_high_idx = np.argmax(df['high'].values)
                max_high_price = df.loc[max_high_idx, 'high']
                
                # 计算每个距离为0的点的综合评分（跌幅 + 时间权重）
                best_idx = zero_distance_indices[0]
                best_score = 0
                
                for idx in zero_distance_indices:
                    if idx > max_high_idx:  # 只考虑山峰后的低点
                        decline = (max_high_price - low_prices[idx]) / max_high_price * 100
                        # 时间权重：更近期的低点获得更高权重
                        time_weight = (idx - max_high_idx) / (len(low_prices) - max_high_idx) * 100  # 时间权重0-100
                        score = decline + time_weight  # 综合评分
                        
                        if score > best_score:
                            best_score = score
                            best_idx = idx
                
                # 如果找到了山峰后的低点，使用它
                if best_score > 0:
                    decline = (max_high_price - low_prices[best_idx]) / max_high_price * 100
                    logger.debug(f"✅ 找到山峰后最佳低点: 索引={best_idx}, 价格={low_prices[best_idx]:.2f}, 跌幅={decline:.2f}%, 评分={best_score:.2f}")
                    return best_idx, low_prices[best_idx]
                else:
                    # 如果没有山峰后的低点，使用最后一个距离为0的点
                    last_zero_idx = zero_distance_indices[-1]
                    last_zero_price = low_prices[last_zero_idx]
                    logger.debug(f"✅ 使用最近一次最低点: 索引={last_zero_idx}, 价格={last_zero_price:.2f}")
                    return last_zero_idx, last_zero_price
            
            # 执行TROUGHBARS算法
            low_prices = df['low'].values
            trough_distances = troughbars(low_prices, zigzag_period)
            
            # 找到最近一次最低点
            final_low_idx, final_low_price = find_lowest_price_with_barslast(low_prices, trough_distances)
            final_low_date = df.loc[final_low_idx, "date"]
            
            # 计算从历史最高价的跌幅
            max_high_idx = df['high'].idxmax()
            max_high_price = df.loc[max_high_idx, 'high']
            actual_decline = (max_high_price - final_low_price) / max_high_price * 100
            
            logger.debug(f"✅ TROUGHBARS检测到阶段低点: 日期={final_low_date}, "
                       f"价格={final_low_price:.2f}, 跌幅={actual_decline:.2f}%")
            
            # 格式化日期
            if hasattr(final_low_date, 'strftime'):
                final_low_date_str = final_low_date.strftime("%Y-%m-%d")
            else:
                final_low_date_str = str(final_low_date)
            
            # 返回单一低点
            stage_lows = [(final_low_idx, final_low_price, final_low_date_str)]
            
            logger.debug(f"✅ 最终阶段低点: 日期={final_low_date_str}, 价格={final_low_price:.2f}")
            return stage_lows
            
        except Exception as e:
            logger.error(f"❌ TROUGHBARS阶段低点检测失败: {e}")
            # 备选方案：返回全局最低点
            try:
                global_min_idx = df['low'].idxmin()
                global_min_price = df.loc[global_min_idx, 'low']
                global_min_date = df.loc[global_min_idx, 'date'].strftime('%Y-%m-%d')
                return [(global_min_idx, global_min_price, global_min_date)]
            except:
                return []
    
    def create_unified_chart(self, stock_code: str, stock_name: str, df: pd.DataFrame, 
                           stage_lows: List[Tuple[int, float, str]], output_file: str) -> bool:
        """创建统一版高质量图表 - 使用mplfinance绘制专业K线图"""
        try:
            # 使用线程锁确保matplotlib操作的线程安全
            with matplotlib_lock:
                # 在多线程环境下，确保matplotlib操作的线程安全
                import matplotlib
                matplotlib.use('Agg')  # 确保使用非交互式后端
                
                import mplfinance as mpf
                
                # 获取最低点位置，只显示从最低点开始往后的数据
                if stage_lows:
                    lowest_idx, _, _ = stage_lows[0]  # 获取最低点的索引
                    # 截取从最低点开始的数据
                    df_display = df.iloc[lowest_idx:].copy()
                    # 不要重置索引，保持原始索引
                else:
                    # 如果没有检测到低点，显示全部数据
                    df_display = df.copy()
                
                # 准备mplfinance需要的数据格式
                df_mpf = df_display.copy()
                df_mpf['date'] = pd.to_datetime(df_mpf['date'])
                df_mpf.set_index('date', inplace=True)
                
                # 确保列名符合mplfinance要求
                df_mpf = df_mpf[['open', 'high', 'low', 'close']].copy()
                
                # 检查数据是否为空
                if df_mpf.empty:
                    logger.warning(f"⚠️ 处理后的数据为空: {stock_code}")
                    return False
                
                logger.debug(f"📊 mplfinance数据: {len(df_mpf)} 行, 列: {list(df_mpf.columns)}")
                
                # 准备额外的绘图元素
                additional_plots = []
                
                # 1. 添加阶段低点水平线
                for i, (idx, price, date_str) in enumerate(stage_lows):
                    # 创建水平线数据
                    hline_data = [price] * len(df_mpf)
                    additional_plots.append(mpf.make_addplot(hline_data, color='blue', linestyle='-', width=2, alpha=0.8))
                
                # 2. 添加百分比涨幅线
                if stage_lows:
                    base_price = min(price for _, price, _ in stage_lows)  # 使用最低价作为基准
                    max_price = df_mpf['high'].max()
                    
                    # 先画原有的百分比线，找出K线覆盖范围内最上方的百分比线
                    visible_percent_lines = []
                    for i, percent_str in enumerate(self.percent_list):
                        try:
                            percent = float(percent_str.rstrip('%')) / 100
                            target_price = base_price * (1 + percent)
                            
                            # 所有百分比线都限制在K线方框内（最高价的100%以内）
                            if target_price <= max_price:  # 限制在K线最高价以内
                                visible_percent_lines.append((percent_str, target_price))
                                # 创建水平线数据
                                hline_data = [target_price] * len(df_mpf)
                                additional_plots.append(mpf.make_addplot(hline_data, color='hotpink', linestyle='--', width=3, alpha=0.8))
                        except (ValueError, TypeError):
                            continue
                
                
                # 获取行业信息
                industry = ""
                if stock_code in self.stock_info:
                    industry = self.stock_info[stock_code].get('industry', '')
                
                # 构建标题
                title_parts = [stock_code, stock_name]
                if industry and industry != "未知行业":
                    title_parts.append(f"({industry})")
                title = " ".join(title_parts) + " - Stage Low Points Analysis"
                
                # 设置中文字体
                import matplotlib.pyplot as plt
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                
                # 设置mplfinance样式
                style = mpf.make_mpf_style(
                    base_mpf_style='charles',
                    gridstyle='-',
                    gridcolor='lightgray',
                    y_on_right=True,
                    facecolor='white',
                    edgecolor='black',
                    figcolor='white',
                    rc={'font.size': 12, 'axes.titlesize': 20, 'axes.labelsize': 14, 
                        'font.sans-serif': ['SimHei', 'Arial Unicode MS', 'DejaVu Sans'],
                        'axes.unicode_minus': False}
                )
                
                # 创建图表
                fig, axes = mpf.plot(
                    df_mpf,
                    type='candle',
                    style=style,
                    title=title,
                    ylabel='Price',
                    volume=False,
                    addplot=additional_plots if additional_plots else None,
                    figsize=(20, 12),
                    tight_layout=True,
                    returnfig=True,
                    panel_ratios=(1,),  # 只显示主图
                    show_nontrading=False,  # 不显示非交易日
                    datetime_format='%Y-%m',  # 日期格式
                    xrotation=45  # X轴标签旋转
                )
                
                # 添加价格标注
                ax = axes[0]  # 获取主图轴
                
                # 标注阶段低点价格
                for i, (idx, price, date_str) in enumerate(stage_lows):
                    ax.text(1.02, price, f'{price:.2f}', 
                           fontsize=16, color='blue', fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                           transform=ax.get_yaxis_transform(), ha='left', va='center')
                
                # 标注百分比涨幅线
                if stage_lows:
                    base_price = min(price for _, price, _ in stage_lows)
                    max_price = df_mpf['high'].max()
                    
                    # 标注K线覆盖范围内的百分比线
                    for percent_str in self.percent_list:
                        try:
                            percent = float(percent_str.rstrip('%')) / 100
                            target_price = base_price * (1 + percent)
                            
                            if target_price <= max_price:  # 限制在K线最高价以内
                                ax.text(1.02, target_price, f'+{percent_str}', 
                                       fontsize=18, color='#8B7355', fontweight='bold',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='#8B7355', linewidth=2),
                                       transform=ax.get_yaxis_transform(), ha='left', va='center')
                        except (ValueError, TypeError):
                            continue
                    
                
                # 重新保存带标注的图表
                plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
                
                # 关闭图形以释放内存
                plt.close(fig)
                
                # 调整图片尺寸到精确的目标尺寸
                try:
                    from PIL import Image
                    with Image.open(output_file) as img:
                        # 调整到精确的目标尺寸 (3991 x 2392)
                        target_width = 3991
                        target_height = 2392
                        resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                        resized_img.save(output_file, 'PNG', quality=95)
                        logger.debug(f"✅ 图片尺寸已调整: {target_width}x{target_height}")
                except ImportError:
                    logger.warning("⚠️ PIL未安装，无法调整图片尺寸")
                except Exception as e:
                    logger.warning(f"⚠️ 调整图片尺寸失败: {e}")
                
                # 检查文件是否成功生成
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    if file_size > 1000:  # 至少1KB
                        logger.debug(f"✅ 图表文件生成成功: {output_file} ({file_size} bytes)")
                        return True
                    else:
                        logger.warning(f"⚠️ 生成的图片文件过小: {output_file} ({file_size} bytes)")
                        return False
                else:
                    logger.error(f"❌ 图表文件未生成: {output_file}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ 图表创建失败 {stock_code}: {type(e).__name__}: {e}")
            import traceback
            logger.debug(f"详细错误信息: {traceback.format_exc()}")
            return False
        finally:
            # 确保释放matplotlib资源
            try:
                plt.close('all')  # 关闭所有图形
            except Exception as cleanup_error:
                logger.debug(f"资源清理异常: {cleanup_error}")
                pass
    
    def process_stock_list(self, stock_list: List[Tuple[str, str, str, str]], 
                          output_dir: str = None, data_dir: str = "../data", workers: int = 4):
        """处理指定的股票列表"""
        # 如果未指定输出目录，使用带日期的默认目录
        if output_dir is None:
            current_date = datetime.now().strftime('%Y%m%d')
            output_dir = f'{current_date}-drawLineRes'
        
        logger.info(f"🚀 开始处理指定股票列表")
        logger.info(f"📁 数据目录: {data_dir}")
        logger.info(f"📁 输出目录: {output_dir}")
        logger.info(f"🧵 线程数: {workers}")
        
        if not stock_list:
            logger.error("❌ 股票列表为空")
            return
        
        self.total_count = len(stock_list)
        self.processed_count = 0
        
        logger.info(f"📊 待处理股票数量: {self.total_count}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 多线程处理
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # 提交任务
            future_to_stock = {
                executor.submit(self._process_single_stock, code, name, output_dir, data_dir, file_prefix): (code, name, industry, file_prefix)
                for code, name, industry, file_prefix in stock_list
            }
            
            # 收集结果
            for future in as_completed(future_to_stock):
                result = future.result()
                results.append(result)
        
        # 统计结果
        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r['success'])
        failed_count = len(results) - success_count
        
        logger.info(f"🎉 股票列表处理完成!")
        logger.info(f"📊 总计: {len(results)}只股票")
        logger.info(f"✅ 成功: {success_count}只")
        logger.info(f"❌ 失败: {failed_count}只")
        logger.info(f"⏱️ 总耗时: {total_time:.2f}秒")
        logger.info(f"⚡ 平均速度: {len(results)/total_time:.2f}只/秒")
        
        # 保存处理结果
        results_file = os.path.join(output_dir, "processing_results.json")
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"📄 处理结果已保存: {results_file}")
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {e}")
        
        # 显示失败的股票
        failed_stocks = [r for r in results if not r['success']]
        if failed_stocks:
            logger.warning(f"⚠️ 失败的股票:")
            for r in failed_stocks[:10]:  # 只显示前10个
                logger.warning(f"   {r['stock_code']} {r['stock_name']}: {r['error']}")
            if len(failed_stocks) > 10:
                logger.warning(f"   ... 还有{len(failed_stocks)-10}只股票失败")

    def _process_single_stock(self, stock_code: str, stock_name: str, 
                           output_dir: str, data_dir: str, file_prefix: str = "") -> dict:
        """处理单只股票（内部方法）"""
        start_time = time.time()
        result = {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'success': False,
            'elapsed_time': 0,
            'error': None,
            'stage_lows_count': 0
        }
        
        try:
            # 1. 验证并加载数据
            df = self.validate_and_load_data(stock_code, data_dir)
            if df is None:
                result['error'] = "数据加载失败"
                return result
            
            # 2. 检测阶段低点
            stage_lows = self.find_stage_lows_unified(df)
            if not stage_lows:
                result['error'] = "未检测到阶段低点"
                return result
            
            result['stage_lows_count'] = len(stage_lows)
            
            # 3. 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 4. 生成图表
            # 根据文件前缀生成带前缀的文件名
            if file_prefix and file_prefix != "UNKNOWN":
                output_file = os.path.join(output_dir, f"{file_prefix}_{stock_code}_{stock_name}.png")
            else:
                output_file = os.path.join(output_dir, f"{stock_code}_{stock_name}.png")
            success = self.create_unified_chart(stock_code, stock_name, df, stage_lows, output_file)
            
            if success:
                result['success'] = True
                
                # 更新进度
                with progress_lock:
                    self.processed_count += 1
                    logger.info(f"✅ [{self.processed_count}/{self.total_count}] {stock_code} {stock_name} - {len(stage_lows)}个低点")
            else:
                result['error'] = "图表创建失败"
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ 处理股票失败 {stock_code}: {e}")
        
        finally:
            result['elapsed_time'] = time.time() - start_time
        
        return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="基础层画线脚本 - 读取resByFilter中的股票",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 默认行为：读取当前日期的resByFilter中的股票
  python draw_lines_unified.py
  
  # 读取指定日期的resByFilter中的股票
  python draw_lines_unified.py --date 2025-01-15
  
  # 指定线程数
  python draw_lines_unified.py --workers 6
        """
    )
    
    # 生成带日期的默认输出目录
    current_date = datetime.now().strftime('%Y%m%d')
    
    # 参数
    parser.add_argument('--date', type=str, 
                       help='日期参数，格式为YYYY-MM-DD，用于构建resByFilter目录')
    parser.add_argument('--workers', type=int, default=4,
                       help='并发处理的线程数 (默认: 4)')
    
    args = parser.parse_args()
    
    # 处理日期参数
    if args.date:
        try:
            # 验证日期格式并转换
            date_obj = datetime.strptime(args.date, '%Y-%m-%d')
            date_str = date_obj.strftime('%Y%m%d')
        except ValueError:
            logger.error(f"❌ 日期格式错误: {args.date}，请使用YYYY-MM-DD格式")
            sys.exit(1)
    else:
        date_str = current_date
    
    # 创建统一画线器
    drawer = UnifiedLineDrawer()
    
    # 读取指定日期的resByFilter中的股票
    filter_dir = f"../{date_str}-resByFilter"
    if not os.path.exists(filter_dir):
        logger.error(f"❌ 目录不存在: {filter_dir}")
        logger.info(f"💡 提示：请确保存在 {filter_dir} 目录")
        sys.exit(1)
    
    # 查找所有CSV文件（PDI和ADX结果文件）
    csv_files = glob.glob(os.path.join(filter_dir, "*.csv"))
    if not csv_files:
        logger.error(f"❌ 在目录 {filter_dir} 中未找到CSV文件")
        logger.info(f"💡 提示：请在 {filter_dir} 目录中放置股票列表CSV文件")
        sys.exit(1)
    
    logger.info(f"📁 找到 {len(csv_files)} 个CSV文件")
    
    # 读取所有CSV文件中的股票，并去重
    all_stocks = {}  # 使用字典去重，key为股票代码
    
    for file_path in csv_files:
        logger.info(f"📄 读取文件: {file_path}")
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            
            # 从文件名提取前缀类型（保持原始数字）
            file_name = os.path.basename(file_path)
            file_prefix = ""
            
            # 使用正则表达式提取ADX或PDI后的数字
            import re
            
            # 定义匹配模式
            patterns = [
                (r'^ADX(\d+)', 'ADX'),  # 匹配开头的ADX
                (r'^PDI(\d+)', 'PDI'),  # 匹配开头的PDI
                (r'ADX(\d+)', 'ADX'),    # 匹配任意位置的ADX
                (r'PDI(\d+)', 'PDI')     # 匹配任意位置的PDI
            ]
            
            # 按优先级尝试匹配
            for pattern, prefix_type in patterns:
                match = re.search(pattern, file_name.upper())
                if match:
                    file_prefix = f"{prefix_type}{match.group(1)}"
                    break
            
            logger.info(f"📊 文件类型: {file_prefix}")
            
            # 从CSV文件中提取股票信息
            for _, row in df.iterrows():
                code = str(row.get('code', ''))
                name = str(row.get('name', code))
                industry = str(row.get('industry', '未知行业'))
                
                # 标准化股票代码（补零到6位）
                if code:
                    normalized_code = code.zfill(6)
                    if normalized_code not in all_stocks:
                        # 扩展股票信息，包含文件前缀
                        all_stocks[normalized_code] = (normalized_code, name, industry, file_prefix)
                        
        except Exception as e:
            logger.error(f"❌ 读取文件 {file_path} 失败: {e}")
            continue
    
    if not all_stocks:
        logger.error(f"❌ 未读取到有效的股票数据")
        sys.exit(1)
    
    # 转换为列表
    stock_list = list(all_stocks.values())
    logger.info(f"📋 去重后共有 {len(stock_list)} 只股票")
    
    # 生成输出目录
    output_dir = f"{date_str}-drawLineRes"
    
    # 批量处理股票列表
    drawer.process_stock_list(stock_list, output_dir, "../data", args.workers)
    
    logger.info("🎉 程序执行完成!")


if __name__ == "__main__":
    main()