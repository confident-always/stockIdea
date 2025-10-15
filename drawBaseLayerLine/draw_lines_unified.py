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
        """加载配置文件中的百分比数据"""
        try:
            config_path = Path(self.config_file)
            if not config_path.exists():
                # 尝试在上级目录查找
                config_path = Path("..") / self.config_file
                if not config_path.exists():
                    logger.warning(f"⚠️ 配置文件 {self.config_file} 不存在，使用默认配置")
                    return ["3%", "16%", "25%", "34%", "50%", "67%", "128%", "228%", "247%", "323%", "457%", "589%", "636%", "770%", "823%", "935%"]
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                percent_dic = config.get('percent_dic', [])
                logger.info(f"✅ 成功加载配置文件: {config_path}")
                return percent_dic
        except Exception as e:
            logger.error(f"❌ 加载配置文件失败: {e}")
            # 使用默认配置
            default_percents = ["3%", "16%", "25%", "34%", "50%", "67%", "128%", "228%", "247%", "323%", "457%", "589%", "636%", "770%", "823%", "935%"]
            logger.info(f"使用默认百分比配置")
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
        """统一版阶段低点检测 - 基于历史最高价后的最低点，确保符合山峰定义"""
        try:
            if len(df) < 50:
                logger.warning("⚠️ 数据不足，无法检测阶段低点")
                return []
            
            # 找到历史最高价位置
            max_high_idx = df['high'].idxmax()
            max_high_price = df.loc[max_high_idx, 'high']
            max_high_date = df.loc[max_high_idx, 'date']
            
            # 计算70%跌幅位置（山峰定义的阈值）
            decline_70_threshold = max_high_price * 0.3
            
            # 从历史最高价之后开始寻找最低点
            after_peak_df = df[df.index > max_high_idx].copy()
            
            final_low_idx = None
            final_low_price = float('inf')
            final_low_date = None
            
            if len(after_peak_df) > 0:
                # 在历史最高价之后寻找最低点
                min_idx_after_peak = after_peak_df['low'].idxmin()
                min_price_after_peak = after_peak_df.loc[min_idx_after_peak, 'low']
                min_date_after_peak = after_peak_df.loc[min_idx_after_peak, 'date']
                
                # 检查是否符合70%跌幅条件（山峰定义）
                if min_price_after_peak <= decline_70_threshold:
                    # 符合山峰定义，使用山峰后的最低点
                    final_low_idx = min_idx_after_peak
                    final_low_price = min_price_after_peak
                    final_low_date = min_date_after_peak
                    
                    logger.debug(f"✅ 符合山峰定义: 最高价={max_high_price:.2f}({max_high_date.strftime('%Y-%m-%d')}), "
                               f"70%跌幅阈值={decline_70_threshold:.2f}, 山峰后最低价={final_low_price:.2f}")
                else:
                    # 不符合山峰定义，使用zigzag算法寻找转折点
                    logger.debug(f"⚠️ 不符合山峰定义: 山峰后最低价{min_price_after_peak:.2f} > 70%跌幅阈值{decline_70_threshold:.2f}")
                    
                    # 实现zigzag转折检测算法作为备选方案
                    def detect_zigzag_turning_points(prices: np.ndarray, threshold: float = 0.6) -> List[int]:
                        """检测zigzag转折点，threshold为转折幅度阈值（60%对应0.6）"""
                        if len(prices) < 3:
                            return []
                        
                        turning_points = []
                        current_trend = None  # 'up' or 'down'
                        last_extreme_idx = 0
                        last_extreme_price = prices[0]
                        
                        for i in range(1, len(prices)):
                            current_price = prices[i]
                            
                            # 计算相对于上一个极值点的变化幅度
                            if last_extreme_price > 0:
                                change_ratio = abs(current_price - last_extreme_price) / last_extreme_price
                            else:
                                change_ratio = 0
                            
                            # 检测转折点
                            if change_ratio >= threshold:
                                if current_price > last_extreme_price:
                                    # 上涨超过阈值
                                    if current_trend != 'up':
                                        # 趋势转为上涨，记录前一个低点
                                        if current_trend == 'down':
                                            turning_points.append(last_extreme_idx)
                                        current_trend = 'up'
                                        last_extreme_idx = i
                                        last_extreme_price = current_price
                                else:
                                    # 下跌超过阈值
                                    if current_trend != 'down':
                                        # 趋势转为下跌，记录前一个高点
                                        if current_trend == 'up':
                                            turning_points.append(last_extreme_idx)
                                        current_trend = 'down'
                                        last_extreme_idx = i
                                        last_extreme_price = current_price
                            else:
                                # 更新当前极值点
                                if current_trend == 'up' and current_price > last_extreme_price:
                                    last_extreme_idx = i
                                    last_extreme_price = current_price
                                elif current_trend == 'down' and current_price < last_extreme_price:
                                    last_extreme_idx = i
                                    last_extreme_price = current_price
                        
                        return turning_points
                    
                    # 检测zigzag转折点
                    turning_points = detect_zigzag_turning_points(df['close'].values, threshold=0.6)
                    
                    if turning_points:
                        # 从最后一个转折点开始寻找最低点
                        last_turning_point = turning_points[-1]
                        search_start = max(0, last_turning_point)
                        
                        # 在转折点之后寻找最低点
                        for i in range(search_start, len(df)):
                            current_low = df.loc[i, 'low']
                            if current_low < final_low_price:
                                final_low_price = current_low
                                final_low_idx = i
                        
                        if final_low_idx is not None:
                            final_low_date = df.loc[final_low_idx, 'date']
            
            # 如果仍然没有找到有效的低点，使用全局最低点作为备选
            if final_low_idx is None:
                final_low_idx = df['low'].idxmin()
                final_low_price = df.loc[final_low_idx, 'low']
                final_low_date = df.loc[final_low_idx, 'date']
                logger.debug(f"⚠️ 使用全局最低点作为备选")
            
            # 格式化日期
            final_low_date_str = final_low_date.strftime('%Y-%m-%d')
            
            # 返回单一低点
            stage_lows = [(final_low_idx, final_low_price, final_low_date_str)]
            
            logger.debug(f"✅ 检测到1个最终低点: 日期={final_low_date_str}, 价格={final_low_price:.2f}")
            return stage_lows
            
        except Exception as e:
            logger.error(f"❌ 阶段低点检测失败: {e}")
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
        """创建统一版高质量图表"""
        fig = None
        try:
            # 使用线程锁确保matplotlib操作的线程安全
            with matplotlib_lock:
                # 在多线程环境下，确保matplotlib操作的线程安全
                import matplotlib
                matplotlib.use('Agg')  # 确保使用非交互式后端
                
                # 创建高质量图表
                fig, ax = plt.subplots(figsize=(20, 12), dpi=200)
                
                # 获取最低点位置，只显示从最低点开始往后的数据
                if stage_lows:
                    lowest_idx, _, _ = stage_lows[0]  # 获取最低点的索引
                    # 截取从最低点开始的数据
                    df_display = df.iloc[lowest_idx:].copy()
                    df_display = df_display.reset_index(drop=True)
                else:
                    # 如果没有检测到低点，显示全部数据
                    df_display = df.copy()
                
                # 设置日期格式
                dates = df_display['date']
                
                # 1. 绘制K线图（简化版）
                for i in range(len(df_display)):
                    try:
                        date = dates.iloc[i]
                        open_price = df_display['open'].iloc[i]
                        high_price = df_display['high'].iloc[i]
                        low_price = df_display['low'].iloc[i]
                        close_price = df_display['close'].iloc[i]
                        
                        # 数据验证
                        if pd.isna(open_price) or pd.isna(high_price) or pd.isna(low_price) or pd.isna(close_price):
                            continue
                        if high_price < low_price or high_price <= 0 or low_price <= 0:
                            continue
                        
                        # 确定颜色
                        color = 'red' if close_price >= open_price else 'green'
                        
                        # 绘制高低线
                        ax.plot([date, date], [low_price, high_price], color='black', linewidth=0.5)
                        
                        # 绘制实体（每10根K线绘制一根，提高性能）
                        if i % 10 == 0 or i == len(df_display) - 1:
                            body_height = abs(close_price - open_price)
                            body_bottom = min(open_price, close_price)
                            
                            # 使用矩形绘制实体
                            rect = plt.Rectangle((date, body_bottom), pd.Timedelta(days=1), body_height, 
                                               facecolor=color, alpha=0.7, linewidth=0.5)
                            ax.add_patch(rect)
                    except Exception as e:
                        logger.debug(f"跳过K线数据 {i}: {e}")
                        continue
                
                # 2. 绘制阶段低点水平线（蓝色直线）
                for i, (idx, price, date_str) in enumerate(stage_lows):
                    # 绘制蓝色水平线
                    ax.axhline(y=price, color='blue', linestyle='-', linewidth=2, alpha=0.8)
                    
                    # 标注价格
                    ax.text(dates.iloc[-1], price, f'{price:.2f}', 
                           fontsize=10, color='blue', fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                # 3. 绘制百分比涨幅线（粉红色线段，加粗显示）
                if stage_lows:
                    base_price = min(price for _, price, _ in stage_lows)  # 使用最低价作为基准
                    
                    for i, percent_str in enumerate(self.percent_list):
                        try:
                            percent = float(percent_str.rstrip('%')) / 100
                            target_price = base_price * (1 + percent)
                            
                            # 检查目标价格是否在合理范围内
                            max_price = df_display['high'].max()
                            if target_price <= max_price * 1.5:  # 不超过历史最高价的1.5倍
                                # 绘制粉红色虚线（加粗）
                                ax.axhline(y=target_price, color='hotpink', linestyle='--', linewidth=3, alpha=0.8)
                                
                                # 标注百分比（棕灰色字体，显示在右边）
                                ax.text(dates.iloc[-1], target_price, f'+{percent_str}', 
                                       fontsize=12, color='#8B7355', fontweight='bold',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='#8B7355', linewidth=2),
                                       ha='left', va='center')
                        except (ValueError, TypeError):
                            continue
                
                # 4. 设置图表属性
                # 获取行业信息
                industry = ""
                if stock_code in self.stock_info:
                    industry = self.stock_info[stock_code].get('industry', '')
                
                # 构建标题
                title_parts = [stock_code, stock_name]
                if industry and industry != "未知行业":
                    title_parts.append(f"({industry})")
                title = " ".join(title_parts) + " - Stage Low Points Analysis"
                
                ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Price', fontsize=12)
                
                # 设置Y轴范围，基于显示的数据
                if not df_display.empty:
                    y_min = df_display['low'].min() * 0.95
                    y_max = df_display['high'].max() * 1.1
                    ax.set_ylim(y_min, y_max)
                
                # 设置日期格式
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                
                # 网格
                ax.grid(True, alpha=0.3)
                
                # 自动调整布局
                plt.tight_layout()
                
                # 保存图表
                plt.savefig(output_file, dpi=200, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                
                # 验证输出文件
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    if file_size > 50000:  # 至少50KB
                        logger.debug(f"✅ 图表创建成功: {output_file} ({file_size} bytes)")
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
                if fig is not None:
                    plt.close(fig)
                plt.close('all')  # 关闭所有图形
            except Exception as cleanup_error:
                logger.debug(f"资源清理异常: {cleanup_error}")
                pass
    
    def process_stock_list(self, stock_list: List[Tuple[str, str, str]], 
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
                executor.submit(self._process_single_stock, code, name, output_dir, data_dir): (code, name, industry)
                for code, name, industry in stock_list
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
                           output_dir: str, data_dir: str) -> dict:
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
            
            # 从CSV文件中提取股票信息
            for _, row in df.iterrows():
                code = str(row.get('code', ''))
                name = str(row.get('name', code))
                industry = str(row.get('industry', '未知行业'))
                
                # 标准化股票代码（补零到6位）
                if code:
                    normalized_code = code.zfill(6)
                    if normalized_code not in all_stocks:
                        all_stocks[normalized_code] = (normalized_code, name, industry)
                        
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