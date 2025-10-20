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
                    self.anchor_m_config = {
                        'enabled': True,
                        'zigzag_percent': 10,
                        'pivot_window': 3,
                        'm_range': {'start': 13.0, 'end': 9.0, 'step': -0.1},
                        'max_k': 20,
                        'match_tolerance_ratio': 0.006,
                        'min_matches': 3,
                        'tiebreaker_prefer_higher_M': True,
                        'line_style': {'color': '#8A2BE2', 'linewidth': 1.0, 'alpha': 0.85},
                        'text_style': {'fontsize': 8, 'x_offset': 5},
                        'annotate_format': 'K={K} 价格={price}',
                        'anchor_fallback_window_days': 60
                    }
                    return default_percents
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                percent_dic = config.get('percent_dic', [])
                self.zigzag_period = config.get('zigzag_period', 20)
                self.zigzag_threshold = config.get('zigzag_threshold', 0.05)
                
                # 加载 anchorMLines 配置
                self.anchor_m_config = config.get('anchorMLines', {})
                if not self.anchor_m_config:
                    # 默认配置
                    self.anchor_m_config = {
                        'enabled': True,
                        'zigzag_percent': 10,
                        'pivot_window': 3,
                        'm_range': {'start': 13.0, 'end': 9.0, 'step': -0.1},
                        'max_k': 20,
                        'match_tolerance_ratio': 0.006,
                        'min_matches': 3,
                        'tiebreaker_prefer_higher_M': True,
                        'line_style': {'color': '#8A2BE2', 'linewidth': 1.0, 'alpha': 0.85},
                        'text_style': {'fontsize': 8, 'x_offset': 5},
                        'annotate_format': 'K={K} 价格={price}',
                        'anchor_fallback_window_days': 60
                    }
                
                logger.info(f"✅ 成功加载配置文件: {config_path}")
                logger.info(f"🔧 ZigZag周期: {self.zigzag_period}, 阈值: {self.zigzag_threshold}")
                logger.info(f"🔧 AnchorMLines功能: {'启用' if self.anchor_m_config.get('enabled', True) else '禁用'}")
                return percent_dic
        except Exception as e:
            logger.error(f"❌ 加载配置文件失败: {e}")
            # 使用默认配置
            default_percents = ["3%", "16%", "25%", "34%", "50%", "67%", "128%", "228%", "247%", "323%", "457%", "589%", "636%", "770%", "823%", "935%"]
            self.zigzag_period = 20
            self.zigzag_threshold = 0.05
            self.anchor_m_config = {
                'enabled': True,
                'zigzag_percent': 10,
                'pivot_window': 3,
                'm_range': {'start': 13.0, 'end': 9.0, 'step': -0.1},
                'max_k': 20,
                'match_tolerance_ratio': 0.006,
                'min_matches': 3,
                'tiebreaker_prefer_higher_M': True,
                'line_style': {'color': '#8A2BE2', 'linewidth': 1.0, 'alpha': 0.85},
                'text_style': {'fontsize': 8, 'x_offset': 5},
                'annotate_format': 'K={K} 价格={price}',
                'anchor_fallback_window_days': 60
            }
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
    
    def zigzag(self, high_prices: np.ndarray, low_prices: np.ndarray, 
               threshold_pct: float = 0.49) -> List[Tuple[int, float, str]]:
        """
        实现ZigZag指标算法，找到显著的转折点
        
        ZigZag算法原理：
        - 只有当价格变化超过设定阈值时才确认转折点
        - 过滤掉小幅波动，保留主要趋势
        - threshold_pct: 49% 表示价格变化需超过49%才确认转折
        
        Args:
            high_prices: 最高价数组
            low_prices: 最低价数组
            threshold_pct: 转折阈值（百分比，如0.49表示49%）
        
        Returns:
            转折点列表 [(索引, 价格, 类型)]，类型为'high'或'low'
        """
        if len(high_prices) < 3:
            return []
        
        pivots = []  # 存储转折点
        
        # 从第一个点开始
        last_pivot_idx = 0
        last_pivot_price = low_prices[0]
        last_pivot_type = 'low'  # 假设从低点开始
        
        # 初始化：找到真正的第一个转折点
        # 先找第一个高点
        searching_for = 'high'
        
        for i in range(1, len(high_prices)):
            if searching_for == 'high':
                # 寻找高点
                current_high = high_prices[i]
                # 计算从最后一个低点到当前的涨幅
                if last_pivot_type == 'low':
                    pct_change = (current_high - last_pivot_price) / last_pivot_price
                    if pct_change >= threshold_pct:
                        # 找到一个显著的高点
                        pivots.append((last_pivot_idx, last_pivot_price, 'low'))
                        last_pivot_idx = i
                        last_pivot_price = current_high
                        last_pivot_type = 'high'
                        searching_for = 'low'
                else:
                        # 更新潜在的起点（如果找到更低的低点）
                        if low_prices[i] < last_pivot_price:
                            last_pivot_idx = i
                            last_pivot_price = low_prices[i]
                            
            else:  # searching_for == 'low'
                # 寻找低点
                current_low = low_prices[i]
                # 计算从最后一个高点到当前的跌幅
                if last_pivot_type == 'high':
                    pct_change = (last_pivot_price - current_low) / last_pivot_price
                    if pct_change >= threshold_pct:
                        # 找到一个显著的低点
                        pivots.append((last_pivot_idx, last_pivot_price, 'high'))
                        last_pivot_idx = i
                        last_pivot_price = current_low
                        last_pivot_type = 'low'
                        searching_for = 'high'
                    else:
                        # 更新潜在的高点（如果找到更高的高点）
                        if high_prices[i] > last_pivot_price:
                            last_pivot_idx = i
                            last_pivot_price = high_prices[i]
        
        # 添加最后一个转折点
        if pivots and last_pivot_idx != pivots[-1][0]:
            pivots.append((last_pivot_idx, last_pivot_price, last_pivot_type))
        
        return pivots

    def troughbars(self, data: np.ndarray, period: int, n: int) -> np.ndarray:
        """
        实现通达信TROUGHBARS函数
        TROUGHBARS(X,N,M) 返回N周期内X的第M个波谷到当前位置的周期数
        
        Args:
            data: 价格数据数组 (通常是最低价)
            period: 查找周期 N  
            n: 第几个波谷 M
        
        Returns:
            每个位置到第n个波谷的距离数组
        """
        result = np.full(len(data), np.nan)
        
        for i in range(len(data)):
            # 获取当前位置前N个周期的数据（包含当前位置）
            start_idx = max(0, i - period + 1)
            end_idx = i + 1
            window_data = data[start_idx:end_idx]
            
            if len(window_data) < 3:  # 至少需要3个点才能找到波谷
                continue
            
            # 寻找波谷（局部最小值）
            troughs = []
            
            # 检查窗口内的每个点是否为波谷
            for j in range(len(window_data)):
                actual_idx = start_idx + j
                
                # 边界处理：第一个和最后一个点也可能是波谷
                is_trough = False
                
                if j == 0:  # 第一个点
                    if len(window_data) > 1 and window_data[j] <= window_data[j+1]:
                        is_trough = True
                elif j == len(window_data) - 1:  # 最后一个点（当前点）
                    if window_data[j] <= window_data[j-1]:
                        is_trough = True
                else:  # 中间点
                    if window_data[j] <= window_data[j-1] and window_data[j] <= window_data[j+1]:
                        is_trough = True
                
                if is_trough:
                    troughs.append((actual_idx, window_data[j]))
            
            # 按价格排序，找到第n个最低的波谷
            if len(troughs) >= n:
                troughs.sort(key=lambda x: x[1])  # 按价格从低到高排序
                nth_trough_idx = troughs[n-1][0]  # 第n个波谷的索引
                result[i] = i - nth_trough_idx  # 距离当前位置的周期数
            
        return result

    def barslast(self, condition: np.ndarray) -> np.ndarray:
        """
        实现通达信BARSLAST函数
        BARSLAST(X) 返回上一次X条件成立到当前的周期数
        
        Args:
            condition: 布尔条件数组
        
        Returns:
            距离上次条件成立的周期数数组
        """
        result = np.full(len(condition), np.nan)
        last_true_idx = -1
        
        for i in range(len(condition)):
            if condition[i]:
                last_true_idx = i
                result[i] = 0
            elif last_true_idx >= 0:
                result[i] = i - last_true_idx
        
        return result

    def ref(self, data: np.ndarray, periods: np.ndarray) -> np.ndarray:
        """
        实现通达信REF函数
        REF(X,A) 引用A周期前的X值
        
        Args:
            data: 数据数组
            periods: 引用周期数数组
        
        Returns:
            引用的历史数据数组
        """
        result = np.full(len(data), np.nan)
        
        for i in range(len(data)):
            if not np.isnan(periods[i]):
                ref_idx = int(i - periods[i])
                if 0 <= ref_idx < len(data):
                    result[i] = data[ref_idx]
        
        return result

    # ==================== AnchorM Lines 功能函数 ====================
    
    def compute_zigzag_small(self, highs: np.ndarray, lows: np.ndarray, 
                             threshold_pct: float) -> List[Tuple[int, float, str]]:
        """
        计算小级别ZigZag转折点(用于M线评分)
        
        Args:
            highs: 最高价数组
            lows: 最低价数组
            threshold_pct: 阈值百分比(如0.10表示10%)
        
        Returns:
            转折点列表 [(索引, 价格, 类型)]
        """
        return self.zigzag(highs, lows, threshold_pct)
    
    def get_local_extremes_around_turns(self, highs: np.ndarray, lows: np.ndarray,
                                       turn_indices: List[int], window: int) -> List[Tuple[float, int]]:
        """
        获取ZigZag拐点附近的局部极值
        
        Args:
            highs: 最高价数组
            lows: 最低价数组
            turn_indices: 拐点索引列表
            window: 窗口大小(±window)
        
        Returns:
            极值列表 [(价格, 索引)]
        """
        extremes = []
        
        for turn_idx in turn_indices:
            start_idx = max(0, turn_idx - window)
            end_idx = min(len(highs), turn_idx + window + 1)
            
            # 获取窗口内的最高价和最低价
            window_highs = highs[start_idx:end_idx]
            window_lows = lows[start_idx:end_idx]
            
            # 局部最高价
            local_max_idx = start_idx + np.argmax(window_highs)
            local_max_price = highs[local_max_idx]
            extremes.append((local_max_price, local_max_idx))
            
            # 局部最低价
            local_min_idx = start_idx + np.argmin(window_lows)
            local_min_price = lows[local_min_idx]
            extremes.append((local_min_price, local_min_idx))
        
        # 去重(相同价格和索引)
        extremes = list(set(extremes))
        extremes.sort(key=lambda x: x[0])  # 按价格排序
        
        return extremes
    
    def generate_B_series(self, A: float, M: float, max_k: int, 
                          max_price: float) -> Tuple[List[float], List[int]]:
        """
        生成B序列: B_k = A + (A × M) × k
        
        Args:
            A: 锚点低价
            M: M值(百分比,如0.127表示12.7%)
            max_k: 最大K值
            max_price: 最高价上沿(用于判断停止)
        
        Returns:
            (B值列表, K值列表)
        """
        N = A * M
        B_values = []
        K_values = []
        
        for k in range(1, max_k + 1):
            B_k = A + N * k
            if B_k > max_price * 1.01:  # 超过最高价+1%
                break
            B_values.append(B_k)
            K_values.append(k)
        
        return B_values, K_values
    
    def score_M(self, B_values: List[float], extremes: List[Tuple[float, int]], 
                match_tolerance_ratio: float) -> Dict:
        """
        对某个M值进行评分
        
        Args:
            B_values: B序列
            extremes: 极值列表 [(价格, 索引)]
            match_tolerance_ratio: 匹配容差比(如0.006表示0.6%)
        
        Returns:
            {'avg_score': float, 'matches_count': int, 'per_k_matches': List}
        """
        if not B_values or not extremes:
            return {'avg_score': 0, 'matches_count': 0, 'per_k_matches': []}
        
        extreme_prices = [e[0] for e in extremes]
        extreme_prices_sorted = sorted(extreme_prices)
        
        scores = []
        per_k_matches = []
        
        for k_idx, B_k in enumerate(B_values):
            # 找到与B_k最接近的两个极值(一上一下)
            upper = None
            lower = None
            
            for e_price in extreme_prices_sorted:
                if e_price >= B_k:
                    upper = e_price
                    break
            
            for e_price in reversed(extreme_prices_sorted):
                if e_price <= B_k:
                    lower = e_price
                    break
            
            # 计算得分
            selected_extremes = []
            if upper is not None:
                selected_extremes.append(upper)
            if lower is not None and lower != upper:
                selected_extremes.append(lower)
            
            if not selected_extremes:
                # 如果没有匹配的极值,取最近的一个
                distances = [(abs(e_price - B_k), e_price) for e_price in extreme_prices]
                distances.sort()
                if distances:
                    selected_extremes.append(distances[0][1])
            
            # 计算该B_k的得分
            k_scores = []
            for e_price in selected_extremes:
                r = abs(e_price - B_k) / B_k
                s_e = 100 * max(0, 1 - min(r / match_tolerance_ratio, 1))
                k_scores.append(s_e)
            
            if k_scores:
                avg_k_score = sum(k_scores) / len(k_scores)
                scores.append(avg_k_score)
                per_k_matches.append({
                    'k': k_idx + 1,
                    'B_k': B_k,
                    'matched_extremes': selected_extremes,
                    'score': avg_k_score
                })
        
        avg_score = sum(scores) / len(scores) if scores else 0
        matches_count = len([s for s in scores if s > 0])
        
        return {
            'avg_score': avg_score,
            'matches_count': matches_count,
            'per_k_matches': per_k_matches
        }
    
    def select_best_M(self, M_results: Dict[float, Dict], min_matches: int,
                     prefer_higher_M: bool = True) -> Tuple[Optional[float], Optional[Dict]]:
        """
        从所有M候选中选择最佳M
        
        Args:
            M_results: {M值: 评分结果}字典
            min_matches: 最小匹配数要求
            prefer_higher_M: 并列时优先选择更大的M
        
        Returns:
            (最佳M值, 最佳结果详情)
        """
        # 过滤匹配数不足的M
        valid_M = {M: result for M, result in M_results.items() 
                   if result['matches_count'] >= min_matches}
        
        if not valid_M:
            return None, None
        
        # 按平均分排序
        sorted_M = sorted(valid_M.items(), 
                         key=lambda x: (x[1]['avg_score'], 
                                       x[1]['matches_count'],
                                       x[0] if prefer_higher_M else -x[0]),
                         reverse=True)
        
        best_M, best_result = sorted_M[0]
        return best_M, best_result
    
    def compute_anchor_M_lines(self, df: pd.DataFrame, anchor_low: float, 
                              anchor_date: pd.Timestamp) -> Optional[Dict]:
        """
        计算最佳M值与B序列
        
        Args:
            df: K线数据
            anchor_low: 锚点低价A
            anchor_date: 锚定日期
        
        Returns:
            最佳M线结果字典,失败返回None
        """
        try:
            config = self.anchor_m_config
            
            if not config.get('enabled', True):
                return None
            
            # 获取锚定日期之后的数据
            df_after = df[df['date'] > anchor_date].copy()
            
            if len(df_after) < 10:
                logger.debug(f"⚠️ 锚定日期之后数据不足: {len(df_after)}天")
                return None
            
            # 计算小级别ZigZag
            zigzag_percent = config.get('zigzag_percent', 10) / 100.0
            highs_after = df_after['high'].values
            lows_after = df_after['low'].values
            
            turns = self.compute_zigzag_small(highs_after, lows_after, zigzag_percent)
            
            if not turns:
                logger.debug(f"⚠️ 锚定日期后未找到ZigZag转折点")
                return None
            
            # 获取拐点索引(相对于df_after)
            turn_indices = [t[0] for t in turns]
            
            # 获取局部极值
            pivot_window = config.get('pivot_window', 3)
            extremes = self.get_local_extremes_around_turns(
                highs_after, lows_after, turn_indices, pivot_window
            )
            
            if not extremes:
                logger.debug(f"⚠️ 未找到局部极值")
                return None
            
            # 遍历M值
            m_range = config.get('m_range', {'start': 13.0, 'end': 9.0, 'step': -0.1})
            M_start = m_range['start']
            M_end = m_range['end']
            M_step = abs(m_range['step'])
            
            M_values = []
            M_current = M_start
            while M_current >= M_end - 0.001:  # 浮点数容差
                M_values.append(M_current)
                M_current -= M_step
            
            max_k = config.get('max_k', 20)
            max_price = df_after['high'].max()
            match_tolerance = config.get('match_tolerance_ratio', 0.006)
            
            M_results = {}
            
            for M_pct in M_values:
                M = M_pct / 100.0  # 转换为小数
                B_values, K_values = self.generate_B_series(anchor_low, M, max_k, max_price)
                
                if not B_values:
                    continue
                
                score_result = self.score_M(B_values, extremes, match_tolerance)
                M_results[M_pct] = {
                    'B_values': B_values,
                    'K_values': K_values,
                    'avg_score': score_result['avg_score'],
                    'matches_count': score_result['matches_count'],
                    'per_k_matches': score_result['per_k_matches']
                }
            
            # 选择最佳M
            min_matches = config.get('min_matches', 3)
            prefer_higher_M = config.get('tiebreaker_prefer_higher_M', True)
            
            best_M, best_result = self.select_best_M(M_results, min_matches, prefer_higher_M)
            
            if best_M is None:
                logger.debug(f"⚠️ 未找到满足条件的M值(最小匹配数={min_matches})")
                return None
            
            logger.debug(f"✅ 最佳M={best_M:.1f}%, 平均分={best_result['avg_score']:.2f}, "
                        f"匹配数={best_result['matches_count']}")
            
            return {
                'best_M': best_M,
                'B_values': best_result['B_values'],
                'K_values': best_result['K_values'],
                'avg_score': best_result['avg_score'],
                'matches_count': best_result['matches_count'],
                'per_k_matches': best_result['per_k_matches'],
                'anchor_low': anchor_low,
                'anchor_date': anchor_date,
                'extremes': extremes
            }
            
        except Exception as e:
            logger.error(f"❌ 计算AnchorM线失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def find_stage_lows_unified(self, df: pd.DataFrame) -> List[Tuple[int, float, str]]:
        """
        统一版阶段低点检测 - 使用ZigZag(L,49)算法
        参数49从lineConfig.json的zigzag_period读取，表示49%的价格变化阈值
        """
        try:
            # 将zigzag_period转换为百分比阈值（49 -> 0.49）
            threshold_pct = self.zigzag_period / 100.0
            logger.debug(f"🔍 开始ZigZag阶段低点检测 (阈值={self.zigzag_period}%)")
            
            # 准备数据
            high_prices = df['high'].values
            low_prices = df['low'].values
            dates = df['date'].values
            
            # 1. 使用ZigZag算法找到所有转折点
            pivots = self.zigzag(high_prices, low_prices, threshold_pct)
            
            stage_lows = []
            
            if not pivots:
                logger.warning("⚠️ ZigZag未找到转折点，使用全局最低点")
                min_idx = df['low'].idxmin()
                min_price = df.loc[min_idx, 'low']
                min_date = df.loc[min_idx, 'date']
                
                if hasattr(min_date, 'strftime'):
                    min_date_str = min_date.strftime("%Y-%m-%d")
                else:
                    min_date_str = str(min_date)
                
                stage_lows = [(min_idx, min_price, min_date_str)]
            else:
                # 2. 从ZigZag转折点中筛选出低点（'low'类型）
                low_pivots = [(idx, price, pivot_type) for idx, price, pivot_type in pivots if pivot_type == 'low']
                
                if not low_pivots:
                    logger.warning("⚠️ ZigZag未找到低点转折，使用全局最低点")
                    min_idx = df['low'].idxmin()
                    min_price = df.loc[min_idx, 'low']
                    min_date = df.loc[min_idx, 'date']
                else:
                    # 3. 使用最后一个（最近的）低点转折作为初始阶段低点
                    idx, price, _ = low_pivots[-1]
                    low_date = df.loc[idx, 'date']
                    
                    logger.debug(f"✅ ZigZag找到 {len(low_pivots)} 个低点转折")
                    logger.debug(f"✅ ZigZag最近低点: 索引={idx}, 价格={price:.2f}")
                    
                    # 4. 优化低点锚定：检查该低点之后是否有更低的价格
                    # 在该低点之后的所有交易日中查找更低的价格
                    if idx < len(df) - 1:  # 如果不是最后一个交易日
                        after_low_df = df.iloc[idx+1:]  # 获取该低点之后的数据
                        
                        # 查找之后的最低价
                        after_min_idx = after_low_df['low'].idxmin()
                        after_min_price = after_low_df.loc[after_min_idx, 'low']
                        
                        # 如果之后有更低的价格，使用该更低价格
                        if after_min_price < price:
                            logger.debug(f"🔽 发现更低价格: 原价格={price:.2f}, 新价格={after_min_price:.2f}")
                            idx = after_min_idx
                            price = after_min_price
                            low_date = df.loc[after_min_idx, 'date']
                            logger.debug(f"✅ 更新锚定低点: 索引={idx}, 日期={low_date}, 价格={price:.2f}")
                    
                    min_idx = idx
                    min_price = price
                    min_date = low_date
                
                # 格式化日期
                if hasattr(min_date, 'strftime'):
                    min_date_str = min_date.strftime("%Y-%m-%d")
                else:
                    min_date_str = str(min_date)
                
                stage_lows = [(min_idx, min_price, min_date_str)]
            
            logger.debug(f"✅ 最终阶段低点: 索引={stage_lows[0][0]}, 日期={stage_lows[0][2]}, 价格={stage_lows[0][1]:.2f}")
            return stage_lows
            
        except Exception as e:
            logger.error(f"❌ ZigZag阶段低点检测失败: {e}")
            import traceback
            logger.debug(f"详细错误信息: {traceback.format_exc()}")
            
            # 备选方案：返回全局最低点
            try:
                global_min_idx = df['low'].idxmin()
                global_min_price = df.loc[global_min_idx, 'low']
                global_min_date = df.loc[global_min_idx, 'date']
                
                if hasattr(global_min_date, 'strftime'):
                    global_min_date_str = global_min_date.strftime('%Y-%m-%d')
                else:
                    global_min_date_str = str(global_min_date)
                
                return [(global_min_idx, global_min_price, global_min_date_str)]
            except Exception as backup_e:
                logger.error(f"❌ 备选方案也失败: {backup_e}")
                return []
    
    def create_unified_chart(self, stock_code: str, stock_name: str, df: pd.DataFrame, 
                           stage_lows: List[Tuple[int, float, str]], output_file: str) -> Tuple[bool, Optional[Dict]]:
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
                    
                    # 先画K线覆盖范围内的百分比线，找出最上方的百分比线
                    visible_percent_lines = []
                    highest_visible_idx = -1  # 记录最高可见百分比线的索引
                    
                    for i, percent_str in enumerate(self.percent_list):
                        try:
                            percent = float(percent_str.rstrip('%')) / 100
                            target_price = base_price * (1 + percent)
                            
                            # K线覆盖范围内的百分比线
                            if target_price <= max_price:
                                visible_percent_lines.append((percent_str, target_price))
                                highest_visible_idx = i  # 更新最高可见线索引
                                # 创建水平线数据
                                hline_data = [target_price] * len(df_mpf)
                                additional_plots.append(mpf.make_addplot(hline_data, color='hotpink', linestyle='--', width=3, alpha=0.8))
                        except (ValueError, TypeError):
                            continue
                
                    # 在K线覆盖不到的区域再画一根百分比线（如果还有下一根）
                    if highest_visible_idx >= 0 and highest_visible_idx + 1 < len(self.percent_list):
                        try:
                            next_percent_str = self.percent_list[highest_visible_idx + 1]
                            next_percent = float(next_percent_str.rstrip('%')) / 100
                            next_target_price = base_price * (1 + next_percent)
                            
                            # 画出K线上方的下一根百分比线
                            hline_data = [next_target_price] * len(df_mpf)
                            additional_plots.append(mpf.make_addplot(hline_data, color='hotpink', linestyle='--', width=3, alpha=0.8))
                            visible_percent_lines.append((next_percent_str, next_target_price))
                            
                            logger.debug(f"✅ 在K线上方添加额外百分比线: +{next_percent_str}")
                        except (ValueError, TypeError):
                            pass
                
                
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
                
                # 设置mplfinance样式 - 中国标准配色：红涨绿跌
                style = mpf.make_mpf_style(
                    base_mpf_style='charles',
                    marketcolors=mpf.make_marketcolors(
                        up='red',        # 上涨为红色
                        down='green',    # 下跌为绿色
                        edge='inherit',  # 边框颜色继承蜡烛颜色
                        wick='inherit',  # 影线颜色继承蜡烛颜色
                        volume='inherit' # 成交量颜色继承蜡烛颜色
                    ),
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
                
                # 计算需要调整的Y轴范围
                if stage_lows:
                    base_price = min(price for _, price, _ in stage_lows)
                    max_price = df_mpf['high'].max()
                    min_price = df_mpf['low'].min()
                    
                    # 计算最高的百分比线价格（包括额外的一根）
                    highest_percent_price = max_price
                    highest_visible_idx = -1
                    
                    for i, percent_str in enumerate(self.percent_list):
                        try:
                            percent = float(percent_str.rstrip('%')) / 100
                            target_price = base_price * (1 + percent)
                            if target_price <= max_price:
                                highest_visible_idx = i
                                highest_percent_price = target_price
                        except (ValueError, TypeError):
                            continue
                    
                    # 如果有额外的百分比线，计算其价格
                    if highest_visible_idx >= 0 and highest_visible_idx + 1 < len(self.percent_list):
                        try:
                            next_percent_str = self.percent_list[highest_visible_idx + 1]
                            next_percent = float(next_percent_str.rstrip('%')) / 100
                            next_target_price = base_price * (1 + next_percent)
                            highest_percent_price = next_target_price
                        except (ValueError, TypeError):
                            pass
                    
                    # 调整Y轴范围，确保最高的百分比线在方框内
                    # 留出一些上下边距（约5%）
                    y_margin = (highest_percent_price - min_price) * 0.05
                    ax.set_ylim(min_price - y_margin, highest_percent_price + y_margin)
                    
                    logger.debug(f"📊 调整Y轴范围: {min_price:.2f} - {highest_percent_price:.2f}")
                
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
                    
                    # 标注K线覆盖范围内的百分比线，并找出最高的
                    highest_visible_idx = -1
                    for i, percent_str in enumerate(self.percent_list):
                        try:
                            percent = float(percent_str.rstrip('%')) / 100
                            target_price = base_price * (1 + percent)
                            
                            if target_price <= max_price:  # K线覆盖范围内
                                highest_visible_idx = i
                                ax.text(1.02, target_price, f'+{percent_str}', 
                                       fontsize=18, color='#8B7355', fontweight='bold',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='#8B7355', linewidth=2),
                                       transform=ax.get_yaxis_transform(), ha='left', va='center')
                        except (ValueError, TypeError):
                            continue
                    
                    # 标注K线上方的额外百分比线
                    if highest_visible_idx >= 0 and highest_visible_idx + 1 < len(self.percent_list):
                        try:
                            next_percent_str = self.percent_list[highest_visible_idx + 1]
                            next_percent = float(next_percent_str.rstrip('%')) / 100
                            next_target_price = base_price * (1 + next_percent)
                            
                            ax.text(1.02, next_target_price, f'+{next_percent_str}', 
                                   fontsize=18, color='#8B7355', fontweight='bold',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='#8B7355', linewidth=2),
                                   transform=ax.get_yaxis_transform(), ha='left', va='center')
                        except (ValueError, TypeError):
                            pass
                
                # ==================== 绘制AnchorM线 ====================
                m_lines_result = None
                if self.anchor_m_config.get('enabled', True) and stage_lows:
                    try:
                        # 直接使用find_stage_lows_unified计算的阶段低点作为锚定低点
                        # 该低点已经通过ZigZag(L,49)算法和低点优化逻辑处理过了
                        anchor_idx, anchor_low, anchor_date = stage_lows[0]
                        logger.debug(f"📍 AnchorM锚点: 日期={anchor_date}, 价格={anchor_low:.2f}")
                        
                        # 计算最佳M值和B序列
                        m_lines_result = self.compute_anchor_M_lines(df, anchor_low, anchor_date)
                        
                        if m_lines_result:
                            best_M = m_lines_result['best_M']
                            B_values = m_lines_result['B_values']
                            K_values = m_lines_result['K_values']
                            per_k_matches = m_lines_result['per_k_matches']
                            
                            # 绘制紫色横线(用于评分的极值)
                            line_style = self.anchor_m_config.get('line_style', {})
                            line_color = line_style.get('color', '#8A2BE2')
                            line_width = line_style.get('linewidth', 1.0)
                            line_alpha = line_style.get('alpha', 0.85)
                            
                            text_style = self.anchor_m_config.get('text_style', {})
                            text_fontsize = text_style.get('fontsize', 8)
                            annotate_format = self.anchor_m_config.get('annotate_format', 'K={K} 价格={price}')
                            
                            # 绘制紫色横线 - 每个K值对应一条线(B_k价格)
                            # 使用最佳M值对应的B序列和K值
                            for k_val, B_k_price in zip(K_values, B_values):
                                # 绘制横线(加粗) - 使用B_k的价格
                                ax.axhline(y=B_k_price, color=line_color, 
                                          linestyle='-', linewidth=line_width, 
                                          alpha=line_alpha, zorder=2.5)
                                
                                # 标注价格和K值(放在左边,加粗)
                                label_text = annotate_format.replace('{K}', str(k_val)).replace('{price}', f'{B_k_price:.2f}')
                                ax.text(-0.02, B_k_price, label_text,
                                       fontsize=text_fontsize, color=line_color, fontweight='bold',
                                       bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.85, edgecolor=line_color, linewidth=2),
                                       transform=ax.get_yaxis_transform(), ha='right', va='center')
                            
                            # 在图片左上角添加M值和B序列信息
                            # 构建文本内容
                            text_lines = [f"M={best_M:.1f}%"]
                            # B序列可能很长,限制显示前10个
                            B_display = [f'{b:.2f}' for b in B_values[:10]]
                            if len(B_values) > 10:
                                B_display.append('...')
                            text_lines.append(f"B: [{', '.join(B_display)}]")
                            text_lines.append(f"Score: {m_lines_result['avg_score']:.1f}")
                            text_lines.append(f"Matches: {m_lines_result['matches_count']}")
                            
                            # 在左上角添加文本框(放大字体)
                            text_content = '\n'.join(text_lines)
                            ax.text(0.01, 0.98, text_content,
                                   transform=ax.transAxes,
                                   fontsize=12, color='purple', fontweight='bold',
                                   bbox=dict(boxstyle="round,pad=0.6", facecolor='white', alpha=0.95, edgecolor='purple', linewidth=2.5),
                                   ha='left', va='top', family='monospace')
                            
                            logger.debug(f"✅ 已绘制AnchorM线: M={best_M:.1f}%, {len(B_values)}条B_k线")
                        else:
                            logger.debug(f"⚠️ 未能计算AnchorM线")
                    except Exception as e:
                        logger.warning(f"⚠️ 绘制AnchorM线时出错: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                
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
                        return True, m_lines_result
                    else:
                        logger.warning(f"⚠️ 生成的图片文件过小: {output_file} ({file_size} bytes)")
                        return False, None
                else:
                    logger.error(f"❌ 图表文件未生成: {output_file}")
                    return False, None
                    
        except Exception as e:
            logger.error(f"❌ 图表创建失败 {stock_code}: {type(e).__name__}: {e}")
            import traceback
            logger.debug(f"详细错误信息: {traceback.format_exc()}")
            return False, None
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
        
        # 清空并重新创建输出目录
        if os.path.exists(output_dir):
            import shutil
            logger.info(f"🗑️  清空输出目录: {output_dir}")
            try:
                shutil.rmtree(output_dir)
                logger.info(f"✅ 已清空输出目录")
            except Exception as e:
                logger.warning(f"⚠️  清空输出目录时出错: {e}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"📁 创建输出目录: {output_dir}")
        
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
            success, m_lines_result = self.create_unified_chart(stock_code, stock_name, df, stage_lows, output_file)
            
            if success:
                result['success'] = True
                
                # 添加AnchorM线结果
                if m_lines_result:
                    result['anchorMLines'] = {
                        'best_M': m_lines_result['best_M'],
                        'avg_score': m_lines_result['avg_score'],
                        'matches_count': m_lines_result['matches_count'],
                        'B_values': m_lines_result['B_values'][:10],  # 只保存前10个
                        'anchor_low': m_lines_result['anchor_low'],
                        'anchor_date': str(m_lines_result['anchor_date'])
                    }
                
                # 更新进度
                with progress_lock:
                    self.processed_count += 1
                    m_info = f", M={m_lines_result['best_M']:.1f}%" if m_lines_result else ""
                    logger.info(f"✅ [{self.processed_count}/{self.total_count}] {stock_code} {stock_name} - {len(stage_lows)}个低点{m_info}")
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