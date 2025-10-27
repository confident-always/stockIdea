#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中间层画线脚本 - 整合基础层功能并添加AnchorBack线
包含完整的基础画线功能 + AnchorBack线分析

核心算法差异:
- AnchorBack算法: B = A + N × K (加法模型)
  其中:
  A = 锚定低点附近5根K线收盘价中的最低值
  N = 步长参数 (0.23 ~ 0.68, 步长0.01)
  K = 奇数序列 (1, 3, 5, 7, 9, ...)
  
功能特性:
1. 从resByFilter中提取所有股票数据
2. 智能数据验证和清洗
3. ZigZag阶段低点检测
4. 高质量K线图表绘制（红涨绿跌）
5. AnchorBack线动态优化和绘制
6. 多线程批量处理
7. 完整的错误处理和日志记录

使用方法:
    # 处理指定日期的股票
    python draw_line_back.py --date 2025-10-20
    
    # 指定线程数
    python draw_line_back.py --date 2025-10-20 --workers 4
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# 设置matplotlib后端和字体
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，支持多线程
import matplotlib.pyplot as plt

# 字体配置 - 使用macOS系统支持的中文字体
plt.rcParams['font.family'] = ['Heiti TC', 'PingFang HK', 'Arial Unicode MS', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('draw_lines_back.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 线程锁
progress_lock = threading.Lock()
matplotlib_lock = threading.Lock()
class BackLineDrawer:
    """中间层画线器 - 整合基础层功能并添加AnchorBack线"""
    
    def __init__(self, config_file: str = "lineConfig.json"):
        """初始化中间层画线器"""
        self.config_file = config_file
        self.percent_list, self.anchor_back_config = self._load_config()
        self.stock_info = self._load_stock_info()
        self.processed_count = 0
        self.total_count = 0
        logger.info(f"✅ 中间层画线器初始化完成")
        logger.info(f"📊 加载{len(self.percent_list)}个百分比配置: {self.percent_list}")
        logger.info(f"📈 加载{len(self.stock_info)}只股票信息")
        logger.info(f"🔧 AnchorBack功能: {'启用' if self.anchor_back_config.get('enabled', True) else '禁用'}")
    
    def _load_config(self) -> Tuple[List[str], Dict]:
        """加载配置文件中的百分比数据、ZigZag参数和AnchorBack配置"""
        try:
            config_path = Path(self.config_file)
            if not config_path.exists():
                config_path = Path("..") / self.config_file
                if not config_path.exists():
                    logger.warning(f"⚠️ 配置文件 {self.config_file} 不存在，使用默认配置")
                    default_percents = ["3%", "16%", "25%", "34%", "50%", "67%", "128%", "228%", "247%", "323%", "457%", "589%", "636%", "770%", "823%", "935%"]
                    self.zigzag_period = 49
                    default_anchor_back = {
                        'enabled': True,
                        'zigzag_percent': 15,
                        'pivot_window': 5,
                        'n_range': {'start': 0.23, 'end': 0.68, 'step': 0.01},
                        'k_list': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                        'match_tolerance_ratio': 0.006,
                        'min_matches': 1,
                        'tiebreaker_prefer_higher_N': True,
                        'line_style': {'color': '#1E90FF', 'linewidth': 3.0, 'alpha': 0.9},
                        'text_style': {'fontsize': 14},
                        'annotate_format': 'K={K} 价格={price}'
                    }
                    return default_percents, default_anchor_back
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                percent_dic = config.get('percent_dic', [])
                self.zigzag_period = config.get('zigzag_period', 49)
                anchor_back_config = config.get('anchorBackLines', {})
                
                if not anchor_back_config:
                    anchor_back_config = {'enabled': False}
                
                logger.info(f"✅ 成功加载配置文件: {config_path}")
                logger.info(f"🔧 ZigZag周期: {self.zigzag_period}%")
                return percent_dic, anchor_back_config
        except Exception as e:
            logger.error(f"❌ 加载配置文件失败: {e}")
            default_percents = ["3%", "16%", "25%", "34%", "50%", "67%", "128%", "228%", "247%", "323%", "457%", "589%", "636%", "770%", "823%", "935%"]
            self.zigzag_period = 49
            default_anchor_back = {'enabled': False}
            return default_percents, default_anchor_back
    
    def _load_stock_info(self) -> Dict[str, Dict[str, str]]:
        """从stocklist.csv加载股票信息（名称、行业、市盈率、总股本）"""
        stock_info = {}
        try:
            possible_paths = ["../stocklist.csv", "stocklist.csv", "./stocklist.csv"]
            stocklist_file = None
            
            for path in possible_paths:
                if os.path.exists(path):
                    stocklist_file = path
                    break
            
            if stocklist_file:
                try:
                    df = pd.read_csv(stocklist_file)
                    logger.info(f"📁 从{stocklist_file}加载股票信息")
                    
                    required_cols = ['symbol', 'name', 'industry', 'pe', 'total_share']
                    if all(col in df.columns for col in required_cols):
                        for _, row in df.iterrows():
                            code = str(row['symbol']).zfill(6)
                            name = str(row['name'])
                            industry = str(row['industry']) if pd.notna(row['industry']) else "未知行业"
                            pe = row['pe'] if pd.notna(row['pe']) else 0
                            total_share = row['total_share'] if pd.notna(row['total_share']) else 0
                            
                            stock_info[code] = {
                                'name': name, 
                                'industry': industry,
                                'pe': pe,
                                'total_share': total_share  # 总股本（亿股），用于计算总市值
                            }
                        
                        logger.info(f"✅ 从stocklist.csv加载股票信息完成，共{len(stock_info)}只股票")
                        return stock_info
                    else:
                        logger.warning(f"⚠️ stocklist.csv缺少必要列")
                except Exception as e:
                    logger.warning(f"⚠️ 读取stocklist.csv失败: {e}")
            
            logger.info("📁 stocklist.csv不可用")
            return stock_info
            
        except Exception as e:
            logger.error(f"❌ 加载股票信息失败: {e}")
            return stock_info
    
    def validate_and_load_data(self, stock_code: str, data_dir: str) -> Optional[pd.DataFrame]:
        """验证并加载股票数据"""
        try:
            normalized_code = str(stock_code).zfill(6)
            data_path = Path(data_dir)
            csv_file = data_path / f"{normalized_code}.csv"
            
            if not csv_file.exists():
                logger.warning(f"⚠️ 数据文件不存在: {csv_file}")
                return None
            
            file_size = csv_file.stat().st_size
            if file_size < 1000:
                logger.warning(f"⚠️ 数据文件过小: {csv_file} ({file_size} bytes)")
                return None
            
            df = pd.read_csv(csv_file)
            
            required_columns = ['date', 'open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"⚠️ 缺少必要列 {missing_columns}: {stock_code}")
                return None
            
            if len(df) < 100:
                logger.warning(f"⚠️ 数据行数不足: {stock_code} ({len(df)} rows)")
                return None
            
            df = df.dropna(subset=required_columns)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if (df[col] <= 0).any():
                    logger.warning(f"⚠️ 发现非正价格数据: {stock_code}")
                    df = df[df[col] > 0]
            
            invalid_rows = (df['high'] < df['low']) | (df['high'] < df['open']) | (df['high'] < df['close']) | \
                          (df['low'] > df['open']) | (df['low'] > df['close'])
            if invalid_rows.any():
                logger.warning(f"⚠️ 发现价格逻辑错误: {stock_code}")
                df = df[~invalid_rows]
            
            if len(df) < 50:
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
        """
        if len(high_prices) < 3:
            return []
        
        pivots = []
        last_pivot_idx = 0
        last_pivot_price = low_prices[0]
        last_pivot_type = 'low'
        searching_for = 'high'
        
        for i in range(1, len(high_prices)):
            if searching_for == 'high':
                current_high = high_prices[i]
                if last_pivot_type == 'low':
                    pct_change = (current_high - last_pivot_price) / last_pivot_price
                    if pct_change >= threshold_pct:
                        pivots.append((last_pivot_idx, last_pivot_price, 'low'))
                        last_pivot_idx = i
                        last_pivot_price = current_high
                        last_pivot_type = 'high'
                        searching_for = 'low'
                    else:
                        if low_prices[i] < last_pivot_price:
                            last_pivot_idx = i
                            last_pivot_price = low_prices[i]
            else:
                current_low = low_prices[i]
                if last_pivot_type == 'high':
                    pct_change = (last_pivot_price - current_low) / last_pivot_price
                    if pct_change >= threshold_pct:
                        pivots.append((last_pivot_idx, last_pivot_price, 'high'))
                        last_pivot_idx = i
                        last_pivot_price = current_low
                        last_pivot_type = 'low'
                        searching_for = 'high'
                    else:
                        if high_prices[i] > last_pivot_price:
                            last_pivot_idx = i
                            last_pivot_price = high_prices[i]
        
        if pivots and last_pivot_idx != pivots[-1][0]:
            pivots.append((last_pivot_idx, last_pivot_price, last_pivot_type))
        
        return pivots

    def find_stage_lows_unified(self, df: pd.DataFrame) -> List[Tuple[int, float, str]]:
        """
        统一版阶段低点检测 - 使用ZigZag(L,49)算法
        """
        try:
            threshold_pct = self.zigzag_period / 100.0
            logger.debug(f"🔍 开始ZigZag阶段低点检测 (阈值={self.zigzag_period}%)")
            
            high_prices = df['high'].values
            low_prices = df['low'].values
            
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
                low_pivots = [(idx, price, pivot_type) for idx, price, pivot_type in pivots if pivot_type == 'low']
                
                if not low_pivots:
                    logger.warning("⚠️ ZigZag未找到低点转折，使用全局最低点")
                    min_idx = df['low'].idxmin()
                    min_price = df.loc[min_idx, 'low']
                    min_date = df.loc[min_idx, 'date']
                else:
                    idx, price, _ = low_pivots[-1]
                    low_date = df.loc[idx, 'date']
                    
                    logger.debug(f"✅ ZigZag找到 {len(low_pivots)} 个低点转折")
                    logger.debug(f"✅ ZigZag最近低点: 索引={idx}, 价格={price:.2f}")
                    
                    if idx < len(df) - 1:
                        after_low_df = df.iloc[idx+1:]
                        after_min_idx = after_low_df['low'].idxmin()
                        after_min_price = after_low_df.loc[after_min_idx, 'low']
                        
                        if after_min_price < price:
                            logger.debug(f"🔽 发现更低价格: 原价格={price:.2f}, 新价格={after_min_price:.2f}")
                            idx = after_min_idx
                            price = after_min_price
                            low_date = df.loc[after_min_idx, 'date']
                            logger.debug(f"✅ 更新锚定低点: 索引={idx}, 日期={low_date}, 价格={price:.2f}")
                    
                    min_idx = idx
                    min_price = price
                    min_date = low_date
                
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
    
    # ==================== AnchorBack Lines 功能函数 ====================
    
    def get_local_extremes_around_turns(self, highs: np.ndarray, lows: np.ndarray,
                                       opens: np.ndarray, closes: np.ndarray,
                                       turns: List[Tuple[int, float, str]], window: int) -> List[Tuple[float, int]]:
        """获取ZigZag拐点附近的局部极值
        
        Args:
            highs: 最高价数组
            lows: 最低价数组
            opens: 开盘价数组（保留参数但不使用）
            closes: 收盘价数组
            turns: ZigZag转折点列表 [(index, price, type)]，type为'high'或'low'
            window: 搜索窗口大小
        
        Returns:
            极值点列表 [(price, index)]
            
        重要逻辑：
            - 当转折点类型为'high'时，只取该转折点附近窗口内的局部最高价
            - 当转折点类型为'low'时，取窗口内所有收盘价中的最小值
              （只使用收盘价，不使用开盘价和最低价，收盘价代表当天买卖双方最终共识）
        """
        extremes = []
        
        for turn_idx, turn_price, turn_type in turns:
            start_idx = max(0, turn_idx - window)
            end_idx = min(len(highs), turn_idx + window + 1)
            
            if turn_type == 'high':
                # 高点转折：只取局部最高价
                window_highs = highs[start_idx:end_idx]
                local_max_idx = start_idx + np.argmax(window_highs)
                local_max_price = highs[local_max_idx]
                extremes.append((local_max_price, local_max_idx))
            else:  # turn_type == 'low'
                # 低点转折：取窗口内所有收盘价的最小值
                window_closes = closes[start_idx:end_idx]
                local_min_idx = start_idx + np.argmin(window_closes)
                local_min_price = closes[local_min_idx]
                extremes.append((local_min_price, local_min_idx))
        
        # 去重并排序
        extremes = list(set(extremes))
        extremes.sort(key=lambda x: x[0])
        
        return extremes
    
    def generate_B_series_back(self, A: float, N: float, k_list: List[int], 
                               max_price: float) -> Tuple[List[float], List[int]]:
        """生成B序列: B_k = A + N × K (AnchorBack算法)
        
        Args:
            A: 锚定低点价格（5根K线收盘价最低值）
            N: 步长参数
            k_list: K值列表（奇数序列，仅用于确定奇数规则）
            max_price: 数据中的最高价
            
        Returns:
            B_values: B序列价格列表
            K_values: K值列表
            
        策略：
        1. 自动计算需要多少个K值才能覆盖到最高价
        2. 在此基础上再额外生成3个K值
        3. 不受 k_list 限制（忽略该参数，仅用于确定奇数规则）
        """
        if N <= 0:
            return [], []
        
        # 计算覆盖到最高价需要的K值（奇数序列）
        # B_k = A + N × K，当 B_k >= max_price 时，K = (max_price - A) / N
        k_to_reach_max = int((max_price - A) / N)
        
        # 确保K是奇数
        if k_to_reach_max % 2 == 0:
            k_to_reach_max += 1
        
        # 在覆盖最高价的基础上再加3个奇数（即 +2, +4, +6）
        k_final = k_to_reach_max + 6
        
        # 安全限制：防止N值过小导致K值过大（例如 > 1000）
        if k_final > 500:
            logger.warning(f"⚠️ K值过大({k_final})，N值可能过小({N:.2f})，限制为501（最大奇数）")
            k_final = 501  # 确保是奇数
        
        B_values = []
        K_values = []
        
        # 生成奇数序列：1, 3, 5, 7, 9, ...
        k = 1
        while k <= k_final:
            B_k = A + N * k
            B_values.append(B_k)
            K_values.append(k)
            k += 2  # 每次加2，保持奇数
        
        return B_values, K_values
    
    def score_N(self, B_values: List[float], K_values: List[int], extremes: List[Tuple[float, int]], 
                match_tolerance_ratio: float, time_decay_min_weight: float = 0.3) -> Dict:
        """对某个N值进行评分（含时间衰减因子）
        
        Args:
            B_values: B序列价格列表
            K_values: K值列表
            extremes: List[Tuple[price, idx]] - 价格和索引（距锚定点的天数）
            match_tolerance_ratio: 匹配容差比例
            time_decay_min_weight: 时间衰减最小权重 (0-1)，越小衰减越强
        
        时间衰减规则:
            - 锚定点位置（idx=0）: 权重 = 1.0
            - 最远点（idx=max）: 权重 = time_decay_min_weight
            - 中间点: 线性插值
        """
        if not B_values or not extremes:
            return {'avg_score': 0, 'matches_count': 0, 'per_k_matches': []}
        
        # 创建价格到索引的映射（用于查找时间信息）
        price_to_idx = {e[0]: e[1] for e in extremes}
        extreme_prices = [e[0] for e in extremes]
        extreme_prices_sorted = sorted(extreme_prices)
        
        # 计算时间衰减系数：最远的极值点索引
        max_idx = max(e[1] for e in extremes) if extremes else 1
        
        scores = []
        per_k_matches = []
        
        for idx, (B_k, k_val) in enumerate(zip(B_values, K_values)):
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
            
            selected_extremes = []
            if upper is not None:
                selected_extremes.append(upper)
            if lower is not None and lower != upper:
                selected_extremes.append(lower)
            
            if not selected_extremes:
                distances = [(abs(e_price - B_k), e_price) for e_price in extreme_prices]
                distances.sort()
                if distances:
                    selected_extremes.append(distances[0][1])
            
            k_scores = []
            for e_price in selected_extremes:
                # 基础匹配得分（价格相似度）
                r = abs(e_price - B_k) / B_k
                base_score = 100 * max(0, 1 - min(r / match_tolerance_ratio, 1))
                
                # 时间衰减因子：离锚定点越近，权重越高
                e_idx = price_to_idx.get(e_price, max_idx)
                if max_idx > 0:
                    # 时间权重：从 1.0 (锚定点) 线性衰减到 time_decay_min_weight (最远点)
                    decay_range = 1.0 - time_decay_min_weight
                    time_weight = 1.0 - decay_range * (e_idx / max_idx)
                else:
                    time_weight = 1.0
                
                # 最终得分 = 基础得分 × 时间权重
                final_score = base_score * time_weight
                k_scores.append(final_score)
            
            if k_scores:
                avg_k_score = sum(k_scores) / len(k_scores)
                scores.append(avg_k_score)
                per_k_matches.append({
                    'k': k_val,
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
    
    def select_best_N(self, N_results: Dict[float, Dict], min_matches: int,
                     prefer_higher_N: bool = True) -> Tuple[Optional[float], Optional[Dict]]:
        """从所有N候选中选择最佳N"""
        valid_N = {N: result for N, result in N_results.items() 
                   if result['matches_count'] >= min_matches}
        
        if not valid_N:
            return None, None
        
        sorted_N = sorted(valid_N.items(), 
                         key=lambda x: (x[1]['avg_score'], 
                                       x[1]['matches_count'],
                                       x[0] if prefer_higher_N else -x[0]),
                         reverse=True)
        
        best_N, best_result = sorted_N[0]
        return best_N, best_result
    
    def compute_anchor_back_lines(self, df: pd.DataFrame, anchor_idx: int, anchor_date, 
                                  stock_code: str = "") -> Optional[Dict]:
        """计算最佳N值与B序列 (AnchorBack算法)
        
        核心算法: B_k = A + N × K
        其中:
        - A: 锚定低点附近5根K线收盘价的最低值
        - N: 步长参数 (0.23 ~ 0.68, 步长0.01)
        - K: 奇数序列 (1, 3, 5, 7, 9, ...)
        """
        try:
            config = self.anchor_back_config
            
            if not config.get('enabled', True):
                return None
            
            # 1. 计算锚定点A：锚定低点附近5根K线收盘价的最低值
            pivot_window = config.get('pivot_window', 5)
            start_idx = max(0, anchor_idx - pivot_window // 2)
            end_idx = min(len(df), anchor_idx + pivot_window // 2 + 1)
            
            window_data = df.iloc[start_idx:end_idx]
            anchor_close_price = window_data['close'].min()  # 窗口内最低收盘价
            anchor_A = float(anchor_close_price)
            
            logger.debug(f"🎯 [{stock_code}] 锚定点A={anchor_A:.2f} (窗口[{start_idx}:{end_idx}]内最低收盘价)")
            
            # 确保 anchor_date 是 pd.Timestamp
            if isinstance(anchor_date, str):
                anchor_date = pd.to_datetime(anchor_date)
            
            df_after = df[df['date'] > anchor_date].copy()
            
            if len(df_after) < 10:
                logger.info(f"⚠️ [{stock_code}] 锚定日期之后数据不足: {len(df_after)}天，跳过AnchorBack线")
                return None
            
            # 2. 使用配置的ZigZag参数寻找转折点
            zigzag_percent = config.get('zigzag_percent', 15) / 100.0
            highs_after = df_after['high'].values
            lows_after = df_after['low'].values
            opens_after = df_after['open'].values
            closes_after = df_after['close'].values
            
            turns = self.zigzag(highs_after, lows_after, zigzag_percent)
            
            if not turns:
                logger.info(f"⚠️ [{stock_code}] 锚定日期后未找到ZigZag转折点，跳过AnchorBack线")
                return None
            
            # 3. 提取局部极值点
            pivot_window = config.get('pivot_window', 5)
            extremes = self.get_local_extremes_around_turns(
                highs_after, lows_after, opens_after, closes_after, turns, pivot_window
            )
            
            if not extremes:
                logger.info(f"⚠️ [{stock_code}] 未找到局部极值，跳过AnchorBack线")
                return None
            
            # 4. 遍历N值范围
            n_range = config.get('n_range', {'start': 0.23, 'end': 0.68, 'step': 0.01})
            N_start = n_range['start']
            N_end = n_range['end']
            N_step = n_range['step']
            
            N_values = []
            N_current = N_start
            while N_current <= N_end + 0.001:
                N_values.append(round(N_current, 2))
                N_current += N_step
            
            k_list = config.get('k_list', [1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
            max_price = df_after['high'].max()
            match_tolerance = config.get('match_tolerance_ratio', 0.006)
            time_decay_min_weight = config.get('time_decay_min_weight', 0.3)
            
            N_results = {}
            
            for N in N_values:
                B_values, K_values = self.generate_B_series_back(anchor_A, N, k_list, max_price)
                
                if not B_values:
                    continue
                
                score_result = self.score_N(B_values, K_values, extremes, match_tolerance, time_decay_min_weight)
                N_results[N] = {
                    'B_values': B_values,
                    'K_values': K_values,
                    'avg_score': score_result['avg_score'],
                    'matches_count': score_result['matches_count'],
                    'per_k_matches': score_result['per_k_matches']
                }
            
            # 5. 选择最佳N值（动态调整策略）
            min_matches = config.get('min_matches', 1)
            prefer_higher_N = config.get('tiebreaker_prefer_higher_N', True)
            
            # 首先尝试获取满足min_matches的最佳N值
            best_N, best_result = self.select_best_N(N_results, min_matches, prefer_higher_N)
            
            # 动态调整策略：如果匹配数<2，尝试降低N值以获得>=2个匹配
            if best_N is None or (best_result and best_result['matches_count'] < 2):
                current_min_matches = best_result['matches_count'] if best_result else 0
                
                if N_results:
                    # 找出所有匹配数>=2的N值
                    valid_N_ge2 = {N: result for N, result in N_results.items() 
                                   if result['matches_count'] >= 2}
                    
                    if valid_N_ge2:
                        # 如果有匹配数>=2的N值，选择其中最佳的（优先选择更小的N值）
                        sorted_N = sorted(valid_N_ge2.items(), 
                                        key=lambda x: (x[1]['avg_score'], 
                                                      x[1]['matches_count'],
                                                      -x[0]),  # 负号表示优先选择更小的N
                                        reverse=True)
                        best_N, best_result = sorted_N[0]
                        logger.info(f"📊 [{stock_code}] 动态调整：降低N值以获得>=2个匹配 → N={best_N:.2f}, 匹配数={best_result['matches_count']}")
                    else:
                        # 如果没有匹配数>=2的，至少选择匹配数最多的
                        sorted_by_matches = sorted(N_results.items(), 
                                                  key=lambda x: (x[1]['matches_count'], 
                                                                x[1]['avg_score'],
                                                                -x[0]),  # 优先小N值
                                                  reverse=True)
                        best_N, best_result = sorted_by_matches[0]
                        if best_result['matches_count'] >= 1:
                            logger.info(f"📊 [{stock_code}] 动态调整：未找到>=2个匹配，使用最佳结果 → N={best_N:.2f}, 匹配数={best_result['matches_count']}")
                        else:
                            logger.info(f"⚠️ [{stock_code}] 所有N值匹配数均<1，跳过AnchorBack线")
                            return None
                else:
                    logger.info(f"⚠️ [{stock_code}] 未找到任何有效的N值，跳过AnchorBack线")
                    return None
            
            logger.debug(f"✅ 最佳N={best_N:.2f}, 平均分={best_result['avg_score']:.2f}, "
                        f"匹配数={best_result['matches_count']}")
            
            return {
                'best_N': best_N,
                'B_values': best_result['B_values'],
                'K_values': best_result['K_values'],
                'avg_score': best_result['avg_score'],
                'matches_count': best_result['matches_count'],
                'per_k_matches': best_result['per_k_matches'],
                'anchor_A': anchor_A,
                'anchor_date': anchor_date,
                'anchor_idx': anchor_idx,
                'extremes': extremes
            }
            
        except Exception as e:
            logger.error(f"❌ 计算AnchorBack线失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def create_back_chart(self, stock_code: str, stock_name: str, df: pd.DataFrame,
                          output_file: str) -> Tuple[bool, Optional[Dict]]:
        """
        创建中间层图表：基础K线图 + AnchorBack线
        """
        try:
            # 1. 检测阶段低点
            stage_lows = self.find_stage_lows_unified(df)
            if not stage_lows:
                logger.warning(f"⚠️ 未检测到阶段低点: {stock_code}")
                return False, None
            
            # 2. 计算AnchorBack线数据（如果启用）
            back_lines_result = None
            if self.anchor_back_config.get('enabled', True):
                anchor_idx, anchor_low, anchor_date = stage_lows[0]
                back_lines_result = self.compute_anchor_back_lines(df, anchor_idx, anchor_date, stock_code)
            
            # 3. 绘制图表
            with matplotlib_lock:
                import mplfinance as mpf
                
                # 准备数据：确保包含锚定点
                if stage_lows:
                    lowest_idx, _, _ = stage_lows[0]
                    # 如果有AnchorBack结果，需要确保显示范围包含锚定点
                    start_idx = lowest_idx
                    if back_lines_result and back_lines_result.get('anchor_idx') is not None:
                        anchor_idx = back_lines_result['anchor_idx']
                        pivot_window = self.anchor_back_config.get('pivot_window', 5)
                        # 向前扩展窗口以包含锚定点
                        anchor_start = max(0, anchor_idx - pivot_window // 2)
                        start_idx = min(start_idx, anchor_start)
                        logger.debug(f"📊 [{stock_code}] 调整显示起点: {lowest_idx} → {start_idx} (包含锚定点)")
                    
                    df_display = df.iloc[start_idx:].copy()
                else:
                    df_display = df.copy()
                
                # 限制显示的K线数量，避免"数据太多"警告
                max_candles = 750
                if len(df_display) > max_candles:
                    logger.info(f"📊 [{stock_code}] 数据量大({len(df_display)}根K线)，"
                               f"只显示最近{max_candles}根")
                    df_display = df_display.iloc[-max_candles:].copy()
                
                df_mpf = df_display.copy()
                df_mpf['date'] = pd.to_datetime(df_mpf['date'])
                df_mpf.set_index('date', inplace=True)
                df_mpf = df_mpf[['open', 'high', 'low', 'close']].copy()
                
                if df_mpf.empty:
                    logger.warning(f"⚠️ 处理后的数据为空: {stock_code}")
                    return False, None
                
                logger.info(f"📊 [{stock_code}] 绘制{len(df_mpf)}根K线")
                
                # 准备额外的绘图元素
                additional_plots = []
                
                # 添加阶段低点水平线
                for i, (idx, price, date_str) in enumerate(stage_lows):
                    hline_data = [price] * len(df_mpf)
                    additional_plots.append(mpf.make_addplot(hline_data, color='blue', linestyle='-', width=2, alpha=0.8))
                
                # 添加百分比涨幅线
                if stage_lows:
                    base_price = min(price for _, price, _ in stage_lows)
                    max_price = df_mpf['high'].max()
                    
                    visible_percent_lines = []
                    highest_visible_idx = -1
                    
                    for i, percent_str in enumerate(self.percent_list):
                        try:
                            percent = float(percent_str.rstrip('%')) / 100
                            target_price = base_price * (1 + percent)
                            
                            if target_price <= max_price:
                                visible_percent_lines.append((percent_str, target_price))
                                highest_visible_idx = i
                                hline_data = [target_price] * len(df_mpf)
                                additional_plots.append(mpf.make_addplot(hline_data, color='hotpink', linestyle='--', width=3, alpha=0.8))
                        except (ValueError, TypeError):
                            continue
                
                    if highest_visible_idx >= 0 and highest_visible_idx + 1 < len(self.percent_list):
                        try:
                            next_percent_str = self.percent_list[highest_visible_idx + 1]
                            next_percent = float(next_percent_str.rstrip('%')) / 100
                            next_target_price = base_price * (1 + next_percent)
                            hline_data = [next_target_price] * len(df_mpf)
                            additional_plots.append(mpf.make_addplot(hline_data, color='hotpink', linestyle='--', width=3, alpha=0.8))
                            visible_percent_lines.append((next_percent_str, next_target_price))
                        except (ValueError, TypeError):
                            pass
                
                # 构建标题（包含行业、总市值、市盈率）
                industry = ""
                pe_val = 0
                total_share = 0
                total_market_cap = 0
                
                if stock_code in self.stock_info:
                    info = self.stock_info[stock_code]
                    industry = info.get('industry', '')
                    pe_val = float(info.get('pe', 0))
                    total_share = float(info.get('total_share', 0))
                    
                    # 计算总市值 = 总股本（亿股）× 当前股价（元）
                    if total_share > 0 and len(df_mpf) > 0:
                        current_price = float(df_mpf['close'].iloc[-1])
                        total_market_cap = total_share * current_price  # 总市值（亿元）
                
                title_parts = [stock_code, stock_name]
                
                # 添加行业
                if industry and industry != "未知行业":
                    title_parts.append(f"({industry})")
                
                # 添加总市值
                if total_market_cap > 0:
                    if total_market_cap >= 1000:
                        title_parts.append(f"总市值:{total_market_cap:.0f}亿")
                    else:
                        title_parts.append(f"总市值:{total_market_cap:.1f}亿")
                
                # 添加市盈率
                if pe_val > 0:
                    title_parts.append(f"PE:{pe_val:.2f}")
                elif pe_val == 0:
                    title_parts.append("PE:亏损")
                
                title = " ".join(title_parts) + " - AnchorBack"
                
                # 设置样式
                plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang HK', 'Arial Unicode MS', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                
                style = mpf.make_mpf_style(
                    base_mpf_style='charles',
                    marketcolors=mpf.make_marketcolors(
                        up='red', down='green', edge='inherit', wick='inherit', volume='inherit'
                    ),
                    gridstyle='-', gridcolor='lightgray', y_on_right=True,
                    facecolor='white', edgecolor='black', figcolor='white',
                    rc={'font.size': 12, 'axes.titlesize': 20, 'axes.labelsize': 14, 
                        'font.sans-serif': ['Heiti TC', 'PingFang HK', 'Arial Unicode MS', 'DejaVu Sans'],
                        'axes.unicode_minus': False}
                )
                
                # 创建图表
                fig, axes = mpf.plot(
                    df_mpf, type='candle', style=style, title=title, ylabel='Price',
                    volume=False, addplot=additional_plots if additional_plots else None,
                    figsize=(20, 12), tight_layout=True, returnfig=True,
                    panel_ratios=(1,), show_nontrading=False, 
                    datetime_format='%Y-%m', xrotation=45
                )
                
                ax = axes[0]
                
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
                    
                    highest_visible_idx = -1
                    for i, percent_str in enumerate(self.percent_list):
                        try:
                            percent = float(percent_str.rstrip('%')) / 100
                            target_price = base_price * (1 + percent)
                            
                            if target_price <= max_price:
                                highest_visible_idx = i
                                ax.text(1.02, target_price, f'+{percent_str}', 
                                       fontsize=18, color='#8B7355', fontweight='bold',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                                alpha=0.9, edgecolor='#8B7355', linewidth=2),
                                       transform=ax.get_yaxis_transform(), ha='left', va='center')
                        except (ValueError, TypeError):
                            continue
                    
                    if highest_visible_idx >= 0 and highest_visible_idx + 1 < len(self.percent_list):
                        try:
                            next_percent_str = self.percent_list[highest_visible_idx + 1]
                            next_percent = float(next_percent_str.rstrip('%')) / 100
                            next_target_price = base_price * (1 + next_percent)
                            
                            ax.text(1.02, next_target_price, f'+{next_percent_str}', 
                                   fontsize=18, color='#8B7355', fontweight='bold',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                            alpha=0.9, edgecolor='#8B7355', linewidth=2),
                                   transform=ax.get_yaxis_transform(), ha='left', va='center')
                        except (ValueError, TypeError):
                            pass
                
                # 4. 添加AnchorBack线并标注锚定低点
                if back_lines_result:
                    best_N = back_lines_result['best_N']
                    B_values = back_lines_result['B_values']
                    K_values = back_lines_result['K_values']
                    anchor_A = back_lines_result['anchor_A']
                    anchor_idx = back_lines_result['anchor_idx']
                    
                    line_style = self.anchor_back_config.get('line_style', {})
                    line_color = line_style.get('color', '#1E90FF')
                    line_width = line_style.get('linewidth', 3.0)
                    line_alpha = line_style.get('alpha', 0.9)
                    
                    text_style = self.anchor_back_config.get('text_style', {})
                    text_fontsize = text_style.get('fontsize', 14)
                    annotate_format = self.anchor_back_config.get('annotate_format', 'K={K} 价格={price}')
                    
                    # 绘制蓝色横线
                    for k_val, B_k_price in zip(K_values, B_values):
                        ax.axhline(y=B_k_price, color=line_color, 
                                  linestyle='-', linewidth=line_width, 
                                  alpha=line_alpha, zorder=2.5)
                        
                        label_text = annotate_format.replace('{K}', str(k_val)).replace('{price}', f'{B_k_price:.2f}')
                        ax.text(-0.02, B_k_price, label_text,
                               fontsize=text_fontsize, color=line_color, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.85, 
                                        edgecolor=line_color, linewidth=2),
                               transform=ax.get_yaxis_transform(), ha='right', va='center')
                    
                    # 标注锚定低点（最低收盘价）
                    if stage_lows:
                        # 获取锚定点的日期（使用原始df）
                        pivot_window = self.anchor_back_config.get('pivot_window', 5)
                        start_idx = max(0, anchor_idx - pivot_window // 2)
                        end_idx = min(len(df), anchor_idx + pivot_window // 2 + 1)
                        
                        # 找到窗口内的最低收盘价及其索引
                        window_data = df.iloc[start_idx:end_idx]
                        min_close_idx = window_data['close'].idxmin()
                        anchor_close_price = df.loc[min_close_idx, 'close']
                        anchor_close_date = df.loc[min_close_idx, 'date']
                        
                        # 检查锚定点日期是否在显示范围内（df_mpf已经设置了date为索引）
                        anchor_date_dt = pd.to_datetime(anchor_close_date)
                        
                        if anchor_date_dt in df_mpf.index:
                            # 绘制红色圆点（提高zorder确保在最上层）
                            ax.plot(anchor_date_dt, anchor_close_price, 
                                   marker='o', markersize=15, color='red', 
                                   markeredgecolor='white', markeredgewidth=2,
                                   zorder=10)
                            
                            # 添加带箭头的标注（使用英文避免中文乱码）
                            ax.annotate(f'Anchor\n{anchor_close_price:.2f}',
                                       xy=(anchor_date_dt, anchor_close_price),
                                       xytext=(15, -40),
                                       textcoords='offset points',
                                       fontsize=13, color='red', fontweight='bold',
                                       bbox=dict(boxstyle='round,pad=0.6', 
                                               facecolor='yellow', edgecolor='red', linewidth=2.5),
                                       arrowprops=dict(arrowstyle='->', color='red', lw=3),
                                       zorder=10)
                            logger.info(f"✅ [{stock_code}] 标注锚定点: 日期={anchor_date_dt}, 价格={anchor_close_price:.2f}")
                        else:
                            logger.warning(f"⚠️ [{stock_code}] 锚定点日期 {anchor_date_dt} 不在显示范围内，跳过标注")
                    
                    # 在图片左上角添加N值信息 - 只显示匹配的B值
                    text_lines = [f"N={best_N:.2f}"]
                    
                    # 提取得分 > 0 的 B 值（与极值点匹配的）
                    if 'per_k_matches' in back_lines_result:
                        matched_B = []
                        for match in back_lines_result['per_k_matches']:
                            if match.get('score', 0) > 0:
                                k_val = match['k']
                                B_k = match['B_k']
                                score = match['score']
                                matched_B.append(f"K{k_val}:{B_k:.2f}({score:.0f})")
                                if len(matched_B) >= 10:  # 最多显示10个
                                    break
                        
                        if matched_B:
                            if len(back_lines_result['per_k_matches']) > len(matched_B):
                                matched_B.append('...')
                            text_lines.append(f"Match_B: [{', '.join(matched_B)}]")
                        else:
                            text_lines.append(f"Match_B: [无匹配]")
                    
                    text_lines.append(f"AvgScore: {back_lines_result['avg_score']:.1f}")
                    text_lines.append(f"Matches: {back_lines_result['matches_count']}/{len(B_values)}")
                    
                    text_content = '\n'.join(text_lines)
                    ax.text(0.01, 0.98, text_content,
                           transform=ax.transAxes,
                           fontsize=11, color='#1E90FF', fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.95, 
                                    edgecolor='#1E90FF', linewidth=2.5),
                           ha='left', va='top', family='monospace')
                    
                    logger.info(f"✅ [{stock_code}] 绘制AnchorBack线: N={best_N:.2f}, {len(B_values)}条线")
                
                # 4.5 统一调整Y轴范围（考虑百分比线和AnchorBack线）
                if stage_lows:
                    base_price = min(price for _, price, _ in stage_lows)
                    max_price = df_mpf['high'].max()
                    min_price = df_mpf['low'].min()
                    
                    # 计算最高的百分比线
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
                    
                    # 加上K线上方的额外百分比线
                    if highest_visible_idx >= 0 and highest_visible_idx + 1 < len(self.percent_list):
                        try:
                            next_percent_str = self.percent_list[highest_visible_idx + 1]
                            next_percent = float(next_percent_str.rstrip('%')) / 100
                            next_target_price = base_price * (1 + next_percent)
                            highest_percent_price = next_target_price
                        except (ValueError, TypeError):
                            pass
                    
                    # 考虑AnchorBack线的最高价格
                    highest_line_price = highest_percent_price
                    if back_lines_result and back_lines_result['B_values']:
                        highest_back_price = max(back_lines_result['B_values'])
                        highest_line_price = max(highest_percent_price, highest_back_price)
                    
                    # 设置Y轴范围，确保所有线都可见
                    y_margin = (highest_line_price - min_price) * 0.05
                    ax.set_ylim(min_price - y_margin, highest_line_price + y_margin)
                    logger.debug(f"📊 [{stock_code}] Y轴范围: {min_price:.2f} - {highest_line_price:.2f}")
                
                # 4.6 绘制最后一个交易日的收盘价横线
                last_close_price = df_mpf['close'].iloc[-1]
                ax.axhline(y=last_close_price, color='red', linestyle='-', linewidth=3, alpha=0.8, zorder=3)
                
                # 在右侧标注收盘价
                ax.text(1.02, last_close_price, f'{last_close_price:.2f}', 
                       fontsize=16, color='red', fontweight='bold',
                       transform=ax.get_yaxis_transform(), ha='left', va='center')
                logger.debug(f"📊 [{stock_code}] 最后交易日收盘价横线: {last_close_price:.2f}")
                
                # 5. 保存图表
                plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
                plt.close(fig)
                
                # 6. 调整图片尺寸
                try:
                    from PIL import Image
                    with Image.open(output_file) as img:
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
                    if file_size > 1000:
                        logger.debug(f"✅ 图表生成成功: {output_file} ({file_size} bytes)")
                        return True, back_lines_result
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
            try:
                plt.close('all')
            except:
                pass
    
    def process_stock_list(self, stock_list: List[Tuple[str, str, str, str]], 
                          output_dir: Optional[str] = None, data_dir: str = "../data", workers: int = 4):
        """处理指定的股票列表"""
        if output_dir is None:
            current_date = datetime.now().strftime('%Y%m%d')
            output_dir = f'{current_date}-drawLineBack'
        
        logger.info(f"🚀 开始处理股票列表（中间层 - AnchorBack）")
        logger.info(f"📁 数据目录: {data_dir}")
        logger.info(f"📁 输出目录: {output_dir}")
        logger.info(f"🧵 线程数: {workers}")
        
        if not stock_list:
            logger.error("❌ 股票列表为空")
            return
        
        self.total_count = len(stock_list)
        self.processed_count = 0
        
        logger.info(f"📊 待处理股票数量: {self.total_count}")
        
        # 创建输出目录（覆盖模式，不清空）
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(output_dir) and os.listdir(output_dir):
            logger.info(f"📁 输出目录已存在: {output_dir}（将覆盖同名文件）")
        else:
            logger.info(f"📁 创建输出目录: {output_dir}")
        
        # 多线程处理
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_stock = {
                executor.submit(self._process_single_stock, code, name, output_dir, data_dir, file_prefix): (code, name, industry, file_prefix)
                for code, name, industry, file_prefix in stock_list
            }
            
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
            for r in failed_stocks[:10]:
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
            'error': None
        }
        
        try:
            # 1. 加载数据
            df = self.validate_and_load_data(stock_code, data_dir)
            if df is None:
                result['error'] = "数据加载失败"
                return result
            
            # 2. 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 3. 生成图表
            if file_prefix and file_prefix != "UNKNOWN":
                output_file = os.path.join(output_dir, f"{file_prefix}_{stock_code}_{stock_name}_2back.png")
            else:
                output_file = os.path.join(output_dir, f"{stock_code}_{stock_name}_2back.png")
            
            success, back_lines_result = self.create_back_chart(stock_code, stock_name, df, output_file)
            
            if success:
                result['success'] = True
                
                # 添加AnchorBack线结果
                if back_lines_result:
                    result['anchorBackLines'] = {
                        'best_N': back_lines_result['best_N'],
                        'avg_score': back_lines_result['avg_score'],
                        'matches_count': back_lines_result['matches_count'],
                        'B_values': back_lines_result['B_values'][:10],
                        'anchor_A': back_lines_result['anchor_A'],
                        'anchor_date': str(back_lines_result['anchor_date'])
                    }
                
                # 更新进度
                with progress_lock:
                    self.processed_count += 1
                    n_info = f", N={back_lines_result['best_N']:.2f}" if back_lines_result else ""
                    logger.info(f"✅ [{self.processed_count}/{self.total_count}] {stock_code} {stock_name}{n_info}")
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
        description="中间层画线脚本 - 基础图表 + AnchorBack线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理当前日期的resByFilter中的股票
  python draw_line_back.py
  
  # 处理指定日期的resByFilter中的股票
  python draw_line_back.py --date 2025-10-20
  
  # 指定线程数
  python draw_line_back.py --date 2025-10-20 --workers 6
  
  # 处理指定股票代码
  python draw_line_back.py --codes 000001 600000 002603
        """
    )
    
    current_date = datetime.now().strftime('%Y%m%d')
    
    parser.add_argument('--date', type=str, 
                       help='日期参数，格式为YYYY-MM-DD，用于构建resByFilter目录')
    parser.add_argument('--workers', type=int, default=4,
                       help='并发处理的线程数 (默认: 4)')
    parser.add_argument('--codes', nargs='+', type=str,
                       help='股票代码列表，多个代码用空格分隔（如：000001 600000）')
    
    args = parser.parse_args()
    
    # 处理日期参数
    if args.date:
        try:
            date_obj = datetime.strptime(args.date, '%Y-%m-%d')
            date_str = date_obj.strftime('%Y%m%d')
        except ValueError:
            logger.error(f"❌ 日期格式错误: {args.date}，请使用YYYY-MM-DD格式")
            sys.exit(1)
    else:
        date_str = current_date
    
    # 创建中间层画线器
    drawer = BackLineDrawer()
    
    # 如果指定了股票代码，直接从data目录读取
    if args.codes:
        logger.info(f"📊 处理指定的股票代码: {', '.join(args.codes)}")
        stock_list = []
        
        # 读取stocklist.csv获取股票名称和行业信息
        stocklist_path = "../stocklist.csv"
        stock_info_dict = {}
        if os.path.exists(stocklist_path):
            try:
                stocklist_df = pd.read_csv(stocklist_path)
                for _, row in stocklist_df.iterrows():
                    code = str(row.get('symbol', '')).zfill(6)
                    name = str(row.get('name', code))
                    industry = str(row.get('industry', '未知行业'))
                    stock_info_dict[code] = (name, industry)
                logger.info(f"✅ 已加载 {len(stock_info_dict)} 只股票的基础信息")
            except Exception as e:
                logger.warning(f"⚠️ 读取stocklist.csv失败: {e}")
        
        # 处理每个股票代码
        for code in args.codes:
            normalized_code = code.zfill(6)
            
            # 获取股票名称和行业
            if normalized_code in stock_info_dict:
                name, industry = stock_info_dict[normalized_code]
            else:
                name = normalized_code
                industry = "未知行业"
                logger.warning(f"⚠️ 未找到股票 {normalized_code} 的基础信息，使用默认值")
            
            stock_list.append((normalized_code, name, industry, ""))
        
        logger.info(f"📋 共有 {len(stock_list)} 只股票待处理")
        
        # 生成输出目录
        output_dir = f"{date_str}-drawLineBack"
        
        # 批量处理股票列表
        drawer.process_stock_list(stock_list, output_dir, "../data", args.workers)
        
    else:
        # 原有逻辑：从resByFilter读取股票
        filter_dir = f"../{date_str}-resByFilter"
        if not os.path.exists(filter_dir):
            logger.error(f"❌ 目录不存在: {filter_dir}")
            logger.info(f"💡 提示：请确保存在 {filter_dir} 目录，或使用 --codes 参数指定股票代码")
            sys.exit(1)
        
        # 查找所有CSV文件
        csv_files = glob.glob(os.path.join(filter_dir, "*.csv"))
        if not csv_files:
            logger.error(f"❌ 在目录 {filter_dir} 中未找到CSV文件")
            sys.exit(1)
        
        logger.info(f"📁 找到 {len(csv_files)} 个CSV文件")
        
        # 读取所有CSV文件中的股票，并去重
        all_stocks = {}
        
        for file_path in csv_files:
            logger.info(f"📄 读取文件: {file_path}")
            try:
                df = pd.read_csv(file_path)
                
                # 从文件名提取前缀
                file_name = os.path.basename(file_path)
                file_prefix = ""
                
                import re
                patterns = [
                    (r'^ADX(\d+)', 'ADX'),
                    (r'^PDI(\d+)', 'PDI'),
                    (r'ADX(\d+)', 'ADX'),
                    (r'PDI(\d+)', 'PDI')
                ]
                
                for pattern, prefix_type in patterns:
                    match = re.search(pattern, file_name.upper())
                    if match:
                        file_prefix = f"{prefix_type}{match.group(1)}"
                        break
                
                logger.info(f"📊 文件类型: {file_prefix}")
                
                # 提取股票信息
                for _, row in df.iterrows():
                    code = str(row.get('code', ''))
                    name = str(row.get('name', code))
                    industry = str(row.get('industry', '未知行业'))
                    
                    if code:
                        normalized_code = code.zfill(6)
                        if normalized_code not in all_stocks:
                            all_stocks[normalized_code] = (normalized_code, name, industry, file_prefix)
                            
            except Exception as e:
                logger.error(f"❌ 读取文件 {file_path} 失败: {e}")
                continue
        
        if not all_stocks:
            logger.error(f"❌ 未读取到有效的股票数据")
            sys.exit(1)
        
        stock_list = list(all_stocks.values())
        logger.info(f"📋 去重后共有 {len(stock_list)} 只股票")
        
        # 生成输出目录
        output_dir = f"{date_str}-drawLineBack"
        
        # 批量处理股票列表
        drawer.process_stock_list(stock_list, output_dir, "../data", args.workers)
    
    logger.info("🎉 程序执行完成!")


if __name__ == "__main__":
    main()

