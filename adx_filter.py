#!/usr/bin/env python3
"""
ADX过滤脚本
根据ADXfilteringconfig配置，过滤res目录中的ADX结果
检查股票在ADX上穿日期后的涨跌幅是否符合设定范围
"""

import pandas as pd
import json
import os
import glob
import logging
import sys
from datetime import datetime, timedelta
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('adx_filter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_file='configs.json'):
    """
    加载配置文件
    
    Args:
        config_file: 配置文件路径
    
    Returns:
        dict: 包含ADX和PDI配置的字典
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 提取ADX和PDI过滤配置
        adx_config = config.get('ADXfilteringconfig', {})
        pdi_config = config.get('PDIfilteringconfig', {})
        
        return {
            'ADX': adx_config,
            'PDI': pdi_config
        }
    except FileNotFoundError:
        logger.error(f"配置文件 {config_file} 未找到")
        return {
            'ADX': {'active': False},
            'PDI': {'active': False}
        }
    except json.JSONDecodeError as e:
        logger.error(f"配置文件 {config_file} 格式错误: {e}")
        return {
            'ADX': {'active': False},
            'PDI': {'active': False}
        }

def parse_percentage(pct_str):
    """解析百分比字符串为浮点数，支持none值表示不过滤"""
    if pct_str is None or (isinstance(pct_str, str) and pct_str.lower() == 'none'):
        return None
    if isinstance(pct_str, str) and pct_str.endswith('%'):
        return float(pct_str[:-1]) / 100.0
    return 0.0

def parse_range_percent(range_str):
    """解析形如 "A-B%" 的区间百分比配置，返回 (A, B) 的浮点区间（按比例值），或 None 表示不过滤。
    例如："16-600%" => (0.16, 6.0)
    支持大小写的"none"返回 None。
    """
    if range_str is None:
        return None
    if isinstance(range_str, str) and range_str.strip().lower() == 'none':
        return None
    if isinstance(range_str, str) and range_str.endswith('%') and '-' in range_str:
        try:
            body = range_str[:-1]
            parts = body.split('-')
            if len(parts) != 2:
                return None
            low = float(parts[0]) / 100.0
            high = float(parts[1]) / 100.0
            return (low, high)
        except Exception:
            logger.warning(f"区间百分比解析失败: {range_str}")
            return None
    logger.warning(f"未识别的区间配置: {range_str}")
    return None

def find_kline_file(stock_code):
    """查找股票对应的K线数据文件"""
    # 尝试不同的文件名格式
    possible_files = [
        f"data/{stock_code}.csv",
        f"data/{stock_code}_kline.csv",
        f"data/sz_{stock_code}_kline.csv",
        f"data/sh_{stock_code}_kline.csv"
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            return file_path
    
    logger.warning(f"未找到股票 {stock_code} 的K线数据文件")
    return None

def load_kline_data(file_path):
    """加载K线数据"""
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df
    except Exception as e:
        logger.error(f"加载K线数据失败 {file_path}: {e}")
        return None

def calculate_pre_high_metrics(kline_df, signal_date, pre_high_days=10):
    """
    计算前高价格相关指标
    
    功能说明：
    1. 计算上穿前N个交易日的开盘价和收盘价中的最高价M
    2. 计算上穿后最高收盘价H发生日往后的最低收盘价J相对于M的涨幅K（前高最小涨幅）
    3. 计算前最高价M到当前收盘价的涨幅A（前高当前涨幅）
    4. 计算前最高价M到发生上穿后最高收盘价H的涨幅B（前高最大涨幅）
    
    Args:
        kline_df: K线数据DataFrame，包含date, open, high, low, close列
        signal_date: 上穿信号日期
        pre_high_days: 上穿前查看的交易日数，默认10天
    
    Returns:
        dict: 包含以下键值对的字典
            - pre_high_price: 前高价格M
            - pre_high_min_range: 前高最小涨幅K（可能为负值，表示跌幅）
            - pre_high_current_range: 前高当前涨幅A
            - pre_high_max_range: 前高最大涨幅B
            如果计算失败，返回None
    """
    try:
        signal_date = pd.to_datetime(signal_date)
        
        # 1. 计算上穿前N个交易日的开盘价和收盘价中的最高价M
        # 找到信号日期在K线数据中的位置
        signal_idx = kline_df[kline_df['date'] == signal_date].index
        if len(signal_idx) == 0:
            logger.warning(f"未找到信号日期 {signal_date} 的K线数据")
            return None
        
        signal_idx = signal_idx[0]
        # 获取信号日期前N个交易日的数据
        start_idx = max(0, signal_idx - pre_high_days)
        pre_signal_data = kline_df.iloc[start_idx:signal_idx]
        
        if pre_signal_data.empty:
            logger.warning(f"在上穿前{pre_high_days}个交易日内未找到K线数据")
            return None
        
        # 计算开盘价和收盘价中的最高价
        open_max = pre_signal_data['open'].max()
        close_max = pre_signal_data['close'].max()
        pre_high_price = max(open_max, close_max)  # M
        
        # 2. 获取上穿后的数据
        post_signal_data = kline_df[kline_df['date'] > signal_date]
        
        # 5. 计算当前收盘价（最后一天的收盘价）
        current_price = kline_df['close'].iloc[-1]
        
        if post_signal_data.empty:
            # 当上穿日期为最后一天时，直接用最后一天收盘价计算涨跌幅
            logger.info(f"上穿日期{signal_date}为最后一天，直接使用最后一天收盘价计算涨跌幅")
            post_high_price = current_price  # H = 当前收盘价
            post_min_price = current_price   # J = 当前收盘价
        else:
            # 3. 计算上穿后最高收盘价H
            post_high_price = post_signal_data['close'].max()  # H
            post_high_date = post_signal_data[post_signal_data['close'] == post_high_price]['date'].iloc[0]
            
            # 4. 计算上穿后最高收盘价H发生日往后的最低收盘价J
            post_high_data = kline_df[kline_df['date'] >= post_high_date]
            if post_high_data.empty:
                post_min_price = post_high_price  # 如果没有后续数据，使用最高价本身
            else:
                post_min_price = post_high_data['close'].min()  # J
        
        # 6. 计算各项涨幅
        if pre_high_price > 0:
            # 前高最小涨幅K：J相对于M的涨幅（可能为负值）
            pre_high_min_range = (post_min_price - pre_high_price) / pre_high_price
            
            # 前高当前涨幅A：当前收盘价相对于M的涨幅
            pre_high_current_range = (current_price - pre_high_price) / pre_high_price
            
            # 前高最大涨幅B：上穿后最高收盘价H相对于M的涨幅
            pre_high_max_range = (post_high_price - pre_high_price) / pre_high_price
            
            return {
                'pre_high_price': pre_high_price,
                'pre_high_min_range': pre_high_min_range,
                'pre_high_current_range': pre_high_current_range,
                'pre_high_max_range': pre_high_max_range
            }
        else:
            logger.warning(f"前高价格为0或负数，无法计算涨幅")
            return None
            
    except Exception as e:
        logger.error(f"计算前高价格相关指标失败: {e}")
        return None

def calculate_max_range_during_crossover(kline_df, signal_date, lookback_days=3):
    """
    计算上穿前后lookback_days个交易日内的最大涨幅
    
    Args:
        kline_df: K线数据DataFrame
        signal_date: 上穿信号日期
        lookback_days: 上穿前后查看的交易日数
    
    Returns:
        float: 最大涨幅百分比，计算失败时返回None
    """
    try:
        signal_date = pd.to_datetime(signal_date)
        
        # 找到信号日期在K线数据中的位置
        signal_idx = kline_df[kline_df['date'] == signal_date].index
        if len(signal_idx) == 0:
            logger.warning(f"未找到信号日期 {signal_date} 的K线数据")
            return None
        
        signal_idx = signal_idx[0]
        
        # 计算查看范围：上穿前后lookback_days个交易日
        start_idx = max(0, signal_idx - lookback_days)
        end_idx = min(len(kline_df) - 1, signal_idx + lookback_days)
        
        # 获取指定交易日范围内的数据
        range_data = kline_df.iloc[start_idx:end_idx + 1]
        
        if range_data.empty:
            logger.warning(f"在上穿前后{lookback_days}个交易日内未找到K线数据")
            return None
        
        # 找到期间内的最高收盘价和最低开盘价
        max_close = range_data['close'].max()
        min_open = range_data['open'].min()
        
        # 计算最大涨幅
        if min_open > 0:
            max_range_up = (max_close - min_open) / min_open
            return max_range_up
        else:
            logger.warning(f"最低开盘价为0或负数，无法计算涨幅")
            return None
            
    except Exception as e:
        logger.error(f"计算上穿期间最大涨幅失败: {e}")
        return None

def calculate_price_change_after_signal(kline_df, signal_date):
    """
    计算ADX上穿信号日期后的最大涨跌幅
    新算法：以信号日期当天的(开盘价+收盘价)/2为基准，计算到K线数据最后一天的最大涨跌幅
    
    Args:
        kline_df: K线数据DataFrame
        signal_date: ADX上穿信号日期
    
    Returns:
        tuple: (max_up_pct, max_down_pct, base_price, signal_date) 最大涨幅、最大跌幅、枢轴点价格、信号日期
    """
    try:
        signal_date = pd.to_datetime(signal_date)
        
        # 找到信号日期当天的数据
        signal_day_data = kline_df[kline_df['date'] == signal_date]
        
        if signal_day_data.empty:
            logger.warning(f"未找到信号日期 {signal_date} 的K线数据")
            return None, None, None, None
        
        signal_row = signal_day_data.iloc[0]
        # 使用信号日期当天的(最高价+最低价+2*收盘价)/4作为枢轴点价格
        # base_price = (signal_row['high'] + signal_row['low'] + 2 * signal_row['close']) / 4
      
        #或者使用信号日期当天的收盘价作为枢轴点价格
        base_price = signal_row['close']
        
        # 或者使用信号日期当天的开盘价作为枢轴点价格
        # base_price = signal_row['open']


        # 获取信号日期后的所有数据（不包含信号日当天）
        future_data = kline_df[kline_df['date'] > signal_date]
        
        if future_data.empty:
            logger.warning(f"信号日期 {signal_date} 后没有K线数据")
            logger.info(f"上穿日期{signal_date}为最后一天，直接使用最后一天收盘价计算枢轴点涨跌幅")
            # 当上穿日期为最后一天时，将最大最小价格都设置为当天收盘价
            current_close = signal_row['close']
            max_high = current_close
            min_low = current_close
        else:
            # 计算期间内的收盘最高价和最低价
            max_high = future_data['close'].max()
            min_low = future_data['close'].min()
        
        # 计算相对于枢轴点价格的涨跌幅
        max_up_pct = (max_high - base_price) / base_price
        max_down_pct = (min_low - base_price) / base_price  # 修改为负值表示跌幅
        
        logger.debug(f"股票信号日期: {signal_date}, 枢轴点价格: {base_price:.2f} (开盘:{signal_row['open']:.2f}, 收盘:{signal_row['close']:.2f}), "
                    f"期间最高: {max_high:.2f}, 期间最低: {min_low:.2f}, "
                    f"最大涨幅: {max_up_pct:.2%}, 最大跌幅: {max_down_pct:.2%}")
        
        return max_up_pct, max_down_pct, base_price, signal_date
        
    except Exception as e:
        logger.error(f"计算价格变化失败: {e}")
        return None, None, None, None

def calculate_current_range_from_base(kline_df, signal_date):
    """
    计算枢轴点价格到最后一日的当前涨幅
    
    Args:
        kline_df: K线数据DataFrame
        signal_date: ADX上穿信号日期
    
    Returns:
        float: 当前涨幅（枢轴点价格到最后一日收盘价的涨幅）
    """
    try:
        signal_date = pd.to_datetime(signal_date)
        
        # 找到信号日期当天的数据
        signal_day_data = kline_df[kline_df['date'] == signal_date]
        
        if signal_day_data.empty:
            logger.warning(f"未找到信号日期 {signal_date} 的K线数据")
            return 0.0
        
        # 使用信号日期当天的(最高价+最低价+2*收盘价)/4作为枢轴点价格
        signal_row = signal_day_data.iloc[0]
        base_price = (signal_row['high'] + signal_row['low'] + 2 * signal_row['close']) / 4
        
        # 检查信号日期是否为最后一天
        future_data = kline_df[kline_df['date'] > signal_date]
        
        if future_data.empty:
            # 当上穿日期为最后一天时，直接使用当天收盘价作为当前价格
            logger.info(f"上穿日期{signal_date}为最后一天，直接使用当天收盘价计算当前涨幅")
            current_price = signal_row['close']
        else:
            # 获取最后一日的收盘价
            current_price = kline_df.iloc[-1]['close']
        
        # 计算当前涨幅
        current_range = (current_price - base_price) / base_price
        
        return current_range
        
    except Exception as e:
        logger.error(f"计算当前涨幅时出错: {e}")
        return 0.0


def calculate_hislow_point_range(kline_df: pd.DataFrame, lookback_months: int | None = None) -> float | None:
    """
    计算最后一天收盘价相对于历史最低价的涨幅。

    - 当提供 lookback_months 时，仅统计最后日期往前 N 月内的最低价。
    - 返回比例值，例如 0.16 表示 16%。若历史最低价<=0或数据缺失则返回 None。
    """
    try:
        if kline_df is None or kline_df.empty:
            return None
        # 确保日期为 datetime
        dates = pd.to_datetime(kline_df['date'])
        last_close = float(kline_df['close'].iloc[-1])
        if lookback_months is not None and lookback_months > 0:
            last_date = dates.iloc[-1]
            start_date = last_date - pd.DateOffset(months=int(lookback_months))
            window_df = kline_df[dates >= start_date]
        else:
            window_df = kline_df
        if window_df is None or window_df.empty:
            logger.warning("指定窗口内无数据，无法计算历史低点涨幅")
            return None
        hist_low = float(window_df['low'].min())
        if hist_low <= 0:
            logger.warning("历史最低价<=0，无法计算历史低点涨幅")
            return None
        return (last_close - hist_low) / hist_low
    except Exception as e:
        logger.error(f"计算历史低点涨幅失败: {e}")
        return None


def filter_results(input_file, output_file, range_up, range_down, file_type, lookback_days=3, max_range_up=0, pre_high_days=10, pre_high_min_range=0, pre_high_max_range=0, hislow_range: tuple | None = None, hislow_lookback_months: int | None = None, target_date=None):
    """
    过滤结果文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        range_up: 最大涨幅阈值（可以为负值表示跌幅，None表示不过滤）
        range_down: 最小涨幅阈值（可以为负值表示跌幅，None表示不过滤）
        file_type: 文件类型 ('ADX' 或 'PDI')
        lookback_days: 上穿前后查看的天数
        max_range_up: 上穿期间最大涨幅阈值（None表示不过滤）
        pre_high_days: 前高天数N
        pre_high_min_range: 前高最小涨幅阈值K（可以为负值表示跌幅，None表示不过滤）
        pre_high_max_range: 前高最大涨幅阈值（前高价格M到上穿后最高收盘价H的涨幅，可以为负值表示跌幅，None表示不过滤）
    """
    try:
        # 读取结果文件
        df = pd.read_csv(input_file)
        file_type = "PDI" if "PDI" in os.path.basename(input_file) else "ADX"
        logger.info(f"读取{file_type}结果文件: {input_file}, 共 {len(df)} 条记录")
        
        filtered_results = []
        
        for idx, row in df.iterrows():
            original_code = str(row['code'])
            signal_date = row['date']
            
            # 如果指定了target_date，则使用target_date替换原始日期
            if target_date:
                signal_date = target_date
                logger.debug(f"股票 {original_code} 日期从 {row['date']} 修正为 {target_date}")
            
            # 直接从data文件夹中查找匹配的CSV文件
            data_dir = 'data'
            matched_code = None
            kline_file = None
            
            # 尝试不同的代码格式来匹配data文件夹中的CSV文件
            possible_codes = [
                original_code,  # 原始代码
                original_code.zfill(6),  # 6位补零
                original_code.lstrip('0') if len(original_code) > 1 else original_code  # 去掉前导零
            ]
            
            for code in possible_codes:
                csv_file = os.path.join(data_dir, f"{code}.csv")
                if os.path.exists(csv_file):
                    matched_code = code
                    kline_file = csv_file
                    break
            
            if not kline_file:
                logger.debug(f"未找到股票 {original_code} 的K线数据文件")
                continue
            
            # 加载K线数据
            kline_df = load_kline_data(kline_file)
            if kline_df is None:
                continue
            
            # 计算价格变化
            max_up_pct, max_down_pct, base_price, signal_date_used = calculate_price_change_after_signal(kline_df, signal_date)
            if max_up_pct is None or max_down_pct is None or base_price is None:
                continue
            
            # 计算上穿期间最大涨幅
            crossover_max_range = calculate_max_range_during_crossover(kline_df, signal_date, lookback_days)
            if crossover_max_range is None:
                logger.warning(f"股票 {matched_code} 无法计算上穿期间最大涨幅")
                continue
            
            # 计算前高价格相关指标
            pre_high_metrics = calculate_pre_high_metrics(kline_df, signal_date, pre_high_days)
            if pre_high_metrics is None:
                logger.warning(f"股票 {matched_code} 无法计算前高价格相关指标")
                continue

            # 计算历史低点涨幅（按配置的 lookback_months 窗口）
            hislow_pct = calculate_hislow_point_range(kline_df, hislow_lookback_months)
            # 历史低点涨幅过滤（None 表示不过滤）
            hislow_ok = hislow_range is None or (hislow_pct is not None and hislow_range[0] <= hislow_pct <= hislow_range[1])
            
            # 检查是否符合过滤条件
            # 计算当前涨幅（枢轴点价格到最后一日的涨幅）
            current_range = calculate_current_range_from_base(kline_df, signal_date)
            
            # 1. 原有的涨跌幅过滤条件（range_up为最大涨幅限制，range_down为最小涨幅要求，None表示不过滤）
            up_ok = range_up is None or max_up_pct <= range_up
            # minRange逻辑：当为正值时表示最小涨幅要求，当为负值时表示最大跌幅限制
            # 例如：minRange=0%表示不允许跌幅，minRange=-1%表示最大跌幅不超过-1%
            down_ok = range_down is None or max_down_pct >= range_down
            
            # 2. 新增的上穿期间最大涨幅过滤条件（None表示不过滤）
            crossover_ok = max_range_up is None or crossover_max_range >= max_range_up
            
            # 3. 新增的前高最小涨幅过滤条件（preHighMinRange逻辑同minRange）
            # 当为正值时表示最小涨幅要求，当为负值时表示最大跌幅限制，None表示不过滤
            # 例如：preHighMinRange=0%表示不允许跌幅，preHighMinRange=-1%表示最大跌幅不超过-1%
            pre_high_ok = pre_high_min_range is None or pre_high_metrics['pre_high_min_range'] >= pre_high_min_range
            
            # 4. 新增的前高最大涨幅过滤条件（pre_high_max_range可以为负值表示跌幅，None表示不过滤）
            pre_high_max_ok = pre_high_max_range is None or pre_high_metrics['pre_high_max_range'] <= pre_high_max_range
            
            if up_ok and down_ok and crossover_ok and pre_high_ok and pre_high_max_ok and hislow_ok:
                # 创建新的行数据，包含原有字段和新增的字段
                new_row = row.copy()
                new_row['code'] = matched_code  # 使用从data文件夹中匹配到的代码
                # 如果指定了target_date，则更新日期字段
                if target_date:
                    new_row['date'] = target_date
                # 历史低点涨幅（在CSV中位于枢轴点价格之前），并注明窗口月数
                hislow_label = f"历史低点涨幅({hislow_lookback_months}月)" if hislow_lookback_months else "历史低点涨幅"
                new_row[hislow_label] = f"{round((hislow_pct or 0) * 100, 2)}%"
                new_row['枢轴点价格'] = round(base_price, 2)
                new_row['当前涨幅'] = f"{round(current_range * 100, 2)}%"  # 转换为百分比并添加百分号
                new_row['最小涨幅'] = f"{round(max_down_pct * 100, 2)}%"  # 转换为百分比并添加百分号
                new_row['最大涨幅'] = f"{round(max_up_pct * 100, 2)}%"  # 转换为百分比并添加百分号
                new_row['上穿期间最大涨幅'] = f"{round(crossover_max_range * 100, 2)}%"  # 转换为百分比并添加百分号
                new_row['前高价格'] = round(pre_high_metrics['pre_high_price'], 2)  # 前高价格
                new_row['前高当前涨幅'] = f"{round(pre_high_metrics['pre_high_current_range'] * 100, 2)}%"  # 转换为百分比并添加百分号
                new_row['前高最小涨幅'] = f"{round(pre_high_metrics['pre_high_min_range'] * 100, 2)}%"  # 转换为百分比并添加百分号
                new_row['前高最大涨幅'] = f"{round(pre_high_metrics['pre_high_max_range'] * 100, 2)}%"  # 转换为百分比并添加百分号
                
                filtered_results.append(new_row)
                logger.info(f"股票 {matched_code} 通过过滤: 信号日期 {signal_date}, "
                           f"枢轴点价格 {base_price:.2f}, 当前涨幅 {current_range:.2%}, 最大涨幅 {max_up_pct:.2%}, 最小涨幅 {max_down_pct:.2%}, "
                           f"上穿期间最大涨幅 {crossover_max_range:.2%}, "
                           f"前高价格 {pre_high_metrics['pre_high_price']:.2f}, "
                           f"前高最小涨幅 {pre_high_metrics['pre_high_min_range']:.2%}, "
                           f"前高最大涨幅 {pre_high_metrics['pre_high_max_range']:.2%}")
            else:
                logger.info(f"股票 {matched_code} 被过滤: 信号日期 {signal_date}, "
                           f"枢轴点价格 {base_price:.2f}, 当前涨幅 {current_range:.2%}, 最大涨幅 {max_up_pct:.2%}, 最小涨幅 {max_down_pct:.2%}, "
                           f"上穿期间最大涨幅 {crossover_max_range:.2%}, "
                           f"前高价格 {pre_high_metrics['pre_high_price']:.2f}, "
                           f"前高最小涨幅 {pre_high_metrics['pre_high_min_range']:.2%}, "
                           f"前高最大涨幅 {pre_high_metrics['pre_high_max_range']:.2%}, "
                           f"历史低点涨幅({hislow_lookback_months}月) {'' if hislow_pct is None else f'{hislow_pct:.2%}'} "
                           f"(涨幅超限: {max_up_pct > range_up if range_up is not None else False}, "
                           f"跌幅超限: {max_down_pct > range_down if range_down is not None else False}, "
                           f"上穿期间涨幅不足: {crossover_max_range < max_range_up if max_range_up is not None else False}, "
                           f"前高最小涨幅不足: {pre_high_metrics['pre_high_min_range'] < pre_high_min_range if pre_high_min_range is not None else False}, "
                           f"前高最大涨幅超限: {pre_high_metrics['pre_high_max_range'] > pre_high_max_range if pre_high_max_range is not None else False}, "
                           f"历史低点涨幅不在区间: {False if hislow_range is None or hislow_pct is None else not (hislow_range[0] <= hislow_pct <= hislow_range[1])})")
        
        # 保存过滤结果
        if filtered_results:
            filtered_df = pd.DataFrame(filtered_results)
            # 统一列顺序，确保“历史低点涨幅(XX月)”位于“枢轴点价格”之前
            base_cols = list(df.columns)
            hislow_label = f"历史低点涨幅({hislow_lookback_months}月)" if hislow_lookback_months else "历史低点涨幅"
            extra_cols = [hislow_label, '枢轴点价格', '当前涨幅', '最小涨幅', '最大涨幅', '上穿期间最大涨幅', '前高价格', '前高当前涨幅', '前高最小涨幅', '前高最大涨幅']
            desired_cols = base_cols + [c for c in extra_cols if c not in base_cols]
            filtered_df = filtered_df.reindex(columns=desired_cols)
            filtered_df.to_csv(output_file, index=False)
            logger.info(f"过滤完成，保存 {len(filtered_results)} 条记录到: {output_file}")
        else:
            logger.warning("没有股票通过过滤条件")
            # 创建空文件但保持相同的列结构，包含新增的字段
            hislow_label = f"历史低点涨幅({hislow_lookback_months}月)" if hislow_lookback_months else "历史低点涨幅"
            columns = list(df.columns) + [hislow_label, '枢轴点价格', '当前涨幅', '最小涨幅', '最大涨幅', '上穿期间最大涨幅', '前高价格', '前高当前涨幅', '前高最小涨幅', '前高最大涨幅']
            empty_df = pd.DataFrame(columns=columns)
            empty_df.to_csv(output_file, index=False)
            
    except Exception as e:
        logger.error(f"过滤结果失败: {e}")

def process_single_file(input_file, output_dir, file_type, config, target_date=None):
    """
    处理单个文件的函数，用于并行处理
    
    Args:
        input_file: 输入文件路径
        output_dir: 输出目录
        file_type: 文件类型 ('ADX' 或 'PDI')
        config: 配置字典
        target_date: 目标日期，用于修正CSV文件中的日期字段
    
    Returns:
        str: 处理结果信息
    """
    try:
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)
        
        # 解析配置参数
        range_up = parse_percentage(config.get('maxRange', '0%'))
        range_down = parse_percentage(config.get('minRange', '0%'))
        lookback_days = config.get('lookback_days', 3)
        max_range_up = parse_percentage(config.get('maxRangeUp', '0%'))
        pre_high_days = config.get('preHighDays', 10)
        pre_high_min_range = parse_percentage(config.get('preHighMinRange', '0%'))
        pre_high_max_range = parse_percentage(config.get('preHighMaxRange', '0%'))
        hislow_range = parse_range_percent(config.get('HisLowPointRange', 'none'))
        lookback_months = config.get('lookback_months', 120)
        
        logger.info(f"开始处理{file_type}文件: {input_file}")
        filter_results(input_file, output_file, range_up, range_down, file_type, 
                     lookback_days, max_range_up, pre_high_days, pre_high_min_range, 
                     pre_high_max_range, hislow_range, lookback_months, target_date)
        
        return f"成功处理 {filename}"
    except Exception as e:
        error_msg = f"处理文件 {input_file} 时出错: {str(e)}"
        logger.error(error_msg)
        return error_msg


def main():
    # 生成带日期的默认输出目录
    current_date = datetime.now().strftime('%Y%m%d')
    default_output_dir = f'{current_date}-resByFilter'
    
    # 生成带日期的默认输入目录
    default_input_dir = f'{current_date}-res'
    
    parser = argparse.ArgumentParser(description='ADX和PDI结果过滤器')
    parser.add_argument('--input-dir', default=default_input_dir, help=f'输入目录 (默认: {default_input_dir})')
    parser.add_argument('--output-dir', default=default_output_dir, help=f'输出目录 (默认: {default_output_dir})')
    parser.add_argument('--workers', type=int, default=12, help='并发处理的线程数')
    parser.add_argument('--date', help='指定日期，格式为YYYY-MM-DD，用于修正CSV文件中的日期字段并确定输入目录')
    
    args = parser.parse_args()
    
    # 记录用户是否明确指定了输入目录
    user_specified_input_dir = '--input-dir' in sys.argv
    
    # 如果指定了日期参数，将其转换为目标日期格式并更新输入目录
    target_date = None
    if args.date:
        try:
            # 验证日期格式并转换
            date_obj = datetime.strptime(args.date, '%Y-%m-%d')
            target_date = args.date
            
            # 只有在用户没有明确指定输入目录时，才根据日期参数自动设置
            if not user_specified_input_dir:
                specified_date = date_obj.strftime('%Y%m%d')
                args.input_dir = f'{specified_date}-res'
                logger.info(f"根据指定日期 {target_date}，输入目录设置为: {args.input_dir}")
            
            logger.info(f"将使用指定日期 {target_date} 修正CSV文件中的日期字段")
        except ValueError:
            logger.error(f"日期格式错误: {args.date}，请使用YYYY-MM-DD格式")
            return
    
    # 加载配置
    configs = load_config()
    adx_config = configs['ADX']
    pdi_config = configs['PDI']
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找所有ADX和PDI结果文件
    adx_files = glob.glob(os.path.join(args.input_dir, 'ADX*.csv'))
    pdi_files = glob.glob(os.path.join(args.input_dir, 'PDI*.csv'))
    
    if not adx_files and not pdi_files:
        logger.warning(f"在 {args.input_dir} 目录下未找到ADX或PDI结果文件")
        return
    
    logger.info(f"找到 {len(adx_files)} 个ADX结果文件, {len(pdi_files)} 个PDI结果文件")
    
    # 准备并行处理的任务列表
    tasks = []
    
    # 处理ADX文件
    if adx_files and adx_config.get('active', False):
        adx_range_up = parse_percentage(adx_config.get('maxRange', '0%'))
        adx_range_down = parse_percentage(adx_config.get('minRange', '0%'))
        adx_max_range_up = parse_percentage(adx_config.get('maxRangeUp', '0%'))
        adx_pre_high_min_range = parse_percentage(adx_config.get('preHighMinRange', '0%'))
        adx_pre_high_max_range = parse_percentage(adx_config.get('preHighMaxRange', '0%'))
        adx_hislow_range = parse_range_percent(adx_config.get('HisLowPointRange', 'none'))
        
        logger.info(f"ADX过滤配置: 最大涨幅 {'none' if adx_range_up is None else f'{adx_range_up:.2%}'}, "
                   f"最大跌幅 {'none' if adx_range_down is None else f'{adx_range_down:.2%}'}, "
                   f"回看天数 {adx_config.get('lookback_days', 3)}, 上穿期间最大涨幅阈值 {'none' if adx_max_range_up is None else f'{adx_max_range_up:.2%}'}, "
                   f"前高天数 {adx_config.get('preHighDays', 10)}, 前高最小涨幅阈值 {'none' if adx_pre_high_min_range is None else f'{adx_pre_high_min_range:.2%}'}, "
                   f"前高最大涨幅阈值 {'none' if adx_pre_high_max_range is None else f'{adx_pre_high_max_range:.2%}'}, "
                   f"历史低点涨幅区间 {'none' if adx_hislow_range is None else f'{adx_hislow_range[0]:.2%}-{adx_hislow_range[1]:.2%}'}, "
                   f"历史低点涨幅窗口 {adx_config.get('lookback_months', 120)}月")
        
        for input_file in adx_files:
            tasks.append((input_file, args.output_dir, 'ADX', adx_config, target_date))
    elif adx_files:
        logger.info("ADX配置未启用，跳过ADX文件处理")
    
    # 处理PDI文件
    if pdi_files and pdi_config.get('active', False):
        pdi_range_up = parse_percentage(pdi_config.get('maxRange', '0%'))
        pdi_range_down = parse_percentage(pdi_config.get('minRange', '0%'))
        pdi_max_range_up = parse_percentage(pdi_config.get('maxRangeUp', '0%'))
        pdi_pre_high_min_range = parse_percentage(pdi_config.get('preHighMinRange', '0%'))
        pdi_pre_high_max_range = parse_percentage(pdi_config.get('preHighMaxRange', '0%'))
        pdi_hislow_range = parse_range_percent(pdi_config.get('HisLowPointRange', 'none'))
        
        logger.info(f"PDI过滤配置: 最大涨幅 {'none' if pdi_range_up is None else f'{pdi_range_up:.2%}'}, "
                   f"最大跌幅 {'none' if pdi_range_down is None else f'{pdi_range_down:.2%}'}, "
                   f"回看天数 {pdi_config.get('lookback_days', 3)}, 上穿期间最大涨幅阈值 {'none' if pdi_max_range_up is None else f'{pdi_max_range_up:.2%}'}, "
                   f"前高天数 {pdi_config.get('preHighDays', 10)}, 前高最小涨幅阈值 {'none' if pdi_pre_high_min_range is None else f'{pdi_pre_high_min_range:.2%}'}, "
                   f"前高最大涨幅阈值 {'none' if pdi_pre_high_max_range is None else f'{pdi_pre_high_max_range:.2%}'}, "
                   f"历史低点涨幅区间 {'none' if pdi_hislow_range is None else f'{pdi_hislow_range[0]:.2%}-{pdi_hislow_range[1]:.2%}'}, "
                   f"历史低点涨幅窗口 {pdi_config.get('lookback_months', 120)}月")
        
        for input_file in pdi_files:
            tasks.append((input_file, args.output_dir, 'PDI', pdi_config, target_date))
    elif pdi_files:
        logger.info("PDI配置未启用，跳过PDI文件处理")
    
    if not tasks:
        logger.info("没有需要处理的文件")
        return
    
    # 并行处理所有文件
    logger.info(f"开始并行处理 {len(tasks)} 个文件，使用 {args.workers} 个线程")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 提交所有任务
        futures = [
            executor.submit(process_single_file, input_file, output_dir, file_type, config, target_date)
            for input_file, output_dir, file_type, config, target_date in tasks
        ]
        
        # 使用tqdm显示进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
            result = future.result()
            logger.debug(result)
    
    logger.info("所有文件处理完成")

if __name__ == "__main__":
    main()