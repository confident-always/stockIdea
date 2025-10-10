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
from datetime import datetime, timedelta
import argparse

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
        config_file: 配置文件路径，默认为'configs.json'
    
    Returns:
        dict: 配置字典
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
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
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {'ADX': {}, 'PDI': {}}

def parse_percentage(pct_str):
    """解析百分比字符串为浮点数"""
    if isinstance(pct_str, str) and pct_str.endswith('%'):
        return float(pct_str[:-1]) / 100.0
    return 0.0

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

def calculate_max_range_during_crossover(kline_df, signal_date, lookback_days=3):
    """
    计算上穿前后lookback_days日内的最大涨幅
    
    Args:
        kline_df: K线数据DataFrame
        signal_date: 上穿信号日期
        lookback_days: 上穿前后查看的天数
    
    Returns:
        float: 最大涨幅百分比，计算失败时返回None
    """
    try:
        signal_date = pd.to_datetime(signal_date)
        
        # 计算查看范围：上穿前后lookback_days天，总共2*lookback_days+1天
        start_date = signal_date - timedelta(days=lookback_days)
        end_date = signal_date + timedelta(days=lookback_days)
        
        # 获取指定日期范围内的数据
        range_data = kline_df[(kline_df['date'] >= start_date) & (kline_df['date'] <= end_date)]
        
        if range_data.empty:
            logger.warning(f"在日期范围 {start_date} 到 {end_date} 内未找到K线数据")
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
        tuple: (max_up_pct, max_down_pct, base_price, signal_date) 最大涨幅、最大跌幅、基准价格、信号日期
    """
    try:
        signal_date = pd.to_datetime(signal_date)
        
        # 找到信号日期当天的数据
        signal_day_data = kline_df[kline_df['date'] == signal_date]
        
        if signal_day_data.empty:
            logger.warning(f"未找到信号日期 {signal_date} 的K线数据")
            return None, None, None, None
        
        # 使用信号日期当天的(最高价+最低价+2*收盘价)/4作为基准价格
        signal_row = signal_day_data.iloc[0]
        base_price = (signal_row['high'] + signal_row['low'] + 2 * signal_row['close']) / 4
        
        # 获取信号日期后的所有数据（不包含信号日当天）
        future_data = kline_df[kline_df['date'] > signal_date]
        
        if future_data.empty:
            logger.warning(f"信号日期 {signal_date} 后没有K线数据")
            # 如果信号日期后没有K线数据，返回涨跌幅为0
            return 0.0, 0.0, base_price, signal_date
        
        # 计算期间内的收盘最高价和最低价
        max_high = future_data['close'].max()
        min_low = future_data['close'].min()
        
        # 计算相对于基准价格的涨跌幅
        max_up_pct = (max_high - base_price) / base_price
        max_down_pct = (base_price - min_low) / base_price
        
        logger.debug(f"股票信号日期: {signal_date}, 基准价格: {base_price:.2f} (开盘:{signal_row['open']:.2f}, 收盘:{signal_row['close']:.2f}), "
                    f"期间最高: {max_high:.2f}, 期间最低: {min_low:.2f}, "
                    f"最大涨幅: {max_up_pct:.2%}, 最大跌幅: {max_down_pct:.2%}")
        
        return max_up_pct, max_down_pct, base_price, signal_date
        
    except Exception as e:
        logger.error(f"计算价格变化失败: {e}")
        return None, None, None, None

def filter_results(input_file, output_file, range_up, range_down, file_type, lookback_days=3, max_range_up=0):
    """
    过滤结果文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        range_up: 最大涨幅阈值
        range_down: 最大跌幅阈值
        file_type: 文件类型 ('ADX' 或 'PDI')
        lookback_days: 上穿前后查看的天数
        max_range_up: 上穿期间最大涨幅阈值
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
            
            # 检查是否符合过滤条件
            # 1. 原有的涨跌幅过滤条件
            up_ok = range_up == 0 or max_up_pct <= range_up
            down_ok = range_down == 0 or max_down_pct <= range_down
            
            # 2. 新增的上穿期间最大涨幅过滤条件
            crossover_ok = max_range_up == 0 or crossover_max_range >= max_range_up
            
            if up_ok and down_ok and crossover_ok:
                # 创建新的行数据，包含原有字段和新增的四个字段
                new_row = row.copy()
                new_row['code'] = matched_code  # 使用从data文件夹中匹配到的代码
                new_row['基准价格'] = round(base_price, 2)
                new_row['最大涨幅'] = round(max_up_pct * 100, 2)  # 转换为百分比
                new_row['最大跌幅'] = round(max_down_pct * 100, 2)  # 转换为百分比
                new_row['上穿期间最大涨幅'] = round(crossover_max_range * 100, 2)  # 转换为百分比
                
                filtered_results.append(new_row)
                logger.info(f"股票 {matched_code} 通过过滤: 信号日期 {signal_date}, "
                           f"基准价格 {base_price:.2f}, 最大涨幅 {max_up_pct:.2%}, 最大跌幅 {max_down_pct:.2%}, "
                           f"上穿期间最大涨幅 {crossover_max_range:.2%}")
            else:
                logger.info(f"股票 {matched_code} 被过滤: 信号日期 {signal_date}, "
                           f"基准价格 {base_price:.2f}, 最大涨幅 {max_up_pct:.2%}, 最大跌幅 {max_down_pct:.2%}, "
                           f"上穿期间最大涨幅 {crossover_max_range:.2%} "
                           f"(涨幅超限: {max_up_pct > range_up if range_up > 0 else False}, "
                           f"跌幅超限: {max_down_pct > range_down if range_down > 0 else False}, "
                           f"上穿期间涨幅不足: {crossover_max_range < max_range_up if max_range_up > 0 else False})")
        
        # 保存过滤结果
        if filtered_results:
            filtered_df = pd.DataFrame(filtered_results)
            filtered_df.to_csv(output_file, index=False)
            logger.info(f"过滤完成，保存 {len(filtered_results)} 条记录到: {output_file}")
        else:
            logger.warning("没有股票通过过滤条件")
            # 创建空文件但保持相同的列结构，包含新增的四个字段
            columns = list(df.columns) + ['基准价格', '最大涨幅', '最大跌幅', '上穿期间最大涨幅']
            empty_df = pd.DataFrame(columns=columns)
            empty_df.to_csv(output_file, index=False)
            
    except Exception as e:
        logger.error(f"过滤结果失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='ADX和PDI结果过滤器')
    parser.add_argument('--input-dir', default='res', help='输入目录')
    parser.add_argument('--output-dir', default='resByFilter', help='输出目录')
    
    args = parser.parse_args()
    
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
    
    # 处理ADX文件
    if adx_files and adx_config.get('active', False):
        adx_range_up = parse_percentage(adx_config.get('rangeUp', '0%'))
        adx_range_down = parse_percentage(adx_config.get('rangeDown', '0%'))
        adx_lookback_days = adx_config.get('lookback_days', 3)
        adx_max_range_up = parse_percentage(adx_config.get('maxRangeUp', '0%'))
        logger.info(f"ADX过滤配置: 最大涨幅 {adx_range_up:.2%}, 最大跌幅 {adx_range_down:.2%}, "
                   f"回看天数 {adx_lookback_days}, 上穿期间最大涨幅阈值 {adx_max_range_up:.2%}")
        
        for input_file in adx_files:
            filename = os.path.basename(input_file)
            output_file = os.path.join(args.output_dir, filename)
            
            logger.info(f"开始处理ADX文件: {input_file}")
            filter_results(input_file, output_file, adx_range_up, adx_range_down, 'ADX', 
                         adx_lookback_days, adx_max_range_up)
    elif adx_files:
        logger.info("ADX配置未启用，跳过ADX文件处理")
    
    # 处理PDI文件
    if pdi_files and pdi_config.get('active', False):
        pdi_range_up = parse_percentage(pdi_config.get('rangeUp', '0%'))
        pdi_range_down = parse_percentage(pdi_config.get('rangeDown', '0%'))
        pdi_lookback_days = pdi_config.get('lookback_days', 3)
        pdi_max_range_up = parse_percentage(pdi_config.get('maxRangeUp', '0%'))
        logger.info(f"PDI过滤配置: 最大涨幅 {pdi_range_up:.2%}, 最大跌幅 {pdi_range_down:.2%}, "
                   f"回看天数 {pdi_lookback_days}, 上穿期间最大涨幅阈值 {pdi_max_range_up:.2%}")
        
        for input_file in pdi_files:
            filename = os.path.basename(input_file)
            output_file = os.path.join(args.output_dir, filename)
            
            logger.info(f"开始处理PDI文件: {input_file}")
            filter_results(input_file, output_file, pdi_range_up, pdi_range_down, 'PDI',
                         pdi_lookback_days, pdi_max_range_up)
    elif pdi_files:
        logger.info("PDI配置未启用，跳过PDI文件处理")
    
    logger.info("所有文件处理完成")

if __name__ == "__main__":
    main()