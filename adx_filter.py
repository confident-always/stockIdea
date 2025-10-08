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

def load_config():
    """加载配置文件"""
    try:
        with open('configs.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get('ADXfilteringconfig', {})
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}

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
        
        # 使用信号日期当天的(开盘价+收盘价)/2作为基准价格
        signal_row = signal_day_data.iloc[0]
        base_price = (signal_row['open'] + signal_row['close']) / 2
        
        # 获取信号日期后的所有数据（不包含信号日当天）
        future_data = kline_df[kline_df['date'] > signal_date]
        
        if future_data.empty:
            logger.warning(f"信号日期 {signal_date} 后没有K线数据")
            # 如果信号日期后没有K线数据，返回涨跌幅为0
            return 0.0, 0.0, base_price, signal_date
        
        # 计算期间内的最高价和最低价
        max_high = future_data['high'].max()
        min_low = future_data['low'].min()
        
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

def filter_adx_results(input_file, output_file, range_up, range_down):
    """
    过滤ADX结果文件
    
    Args:
        input_file: 输入的ADX结果文件
        output_file: 输出的过滤后文件
        range_up: 允许的最大涨幅
        range_down: 允许的最大跌幅
    """
    try:
        # 读取ADX结果
        adx_df = pd.read_csv(input_file)
        logger.info(f"读取ADX结果文件: {input_file}, 共 {len(adx_df)} 条记录")
        
        filtered_results = []
        
        for idx, row in adx_df.iterrows():
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
            
            # 检查是否符合过滤条件 - 修正过滤逻辑
            # 如果涨幅超过rangeUp，则过滤掉（不保留）
            # 如果跌幅超过rangeDown，则过滤掉（不保留）
            up_ok = range_up == 0 or max_up_pct <= range_up
            down_ok = range_down == 0 or max_down_pct <= range_down
            
            if up_ok and down_ok:
                # 创建新的行数据，包含原有字段和新增的三个字段
                new_row = row.copy()
                new_row['code'] = matched_code  # 使用从data文件夹中匹配到的代码
                new_row['基准价格'] = round(base_price, 2)
                new_row['最大涨幅'] = round(max_up_pct * 100, 2)  # 转换为百分比
                new_row['最大跌幅'] = round(max_down_pct * 100, 2)  # 转换为百分比
                
                filtered_results.append(new_row)
                logger.info(f"股票 {matched_code} 通过过滤: 信号日期 {signal_date}, "
                           f"基准价格 {base_price:.2f}, 最大涨幅 {max_up_pct:.2%}, 最大跌幅 {max_down_pct:.2%}")
            else:
                logger.info(f"股票 {matched_code} 被过滤: 信号日期 {signal_date}, "
                           f"基准价格 {base_price:.2f}, 最大涨幅 {max_up_pct:.2%}, 最大跌幅 {max_down_pct:.2%} "
                           f"(涨幅超限: {max_up_pct > range_up if range_up > 0 else False}, "
                           f"跌幅超限: {max_down_pct > range_down if range_down > 0 else False})")
        
        # 保存过滤结果
        if filtered_results:
            filtered_df = pd.DataFrame(filtered_results)
            filtered_df.to_csv(output_file, index=False)
            logger.info(f"过滤完成，保存 {len(filtered_results)} 条记录到: {output_file}")
        else:
            logger.warning("没有股票通过过滤条件")
            # 创建空文件但保持相同的列结构，包含新增的三个字段
            columns = list(adx_df.columns) + ['基准价格', '最大涨幅', '最大跌幅']
            empty_df = pd.DataFrame(columns=columns)
            empty_df.to_csv(output_file, index=False)
            
    except Exception as e:
        logger.error(f"过滤ADX结果失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='ADX结果过滤器')
    parser.add_argument('--input-dir', default='res', help='输入目录')
    parser.add_argument('--output-dir', default='resByFilter', help='输出目录')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config()
    range_up = parse_percentage(config.get('rangeUp', '0%'))
    range_down = parse_percentage(config.get('rangeDown', '0%'))
    
    logger.info(f"过滤配置: 最大涨幅 {range_up:.2%}, 最大跌幅 {range_down:.2%}")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找所有ADX结果文件
    adx_files = glob.glob(os.path.join(args.input_dir, 'ADX*.csv'))
    
    if not adx_files:
        logger.warning(f"在 {args.input_dir} 目录下未找到ADX结果文件")
        return
    
    logger.info(f"找到 {len(adx_files)} 个ADX结果文件")
    
    # 处理每个文件
    for input_file in adx_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(args.output_dir, filename)
        
        logger.info(f"开始处理: {input_file}")
        filter_adx_results(input_file, output_file, range_up, range_down)
    
    logger.info("所有文件处理完成")

if __name__ == "__main__":
    main()