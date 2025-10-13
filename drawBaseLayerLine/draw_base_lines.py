#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础层画线脚本
功能：
1. 从resByFilter中提取所有股票数据
2. 在历史低点画水平蓝线并标注价格
3. 根据lineConfig.json的percent_dic绘制涨幅百分比线
4. 输出结果图到drawLineRes文件夹
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import glob
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('draw_base_lines.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class BaseLineDrawer:
    """基础层画线类"""
    
    def __init__(self, config_file: str = "lineConfig.json"):
        """初始化"""
        self.config_file = config_file
        self.percent_list = self._load_config()
        
    def _load_config(self) -> List[str]:
        """加载配置文件"""
        try:
            config_path = Path(self.config_file)
            if not config_path.exists():
                # 如果配置文件不存在，尝试在上级目录查找
                config_path = Path("..") / self.config_file
                if not config_path.exists():
                    logger.warning(f"配置文件 {self.config_file} 不存在，使用默认配置")
                    return ["3%", "16%", "25%", "34%", "50%", "67%", "128%", "228%", "247%", "323%", "457%", "589%", "636%", "770%", "823%", "935%"]
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('percent_dic', [])
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return []
    
    def normalize_stock_code(self, stock_code):
        """标准化股票代码"""
        if isinstance(stock_code, str):
            # 移除可能的前缀和后缀
            code = stock_code.replace('SH', '').replace('SZ', '').replace('.SH', '').replace('.SZ', '')
            # 确保是6位数字
            if code.isdigit():
                return code.zfill(6)
        return str(stock_code).zfill(6)
    
    def find_trough_bars(self, low_prices: pd.Series, amplitude_threshold: float = 5.0, lookback: int = 1) -> np.ndarray:
        """
        寻找波谷点（历史低点）
        
        Args:
            low_prices: 最低价序列
            amplitude_threshold: 振幅阈值（百分比）
            lookback: 回看周期
            
        Returns:
            波谷点的索引数组
        """
        if len(low_prices) < 3:
            return np.array([])
        
        trough_bars = []
        
        for i in range(lookback, len(low_prices) - lookback):
            current_low = low_prices.iloc[i]
            
            # 检查是否为局部最低点
            is_local_min = True
            for j in range(max(0, i - lookback), min(len(low_prices), i + lookback + 1)):
                if j != i and low_prices.iloc[j] <= current_low:
                    is_local_min = False
                    break
            
            if is_local_min:
                # 检查振幅是否足够
                left_high = low_prices.iloc[max(0, i - lookback):i].max() if i > 0 else current_low
                right_high = low_prices.iloc[i+1:min(len(low_prices), i + lookback + 1)].max() if i < len(low_prices) - 1 else current_low
                
                max_high = max(left_high, right_high)
                if max_high > current_low:
                    amplitude = (max_high - current_low) / current_low * 100
                    if amplitude >= amplitude_threshold:
                        trough_bars.append(i)
        
        return np.array(trough_bars)
    
    def get_low_price_1(self, low_prices: pd.Series, trough_bars: np.ndarray) -> Tuple[List[float], List[int]]:
        """
        获取历史低点价格和索引
        
        Args:
            low_prices: 最低价序列
            trough_bars: 波谷点索引
            
        Returns:
            (低点价格列表, 低点索引列表)
        """
        if len(trough_bars) == 0:
            # 如果没有找到波谷点，返回全局最低点
            min_idx = low_prices.idxmin()
            return [low_prices.iloc[min_idx]], [min_idx]
        
        low_prices_list = [low_prices.iloc[idx] for idx in trough_bars]
        return low_prices_list, trough_bars.tolist()
    
    def calculate_low_price_1(self, df: pd.DataFrame, trough_bars: pd.Series) -> float:
        """
        计算历史低点价格
        
        Args:
            df: 股票数据DataFrame
            trough_bars: 波谷点序列
            
        Returns:
            历史低点价格
        """
        if trough_bars.empty:
            return df['low'].min()
        
        # 获取所有波谷点的最低价
        trough_lows = df.loc[trough_bars, 'low']
        return trough_lows.min()
    
    def load_stock_data(self, stock_code: str, data_dir: str = "../data") -> Optional[pd.DataFrame]:
        """
        从data目录加载股票历史数据
        
        Args:
            stock_code: 股票代码
            data_dir: 数据目录路径
            
        Returns:
            股票数据DataFrame或None
        """
        try:
            # 标准化股票代码
            normalized_code = self.normalize_stock_code(stock_code)
            
            # 构建文件路径
            file_path = os.path.join(data_dir, f"{normalized_code}.csv")
            
            if not os.path.exists(file_path):
                logger.warning(f"股票数据文件不存在: {file_path}")
                return None
            
            df = pd.read_csv(file_path)
            
            # 标准化列名
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif 'Date' in df.columns:
                df['date'] = pd.to_datetime(df['Date'])
                df = df.drop('Date', axis=1)
            
            # 确保必要的列存在
            required_cols = ['date', 'open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in df.columns:
                    if col.capitalize() in df.columns:
                        df[col] = df[col.capitalize()]
                    else:
                        logger.error(f"文件 {file_path} 缺少必要列: {col}")
                        return None
            
            # 按日期排序
            df = df.sort_values('date').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"加载股票 {stock_code} 数据失败: {e}")
            return None
    
    def draw_stock_chart(self, stock_code: str, output_dir: str):
        """绘制单个股票的K线图和标注线"""
        try:
            # 从data目录加载股票历史数据
            df = self.load_stock_data(stock_code)
            if df is None:
                logger.warning(f"无法加载股票 {stock_code} 的历史数据")
                return
            
            # 寻找波谷点
            trough_bars = self.find_trough_bars(df['low'])
            low_prices, low_indices = self.get_low_price_1(df['low'], trough_bars)
            
            # 如果没有找到低点，使用全局最低点
            if not low_prices:
                min_idx = df['low'].idxmin()
                low_prices = [df.loc[min_idx, 'low']]
                low_indices = [min_idx]
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # 绘制K线图
            dates = df['date']
            opens = df['open']
            highs = df['high']
            lows = df['low']
            closes = df['close']
            
            # 绘制蜡烛图
            for i in range(len(df)):
                date = dates.iloc[i]
                open_price = opens.iloc[i]
                high_price = highs.iloc[i]
                low_price = lows.iloc[i]
                close_price = closes.iloc[i]
                
                # 确定颜色
                color = 'red' if close_price >= open_price else 'green'
                
                # 绘制影线
                ax.plot([date, date], [low_price, high_price], color='black', linewidth=0.5)
                
                # 绘制实体
                body_height = abs(close_price - open_price)
                bottom = min(open_price, close_price)
                
                if body_height > 0:
                    rect = plt.Rectangle((mdates.date2num(date) - 0.3, bottom), 
                                       0.6, body_height, 
                                       facecolor=color, alpha=0.7)
                    ax.add_patch(rect)
                else:
                    # 十字星
                    ax.plot([mdates.date2num(date) - 0.3, mdates.date2num(date) + 0.3], 
                           [open_price, open_price], color=color, linewidth=1)
            
            # 绘制历史低点水平蓝线
            for i, (low_price, low_idx) in enumerate(zip(low_prices, low_indices)):
                # 绘制水平线
                ax.axhline(y=low_price, color='blue', linestyle='-', linewidth=2, alpha=0.7)
                
                # 标注价格
                ax.text(dates.iloc[-1], low_price, f'低点: {low_price:.2f}', 
                       fontsize=10, color='blue', ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # 绘制涨幅百分比线
            base_price = min(low_prices)  # 使用最低的历史低点作为基准
            
            for percent_str in self.percent_list:
                try:
                    percent = float(percent_str.replace('%', ''))
                    target_price = base_price * (1 + percent / 100)
                    
                    # 绘制涨幅线
                    ax.axhline(y=target_price, color='orange', linestyle='--', 
                              linewidth=1, alpha=0.6)
                    
                    # 标注涨幅
                    ax.text(dates.iloc[0], target_price, f'+{percent_str}', 
                           fontsize=8, color='orange', ha='right', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
                    
                except ValueError:
                    continue
            
            # 设置图表属性
            ax.set_title(f'{stock_code} - 基础层画线图', fontsize=16, fontweight='bold')
            ax.set_xlabel('日期', fontsize=12)
            ax.set_ylabel('价格', fontsize=12)
            
            # 格式化x轴日期
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # 设置网格
            ax.grid(True, alpha=0.3)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            output_path = Path(output_dir) / f"{stock_code}_base_lines.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"已生成图表: {output_path}")
            
        except Exception as e:
            logger.error(f"绘制股票 {stock_code} 图表失败: {e}")
            plt.close()
    
    def process_all_stocks(self, input_dir: str = "resByFilter", output_dir: str = "drawLineRes"):
        """
        处理所有股票数据
        
        Args:
            input_dir: 输入目录（包含选股结果文件）
            output_dir: 输出目录
        """
        try:
            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # 查找所有CSV文件
            input_path = Path(input_dir)
            if not input_path.exists():
                logger.error(f"输入目录不存在: {input_dir}")
                return
            
            csv_files = list(input_path.glob("*.csv"))
            if not csv_files:
                logger.warning(f"在目录 {input_dir} 中未找到CSV文件")
                return
            
            logger.info(f"找到 {len(csv_files)} 个CSV文件")
            
            # 处理每个文件
            total_stocks = 0
            for csv_file in csv_files:
                logger.info(f"处理文件: {csv_file}")
                
                df = pd.read_csv(csv_file)
                if df.empty:
                    continue
                
                # 获取股票代码列表
                if 'code' in df.columns:
                    stock_codes = df['code'].unique()
                else:
                    logger.warning(f"文件 {csv_file} 中没有找到 'code' 列")
                    continue
                
                logger.info(f"文件中包含 {len(stock_codes)} 只股票")
                
                # 为每只股票绘制图表
                for stock_code in stock_codes:
                    try:
                        self.draw_stock_chart(stock_code, output_dir)
                        total_stocks += 1
                    except Exception as e:
                        logger.error(f"处理股票 {stock_code} 失败: {e}")
                        continue
            
            logger.info(f"处理完成，共生成 {total_stocks} 个图表")
            
        except Exception as e:
            logger.error(f"处理过程中发生错误: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基础层画线脚本')
    parser.add_argument('--input-dir', default='resByFilter', help='输入目录')
    parser.add_argument('--output-dir', default='drawLineRes', help='输出目录')
    parser.add_argument('--config', default='lineConfig.json', help='配置文件路径')
    
    args = parser.parse_args()
    
    logger.info("开始执行基础层画线脚本")
    
    # 创建画线器
    drawer = BaseLineDrawer(args.config)
    
    # 处理所有股票
    drawer.process_all_stocks(args.input_dir, args.output_dir)
    
    logger.info("基础层画线脚本执行完成")

if __name__ == "__main__":
    main()