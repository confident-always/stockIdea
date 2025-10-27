#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALL画线脚本 - 整合AnchorM（紫色线）和AnchorBack（蓝色线）到一张图
使用mplfinance画线器确保价格对齐准确

策略：
1. 读取股票数据并计算阶段低点
2. 计算AnchorM线数据（紫色）
3. 计算AnchorBack线数据（蓝色）
4. 使用mplfinance一次性绘制K线图 + M线 + B线 + 百分比线

使用方法:
    # 处理指定日期
    python draw_lines_all.py --date 2025-10-24 --workers 4
    
    # 处理指定股票代码
    python draw_lines_all.py --codes 000001 600000
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import argparse
import glob
import pandas as pd
import numpy as np
from typing import Optional, Dict
import mplfinance as mpf

# 设置matplotlib后端和字体
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 字体配置
plt.rcParams['font.family'] = ['Heiti TC', 'PingFang HK', 'Arial Unicode MS', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('draw_lines_all.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 导入画线器类
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draw_lines_mid import MidLineDrawer
from draw_lines_back import BackLineDrawer


def draw_all_for_stock(mid_drawer: MidLineDrawer,
                       back_drawer: BackLineDrawer,
                       stock_code: str,
                       stock_name: str,
                       output_file: str) -> bool:
    """绘制ALL图 - 整合M线和B线到一张图
    
    使用mplfinance画线器确保价格对齐准确
    """
    
    try:
        # 读取CSV数据
        csv_file = f"../data/{stock_code}.csv"
        if not os.path.exists(csv_file):
            logger.warning(f"⚠️ [{stock_code}] 找不到CSV文件: {csv_file}")
            return False
        
        df = pd.read_csv(csv_file)
        if df.empty:
            logger.warning(f"⚠️ [{stock_code}] 数据为空")
            return False
        
        # 确保date列是datetime类型
        df['date'] = pd.to_datetime(df['date'])
        
        # 检测阶段低点
        stage_lows = mid_drawer.find_stage_lows_unified(df)
        if not stage_lows:
            logger.info(f"ℹ️ [{stock_code}] 无阶段低点")
            return False
        
        # 获取锚点信息
        anchor_idx, anchor_low, anchor_date = stage_lows[0]
        
        # 计算AnchorM线数据
        m_lines_result = mid_drawer.compute_anchor_M_lines(df, anchor_low, anchor_date, stock_code)
        
        # 计算AnchorBack线数据
        back_data = back_drawer.compute_anchor_back_lines(df, anchor_idx, anchor_date, stock_code)
        
        # 准备绘图数据范围
        lowest_idx = stage_lows[0][0]
        df_display = df.iloc[lowest_idx:].copy()
        
        # 限制显示的K线数量
        max_candles = 750
        if len(df_display) > max_candles:
            logger.info(f"📊 [{stock_code}] 数据量大({len(df_display)}根K线)，只显示最近{max_candles}根")
            df_display = df_display.iloc[-max_candles:].copy()
        
        # 准备mplfinance数据
        df_mpf = df_display.copy()
        df_mpf['date'] = pd.to_datetime(df_mpf['date'])
        df_mpf.set_index('date', inplace=True)
        df_mpf = df_mpf[['open', 'high', 'low', 'close']].copy()
        
        if df_mpf.empty:
            logger.warning(f"⚠️ [{stock_code}] 处理后的数据为空")
            return False
        
        logger.info(f"📊 [{stock_code}] 绘制{len(df_mpf)}根K线")
        
        # 准备额外的绘图元素
        additional_plots = []
        
        # 1. 添加阶段低点水平线（蓝色实线）
        for i, (idx, price, date_str) in enumerate(stage_lows):
            hline_data = [price] * len(df_mpf)
            additional_plots.append(mpf.make_addplot(hline_data, color='blue', linestyle='-', width=2, alpha=0.8))
        
        # 2. 添加百分比涨幅线（粉色虚线）
        base_price = min(price for _, price, _ in stage_lows)
        max_price = df_mpf['high'].max()
        
        visible_percent_lines = []
        highest_visible_idx = -1
        
        for i, percent_str in enumerate(mid_drawer.percent_list):
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
        
        # 添加K线上方的额外百分比线
        if highest_visible_idx >= 0 and highest_visible_idx + 1 < len(mid_drawer.percent_list):
            try:
                next_percent_str = mid_drawer.percent_list[highest_visible_idx + 1]
                next_percent = float(next_percent_str.rstrip('%')) / 100
                next_target_price = base_price * (1 + next_percent)
                hline_data = [next_target_price] * len(df_mpf)
                additional_plots.append(mpf.make_addplot(hline_data, color='hotpink', linestyle='--', width=3, alpha=0.8))
                visible_percent_lines.append((next_percent_str, next_target_price))
            except (ValueError, TypeError):
                pass
        
        # 构建标题
        industry = ""
        pe_val = 0
        total_share = 0
        total_market_cap = 0
        
        if stock_code in mid_drawer.stock_info:
            info = mid_drawer.stock_info[stock_code]
            industry = info.get('industry', '')
            pe_val = float(info.get('pe', 0))
            total_share = float(info.get('total_share', 0))
            
            if total_share > 0 and len(df_mpf) > 0:
                current_price = float(df_mpf['close'].iloc[-1])
                total_market_cap = total_share * current_price
        
        title_parts = [stock_code, stock_name]
        
        if industry and industry != "未知行业":
            title_parts.append(f"({industry})")
        
        if total_market_cap > 0:
            if total_market_cap >= 1000:
                title_parts.append(f"总市值:{total_market_cap:.0f}亿")
            else:
                title_parts.append(f"总市值:{total_market_cap:.1f}亿")
        
        if pe_val > 0:
            title_parts.append(f"PE:{pe_val:.2f}")
        elif pe_val == 0:
            title_parts.append("PE:亏损")
        
        title = " ".join(title_parts) + " - ALL"
        
        # 设置样式
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        style = mpf.make_mpf_style(
            base_mpf_style='charles',
            marketcolors=mpf.make_marketcolors(
                up='red', down='green', edge='inherit', wick='inherit', volume='inherit'
            ),
            gridstyle='-', gridcolor='lightgray', y_on_right=True,
            facecolor='white', edgecolor='black', figcolor='white',
            rc={'font.size': 12, 'axes.titlesize': 20, 'axes.labelsize': 14, 
                'font.sans-serif': ['SimHei', 'Arial Unicode MS', 'DejaVu Sans'],
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
        
        # 3. 标注阶段低点价格
        for i, (idx, price, date_str) in enumerate(stage_lows):
            ax.text(1.02, price, f'{price:.2f}', 
                   fontsize=16, color='blue', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                   transform=ax.get_yaxis_transform(), ha='left', va='center')
        
        # 4. 标注百分比涨幅线
        max_price = df_mpf['high'].max()
        highest_visible_idx = -1
        
        for i, percent_str in enumerate(mid_drawer.percent_list):
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
        
        if highest_visible_idx >= 0 and highest_visible_idx + 1 < len(mid_drawer.percent_list):
            try:
                next_percent_str = mid_drawer.percent_list[highest_visible_idx + 1]
                next_percent = float(next_percent_str.rstrip('%')) / 100
                next_target_price = base_price * (1 + next_percent)
                
                ax.text(1.02, next_target_price, f'+{next_percent_str}', 
                       fontsize=18, color='#8B7355', fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                alpha=0.9, edgecolor='#8B7355', linewidth=2),
                       transform=ax.get_yaxis_transform(), ha='left', va='center')
            except (ValueError, TypeError):
                pass
        
        # 5. 添加AnchorM线（紫色）
        if m_lines_result:
            best_M = m_lines_result['best_M']
            M_B_values = m_lines_result['B_values']
            M_K_values = m_lines_result['K_values']
            
            m_line_style = mid_drawer.anchor_m_config.get('line_style', {})
            m_line_color = m_line_style.get('color', '#8A2BE2')
            m_line_width = m_line_style.get('linewidth', 3.0)
            m_line_alpha = m_line_style.get('alpha', 0.9)
            
            m_text_style = mid_drawer.anchor_m_config.get('text_style', {})
            m_text_fontsize = m_text_style.get('fontsize', 14)
            m_annotate_format = mid_drawer.anchor_m_config.get('annotate_format', 'K={K} 价格={price}')
            
            # 绘制紫色横线
            for k_val, B_k_price in zip(M_K_values, M_B_values):
                ax.axhline(y=B_k_price, color=m_line_color, 
                          linestyle='-', linewidth=m_line_width, 
                          alpha=m_line_alpha, zorder=2.5)
                
                label_text = m_annotate_format.replace('{K}', str(k_val)).replace('{price}', f'{B_k_price:.2f}')
                ax.text(-0.02, B_k_price, label_text,
                       fontsize=m_text_fontsize, color=m_line_color, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.85, 
                                edgecolor=m_line_color, linewidth=2),
                       transform=ax.get_yaxis_transform(), ha='right', va='center')
            
            # 在图片左上角添加M值信息
            text_lines = [f"M={best_M:.1f}%"]
            
            if 'per_k_matches' in m_lines_result:
                matched_B = []
                for match in m_lines_result['per_k_matches']:
                    if match.get('score', 0) > 0:
                        k_val = match['k']
                        B_k = match['B_k']
                        score = match['score']
                        matched_B.append(f"k{k_val}:{B_k:.2f}({score:.0f})")
                        if len(matched_B) >= 10:
                            break
                
                if matched_B:
                    if len(m_lines_result['per_k_matches']) > len(matched_B):
                        matched_B.append('...')
                    text_lines.append(f"Match_B: [{', '.join(matched_B)}]")
                else:
                    text_lines.append(f"Match_B: [无匹配]")
            
            text_lines.append(f"AvgScore: {m_lines_result['avg_score']:.1f}")
            text_lines.append(f"Matches: {m_lines_result['matches_count']}/{len(M_B_values)}")
            
            text_content = '\n'.join(text_lines)
            ax.text(0.01, 0.98, text_content,
                   transform=ax.transAxes,
                   fontsize=11, color='purple', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.95, 
                            edgecolor='purple', linewidth=2.5),
                   ha='left', va='top', family='monospace')
            
            logger.info(f"✅ [{stock_code}] 绘制AnchorM线: M={best_M:.1f}%, {len(M_B_values)}条线")
        
        # 6. 添加AnchorBack线（蓝色）
        if back_data:
            best_N = back_data['best_N']
            B_B_values = back_data['B_values']
            B_K_values = back_data['K_values']
            
            b_line_style = back_drawer.anchor_back_config.get('line_style', {})
            b_line_color = b_line_style.get('color', '#1E90FF')
            b_line_width = b_line_style.get('linewidth', 3.0)
            b_line_alpha = b_line_style.get('alpha', 0.9)
            
            b_text_style = back_drawer.anchor_back_config.get('text_style', {})
            b_text_fontsize = b_text_style.get('fontsize', 14)
            b_annotate_format = back_drawer.anchor_back_config.get('annotate_format', 'K={K} 价格={price}')
            
            # 绘制蓝色横线（使用axhline确保价格对齐准确）
            for k_val, B_k_price in zip(B_K_values, B_B_values):
                label_text = b_annotate_format.replace('{K}', str(k_val)).replace('{price}', f'{B_k_price:.2f}')
                
                # 先绘制标签(在最左边)
                # 设置zorder=10确保蓝色标签显示在紫色标签上面
                ax.text(-0.15, B_k_price, label_text,
                       fontsize=10, color=b_line_color, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan', alpha=0.85, 
                                edgecolor=b_line_color, linewidth=1.5),
                       transform=ax.get_yaxis_transform(), ha='left', va='center', zorder=10)
                
                # 获取图表的x轴范围,用于计算连接线位置
                xlim = ax.get_xlim()
                x_range = xlim[1] - xlim[0]
                
                # 计算连接线和主蓝线的精确位置
                # 标签框右端约在xlim[0] - 0.04 * x_range位置
                x_label_end = xlim[0]- 0.085 * x_range  
                # 主蓝线从xlim[0]开始(图表左边界)
                x_chart_start = xlim[0]
                
                # 绘制连接线:从标签框右端到图表左边界,完全连接
                ax.plot([x_label_end, x_chart_start], [B_k_price, B_k_price], 
                       color=b_line_color, linestyle='-', linewidth=2.0, 
                       alpha=b_line_alpha, zorder=10, clip_on=False)
                
                # 绘制主蓝线:从图表左边界到右边界外延长
                # 使用plot绘制主蓝线,从图表左边界开始延伸到右边外
                x_main_end = xlim[1] + 0.02* x_range  # 延伸到右边界外
                ax.plot([x_chart_start, x_main_end], [B_k_price, B_k_price], 
                       color=b_line_color, linestyle='-', linewidth=2.0, 
                       alpha=b_line_alpha, zorder=2.4, clip_on=False)
            
            # 在图片右上角添加N值信息
            text_lines = [f"N={best_N:.2f}"]
            
            if 'per_k_matches' in back_data:
                matched_items = []
                for match in back_data['per_k_matches']:
                    if match.get('score', 0) > 0:
                        k_val = match['k']
                        B_k = match['B_k']
                        score = match['score']
                        matched_items.append(f"K{k_val}:{B_k:.2f}({score:.0f})")
                        if len(matched_items) >= 4:
                            break
                
                if matched_items:
                    if len(back_data['per_k_matches']) > len(matched_items):
                        matched_items.append('...')
                    text_lines.append(f"Match_B: [{', '.join(matched_items)}]")
                else:
                    text_lines.append(f"Match_B: []")
            else:
                text_lines.append(f"Match_B: []")
            
            text_lines.append(f"AvgScore: {back_data['avg_score']:.1f}")
            text_lines.append(f"Matches: {back_data['matches_count']}/{len(B_B_values)}")
            
            text_content = '\n'.join(text_lines)
            ax.text(0.99, 0.98, text_content,
                   transform=ax.transAxes,
                   fontsize=11, color=b_line_color, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.95, 
                            edgecolor=b_line_color, linewidth=2.5),
                   ha='right', va='top', family='monospace')
            
            logger.info(f"✅ [{stock_code}] 绘制AnchorBack线: N={best_N:.2f}, {len(B_B_values)}条线")
        
        # 7. 统一调整Y轴范围
        min_price = df_mpf['low'].min()
        
        # 计算最高的百分比线
        highest_percent_price = max_price
        highest_visible_idx = -1
        
        for i, percent_str in enumerate(mid_drawer.percent_list):
            try:
                percent = float(percent_str.rstrip('%')) / 100
                target_price = base_price * (1 + percent)
                if target_price <= max_price:
                    highest_visible_idx = i
                    highest_percent_price = target_price
            except (ValueError, TypeError):
                continue
        
        if highest_visible_idx >= 0 and highest_visible_idx + 1 < len(mid_drawer.percent_list):
            try:
                next_percent_str = mid_drawer.percent_list[highest_visible_idx + 1]
                next_percent = float(next_percent_str.rstrip('%')) / 100
                next_target_price = base_price * (1 + next_percent)
                highest_percent_price = next_target_price
            except (ValueError, TypeError):
                pass
        
        # 考虑M线和B线的最高价格
        highest_line_price = highest_percent_price
        if m_lines_result and m_lines_result['B_values']:
            highest_m_price = max(m_lines_result['B_values'])
            highest_line_price = max(highest_line_price, highest_m_price)
        if back_data and back_data['B_values']:
            highest_b_price = max(back_data['B_values'])
            highest_line_price = max(highest_line_price, highest_b_price)
        
        # 设置Y轴范围
        y_margin = (highest_line_price - min_price) * 0.05
        ax.set_ylim(min_price - y_margin, highest_line_price + y_margin)
        logger.debug(f"📊 [{stock_code}] Y轴范围: {min_price:.2f} - {highest_line_price:.2f}")
        
        # 8. 绘制最后一个交易日的收盘价横线
        last_close_price = df_mpf['close'].iloc[-1]
        ax.axhline(y=last_close_price, color='red', linestyle='-', linewidth=3, alpha=0.8, zorder=3)
        
        ax.text(1.02, last_close_price, f'{last_close_price:.2f}', 
               fontsize=16, color='red', fontweight='bold',
               transform=ax.get_yaxis_transform(), ha='left', va='center')
        logger.debug(f"📊 [{stock_code}] 最后交易日收盘价横线: {last_close_price:.2f}")
        
        # 9. 保存图表
        plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        
        # 10. 调整图片尺寸
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
                return True
            else:
                logger.warning(f"⚠️ 生成的图片文件过小: {output_file} ({file_size} bytes)")
                return False
        else:
            logger.error(f"❌ 图表文件未生成: {output_file}")
            return False
        
    except Exception as e:
        logger.error(f"❌ [{stock_code}] 图表创建失败: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def get_stock_list_from_csv(csv_files: list) -> Dict[str, tuple]:
    """从CSV文件内容中提取股票代码、名称和前缀
    
    Returns:
        Dict[str, tuple]: {股票代码: (股票名称, 前缀)}
    """
    stock_dict = {}
    
    for csv_file in csv_files:
        try:
            logger.info(f"📄 读取文件: {csv_file}")
            df = pd.read_csv(csv_file)
            
            # 从文件名提取前缀
            file_name = os.path.basename(csv_file)
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
            
            # 从CSV内容中提取股票信息
            for _, row in df.iterrows():
                code = str(row.get('code', ''))
                name = str(row.get('name', code))
                
                if code:
                    normalized_code = code.zfill(6)
                    if normalized_code not in stock_dict:
                        stock_dict[normalized_code] = (name, file_prefix)
                        
        except Exception as e:
            logger.warning(f"⚠️ 读取文件失败: {csv_file}, {e}")
            continue
    
    return stock_dict


def main():
    parser = argparse.ArgumentParser(description='生成ALL画线图（整合M线和B线）')
    parser.add_argument('--date', type=str, help='日期（格式：YYYY-MM-DD）')
    parser.add_argument('--codes', nargs='+', type=str, help='股票代码列表')
    parser.add_argument('--workers', type=int, default=1, help='线程数（为兼容流水线保留，当前未使用）')
    
    args = parser.parse_args()
    
    if args.date:
        date_str = args.date.replace('-', '')
    else:
        date_str = datetime.now().strftime('%Y%m%d')
    
    logger.info("=" * 80)
    logger.info(f"📅 处理日期: {args.date if args.date else '当前日期'}")
    logger.info("=" * 80)
    
    # 初始化画线器
    logger.info("🔧 初始化MidLineDrawer和BackLineDrawer...")
    mid_drawer = MidLineDrawer()
    back_drawer = BackLineDrawer()
    
    # 输出目录
    output_dir = f"{date_str}-drawLineAll"
    
    # 创建输出目录（覆盖模式，不清空）
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_dir) and os.listdir(output_dir):
        logger.info(f"📁 输出目录已存在: {output_dir}（将覆盖同名文件）")
    else:
        logger.info(f"📁 创建输出目录: {output_dir}")
    
    # 获取需要处理的股票列表
    if args.codes:
        # 从指定代码获取股票列表（无前缀）
        stock_dict = {}
        for code in args.codes:
            # 优先从stock_info中获取股票名称
            if code in mid_drawer.stock_info:
                name = mid_drawer.stock_info[code].get('name', code)
                stock_dict[code] = (name, "")  # 无前缀
            else:
                # 其次从CSV文件名中提取股票名称
                csv_files = glob.glob(f"../data/{code}*.csv")
                if csv_files:
                    csv_file = csv_files[0]
                    base_name = os.path.basename(csv_file)
                    parts = base_name.replace('.csv', '').split('_')
                    name = '_'.join(parts[1:]) if len(parts) > 1 else code
                    stock_dict[code] = (name, "")  # 无前缀
                else:
                    stock_dict[code] = (code, "")  # 无前缀
    else:
        # 从resByFilter目录获取股票列表（带前缀）
        filter_dir = f"../{date_str}-resByFilter"
        if not os.path.exists(filter_dir):
            logger.error(f"❌ 找不到目录: {filter_dir}")
            return
        
        csv_files = glob.glob(os.path.join(filter_dir, "*.csv"))
        if not csv_files:
            logger.error(f"❌ 在 {filter_dir} 中没有找到CSV文件")
            return
        
        stock_dict = get_stock_list_from_csv(csv_files)
    
    if not stock_dict:
        logger.error(f"❌ 没有找到需要处理的股票")
        return
    
    logger.info(f"📂 找到 {len(stock_dict)} 只股票")
    
    # 处理每只股票
    success_count = 0
    failed_count = 0
    
    for code, (name, prefix) in stock_dict.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"📈 [{code}] {name}" + (f" ({prefix})" if prefix else ""))
        logger.info(f"{'='*60}")
        
        # 构造输出文件路径（带前缀）
        if prefix:
            output_file = os.path.join(output_dir, f"{prefix}_{code}_{name}_3all.png")
        else:
            output_file = os.path.join(output_dir, f"{code}_{name}_3all.png")
        
        # 绘制ALL图
        if draw_all_for_stock(mid_drawer, back_drawer, code, name, output_file):
            success_count += 1
            logger.info(f"✅ [{code}] 绘制成功")
        else:
            failed_count += 1
            logger.warning(f"❌ [{code}] 绘制失败")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ 完成！")
    logger.info(f"📊 成功: {success_count}/{len(stock_dict)} 张")
    if failed_count > 0:
        logger.info(f"❌ 失败: {failed_count} 张")
    logger.info(f"📁 输出目录: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
