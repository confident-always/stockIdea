#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALL画线脚本 - 在Mid图片基础上绘制AnchorBack线
将AnchorM（紫色线）和AnchorBack（蓝色线）合并到一张图中

策略：
1. 加载mid图片作为基础（不存在则生成）
2. 调用BackLineDrawer计算AnchorBack线的数据
3. 在mid图片上直接绘制AnchorBack蓝色线条和右上角N信息框

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
import shutil
import subprocess
import pandas as pd
import numpy as np
from typing import Optional, Dict
from PIL import Image

# 设置matplotlib后端和字体
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

# 导入BackLineDrawer类
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draw_lines_back import BackLineDrawer


def ensure_mid_exists(date_str: str, codes: list = None) -> bool:
    """确保mid图片存在，如果不存在则生成"""
    mid_dir = f"{date_str}-drawLineMid"
    
    if os.path.exists(mid_dir):
        mid_files = glob.glob(os.path.join(mid_dir, "*_1mid.png"))
        if mid_files:
            logger.info(f"✅ Mid图片已存在: {len(mid_files)} 张")
            return True
    
    logger.info(f"⚠️ Mid图片不存在，开始生成...")
    cmd = ["python", "draw_lines_mid.py", "--date", date_str]
    if codes:
        cmd.extend(["--codes"] + codes)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"❌ Mid图片生成失败: {result.stderr}")
            return False
        logger.info(f"✅ Mid图片生成完成")
        return True
    except Exception as e:
        logger.error(f"❌ 运行draw_lines_mid.py失败: {str(e)}")
        return False


def draw_all_for_stock(back_drawer: BackLineDrawer,
                       mid_file: str,
                       output_file: str,
                       code: str) -> bool:
    """在mid图片基础上绘制back线条和信息框"""
    
    try:
        # 读取CSV数据
        csv_file = f"../data/{code}.csv"
        if not os.path.exists(csv_file):
            logger.warning(f"⚠️ 找不到CSV文件: {csv_file}")
            # 没有数据，直接复制mid图片
            shutil.copy2(mid_file, output_file)
            return True
        
        df = pd.read_csv(csv_file)
        if df.empty:
            logger.warning(f"⚠️ {csv_file} 数据为空")
            shutil.copy2(mid_file, output_file)
            return True
        
        # 确保date列是datetime类型
        df['date'] = pd.to_datetime(df['date'])
        
        # 检测阶段低点
        stage_lows = back_drawer.find_stage_lows_unified(df)
        if not stage_lows:
            logger.info(f"ℹ️ {code} 无阶段低点，使用mid图片")
            shutil.copy2(mid_file, output_file)
            return True
        
        # 计算AnchorBack线数据
        anchor_idx, anchor_low, anchor_date = stage_lows[0]
        back_data = back_drawer.compute_anchor_back_lines(df, anchor_idx, anchor_date, code)
        
        if back_data is None:
            logger.info(f"ℹ️ {code} 无AnchorBack数据，使用mid图片")
            shutil.copy2(mid_file, output_file)
            return True
        
        # 先复制mid图片
        shutil.copy2(mid_file, output_file)
        
        # 在mid图片上绘制back线条和信息框
        draw_back_on_image(output_file, back_data, back_drawer.anchor_back_config)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 处理 {code} 失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # 失败时也复制mid图片
        try:
            shutil.copy2(mid_file, output_file)
        except:
            pass
        return False


def draw_back_on_image(image_path: str, 
                      back_data: Dict,
                      config: Dict) -> None:
    """在图片上绘制AnchorBack线条和右上角N信息框"""
    
    try:
        # 打开图片
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # 计算DPI和figsize（匹配原图）
        dpi = 100
        figsize = (img_width / dpi, img_height / dpi)
        
        # 创建figure和axes
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])  # 使用绝对坐标，无边距
        
        # 显示原图
        ax.imshow(img, aspect='auto')
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # Y轴反转（图片坐标系）
        ax.axis('off')
        
        # 获取back数据
        best_N = back_data['best_N']
        B_values = back_data['B_values']
        K_values = back_data['K_values']
        anchor_A = back_data['anchor_A']
        avg_score = back_data['avg_score']
        matches_count = back_data['matches_count']
        
        # 获取线条样式配置（放大显示）
        line_style = config.get('line_style', {})
        line_color = line_style.get('color', '#1E90FF')
        line_width = line_style.get('linewidth', 3.0) * 1.3  # 增加30%线宽
        line_alpha = line_style.get('alpha', 0.9)
        
        text_style = config.get('text_style', {})
        text_fontsize = text_style.get('fontsize', 14) * 1.2  # 增加20%字体大小
        
        # 图表区域（像素坐标）
        # 根据mplfinance标准布局：左100px，右100px，顶135px，底100px
        chart_left = 100
        chart_right = img_width - 100
        chart_top = 135
        chart_bottom = img_height - 100
        
        # 从图片中推断价格范围（使用back数据）
        price_min = anchor_A * 0.85  # 留一些余量
        price_max = max(B_values) * 1.15 if B_values else anchor_A * 2
        
        # 绘制每条蓝色横线和标注
        for k_val, B_k_price in zip(K_values, B_values):
            # 价格转y坐标（像素）
            if price_max > price_min:
                y_px = chart_top + (price_max - B_k_price) / (price_max - price_min) * (chart_bottom - chart_top)
            else:
                y_px = (chart_top + chart_bottom) / 2
            
            # 绘制横线
            ax.plot([chart_left, chart_right], [y_px, y_px],
                   color=line_color, linewidth=4, 
                   alpha=line_alpha, linestyle='-', zorder=10)
            
            # 在左侧添加标注
            label_text = f"K={k_val} 价格={B_k_price:.2f}"
            ax.text(chart_left +80, y_px, label_text,
                   fontsize=text_fontsize, color=line_color, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.4", facecolor='lightcyan', 
                           edgecolor=line_color, linewidth=2, alpha=0.85),
                   ha='right', va='center', zorder=11)
        
        # 绘制右上角N信息框（放大并向左移动）
        # 左上角M框位置：x=130, y=2, width=450, height=75
        info_box_left = 130
        info_box_top = 2
        info_box_width = 1000  # 放大：从450增加到520
        info_box_height = 180  # 放大：从75增加到90
        
        # 右上角位置（向左移动50px）
        box_right = img_width - info_box_left - info_box_width - 220  # 向左移动50px
        box_top = info_box_top
        
        # 绘制信息框背景（放大）
        box = mpatches.FancyBboxPatch(
            (box_right, box_top), info_box_width, info_box_height,
            boxstyle="round,pad=8",  # 增加padding
            edgecolor=line_color, facecolor='white',
            linewidth=3, alpha=0.95, zorder=20  # 增加边框宽度
        )
        ax.add_patch(box)
        
        # 准备信息文本（与M框格式完全一致）
        text_lines = []
        
        # 第1行：N=值
        text_lines.append(f"N={best_N:.2f}")
        
        # 第2行：Match_B: [K值:B值(得分), ...]
        if 'per_k_matches' in back_data:
            matched_items = []
            for match in back_data['per_k_matches']:
                if match.get('score', 0) > 0:
                    k_val = match['k']
                    B_k = match['B_k']
                    score = match['score']
                    # 格式：K值:B值(得分)
                    matched_items.append(f"K{k_val}:{B_k:.2f}({score:.0f})")
                    if len(matched_items) >= 4:  # 显示前4个，与M框一致
                        break
            
            if matched_items:
                # 如果有更多匹配，添加 ...
                if len(back_data['per_k_matches']) > len(matched_items):
                    matched_items.append('...')
                text_lines.append(f"Match_B: [{', '.join(matched_items)}]")
            else:
                text_lines.append(f"Match_B: []")
        else:
            text_lines.append(f"Match_B: []")
        
        # 第3行：AvgScore（不带百分号，与M框一致）
        text_lines.append(f"AvgScore: {avg_score:.1f}")
        
        # 第4行：Matches
        text_lines.append(f"Matches: {matches_count}/{len(B_values)}")
        
        # 绘制文本（分行，字体加粗放大，增加行高）
        text_x = box_right + 15
        text_y = box_top + 15
        line_height = 40 # 增加行高，适应更大的字体
        
        for i, line in enumerate(text_lines):
            ax.text(text_x, text_y + i * line_height, line,
                   fontsize=30, color=line_color, fontweight='bold',  # 增大字体到14
                   ha='left', va='top', zorder=21)
        
        # 保存图片（不使用bbox_inches='tight'，保持原始尺寸）
        plt.savefig(image_path, dpi=dpi, pad_inches=0)
        plt.close(fig)
        
        logger.debug(f"✅ 在图片上绘制了 {len(B_values)} 条back线和N信息框")
        
    except Exception as e:
        logger.error(f"❌ 绘制back线条失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # 失败时保持原图
        plt.close('all')


def main():
    parser = argparse.ArgumentParser(description='生成ALL画线图（在Mid基础上添加Back线）')
    parser.add_argument('--date', type=str, help='日期（格式：YYYY-MM-DD）')
    parser.add_argument('--workers', type=int, default=4, help='线程数（用于生成mid）')
    parser.add_argument('--codes', nargs='+', type=str, help='股票代码列表')
    
    args = parser.parse_args()
    
    if args.date:
        date_str = args.date.replace('-', '')
    else:
        date_str = datetime.now().strftime('%Y%m%d')
    
    logger.info("=" * 80)
    logger.info(f"📅 处理日期: {args.date if args.date else '当前日期'}")
    logger.info("=" * 80)
    
    # 确保mid图片存在
    if not ensure_mid_exists(date_str, args.codes):
        logger.error(f"❌ Mid图片不可用")
        return
    
    # 初始化BackLineDrawer
    logger.info("🔧 初始化BackLineDrawer...")
    back_drawer = BackLineDrawer()
    
    # 目录路径
    mid_dir = f"{date_str}-drawLineMid"
    output_dir = f"{date_str}-drawLineAll"
    
    # 创建输出目录（清除旧文件）
    if os.path.exists(output_dir):
        logger.info(f"🗑️  清除旧文件夹: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"📁 创建输出目录: {output_dir}")
    
    # 查找mid图片
    mid_files = glob.glob(os.path.join(mid_dir, "*_1mid.png"))
    
    if not mid_files:
        logger.error(f"❌ 在 {mid_dir} 中没有找到 *_1mid.png 文件")
        return
    
    logger.info(f"📂 找到 {len(mid_files)} 个mid图片")
    
    # 处理每个图片
    success_count = 0
    failed_count = 0
    
    for i, mid_file in enumerate(mid_files, 1):
        # 获取基本文件名和代码
        base_name = os.path.basename(mid_file).replace('_1mid.png', '')
        parts = base_name.split('_')
        
        # 提取股票代码
        if len(parts) >= 2:
            if parts[0].startswith(('ADX', 'PDI')):
                code = parts[1]
            else:
                code = parts[0]
        else:
            logger.warning(f"⚠️ 无法解析文件名: {base_name}")
            continue
        
        # 构造输出文件路径
        output_file = os.path.join(output_dir, f"{base_name}_3all.png")
        
        # 绘制ALL图
        if draw_all_for_stock(back_drawer, mid_file, output_file, code):
            success_count += 1
            if success_count % 10 == 0:
                logger.info(f"✅ [{success_count}/{len(mid_files)}] 已完成 {success_count} 张图片")
        else:
            failed_count += 1
    
    logger.info("=" * 80)
    logger.info(f"✅ 完成！")
    logger.info(f"📊 成功: {success_count}/{len(mid_files)} 张")
    if failed_count > 0:
        logger.info(f"❌ 失败: {failed_count} 张")
    logger.info(f"📁 输出目录: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
