#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画线流水线脚本 - 先后运行 draw_lines_mid.py 和 draw_lines_back.py
将两种算法的结果输出到同一个目录，通过文件名后缀区分

功能特性：
1. 按顺序运行 AnchorM 和 AnchorBack 两种算法
2. 统一输出目录：{日期}-drawLine
3. 文件命名规则：
   - AnchorM: {前缀}_{代码}_{股票名}_1mid.png
   - AnchorBack: {前缀}_{代码}_{股票名}_2back.png
4. 通过subprocess调用独立脚本，确保配置和逻辑完全独立
5. 自动重命名和合并输出文件

使用方法：
    # 处理指定日期的股票
    python run_draw_lines_pipeline.py --date 2025-10-22
    
    # 指定线程数
    python run_draw_lines_pipeline.py --date 2025-10-22 --workers 4
"""

import os
import sys
import subprocess
import logging
import shutil
import glob
from datetime import datetime
from pathlib import Path
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('run_draw_lines_pipeline.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def run_script_with_params(script_name: str, date: str, workers: int, script_type: str) -> tuple:
    """
    运行指定的画线脚本
    
    Args:
        script_name: 脚本名称
        date: 日期参数（YYYY-MM-DD格式）
        workers: 线程数
        script_type: 脚本类型（'mid' 或 'back'）
        
    Returns:
        tuple: (是否成功, 输出目录)
    """
    logger.info(f"")
    logger.info(f"{'='*80}")
    logger.info(f"🚀 开始运行脚本: {script_name}")
    logger.info(f"📅 日期: {date}")
    logger.info(f"🧵 线程数: {workers}")
    logger.info(f"{'='*80}")
    
    try:
        # 构建命令
        cmd = [
            sys.executable,  # Python解释器
            script_name,
            '--date', date,
            '--workers', str(workers)
        ]
        
        logger.info(f"💻 执行命令: {' '.join(cmd)}")
        
        # 运行脚本
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        # 输出标准输出
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
        
        # 输出标准错误
        if result.stderr:
            for line in result.stderr.strip().split('\n'):
                if line.strip():
                    logger.warning(f"  ⚠️ {line}")
        
        # 检查返回码
        if result.returncode != 0:
            logger.error(f"❌ 脚本 {script_name} 执行失败，返回码: {result.returncode}")
            return False, None
        
        # 推断输出目录
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        date_str = date_obj.strftime('%Y%m%d')
        
        # 根据脚本类型确定输出目录
        if script_type == 'mid':
            output_dir = f"{date_str}-drawLineMid"
        elif script_type == 'back':
            output_dir = f"{date_str}-drawLineBack"
        else:
            output_dir = f"{date_str}-drawLineRes"
        
        logger.info(f"✅ 脚本 {script_name} 执行成功")
        logger.info(f"📁 输出目录: {output_dir}")
        return True, output_dir
        
    except Exception as e:
        logger.error(f"❌ 运行脚本 {script_name} 时发生异常: {e}")
        return False, None


def move_files(source_dir: str, target_dir: str):
    """
    将源目录的文件移动到目标目录（文件已经有正确的后缀）
    
    Args:
        source_dir: 源目录
        target_dir: 目标目录
    """
    if not os.path.exists(source_dir):
        logger.warning(f"⚠️ 源目录不存在: {source_dir}")
        return 0
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 查找所有PNG文件
    png_files = glob.glob(os.path.join(source_dir, "*.png"))
    
    if not png_files:
        logger.warning(f"⚠️ 源目录 {source_dir} 中没有PNG文件")
        return 0
    
    logger.info(f"📦 移动 {len(png_files)} 个图片文件...")
    
    moved_count = 0
    for png_file in png_files:
        try:
            # 获取文件名（不含路径）
            base_name = os.path.basename(png_file)
            
            # 目标文件路径
            target_file = os.path.join(target_dir, base_name)
            
            # 移动文件
            shutil.move(png_file, target_file)
            moved_count += 1
            
        except Exception as e:
            logger.error(f"❌ 移动文件 {png_file} 失败: {e}")
    
    logger.info(f"✅ 已移动 {moved_count} 个文件到 {target_dir}")
    
    # 删除源目录（如果为空）
    try:
        if os.path.exists(source_dir):
            remaining_files = os.listdir(source_dir)
            if not remaining_files:
                os.rmdir(source_dir)
                logger.info(f"🗑️  已删除空目录: {source_dir}")
            else:
                logger.info(f"📁 保留目录（还有 {len(remaining_files)} 个其他文件）: {source_dir}")
    except Exception as e:
        logger.warning(f"⚠️ 删除目录 {source_dir} 失败: {e}")
    
    return moved_count


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="画线流水线脚本 - 先后运行 AnchorM 和 AnchorBack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理当前日期的resByFilter中的股票
  python run_draw_lines_pipeline.py
  
  # 处理指定日期的resByFilter中的股票
  python run_draw_lines_pipeline.py --date 2025-10-22
  
  # 指定线程数
  python run_draw_lines_pipeline.py --date 2025-10-22 --workers 6

输出说明:
  输出目录: {日期}-drawLine/
  文件命名:
    - AnchorM图表: {前缀}_{代码}_{股票名}_1mid.png
    - AnchorBack图表: {前缀}_{代码}_{股票名}_2back.png
        """
    )
    
    parser.add_argument('--date', type=str, 
                       help='日期参数，格式为YYYY-MM-DD')
    parser.add_argument('--workers', type=int, default=4,
                       help='并发处理的线程数 (默认: 4)')
    
    args = parser.parse_args()
    
    # 处理日期参数
    if args.date:
        date_str = args.date
        try:
            # 验证日期格式
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            date_num = date_obj.strftime('%Y%m%d')
        except ValueError:
            logger.error(f"❌ 日期格式错误: {date_str}，请使用YYYY-MM-DD格式")
            sys.exit(1)
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')
        date_num = datetime.now().strftime('%Y%m%d')
    
    # 最终输出目录
    final_output_dir = f"{date_num}-drawLine"
    
    logger.info(f"")
    logger.info(f"{'='*80}")
    logger.info(f"🎬 画线流水线启动")
    logger.info(f"📅 处理日期: {date_str}")
    logger.info(f"🧵 线程数: {args.workers}")
    logger.info(f"📁 最终输出目录: {final_output_dir}")
    logger.info(f"{'='*80}")
    
    # 清空最终输出目录
    if os.path.exists(final_output_dir):
        logger.info(f"🗑️  清空输出目录: {final_output_dir}")
        try:
            shutil.rmtree(final_output_dir)
        except Exception as e:
            logger.warning(f"⚠️ 清空目录失败: {e}")
    
    # 创建最终输出目录
    os.makedirs(final_output_dir, exist_ok=True)
    
    # 步骤1：运行 draw_lines_mid.py (AnchorM算法)
    success_mid, mid_output_dir = run_script_with_params(
        'draw_lines_mid.py',
        date_str,
        args.workers,
        'mid'
    )
    
    if not success_mid:
        logger.error(f"❌ AnchorM 脚本执行失败，流水线中止")
        sys.exit(1)
    
    # 移动 AnchorM 的输出文件（文件已经有 _1mid 后缀）
    mid_moved = 0
    if mid_output_dir:
        logger.info(f"")
        logger.info(f"📦 处理 AnchorM 输出文件...")
        mid_moved = move_files(mid_output_dir, final_output_dir)
    
    # 步骤2：运行 draw_lines_back.py (AnchorBack算法)
    success_back, back_output_dir = run_script_with_params(
        'draw_lines_back.py',
        date_str,
        args.workers,
        'back'
    )
    
    if not success_back:
        logger.error(f"❌ AnchorBack 脚本执行失败")
        sys.exit(1)
    
    # 移动 AnchorBack 的输出文件（文件已经有 _2back 后缀）
    back_moved = 0
    if back_output_dir:
        logger.info(f"")
        logger.info(f"📦 处理 AnchorBack 输出文件...")
        back_moved = move_files(back_output_dir, final_output_dir)
    
    # 统计最终结果
    logger.info(f"")
    logger.info(f"{'='*80}")
    logger.info(f"📊 统计最终输出...")
    
    final_files = glob.glob(os.path.join(final_output_dir, "*.png"))
    mid_files = [f for f in final_files if '_1mid.png' in f]
    back_files = [f for f in final_files if '_2back.png' in f]
    
    logger.info(f"✅ AnchorM (1mid): 移动了 {mid_moved} 个文件")
    logger.info(f"✅ AnchorBack (2back): 移动了 {back_moved} 个文件")
    logger.info(f"📁 最终目录中的文件:")
    logger.info(f"   - _1mid.png: {len(mid_files)} 张")
    logger.info(f"   - _2back.png: {len(back_files)} 张")
    logger.info(f"   - 总计: {len(final_files)} 张")
    logger.info(f"📁 输出目录: {final_output_dir}")
    logger.info(f"{'='*80}")
    logger.info(f"🎉 画线流水线全部完成!")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
