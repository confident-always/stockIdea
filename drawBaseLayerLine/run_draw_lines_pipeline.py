#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画线流水线脚本 - 同时运行 draw_lines_mid.py 和 draw_lines_back.py
将两种算法的结果输出到同一个目录，通过文件名后缀区分

功能特性：
1. 同时运行 AnchorM 和 AnchorBack 两种算法
2. 统一输出目录：{日期}-drawLine
3. 文件命名规则：
   - AnchorM: {前缀}_{代码}_{股票名}_1mid.png
   - AnchorBack: {前缀}_{代码}_{股票名}_2back.png
4. 并行处理，提高效率
5. 统一的日志和结果汇总

使用方法：
    # 处理指定日期的股票
    python run_draw_lines_pipeline.py --date 2025-10-22
    
    # 指定线程数
    python run_draw_lines_pipeline.py --date 2025-10-22 --workers 4
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

# 导入两个画线器（使用别名区分）
from draw_lines_mid import MidLineDrawer as AnchorMDrawer
from draw_lines_back import MidLineDrawer as AnchorBackDrawer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('run_draw_lines_pipeline.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 线程锁
progress_lock = threading.Lock()


class DrawLinesPipeline:
    """画线流水线 - 同时运行两种算法"""
    
    def __init__(self):
        """初始化流水线"""
        self.mid_drawer = AnchorMDrawer()
        self.back_drawer = AnchorBackDrawer()
        self.processed_count = 0
        self.total_count = 0
        logger.info(f"✅ 画线流水线初始化完成")
    
    def process_single_stock(self, stock_code: str, stock_name: str, 
                           output_dir: str, data_dir: str, file_prefix: str = "") -> dict:
        """处理单只股票 - 同时生成两种算法的图表"""
        start_time = time.time()
        result = {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'mid_success': False,
            'back_success': False,
            'elapsed_time': 0,
            'mid_error': None,
            'back_error': None
        }
        
        try:
            # 1. 加载数据（只加载一次，两种算法共用）
            df = self.mid_drawer.validate_and_load_data(stock_code, data_dir)
            if df is None:
                result['mid_error'] = "数据加载失败"
                result['back_error'] = "数据加载失败"
                return result
            
            # 2. 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 3. 生成 AnchorM 图表（1mid）
            if file_prefix and file_prefix != "UNKNOWN":
                mid_output_file = os.path.join(output_dir, f"{file_prefix}_{stock_code}_{stock_name}_1mid.png")
            else:
                mid_output_file = os.path.join(output_dir, f"{stock_code}_{stock_name}_1mid.png")
            
            try:
                mid_success, mid_lines_result = self.mid_drawer.create_mid_chart(
                    stock_code, stock_name, df.copy(), mid_output_file
                )
                result['mid_success'] = mid_success
                
                if mid_success and mid_lines_result:
                    result['anchorMLines'] = {
                        'best_M': mid_lines_result['best_M'],
                        'avg_score': mid_lines_result['avg_score'],
                        'matches_count': mid_lines_result['matches_count']
                    }
            except Exception as e:
                result['mid_error'] = str(e)
                logger.error(f"❌ AnchorM图表创建失败 {stock_code}: {e}")
            
            # 4. 生成 AnchorBack 图表（2back）
            if file_prefix and file_prefix != "UNKNOWN":
                back_output_file = os.path.join(output_dir, f"{file_prefix}_{stock_code}_{stock_name}_2back.png")
            else:
                back_output_file = os.path.join(output_dir, f"{stock_code}_{stock_name}_2back.png")
            
            try:
                back_success, back_lines_result = self.back_drawer.create_mid_chart(
                    stock_code, stock_name, df.copy(), back_output_file
                )
                result['back_success'] = back_success
                
                if back_success and back_lines_result:
                    # draw_lines_back.py 返回的也是 M 值（因为它还使用 AnchorM 的结构）
                    result['anchorBackLines'] = {
                        'best_M': back_lines_result.get('best_M', 0),
                        'avg_score': back_lines_result.get('avg_score', 0),
                        'matches_count': back_lines_result.get('matches_count', 0)
                    }
            except Exception as e:
                result['back_error'] = str(e)
                logger.error(f"❌ AnchorBack图表创建失败 {stock_code}: {e}")
            
            # 5. 更新进度
            with progress_lock:
                self.processed_count += 1
                status_parts = []
                if result['mid_success']:
                    m_val = result.get('anchorMLines', {}).get('best_M', 0)
                    status_parts.append(f"Mid:M={m_val:.1f}%")
                else:
                    status_parts.append("Mid:失败")
                
                if result['back_success']:
                    m_val_back = result.get('anchorBackLines', {}).get('best_M', 0)
                    status_parts.append(f"Back:M={m_val_back:.1f}%")
                else:
                    status_parts.append("Back:失败")
                
                status = " | ".join(status_parts)
                logger.info(f"✅ [{self.processed_count}/{self.total_count}] {stock_code} {stock_name} - {status}")
            
        except Exception as e:
            result['mid_error'] = str(e)
            result['back_error'] = str(e)
            logger.error(f"❌ 处理股票失败 {stock_code}: {e}")
        finally:
            result['elapsed_time'] = time.time() - start_time
        
        return result
    
    def process_stock_list(self, stock_list: List[Tuple[str, str, str, str]], 
                          output_dir: str, data_dir: str = "../data", workers: int = 4):
        """处理股票列表"""
        logger.info(f"🚀 开始画线流水线处理")
        logger.info(f"📁 数据目录: {data_dir}")
        logger.info(f"📁 输出目录: {output_dir}")
        logger.info(f"🧵 线程数: {workers}")
        logger.info(f"📊 算法: AnchorM (紫色) + AnchorBack (蓝色)")
        
        if not stock_list:
            logger.error("❌ 股票列表为空")
            return
        
        self.total_count = len(stock_list)
        self.processed_count = 0
        
        logger.info(f"📊 待处理股票数量: {self.total_count}")
        logger.info(f"📊 预计生成图片数量: {self.total_count * 2} 张")
        
        # 清空并重新创建输出目录
        if os.path.exists(output_dir):
            import shutil
            logger.info(f"🗑️  清空输出目录: {output_dir}")
            try:
                shutil.rmtree(output_dir)
                logger.info(f"✅ 已清空输出目录")
            except Exception as e:
                logger.warning(f"⚠️  清空输出目录时出错: {e}")
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"📁 创建输出目录: {output_dir}")
        
        # 多线程处理
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_stock = {
                executor.submit(self.process_single_stock, code, name, output_dir, data_dir, file_prefix): 
                (code, name, industry, file_prefix)
                for code, name, industry, file_prefix in stock_list
            }
            
            for future in as_completed(future_to_stock):
                result = future.result()
                results.append(result)
        
        # 统计结果
        total_time = time.time() - start_time
        mid_success_count = sum(1 for r in results if r['mid_success'])
        back_success_count = sum(1 for r in results if r['back_success'])
        both_success_count = sum(1 for r in results if r['mid_success'] and r['back_success'])
        
        logger.info(f"")
        logger.info(f"🎉 画线流水线处理完成!")
        logger.info(f"📊 总计股票: {len(results)}只")
        logger.info(f"✅ AnchorM成功: {mid_success_count}只 ({mid_success_count/len(results)*100:.1f}%)")
        logger.info(f"✅ AnchorBack成功: {back_success_count}只 ({back_success_count/len(results)*100:.1f}%)")
        logger.info(f"✅ 两种算法都成功: {both_success_count}只 ({both_success_count/len(results)*100:.1f}%)")
        logger.info(f"⏱️ 总耗时: {total_time:.2f}秒")
        logger.info(f"⚡ 平均速度: {len(results)/total_time:.2f}只/秒")
        
        # 保存处理结果
        results_file = os.path.join(output_dir, "pipeline_results.json")
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"📄 处理结果已保存: {results_file}")
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {e}")
        
        # 显示失败的股票
        failed_stocks = [r for r in results if not (r['mid_success'] or r['back_success'])]
        if failed_stocks:
            logger.warning(f"⚠️ 完全失败的股票（两种算法都失败）:")
            for r in failed_stocks[:10]:
                logger.warning(f"   {r['stock_code']} {r['stock_name']}")
            if len(failed_stocks) > 10:
                logger.warning(f"   ... 还有{len(failed_stocks)-10}只股票完全失败")
        
        # 显示部分失败的股票
        partial_failed = [r for r in results if (r['mid_success'] ^ r['back_success'])]
        if partial_failed:
            logger.warning(f"⚠️ 部分成功的股票（只有一种算法成功）: {len(partial_failed)}只")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="画线流水线脚本 - 同时运行 AnchorM 和 AnchorBack",
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
    
    current_date = datetime.now().strftime('%Y%m%d')
    
    parser.add_argument('--date', type=str, 
                       help='日期参数，格式为YYYY-MM-DD，用于构建resByFilter目录')
    parser.add_argument('--workers', type=int, default=4,
                       help='并发处理的线程数 (默认: 4)')
    
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
    
    # 创建流水线
    pipeline = DrawLinesPipeline()
    
    # 读取指定日期的resByFilter中的股票
    filter_dir = f"../{date_str}-resByFilter"
    if not os.path.exists(filter_dir):
        logger.error(f"❌ 目录不存在: {filter_dir}")
        logger.info(f"💡 提示：请确保存在 {filter_dir} 目录")
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
    
    # 生成统一输出目录
    output_dir = f"{date_str}-drawLine"
    
    # 批量处理股票列表
    pipeline.process_stock_list(stock_list, output_dir, "../data", args.workers)
    
    logger.info("🎉 流水线执行完成!")


if __name__ == "__main__":
    main()

