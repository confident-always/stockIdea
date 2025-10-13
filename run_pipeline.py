#!/usr/bin/env python3
"""
股票数据处理流水线脚本
依次执行README myself.md中的三个脚本命令：
1. fetch_kline_akshare.py - 获取股票历史数据
2. select_stock.py - 进行选股
3. adx_filter.py - 进行涨跌幅过滤
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_command(command, description):
    """
    执行命令并处理结果
    
    Args:
        command (str): 要执行的命令
        description (str): 命令描述
    
    Returns:
        bool: 执行是否成功
    """
    print(f"\n{'='*60}")
    print(f"开始执行: {description}")
    print(f"命令: {command}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    try:
        # 使用shell=True来支持复杂的命令
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=False,  # 让输出直接显示在终端
            text=True,
            cwd=os.getcwd()
        )
        
        print(f"\n✅ {description} 执行成功!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} 执行失败!")
        print(f"错误代码: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ {description} 执行出现异常: {str(e)}")
        return False

def main():
    """主函数"""
    print("🚀 开始执行股票数据处理流水线")
    print(f"工作目录: {os.getcwd()}")
    
    # 定义三个要执行的命令
    commands = [
        {
            "command": "python fetch_kline_akshare.py --start 0 --end today --stocklist ./stocklist.csv --exclude-boards gem star bj --out ./data --workers 12",
            "description": "第一步: 获取股票历史数据 (fetch_kline_akshare.py)"
        },
        {
            "command": "python select_stock.py --data-dir ./data --config ./configs.json --meta-workers 6",
            "description": "第二步: 进行选股 (select_stock.py)"
        },
        {
            "command": "python adx_filter.py --input-dir res --output-dir resByFilter --workers 6",
            "description": "第三步: 进行涨跌幅过滤 (adx_filter.py)"
        }
    ]
    
    # 记录开始时间
    start_time = time.time()
    
    # 依次执行每个命令
    for i, cmd_info in enumerate(commands, 1):
        step_start_time = time.time()
        
        success = run_command(cmd_info["command"], cmd_info["description"])
        
        step_duration = time.time() - step_start_time
        print(f"步骤 {i} 耗时: {step_duration:.2f} 秒")
        
        if not success:
            print(f"\n💥 流水线在第 {i} 步失败，停止执行后续步骤")
            sys.exit(1)
        
        # 在步骤之间添加短暂延迟
        if i < len(commands):
            print(f"\n⏳ 等待 2 秒后继续下一步...")
            time.sleep(2)
    
    # 计算总耗时
    total_duration = time.time() - start_time
    
    print(f"\n🎉 所有步骤执行完成!")
    print(f"总耗时: {total_duration:.2f} 秒 ({total_duration/60:.1f} 分钟)")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()