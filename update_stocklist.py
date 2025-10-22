#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用Tushare的bak_basic接口更新A股股票列表

功能：
1. 从Tushare获取最新的股票数据
2. 包含完整的财务指标
3. 自动更新stocklist.csv

作者：AI Assistant
日期：2025-10-22
参考：https://tushare.pro/document/2?doc_id=262
"""

import tushare as ts
import pandas as pd
import os
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('update_stocklist.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def load_tushare_token():
    """从配置文件加载Tushare token"""
    try:
        token_file = '.tushare_token'
        if os.path.exists(token_file):
            with open(token_file, 'r', encoding='utf-8') as f:
                token = f.read().strip()
                if token:
                    logger.info("✅ 已加载Tushare token")
                    return token
    except Exception as e:
        logger.error(f"❌ 读取token失败: {e}")
    return None


def get_latest_trade_date():
    """获取最新交易日期"""
    try:
        # 获取今天的日期
        today = datetime.now().strftime('%Y%m%d')
        logger.info(f"📅 尝试获取日期: {today}")
        return today
    except Exception as e:
        logger.error(f"❌ 获取日期失败: {e}")
        return None


def fetch_stock_data(token, trade_date):
    """从Tushare获取股票数据"""
    try:
        logger.info("=" * 70)
        logger.info("🚀 开始从Tushare获取股票数据")
        logger.info("=" * 70)
        logger.info("")
        
        # 设置token
        ts.set_token(token)
        pro = ts.pro_api()
        
        logger.info(f"📥 正在获取 {trade_date} 的股票数据...")
        logger.info("   使用接口: bak_basic")
        logger.info("")
        
        # 使用bak_basic接口获取数据
        # 根据文档，包含所有需要的字段
        fields = [
            'trade_date', 'ts_code', 'name', 'industry', 'area',
            'pe', 'float_share', 'total_share', 'total_assets',
            'liquid_assets', 'fixed_assets', 'reserved', 'reserved_pershare',
            'eps', 'bvps', 'pb', 'list_date', 'undp', 'per_undp',
            'rev_yoy', 'profit_yoy', 'gpr', 'npr', 'holder_num'
        ]
        
        df = pro.bak_basic(
            trade_date=trade_date,
            fields=','.join(fields)
        )
        
        if df is None or df.empty:
            logger.warning(f"⚠️ {trade_date} 无数据，可能不是交易日")
            # 尝试前一天
            prev_date = get_previous_date(trade_date)
            logger.info(f"📥 尝试获取 {prev_date} 的数据...")
            df = pro.bak_basic(
                trade_date=prev_date,
                fields=','.join(fields)
            )
        
        if df is not None and not df.empty:
            logger.info(f"✅ 成功获取 {len(df)} 只股票数据")
            logger.info("")
            return df
        else:
            logger.error("❌ 获取数据失败")
            return None
            
    except Exception as e:
        logger.error(f"❌ 获取数据异常: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def get_previous_date(date_str):
    """获取前一天的日期"""
    try:
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        from datetime import timedelta
        prev_date = date_obj - timedelta(days=1)
        return prev_date.strftime('%Y%m%d')
    except:
        return date_str


def process_data(df):
    """处理数据格式"""
    try:
        logger.info("🔄 正在处理数据...")
        
        # 创建symbol列（6位股票代码）
        df['symbol'] = df['ts_code'].str.split('.').str[0]
        
        # 重命名列以匹配原有格式
        df_result = df[[
            'ts_code', 'symbol', 'name', 'area', 'industry',
            'pe', 'float_share', 'total_share', 'total_assets',
            'liquid_assets', 'fixed_assets', 'reserved', 'reserved_pershare',
            'eps', 'bvps', 'pb', 'list_date', 'undp', 'per_undp',
            'rev_yoy', 'profit_yoy', 'gpr', 'npr', 'holder_num'
        ]].copy()
        
        # 处理空值
        df_result['area'] = df_result['area'].fillna('未知')
        df_result['industry'] = df_result['industry'].fillna('未知')
        
        # 按股票代码排序
        df_result = df_result.sort_values('symbol').reset_index(drop=True)
        
        logger.info(f"✅ 数据处理完成")
        logger.info("")
        
        return df_result
        
    except Exception as e:
        logger.error(f"❌ 数据处理失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def backup_file(file_path):
    """备份现有文件"""
    try:
        if os.path.exists(file_path):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{file_path}.backup_{timestamp}"
            import shutil
            shutil.copy2(file_path, backup_path)
            logger.info(f"✅ 已备份: {backup_path}")
            return True
    except Exception as e:
        logger.warning(f"⚠️ 备份失败: {e}")
    return False


def save_to_csv(df, file_path):
    """保存到CSV文件"""
    try:
        logger.info(f"💾 正在保存到 {file_path}...")
        
        df.to_csv(file_path, index=False, encoding='utf-8')
        
        logger.info(f"✅ 保存成功！")
        logger.info("")
        
        # 显示统计信息
        logger.info("📊 数据统计:")
        logger.info(f"   总股票数: {len(df)}")
        logger.info(f"   上海市场: {len(df[df['ts_code'].str.endswith('.SH')])}")
        logger.info(f"   深圳市场: {len(df[df['ts_code'].str.endswith('.SZ')])}")
        logger.info(f"   北京市场: {len(df[df['ts_code'].str.endswith('.BJ')])}")
        logger.info(f"   ST股票: {len(df[df['name'].str.contains('ST', na=False)])}")
        
        # 数据完整性
        has_industry = len(df[df['industry'] != '未知'])
        has_area = len(df[df['area'] != '未知'])
        logger.info(f"   有行业信息: {has_industry} ({has_industry/len(df)*100:.1f}%)")
        logger.info(f"   有地区信息: {has_area} ({has_area/len(df)*100:.1f}%)")
        
        # 财务数据
        has_pe = len(df[df['pe'].notna()])
        logger.info(f"   有市盈率数据: {has_pe} ({has_pe/len(df)*100:.1f}%)")
        
        logger.info("")
        
        # 显示数据示例
        logger.info("📝 数据示例（前5行）:")
        for idx, row in df.head(5).iterrows():
            logger.info(f"   {row['symbol']} | {row['name']:<8} | {row['area']:<6} | {row['industry']:<10} | PE:{row['pe']}")
        
        logger.info("")
        
    except Exception as e:
        logger.error(f"❌ 保存失败: {e}")
        raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='使用Tushare的bak_basic接口更新股票列表',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-o', '--output', type=str, default='stocklist.csv',
                       help='输出文件路径（默认: stocklist.csv）')
    parser.add_argument('--no-backup', action='store_true',
                       help='不备份旧文件')
    parser.add_argument('--date', type=str, default=None,
                       help='指定交易日期（格式: YYYYMMDD），默认今天')
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 70)
        logger.info("🚀 A股股票列表更新工具（Tushare bak_basic）")
        logger.info("=" * 70)
        logger.info("")
        
        # 1. 加载token
        token = load_tushare_token()
        if not token:
            logger.error("❌ 未找到Tushare token")
            logger.error("   请确保 .tushare_token 文件存在")
            return 1
        
        logger.info("")
        
        # 2. 备份
        if not args.no_backup:
            backup_file(args.output)
            logger.info("")
        
        # 3. 获取交易日期
        trade_date = args.date if args.date else get_latest_trade_date()
        if not trade_date:
            logger.error("❌ 无法确定交易日期")
            return 1
        
        # 4. 获取数据
        df = fetch_stock_data(token, trade_date)
        if df is None:
            logger.error("❌ 获取数据失败")
            return 1
        
        # 5. 处理数据
        df = process_data(df)
        if df is None:
            logger.error("❌ 数据处理失败")
            return 1
        
        # 6. 保存数据
        save_to_csv(df, args.output)
        
        logger.info("=" * 70)
        logger.info("🎉 更新完成！")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error("")
        logger.error("=" * 70)
        logger.error(f"❌ 更新失败: {e}")
        logger.error("=" * 70)
        
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

