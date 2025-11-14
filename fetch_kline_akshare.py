#!/usr/bin/env python3
from __future__ import annotations

"""
使用 Akshare 作为数据源抓取 A 股日线（不复权），并将数据保存到 data 文件夹。

特性：
- 读取 stocklist.csv（支持列：ts_code 或 symbol）
- 支持指定日期范围：--start/--end；当传 0 时不向接口传该参数（取 Akshare 默认）
- 支持排除板块：gem(创业板 300/301)、star(科创板 688)、bj(北交所 .BJ/4/8)
- 并发抓取，统一输出为 CSV，列：date, open, close, high, low, volume

示例：
  python fetch_kline_akshare.py \
    --stocklist ./stocklist.csv \
    --start 0 \
    --end today \
    --exclude-boards gem star bj \
    --out ./data \
    --workers 6
"""

import argparse
import datetime as dt
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import akshare as ak
import pandas as pd
from tqdm import tqdm


# --------------------------- 日志 --------------------------- #
LOG_FILE = Path("fetch.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("fetch_akshare")


# --------------------------- 工具函数 --------------------------- #
def _normalize_code(s: str) -> str:
    return str(s).strip().upper()


def to_ak_symbol_from_ts(ts_code: str) -> Optional[str]:
    ts_code = _normalize_code(ts_code)
    if "." not in ts_code:
        return None
    base, exch = ts_code.split(".", 1)
    exch = exch.upper()
    if exch == "SZ":
        return f"sz{base}"
    if exch == "SH":
        return f"sh{base}"
    if exch == "BJ":
        return f"bj{base}"
    return None


def to_ak_symbol_from_symbol(symbol: str) -> Optional[str]:
    symbol = str(symbol).strip()
    if not symbol or not symbol.isdigit():
        return None
    if symbol.startswith(("000", "001", "002", "003", "004", "300", "301")):
        return f"sz{symbol}"
    if symbol.startswith(("600", "601", "603", "605", "688")):
        return f"sh{symbol}"
    if symbol.startswith(("8", "4")):
        return f"bj{symbol}"
    # 兜底：常见深圳
    return f"sz{symbol}"


def exclude_by_boards(symbols: List[str], boards: List[str]) -> List[str]:
    boards = set(boards or [])
    out: List[str] = []
    for s in symbols:
        base = s[2:] if len(s) > 2 else s
        if "gem" in boards and base.startswith(("300", "301")):
            continue
        if "star" in boards and base.startswith(("688", "689")):  # 修复：科创板包括688和689开头
            continue
        if "bj" in boards and (s.startswith("bj") or base.startswith(("8", "4"))):
            continue
        out.append(s)
    return out


def load_codes_from_stocklist(stocklist_path: Path, exclude_boards: List[str]) -> List[str]:
    df = pd.read_csv(stocklist_path)
    cols = {c.lower(): c for c in df.columns}
    ak_symbols: List[str] = []

    if "ts_code" in cols:
        for v in df[cols["ts_code"]].dropna().tolist():
            ak_sym = to_ak_symbol_from_ts(str(v))
            if ak_sym:
                ak_symbols.append(ak_sym)
    elif "symbol" in cols:
        for v in df[cols["symbol"]].dropna().tolist():
            ak_sym = to_ak_symbol_from_symbol(str(v))
            if ak_sym:
                ak_symbols.append(ak_sym)
    else:
        raise ValueError("stocklist.csv 需包含 'ts_code' 或 'symbol' 列")

    ak_symbols = exclude_by_boards(ak_symbols, exclude_boards)
    return sorted(set(ak_symbols))


# --------------------------- 抓取 --------------------------- #
def fetch_one(ak_symbol: str, start: Optional[str], end: Optional[str], out_dir: Path) -> None:
    import time
    import random
    
    max_retries = 1
    base_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            # 添加随机延时，避免频率限制
            if attempt > 0:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.info("重试 %s (第%d次)，等待 %.2f 秒", ak_symbol, attempt + 1, delay)
                time.sleep(delay)
            
            kwargs = {"symbol": ak_symbol}
            if start:
                kwargs["start_date"] = start
            if end:
                kwargs["end_date"] = end
            # Akshare 默认不复权；如需复权可加 adjust="qfq" / "hfq"
            df = ak.stock_zh_a_daily(**kwargs)
            
            # 如果成功获取数据，跳出重试循环
            break
            
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                # 判断是否为可重试的错误
                if "No value to decode" in error_msg or "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                    logger.warning("抓取 %s 遇到可重试错误: %s，将重试", ak_symbol, error_msg)
                    continue
                else:
                    logger.error("抓取失败 %s: %s (不可重试错误)", ak_symbol, error_msg)
                    return
            else:
                logger.error("抓取失败 %s: %s (已重试%d次)", ak_symbol, error_msg, max_retries)
                return

    if df is None or df.empty:
        logger.warning("数据为空: %s", ak_symbol)
        return

    # 统一列名与类型
    rename_map = {
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    df = df.rename(columns=rename_map)[list(rename_map.values())].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    out_fp = out_dir / f"{ak_symbol[2:]}.csv"
    df.to_csv(out_fp, index=False)
    logger.info("保存: %s (%d 行)", out_fp.name, len(df))


def validate(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    if df["date"].isna().any():
        raise ValueError("存在缺失日期！")
    if (df["date"] > pd.Timestamp.today()).any():
        raise ValueError("数据包含未来日期，可能抓取错误！")
    return df


def update_one_to_latest(
    ak_symbol: str,
    out_dir: Path,
) -> None:
    """补充单只股票的数据到最后一天的下一天到当前日期"""
    code = ak_symbol[2:]  # 去掉前缀，保存文件名用
    csv_path = out_dir / f"{code}.csv"

    # 如果文件不存在，跳过
    if not csv_path.exists():
        logger.debug(f"{code} 文件不存在，跳过更新")
        return

    # 读取现有数据
    try:
        existing_df = pd.read_csv(csv_path)
        if existing_df.empty:
            logger.debug(f"{code} 现有数据为空，跳过更新")
            return

        # 确保date列是datetime类型
        existing_df["date"] = pd.to_datetime(existing_df["date"])
        last_date = existing_df["date"].max()

        # 计算开始日期（最后一天的下一天）
        start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
        end_date = dt.date.today().strftime("%Y%m%d")

        # 如果开始日期大于今天，说明数据已经是最新的
        if start_date > end_date:
            logger.debug(f"{code} 数据已是最新，无需更新")
            return

        # 计算需要补充的天数
        days_to_add = (dt.datetime.strptime(end_date, '%Y%m%d').date() - dt.datetime.strptime(start_date, '%Y%m%d').date()).days + 1
        logger.info(f"{code} 补充数据: 最后日期 {last_date.strftime('%Y-%m-%d')}，需补充 {days_to_add} 天 ({start_date} -> {end_date})")

    except Exception as e:
        logger.error(f"{code} 读取现有数据失败: {e}")
        return

    # 获取新数据，带有重试机制
    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            kwargs = {"symbol": ak_symbol}
            kwargs["start_date"] = start_date
            kwargs["end_date"] = end_date

            new_df = ak.stock_zh_a_daily(**kwargs)

            if new_df is None or new_df.empty:
                logger.debug(f"{code} 期间无新数据")
                break

            # 统一列名与类型
            rename_map = {
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
            new_df = new_df.rename(columns=rename_map)[list(rename_map.values())].copy()
            new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce")
            for c in ["open", "close", "high", "low", "volume"]:
                new_df[c] = pd.to_numeric(new_df[c], errors="coerce")

            new_df = validate(new_df)

            # 合并数据
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

            # 保存合并后的数据
            combined_df.to_csv(csv_path, index=False)
            logger.info(f"{code} 补充完成，新增 {len(new_df)} 条数据")
            break

        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                # 判断是否为可重试的错误
                if "No value to decode" in error_msg or "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"{code} 第 {attempt + 1} 次补充数据失败，{delay:.1f} 秒后重试：{e}")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"{code} 第 {attempt + 1} 次补充数据失败（不可重试）：{e}")
                    break
            else:
                logger.error(f"{code} 三次补充数据均失败，已跳过！")
                break


# --------------------------- 主入口 --------------------------- #
def main():
    parser = argparse.ArgumentParser(description="使用 Akshare 抓取 A 股日线（不复权）并保存到 data")
    # 运行模式
    parser.add_argument("--mode", choices=["fetch", "update"], default="fetch",
                       help="运行模式: fetch(全量抓取), update(增量更新到最新日期)")
    # 抓取范围（传 0 表示不向接口传该参数）
    parser.add_argument("--stocklist", type=Path, default=Path("./stocklist.csv"), help="股票清单CSV，需包含 ts_code 或 symbol")
    parser.add_argument("--start", default="0", help="起始日期 YYYYMMDD 或 'today'；传 0 表示不填接口参数")
    parser.add_argument("--end", default="today", help="结束日期 YYYYMMDD 或 'today'；传 0 表示不填接口参数")
    parser.add_argument("--exclude-boards", nargs="*", default=[], choices=["gem", "star", "bj"], help="排除板块：gem/star/bj")
    parser.add_argument("--out", default="./data", help="输出目录")
    parser.add_argument("--workers", type=int, default=6, help="并发线程数")
    args = parser.parse_args()

    # 日期解析（支持 0 表示不填）
    raw_start = str(args.start).strip()
    raw_end = str(args.end).strip()
    start = None if raw_start in ("0", "0.0") else (
        dt.date.today().strftime("%Y%m%d") if raw_start.lower() == "today" else raw_start
    )
    end = None if raw_end in ("0", "0.0") else (
        dt.date.today().strftime("%Y%m%d") if raw_end.lower() == "today" else raw_end
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取股票池
    codes = load_codes_from_stocklist(Path(args.stocklist), list(args.exclude_boards or []))
    if not codes:
        logger.error("stocklist 为空或被过滤后无代码，请检查。")
        sys.exit(1)

    # 根据模式执行不同的操作
    if args.mode == "update":
        # 过滤出实际存在的CSV文件，避免更新不存在的文件
        existing_codes = [code for code in codes if (out_dir / f"{code[2:]}.csv").exists()]

        logger.info(
            "从 stocklist.csv 读取到 %d 只股票（排除板块：%s）",
            len(codes), ",".join(sorted(args.exclude_boards)) or "无",
        )

        logger.info(
            "开始增量更新 %d 支股票 | 数据源:Akshare(日线,不复权) | 补充到最新日期 | 排除:%s",
            len(existing_codes), ",".join(sorted(args.exclude_boards)) or "无",
        )

        # 多线程增量更新 - 只更新 stocklist.csv 中指定且文件存在的股票
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    update_one_to_latest,
                    code,
                    out_dir,
                )
                for code in existing_codes
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="更新进度"):
                pass

        logger.info("增量更新任务完成，数据已保存至 %s", out_dir.resolve())
    else:
        logger.info(
            "开始抓取 %d 支股票 | 数据源:Akshare(日线,不复权) | 日期:%s → %s | 排除:%s",
            len(codes), start or "(未指定)", end or "(未指定)", ",".join(sorted(args.exclude_boards)) or "无",
        )

        # 并发抓取
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    fetch_one,
                    code,
                    start,
                    end,
                    out_dir,
                )
                for code in codes
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="下载进度"):
                pass

        logger.info("全部任务完成，数据已保存至 %s", out_dir.resolve())


if __name__ == "__main__":
    main()