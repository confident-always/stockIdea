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
import sys
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
        if "star" in boards and base.startswith("688"):
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
    try:
        kwargs = {"symbol": ak_symbol}
        if start:
            kwargs["start_date"] = start
        if end:
            kwargs["end_date"] = end
        # Akshare 默认不复权；如需复权可加 adjust="qfq" / "hfq"
        df = ak.stock_zh_a_daily(**kwargs)
    except Exception as e:
        logger.error("抓取失败 %s: %s", ak_symbol, e)
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


# --------------------------- 主入口 --------------------------- #
def main():
    parser = argparse.ArgumentParser(description="使用 Akshare 抓取 A 股日线（不复权）并保存到 akdata")
    parser.add_argument("--stocklist", type=Path, default=Path("./stocklist.csv"), help="股票清单CSV，需包含 ts_code 或 symbol")
    parser.add_argument("--start", default="20190101", help="起始日期 YYYYMMDD 或 'today'；传 0 表示不填接口参数")
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
    # 清空输出目录（删除现有 CSV）
    removed = 0
    for p in out_dir.iterdir():
        try:
            if p.is_file() and p.suffix.lower() == ".csv":
                p.unlink()
                removed += 1
        except Exception as e:
            logger.warning("删除失败 %s: %s", p, e)
    logger.info("已清空输出目录 %s ，删除 %d 个CSV文件", out_dir.resolve(), removed)

    # 读取股票池
    codes = load_codes_from_stocklist(Path(args.stocklist), list(args.exclude_boards or []))
    if not codes:
        logger.error("stocklist 为空或被过滤后无代码，请检查。")
        sys.exit(1)

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