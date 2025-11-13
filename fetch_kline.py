from __future__ import annotations

import argparse
import datetime as dt
import logging
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
import os

import pandas as pd
import tushare as ts
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --------------------------- 全局日志配置 --------------------------- #
LOG_FILE = Path("fetch.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("fetch_from_stocklist")

# --------------------------- 限流/封禁处理配置 --------------------------- #
COOLDOWN_SECS = 600
BAN_PATTERNS = (
    "访问频繁", "请稍后", "超过频率", "频繁访问",
    "too many requests", "429",
    "forbidden", "403",
    "max retries exceeded"
)

def _looks_like_ip_ban(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    return any(pat in msg for pat in BAN_PATTERNS)

class RateLimitError(RuntimeError):
    """表示命中限流/封禁，需要长时间冷却后重试。"""
    pass

def _cool_sleep(base_seconds: int) -> None:
    jitter = random.uniform(0.9, 1.2)
    sleep_s = max(1, int(base_seconds * jitter))
    logger.warning("疑似被限流/封禁，进入冷却期 %d 秒...", sleep_s)
    time.sleep(sleep_s)

# --------------------------- 历史K线（Tushare 日线，不复权） --------------------------- #
pro: Optional[ts.pro_api] = None  # 模块级会话

def set_api(session) -> None:
    """由外部(比如GUI)注入已创建好的 ts.pro_api() 会话"""
    global pro
    pro = session
    

def _to_ts_code(code: str) -> str:
    """把6位code映射到标准 ts_code 后缀。"""
    code = str(code).zfill(6)
    if code.startswith(("60", "68", "9")):
        return f"{code}.SH"
    elif code.startswith(("4", "8")):
        return f"{code}.BJ"
    else:
        return f"{code}.SZ"

def _get_kline_tushare(code: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    ts_code = _to_ts_code(code)
    try:
        kwargs = {
            "ts_code": ts_code,
            "adj": None,
            "freq": "D",
            "api": pro,
        }
        if start:
            kwargs["start_date"] = start
        if end:
            kwargs["end_date"] = end
        df = ts.pro_bar(**kwargs)
    except Exception as e:
        if _looks_like_ip_ban(e):
            raise RateLimitError(str(e)) from e
        raise

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={"trade_date": "date", "vol": "volume"})[
        ["date", "open", "close", "high", "low", "volume"]
    ].copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)

def validate(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    if df["date"].isna().any():
        raise ValueError("存在缺失日期！")
    if (df["date"] > pd.Timestamp.today()).any():
        raise ValueError("数据包含未来日期，可能抓取错误！")
    return df

# --------------------------- 读取 stocklist.csv & 过滤板块 --------------------------- #

def _filter_by_boards_stocklist(df: pd.DataFrame, exclude_boards: set[str]) -> pd.DataFrame:
    """
    exclude_boards 子集：{'gem','star','bj'}
    - gem  : 创业板 300/301（.SZ）
    - star : 科创板 688（.SH）
    - bj   : 北交所（.BJ 或 4/8 开头，以及转换后的bj开头代码）
    """
    code = df["symbol"].astype(str).str.zfill(6)
    ts_code = df["ts_code"].astype(str).str.upper()
    mask = pd.Series(True, index=df.index)

    if "gem" in exclude_boards:
        mask &= ~code.str.startswith(("300", "301"))
    if "star" in exclude_boards:
        mask &= ~code.str.startswith(("688",))
    if "bj" in exclude_boards:
        # 北交所股票：.BJ结尾 或 symbol以4/8开头
        mask &= ~(ts_code.str.endswith(".BJ") | code.str.startswith(("4", "8")))

    return df[mask].copy()

def load_codes_from_stocklist(stocklist_csv: Path, exclude_boards: set[str]) -> List[str]:
    df = pd.read_csv(stocklist_csv)    
    df = _filter_by_boards_stocklist(df, exclude_boards)
    codes = df["symbol"].astype(str).str.zfill(6).tolist()
    codes = list(dict.fromkeys(codes))  # 去重保持顺序
    logger.info("从 %s 读取到 %d 只股票（排除板块：%s）",
                stocklist_csv, len(codes), ",".join(sorted(exclude_boards)) or "无")
    return codes

# --------------------------- 单只抓取（全量覆盖保存） --------------------------- #
def fetch_one(
    code: str,
    start: str,
    end: str,
    out_dir: Path,
):
    csv_path = out_dir / f"{code}.csv"

    for attempt in range(1, 4):
        try:
            new_df = _get_kline_tushare(code, start, end)
            if new_df.empty:
                logger.debug("%s 无数据，生成空表。", code)
                new_df = pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume"])
            new_df = validate(new_df)
            new_df.to_csv(csv_path, index=False)  # 直接覆盖保存
            break
        except Exception as e:
            if _looks_like_ip_ban(e):
                logger.error(f"{code} 第 {attempt} 次抓取疑似被封禁，沉睡 {COOLDOWN_SECS} 秒")
                _cool_sleep(COOLDOWN_SECS)
            else:
                silent_seconds = 15 * attempt
                logger.info(f"{code} 第 {attempt} 次抓取失败，{silent_seconds} 秒后重试：{e}")
                time.sleep(silent_seconds)
    else:
        logger.error("%s 三次抓取均失败，已跳过！", code)

# --------------------------- 增量更新（补充数据到最新日期） --------------------------- #
def update_one_to_latest(
    code: str,
    out_dir: Path,
):
    """补充单只股票的数据到最后一天的下一天到当前日期"""
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

        # 如果开始日期大于等于今天，说明数据已经是最新的
        if start_date >= end_date:
            logger.debug(f"{code} 数据已是最新，无需更新")
            return

        logger.info(f"{code} 开始补充数据: {start_date} -> {end_date}")

    except Exception as e:
        logger.error(f"{code} 读取现有数据失败: {e}")
        return

    # 获取新数据
    for attempt in range(1, 4):
        try:
            new_df = _get_kline_tushare(code, start_date, end_date)
            if new_df.empty:
                logger.debug(f"{code} 期间无新数据")
                break

            new_df = validate(new_df)

            # 合并数据
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

            # 保存合并后的数据
            combined_df.to_csv(csv_path, index=False)
            logger.info(f"{code} 补充完成，新增 {len(new_df)} 条数据")
            break

        except Exception as e:
            if _looks_like_ip_ban(e):
                logger.error(f"{code} 第 {attempt} 次补充数据疑似被封禁，沉睡 {COOLDOWN_SECS} 秒")
                _cool_sleep(COOLDOWN_SECS)
            else:
                silent_seconds = 15 * attempt
                logger.info(f"{code} 第 {attempt} 次补充数据失败，{silent_seconds} 秒后重试：{e}")
                time.sleep(silent_seconds)
    else:
        logger.error(f"{code} 三次补充数据均失败，已跳过！")

# --------------------------- 主入口 --------------------------- #
def main():
    parser = argparse.ArgumentParser(description="从 stocklist.csv 读取股票池并用 Tushare 抓取日线K线（不复权，全量覆盖或增量更新）")
    # 运行模式
    parser.add_argument("--mode", choices=["fetch", "update"], default="fetch",
                       help="运行模式: fetch(全量抓取), update(增量更新到最新日期)")
    # 抓取范围（传 0 表示不向接口传该参数）
    parser.add_argument("--start", default="20190101", help="起始日期 YYYYMMDD 或 'today'；传 0 表示不填接口参数")
    parser.add_argument("--end", default="today", help="结束日期 YYYYMMDD 或 'today'；传 0 表示不填接口参数")
    # 股票清单与板块过滤
    parser.add_argument("--stocklist", type=Path, default=Path("./stocklist.csv"), help="股票清单CSV路径（需含 ts_code 或 symbol）")
    parser.add_argument(
        "--exclude-boards",
        nargs="*",
        default=[],
        choices=["gem", "star", "bj"],
        help="排除板块，可多选：gem(创业板300/301) star(科创板688) bj(北交所.BJ/4/8)"
    )
    # 其它
    parser.add_argument("--out", default="./data", help="输出目录")
    parser.add_argument("--workers", type=int, default=6, help="并发线程数")
    args = parser.parse_args()

    # ---------- Tushare Token ---------- #
    os.environ["NO_PROXY"] = "api.waditu.com,.waditu.com,waditu.com"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    ts_token = os.environ.get("TUSHARE_TOKEN")
    if not ts_token:
        raise ValueError("请先设置环境变量 TUSHARE_TOKEN，例如：export TUSHARE_TOKEN=你的token")
    ts.set_token(ts_token)
    global pro
    pro = ts.pro_api()

    # ---------- 日期解析（支持 0 表示不填） ---------- #
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

    # ---------- 从 stocklist.csv 读取股票池 ---------- #
    exclude_boards = set(args.exclude_boards or [])
    codes = load_codes_from_stocklist(args.stocklist, exclude_boards)

    if not codes:
        logger.error("stocklist 为空或被过滤后无代码，请检查。")
        sys.exit(1)

    # 根据模式执行不同的操作
    if args.mode == "update":
        logger.info(
            "开始增量更新 %d 支股票 | 数据源:Tushare(日线,不复权) | 补充到最新日期 | 排除:%s",
            len(codes), ",".join(sorted(exclude_boards)) or "无",
        )

        # ---------- 多线程增量更新 ---------- #
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    update_one_to_latest,
                    code,
                    out_dir,
                )
                for code in codes
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="更新进度"):
                pass

        logger.info("增量更新任务完成，数据已保存至 %s", out_dir.resolve())
    else:
        logger.info(
            "开始抓取 %d 支股票 | 数据源:Tushare(日线,不复权) | 日期:%s → %s | 排除:%s",
            len(codes), start or "(未指定)", end or "(未指定)", ",".join(sorted(exclude_boards)) or "无",
        )

        # ---------- 多线程抓取（全量覆盖） ---------- #
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
