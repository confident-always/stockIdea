from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import csv
import akshare as ak
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # 将日志写入文件
        logging.FileHandler("select_results.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("select")


# ---------- 工具 ----------

def load_data(data_dir: Path, codes: Iterable[str]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for code in codes:
        fp = data_dir / f"{code}.csv"
        if not fp.exists():
            logger.warning("%s 不存在，跳过", fp.name)
            continue
        df = pd.read_csv(fp, parse_dates=["date"]).sort_values("date")
        frames[code] = df
    return frames


def fetch_company_info_from_em(codes: List[str], max_workers: int | None = None) -> Dict[str, Dict[str, str]]:
    """通过东方财富接口获取公司名称、所属行业与公司简介。

    使用 ak.stock_individual_info_em(symbol) 逐只查询；返回
    映射：code -> {name, industry, brief}。
    参考页面示例：`http://quote.eastmoney.com/concept/sh603777.html?from=classic`
    """
    def _fetch_one(code: str) -> tuple[str, Dict[str, str]]:
        symbol = str(code).strip().zfill(6)
        try:
            df = ak.stock_individual_info_em(symbol=symbol)
        except Exception as e:
            logger.warning("东方财富个股信息查询失败 %s: %s", symbol, e)
            return code, {"name": "", "industry": "", "brief": ""}

        if df is None or df.empty:
            return code, {"name": "", "industry": "", "brief": ""}

        # 常见返回结构：两列 item/value
        try:
            kv = df.copy()
            item_col = next((c for c in ["item", "项目", "字段"] if c in kv.columns), None)
            value_col = next((c for c in ["value", "值", "内容"] if c in kv.columns), None)
            if item_col is None or value_col is None:
                item_col, value_col = kv.columns[:2]
            series = (
                kv.set_index(item_col)[value_col]
                .astype(str)
                .str.strip()
            )
        except Exception:
            series = pd.Series(dtype=str)

        def pick(keys: List[str]) -> str:
            for k in keys:
                if k in series.index and pd.notna(series.get(k, None)):
                    return str(series[k])
            return ""

        name = pick(["公司名称", "公司简称", "股票简称", "证券简称", "名称"])
        industry = pick(["所属行业", "行业"])
        brief = pick(["公司简介", "公司概述", "公司介绍", "公司简介(最新)"])
        return code, {"name": name, "industry": industry, "brief": brief}

    if not codes:
        return {}

    workers = max_workers if (isinstance(max_workers, int) and max_workers > 0) else min(8, len(codes))
    info: Dict[str, Dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch_one, c): c for c in codes}
        for fu in as_completed(futures):
            try:
                code, payload = fu.result()
            except Exception as e:
                code = futures[fu]
                logger.warning("并发获取公司信息失败 %s: %s", code, e)
                payload = {"name": "", "industry": "", "brief": ""}
            info[code] = payload
    return info


def load_config(cfg_path: Path) -> List[Dict[str, Any]]:
    if not cfg_path.exists():
        logger.error("配置文件 %s 不存在", cfg_path)
        sys.exit(1)
    with cfg_path.open(encoding="utf-8") as f:
        cfg_raw = json.load(f)

    # 兼容三种结构：单对象、对象数组、或带 selectors 键
    if isinstance(cfg_raw, list):
        cfgs = cfg_raw
    elif isinstance(cfg_raw, dict) and "selectors" in cfg_raw:
        cfgs = cfg_raw["selectors"]
    else:
        cfgs = [cfg_raw]

    if not cfgs:
        logger.error("configs.json 未定义任何 Selector")
        sys.exit(1)

    return cfgs


def instantiate_selector(cfg: Dict[str, Any]):
    """动态加载 Selector 类并实例化"""
    cls_name: str = cfg.get("class")
    if not cls_name:
        raise ValueError("缺少 class 字段")

    try:
        module = importlib.import_module("Selector")
        cls = getattr(module, cls_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"无法加载 Selector.{cls_name}: {e}") from e

    params = cfg.get("params", {})
    return cfg.get("alias", cls_name), cls(**params)


# ---------- 主函数 ----------

def main():
    p = argparse.ArgumentParser(description="Run selectors defined in configs.json")
    p.add_argument("--data-dir", default="./data", help="CSV 行情目录")
    p.add_argument("--config", default="./configs.json", help="Selector 配置文件")
    p.add_argument("--date", help="交易日 YYYY-MM-DD；缺省=数据最新日期")
    p.add_argument("--tickers", default="all", help="'all' 或逗号分隔股票代码列表")
    p.add_argument("--meta-workers", type=int, default=4, help="公司信息查询并发线程数（东方财富接口）")
    p.add_argument("--adx-periods", default=None, help="运行 ADX 的 DMI 周期列表，逗号分隔，例如 39,105,243；缺省按 39/105/243 生成")
    args = p.parse_args()

    # --- 加载行情 ---
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("数据目录 %s 不存在", data_dir)
        sys.exit(1)

    codes = (
        [f.stem for f in data_dir.glob("*.csv")]
        if args.tickers.lower() == "all"
        else [c.strip() for c in args.tickers.split(",") if c.strip()]
    )
    if not codes:
        logger.error("股票池为空！")
        sys.exit(1)

    data = load_data(data_dir, codes)
    if not data:
        logger.error("未能加载任何行情数据")
        sys.exit(1)

    trade_date = (
        pd.to_datetime(args.date)
        if args.date
        else max(df["date"].max() for df in data.values())
    )
    if not args.date:
        logger.info("未指定 --date，使用最近日期 %s", trade_date.date())

    # --- 加载 Selector 配置 ---
    selector_cfgs = load_config(Path(args.config))

    # --- 逐个 Selector 运行 ---
    for cfg in selector_cfgs:
        if cfg.get("activate", True) is False:
            continue
        try:
            alias, selector = instantiate_selector(cfg)
        except Exception as e:
            logger.error("跳过配置 %s：%s", cfg, e)
            continue

        # 处理 ADXSelector 配置
        if cfg.get("class") == "ADXSelector":
            dmi_period = cfg.get("params", {}).get("dmi_period", 39)
            lookback_days = cfg.get("params", {}).get("lookback_days", 5)
            alias = cfg.get("alias", f"ADX{dmi_period}")
            
            logger.info("")
            logger.info("【参数提示】%s：DMI周期=%s；lookback_days=%s", alias, dmi_period, lookback_days)
            if isinstance(lookback_days, int) and lookback_days > 0:
                logger.info("这是最近 %d 天范围内 ADX 上穿 MDI 的数据", lookback_days)
            else:
                logger.info("这是最近若干天范围内 ADX 上穿 MDI 的数据")

            # 为当前配置实例化 ADXSelector 并输出对应 CSV
            try:
                module = importlib.import_module("Selector")
                cls = getattr(module, "ADXSelector")
                compute_dmi = getattr(module, "compute_dmi", None)
            except Exception as e:
                logger.error("无法加载 ADXSelector：%s", e)
                continue

            # 使用配置中的参数创建选择器
            params = dict(cfg.get("params", {}))
            # 确保使用dmi_period参数名，因为ADXSelector构造函数使用dmi_period参数名
            if "dmi_period" not in params:
                if "period" in params:
                    params["dmi_period"] = params.pop("period")
                else:
                    params["dmi_period"] = dmi_period
            selector_p = cls(**params)

            # 执行选股并获取详细数据（包含 cross_date 和 ADX）
            detailed_picks = []
            if hasattr(selector_p, "select_with_details"):
                try:
                    detailed_picks = selector_p.select_with_details(trade_date, data)
                except Exception as e:
                    logger.error("%s 详情选股失败：%s", alias, e)
                    detailed_picks = []
            else:
                # 兼容不带详情方法的情形：仅输出代码，ADX 留空
                try:
                    codes_picks = selector_p.select(trade_date, data)
                    detailed_picks = [{"code": c, "cross_date": trade_date.date().isoformat(), "ADX": None} for c in codes_picks]
                except Exception as e:
                    logger.error("%s 选股失败：%s", alias, e)
                    detailed_picks = []

            # 打印当期结果
            logger.info("")
            logger.info("============== 选股结果 [%s] ==============", alias)
            logger.info("交易日: %s", trade_date.date())
            logger.info("符合条件股票数: %d", len(detailed_picks))
            # 取消循环打印股票代码、日期、ADX 的控制台输出
            if not detailed_picks:
                logger.info("无符合条件股票")

            # 生成 CSV：使用 alias + 日期命名
            out_dir = Path("res")
            out_dir.mkdir(exist_ok=True)
            out_file = out_dir / f"{alias}_{trade_date.date().isoformat()}.csv"
            try:
                # 获取公司信息（东方财富接口），并构造含公司信息的行数据
                codes_for_info = [r.get("code", "") for r in detailed_picks]
                info_map = fetch_company_info_from_em(codes_for_info, max_workers=args.meta_workers) if detailed_picks else {}

                rows = []
                for r in detailed_picks:
                    code = r.get("code", "")
                    # 将日期列设置为上穿日期（若缺失则回退为交易日）
                    date = r.get("cross_date") or trade_date.date().isoformat()
                    # 强制使用最后一日的 ADX 值
                    adx_val = r.get("ADX", "")
                    try:
                        if compute_dmi and code in data:
                            hist = data[code]
                            hist = hist[hist["date"] <= trade_date]
                            dmi_df = compute_dmi(hist, dmi_period)
                            last_adx = dmi_df["ADX"].iloc[-1]
                            if pd.notna(last_adx):
                                adx_val = round(float(last_adx), 3)
                    except Exception as e:
                        logger.warning("计算最后一日 ADX 失败 %s(%d)：%s", code, dmi_period, e)
                    meta = info_map.get(code, {})
                    rows.append([
                        code,
                        date,
                        adx_val,
                        meta.get("name", ""),
                        meta.get("industry", ""),
                        meta.get("brief", ""),
                    ])

                # 按行业排序（无行业信息的排后）；空结果时仅写表头
                if rows:
                    rows.sort(key=lambda r: (0 if r[4] else 1, r[4], r[0]))

                with open(out_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["code", "date", "ADX", "name", "industry", "brief"])
                    for r in rows:
                        writer.writerow(r)
                logger.info("已生成结果CSV: %s", out_file)

            except Exception as e:
                logger.error("生成 %s CSV 失败：%s", alias, e)

        # 处理 PDISelector 配置
        elif cfg.get("class") == "PDISelector":
            dmi_period = cfg.get("params", {}).get("dmi_period", 39)
            lookback_days = cfg.get("params", {}).get("lookback_days", 5)
            alias = cfg.get("alias", f"PDI{dmi_period}")
            
            logger.info("")
            logger.info("【参数提示】%s：DMI周期=%s；lookback_days=%s", alias, dmi_period, lookback_days)
            if isinstance(lookback_days, int) and lookback_days > 0:
                logger.info("这是最近 %d 天范围内 PDI 上穿 MDI 的数据", lookback_days)
            else:
                logger.info("这是最近若干天范围内 PDI 上穿 MDI 的数据")

            # 为当前配置实例化 PDISelector 并输出对应 CSV
            try:
                module = importlib.import_module("Selector")
                cls = getattr(module, "PDISelector")
                compute_dmi = getattr(module, "compute_dmi", None)
            except Exception as e:
                logger.error("无法加载 PDISelector：%s", e)
                continue

            # 使用配置中的参数创建选择器
            params = dict(cfg.get("params", {}))
            # 确保使用dmi_period参数名，因为PDISelector构造函数使用dmi_period参数名
            if "dmi_period" not in params:
                if "period" in params:
                    params["dmi_period"] = params.pop("period")
                else:
                    params["dmi_period"] = dmi_period
            selector_p = cls(**params)

            # 执行选股并获取详细数据（包含 cross_date 和 PDI）
            detailed_picks = []
            if hasattr(selector_p, "select_with_details"):
                try:
                    detailed_picks = selector_p.select_with_details(trade_date, data)
                except Exception as e:
                    logger.error("%s 详情选股失败：%s", alias, e)
                    detailed_picks = []
            else:
                # 兼容不带详情方法的情形：仅输出代码，PDI 留空
                try:
                    codes_picks = selector_p.select(trade_date, data)
                    detailed_picks = [{"code": c, "cross_date": trade_date.date().isoformat(), "PDI": None} for c in codes_picks]
                except Exception as e:
                    logger.error("%s 选股失败：%s", alias, e)
                    detailed_picks = []

            # 打印当期结果
            logger.info("")
            logger.info("============== 选股结果 [%s] ==============", alias)
            logger.info("交易日: %s", trade_date.date())
            logger.info("符合条件股票数: %d", len(detailed_picks))
            # 取消循环打印股票代码、日期、PDI 的控制台输出
            if not detailed_picks:
                logger.info("无符合条件股票")

            # 生成 CSV：使用 alias + 日期命名
            out_dir = Path("res")
            out_dir.mkdir(exist_ok=True)
            out_file = out_dir / f"{alias}_{trade_date.date().isoformat()}.csv"
            try:
                # 获取公司信息（东方财富接口），并构造含公司信息的行数据
                codes_for_info = [r.get("code", "") for r in detailed_picks]
                info_map = fetch_company_info_from_em(codes_for_info, max_workers=args.meta_workers) if detailed_picks else {}

                rows = []
                for r in detailed_picks:
                    code = r.get("code", "")
                    # 将日期列设置为上穿日期（若缺失则回退为交易日）
                    date = r.get("cross_date") or trade_date.date().isoformat()
                    # 强制使用最后一日的 PDI 值
                    pdi_val = r.get("PDI", "")
                    try:
                        if compute_dmi and code in data:
                            hist = data[code]
                            hist = hist[hist["date"] <= trade_date]
                            dmi_df = compute_dmi(hist, dmi_period)
                            last_pdi = dmi_df["PDI"].iloc[-1]
                            if pd.notna(last_pdi):
                                pdi_val = round(float(last_pdi), 3)
                    except Exception as e:
                        logger.warning("计算最后一日 PDI 失败 %s(%d)：%s", code, dmi_period, e)
                    meta = info_map.get(code, {})
                    rows.append([
                        code,
                        date,
                        pdi_val,
                        meta.get("name", ""),
                        meta.get("industry", ""),
                        meta.get("brief", ""),
                    ])

                # 按行业排序（无行业信息的排后）；空结果时仅写表头
                if rows:
                    rows.sort(key=lambda r: (0 if r[4] else 1, r[4], r[0]))

                with open(out_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["code", "date", "PDI", "name", "industry", "brief"])
                    for r in rows:
                        writer.writerow(r)
                logger.info("已生成结果CSV: %s", out_file)

            except Exception as e:
                logger.error("生成 %s CSV 失败：%s", alias, e)

            # 已针对 ADXSelector 完成导出，跳过后续通用写入逻辑
            continue

        picks = selector.select(trade_date, data)

        # 将结果写入日志，同时输出到控制台
        logger.info("")
        logger.info("============== 选股结果 [%s] ==============", alias)
        logger.info("交易日: %s", trade_date.date())
        logger.info("符合条件股票数: %d", len(picks))
        
        # 如果是ADXSelector，显示详细的 PDI/MDI/ADX 值
        if hasattr(selector, 'select_with_details'):
            detailed_picks = selector.select_with_details(trade_date, data)
        else:
            logger.info("%s", ", ".join(picks) if picks else "无符合条件股票")


if __name__ == "__main__":
    main()
