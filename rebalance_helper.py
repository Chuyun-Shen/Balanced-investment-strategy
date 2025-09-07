# -*- coding: utf-8 -*-
"""
自动化资产再平衡助手（V2：价量动量 + 位阶因子 + 数据源修复）
- 核心变化：
  1) 强度模型：在原“收益率缺口(ERP)+趋势+波动”框架上，加入【价量动量】与【位阶因子】（自动阈值，不写死3800）。
  2) 权益上限：以【目标权益占比】为锚，随月度回撤/VIX动态收缩，避免长期卡住权益买单。
  3) 数据源：CN10Y 优先 TradingEconomics；CSI300 近10年PE分位优先 亿牛，回退 Value500；并加超时/UA/重试。
  4) 回流策略：保留 --realloc auto/defense/none 三档；auto 根据行情好坏决定是否把被裁权益买单回流到防守资产。

依赖：pip install akshare pandas numpy yfinance pandas-datareader requests beautifulsoup4 tradingeconomics
"""
import argparse
import os, warnings, datetime as dt
import pandas as pd
import numpy as np
import re, time
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# ====== 显示选项 ======
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', 20)
pd.set_option('display.precision', 2)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.width', 150)
pd.set_option('display.unicode.ambiguous_as_wide', True)

# ====== 全局参数（按需改） ======
TOTAL_TARGET   = 40000.0
STAGING_WEEKS  = 6
WEEKLY_LIMIT   = 3000.0
USE_INTENSITY  = True
AUTO_ADJUST    = True

# 自动调整参数
MIN_TRADE_AMOUNT = 500.0
MAX_TRADE_PCT    = 0.05
CASH_RESERVE_PCT = 0.02

# 再平衡带宽
BAND_PP_ABS = 1.5
BAND_REL    = 0.20

# 权益上限触发时的重分配策略："defense" | "none" | "auto"
REALLOCATE_POLICY = "auto"

# 整手集合（补全 513100、518880）
BOARD_LOT_CODES = {"510300","510500","588000","513500","513800","513100","518880"}

# 仅减持集合（开放式/LOF）
SELLABLE_CODES = {"161119","007360","001917","164824","009051"}

# 中文名表（备份名）
CODE_NAMES = {
    "161119":"易方达中债新综合债券指数LOF",
    "007360":"易方达中短期美元债(QDII)A",
    "510300":"沪深300ETF",
    "510500":"中证500ETF",
    "588000":"科创50ETF",
    "001917":"招商量化精选A",
    "513500":"标普500ETF(QDII)",
    "513100":"纳指100ETF(QDII)",
    "513800":"日本东证ETF(QDII)",
    "164824":"工银印度基金LOF",
    "009051":"易方达中证红利ETF/联接",
    "518880":"黄金ETF",
}

# ====== 目标权重 ======
targets = {
    # 防守资产 35%
    "161119": 0.25,
    "007360": 0.10,

    # 核心权益 40%
    "510300": 0.15,
    "009051": 0.15,
    "510500": 0.06,
    "001917": 0.04,

    # 海外权益 15%
    "513500": 0.10,
    "513100": 0.02,
    "513800": 0.03,

    # 商品+替代 10%
    "518880": 0.06,
    "588000": 0.02,
    "164824": 0.02,
}

DEBT_FLOOR = 0.35

def apply_debt_floor(targets: dict, debt_floor=DEBT_FLOOR) -> dict:
    t = targets.copy()
    debt_sum = t.get("161119",0)+t.get("007360",0)
    if debt_sum >= debt_floor: 
        return t
    need = debt_floor - debt_sum
    eq_codes = [c for c in t if c not in ("161119","007360")]
    eq_sum = sum(t[c] for c in eq_codes)
    if eq_sum <= 0: 
        t["161119"] += need; return t
    for c in eq_codes:
        t[c] = max(t[c] - need * (t[c]/eq_sum), 0)
    t["161119"] += need
    s = sum(t.values())
    for c in t: t[c] = t[c]/s
    return t

targets = apply_debt_floor(targets, DEBT_FLOOR)

# ====== 当前持仓（外部配置） ======
from config import load_positions
current_positions = load_positions()

# ====== 数据源 ======
import akshare as ak
import yfinance as yf
from pandas_datareader import data as pdr

UA = {"User-Agent":"Mozilla/5.0"}

# --- 工具函数 ---

def _retry(n=2, sleep=0.6):
    def deco(fn):
        def wrap(*a, **k):
            for i in range(n):
                try:
                    return fn(*a, **k)
                except Exception:
                    if i==n-1: raise
                    time.sleep(sleep)
        return wrap
    return deco

# —— 现货/估值 ——

def get_etf_spot():
    df = ak.fund_etf_spot_em()
    df = df[['代码','名称','最新价','涨跌幅','成交额','成交量']].rename(
        columns={'代码':'code','名称':'name','最新价':'price'})
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    return df.set_index('code')

@_retry()
def get_lof_spot():
    try:
        df = ak.fund_lof_spot_em()
        df = df[['代码','名称','最新价']].rename(columns={'代码':'code','名称':'name','最新价':'price'})
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        return df.set_index('code')
    except Exception:
        return pd.DataFrame(columns=['name','price'])

@_retry()
def get_fund_estimation(codes):
    try:
        val = ak.fund_em_value_estimation()
        val = val[['基金代码','基金简称','单位净值','估算净值','估算涨幅']].rename(
            columns={'基金代码':'code','基金简称':'name','估算净值':'price'})
        val = val[val['code'].isin(codes)]
        val['price'] = pd.to_numeric(val['price'], errors='coerce')
        return val.set_index('code')
    except Exception:
        return pd.DataFrame(columns=['name','price'])

@_retry()
def get_csindex_values(index_code):
    try:
        val = ak.stock_zh_index_value_csindex(symbol=index_code)
        last = val.tail(1).copy()
        pe = pd.to_numeric(last['市盈率2'], errors='coerce').iloc[0]
        dy = pd.to_numeric(last['股息率2'], errors='coerce').iloc[0]  # %
        return pe, dy/100.0
    except Exception:
        return np.nan, np.nan

# —— 中国10Y收益率（优先 TE，统一为百分数） ——
@_retry(n=3, sleep=1.0)
def get_cn_yield_10y(timeout=10):
    """
    返回 (cn10y_pct, source_tag)
    - cn10y_pct 为“百分数”，如 1.79 表示 1.79%
    - 采用多源回退；若全部失败，尝试读取上次缓存
    """
    UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
    CACHE_FILE = os.path.join(os.path.dirname(__file__), "cache_cn10y.json")

    # 1) 中国货币网（Shibor.org）——政府债券利率历史数据（日表）
    #    页面直接包含“日期 / 1年期国债收益率 / 10年期国债收益率”
    try:
        url = "https://www.shibor.org/chinese/sddsintigy/"
        html = requests.get(url, headers=UA, timeout=timeout).text
        tables = pd.read_html(html)
        for df in tables:
            cols = [str(c).strip() for c in df.columns]
            if "日期" in cols and any("10年期国债收益率" in c for c in cols):
                df.columns = cols
                df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
                df = df.dropna(subset=["日期"]).sort_values("日期")
                s = pd.to_numeric(df[[c for c in df.columns if "10年期国债收益率" in c][0]], errors="coerce").dropna()
                if not s.empty:
                    val = float(s.iloc[-1])  # 已是百分数
                    _cache_cn10y(CACHE_FILE, val, "shibor")
                    return val, "shibor"
    except Exception:
        pass

    # 2) 中债 ChinaBond 历史查询（近 60 天，10Y）
    #    这个端点可直接返回表格；不同系统有时需要 qxId=hzsylqx；用 10Y 列
    try:
        end = dt.date.today().strftime("%Y-%m-%d")
        start = (dt.date.today() - dt.timedelta(days=60)).strftime("%Y-%m-%d")
        url = ("https://yield.chinabond.com.cn/cbweb-pbc-web/pbc/historyQuery"
               f"?startDate={start}&endDate={end}&gjqx=10&qxId=hzsylqx&locale=cn_ZH")
        html = requests.get(url, headers=UA, timeout=timeout).text
        tables = pd.read_html(html)
        if tables:
            df = tables[0]
            # 标准化列名，兼容 '10 Y'/'10年' 等
            df.columns = [re.sub(r"\s+", "", str(c)) for c in df.columns]
            if "日期" not in df.columns:
                df.rename(columns={df.columns[0]: "日期"}, inplace=True)
            df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
            df = df.dropna(subset=["日期"]).sort_values("日期")
            # 优先找 '10Y'，其次 '10年'、'10年期'
            for col in ("10Y", "10年", "10年期"):
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    if not s.empty:
                        val = float(s.iloc[-1])  # 百分数
                        _cache_cn10y(CACHE_FILE, val, "chinabond_history")
                        return val, "chinabond_history"
    except Exception:
        pass

    # 3) TradingEconomics（网页文本 regex 提取最新值）
    try:
        url = "https://zh.tradingeconomics.com/china/government-bond-yield"
        html = requests.get(url, headers=UA, timeout=timeout).text
        # 常见格式： “降至1.77％” 或 “至1.76%”
        m = re.search(r"(\d+\.\d+)\s*%?", html)
        if m:
            val = float(m.group(1))
            # 合理性过滤：0.5%~8% 之间
            if 0.5 <= val <= 8.0:
                _cache_cn10y(CACHE_FILE, val, "te_web")
                return val, "te_web"
    except Exception:
        pass

    # 4) 英为财情（历史数据表）
    try:
        url = "https://cn.investing.com/rates-bonds/china-10-year-bond-yield-historical-data"
        html = requests.get(url, headers=UA, timeout=timeout).text
        tables = pd.read_html(html)
        if tables:
            df = tables[0]
            # 取第一行“最近值”或最后一行（按日期排序）
            # 英为财情表头/列名可能变化，通用做法：把第一列转日期成功的行作为数据
            df.columns = [str(c).strip() for c in df.columns]
            if "日期" in df.columns:
                df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
                df = df.dropna(subset=["日期"]).sort_values("日期")
                # 数值列通常叫“收盘”或“价格”
                for col in ("收盘", "价格", "收益率"):
                    if col in df.columns:
                        s = pd.to_numeric(df[col], errors="coerce").dropna()
                        if not s.empty and 0.5 <= float(s.iloc[-1]) <= 8.0:
                            val = float(s.iloc[-1])
                            _cache_cn10y(CACHE_FILE, val, "investing")
                            return val, "investing"
    except Exception:
        pass

    # 5) 全部失败：读缓存
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return float(obj["value"]), obj.get("source", "cache")
    except Exception:
        pass

    return float("nan"), "none"

def _cache_cn10y(path, value, source):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"value": float(value), "source": str(source), "ts": dt.datetime.now().isoformat()}, f)
    except Exception:
        pass

# —— 其他宏观/波动 ——
@_retry()
def get_fred_us_ig_1_3y():
    try:
        series = pdr.DataReader('BAMLC1A0C13YEY', 'fred')
        return float(series.dropna().iloc[-1])
    except Exception:
        return np.nan

@_retry()
def get_vix():
    try:
        vix = yf.Ticker("^VIX").history(period="1mo", interval="1d")["Close"].dropna()
        return float(vix.iloc[-1])
    except Exception:
        return np.nan

# —— 价量序列（用 510300 以便有成交量；回退 000300 无量） ——
@_retry()
def get_px_vol_510300(start="2018-01-01"):
    """
    返回 DataFrame(index=日期, ['close','volume','amount','_source'])
    优先: 东财 ETF 历史 (fund_etf_hist_em)
    回退: 新浪 ETF 历史 (fund_etf_hist_sina, 需 'sh510300')
    兜底: 东财指数 000300 的成交量(单位=手)仅供最后兜底
    """
    start_ymd = start.replace("-", "")
    end_ymd = dt.date.today().strftime("%Y%m%d")

    # 1) 东财 ETF 历史
    try:
        df = ak.fund_etf_hist_em(
            symbol="510300", period="daily",
            start_date=start_ymd, end_date=end_ymd, adjust=""
        )
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        df = df.dropna(subset=["日期"]).set_index("日期").sort_index()
        out = pd.DataFrame({
            "close":  pd.to_numeric(df["收盘"],  errors="coerce"),
            "volume": pd.to_numeric(df.get("成交量"), errors="coerce"),  # 可能缺
            "amount": pd.to_numeric(df.get("成交额"), errors="coerce")   # 可能缺
        })
        # 有 volume 或 amount 即可接受
        if out[["close","volume","amount"]].notna().any(axis=None):
            out["_source"] = "eastmoney_etf"
            return out.dropna(subset=["close"])
    except Exception:
        pass

    # 2) 新浪 ETF 历史（带交易所前缀）
    try:
        s = ak.fund_etf_hist_sina(symbol="sh510300")
        s["date"] = pd.to_datetime(s["date"], errors="coerce")
        s = s.dropna(subset=["date"]).set_index("date").sort_index()
        out = s[["close","volume"]].apply(pd.to_numeric, errors="coerce")
        out["amount"] = np.nan
        if not out.dropna(subset=["close"]).empty:
            out["_source"] = "sina_etf"
            return out
    except Exception:
        pass

    # 3) 兜底（不建议与 ETF 混源，仅在前两者完全失败时启用）
    try:
        idx = ak.index_zh_a_hist(symbol="000300", period="daily",
                                 start_date=start_ymd, end_date=end_ymd)
        idx["日期"] = pd.to_datetime(idx["日期"], errors="coerce")
        idx = idx.dropna(subset=["日期"]).set_index("日期").sort_index()
        out = pd.DataFrame({
            "close":  pd.to_numeric(idx["收盘"],  errors="coerce"),
            "volume": pd.to_numeric(idx.get("成交量"), errors="coerce"),  # 单位=手
            "amount": pd.to_numeric(idx.get("成交额"), errors="coerce")
        })
        if not out.dropna(subset=["close"]).empty:
            out["_source"] = "eastmoney_index"  # 标记为指数口径
            return out
    except Exception:
        pass

    # 全失败
    return pd.DataFrame(columns=["close","volume","amount","_source"])


@_retry()
def get_px_000300(start="2018-01-01"):
    start_ymd = start.replace("-", "")
    end_ymd = dt.date.today().strftime("%Y%m%d")
    df = ak.index_zh_a_hist(
        symbol="000300",
        period="daily",
        start_date=start_ymd,
        end_date=end_ymd
    )
    # 文档定义：没有 adjust 参数，日期列为 '日期'
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df.dropna(subset=["日期"]).set_index("日期").sort_index()
    close = pd.to_numeric(df["收盘"], errors="coerce").dropna()
    return close


# ====== 聚合名称与价格 ======
spot = get_etf_spot()
lof_spot = get_lof_spot()
fund_est = get_fund_estimation(list(current_positions.keys()))

NAME_MAP = {}
for df in (spot, lof_spot, fund_est):
    if not df.empty:
        NAME_MAP.update(df['name'].to_dict())
for k, v in CODE_NAMES.items():
    NAME_MAP.setdefault(k, v)


def _normalize_code(x: str) -> str:
    m = re.search(r'(\d{6})', str(x))
    return m.group(1) if m else str(x)


def load_price_book(codes):
    wanted = set(_normalize_code(c) for c in codes)
    price_book, name_book = {}, {}

    def _ingest_df(df, code_col, name_col, price_col):
        if df is None or df.empty:
            return
        tmp = df[[code_col, name_col, price_col]].copy()
        tmp.columns = ["code", "name", "price"]
        tmp["code"] = tmp["code"].map(_normalize_code)
        tmp["price"] = pd.to_numeric(tmp["price"], errors="coerce")
        for _, r in tmp.iterrows():
            c = r["code"]
            if c in wanted and pd.notna(r["price"]) and r["price"] > 0:
                price_book.setdefault(c, float(r["price"]))
                if isinstance(r["name"], str) and r["name"]:
                    name_book.setdefault(c, r["name"])

    try:
        _ingest_df(ak.fund_etf_spot_em(), "代码", "名称", "最新价")
    except Exception:
        pass
    try:
        _ingest_df(ak.fund_lof_spot_em(), "代码", "名称", "最新价")
    except Exception:
        pass
    try:
        _ingest_df(ak.fund_etf_category_sina(symbol="ETF基金"), "代码", "名称", "最新价")
    except Exception:
        pass
    try:
        _ingest_df(ak.fund_etf_category_sina(symbol="LOF基金"), "代码", "名称", "最新价")
    except Exception:
        pass
    try:
        est = ak.fund_em_value_estimation()
        if est is not None and not est.empty:
            est = est[["基金代码", "基金简称", "估算净值"]].rename(
                columns={"基金代码": "代码", "基金简称": "名称", "估算净值": "最新价"}
            )
            _ingest_df(est, "代码", "名称", "最新价")
    except Exception:
        pass
    try:
        ofd = ak.fund_open_fund_daily_em()
        if ofd is not None and not ofd.empty:
            ofd = ofd[["基金代码", "基金简称", "单位净值"]].rename(
                columns={"基金代码": "代码", "基金简称": "名称", "单位净值": "最新价"}
            )
            _ingest_df(ofd, "代码", "名称", "最新价")
    except Exception:
        pass

    for c, n in CODE_NAMES.items():
        name_book.setdefault(c, n)
    return price_book, name_book

PRICE_BOOK, NAME_BOOK = load_price_book(list(set(list(targets.keys()) + list(current_positions.keys()))))


def get_name(code: str) -> str:
    return NAME_BOOK.get(code) or NAME_MAP.get(code) or CODE_NAMES.get(code, code)


def get_price(code: str) -> float:
    return PRICE_BOOK.get(code, np.nan)

# ====== 更简单：CSI300 PE与10年分位（亿牛→Value500） ======

def _to_float_safe(x, default=float('nan')):
    try:
        return float(x)
    except Exception:
        return default

@_retry()
def get_csi300_pe_and_pct10y_simple(timeout=8):
    # 1) 亿牛
    try:
        r = requests.get("https://eniu.com/gu/sz399300", timeout=timeout, headers=UA)
        r.raise_for_status()
        html = r.text
        m_pe = re.search(r"市盈率：\s*([0-9.]+)", html)
        pe = _to_float_safe(m_pe.group(1)) if m_pe else float('nan')
        m_pct = re.search(r"近10年：\s*([0-9.]+)%", html)
        if m_pct:
            pct10y = _to_float_safe(m_pct.group(1)) / 100.0
            return pe, pct10y
    except Exception:
        pass
    # 2) Value500
    try:
        r = requests.get("https://value500.com/000300SHPEPB.asp", timeout=timeout, headers=UA)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True)
        m_pct = re.search(r"近十年所处百分位\s*([0-9.]+)%", text)
        m_pe = re.search(r"沪深300滚动市盈率\s*([0-9.]+)", text)
        pe = _to_float_safe(m_pe.group(1)) if m_pe else float('nan')
        if m_pct:
            pct10y = _to_float_safe(m_pct.group(1)) / 100.0
            return pe, pct10y
    except Exception:
        pass
    return float('nan'), float('nan')

# ====== 指标计算 ======
pe_300_web, pe_pct_web = get_csi300_pe_and_pct10y_simple()
pe_300_off, dy_300 = get_csindex_values("000300")
pe_300 = pe_300_web if (isinstance(pe_300_web, float) and not np.isnan(pe_300_web)) else pe_300_off
pe_pct = pe_pct_web

pe_div, dy_div = get_csindex_values("000922")
pe_500, _      = get_csindex_values("000905")
pe_kc50, _     = get_csindex_values("000688")

cn10y, cn10y_src = get_cn_yield_10y()
us_ig_1_3y = get_fred_us_1_3y = get_fred_us_ig_1_3y()
vix_now    = get_vix()

# 收益率转换与缺口计算（cn10y 已是百分数）
ep_300    = (1.0/pe_300) if pe_300 and pe_300>0 else np.nan
yield_gap = (ep_300 - cn10y/100.0) if (not np.isnan(ep_300) and not np.isnan(cn10y)) else np.nan
div_gap   = (dy_div - cn10y/100.0) if (not np.isnan(dy_div) and not np.isnan(cn10y)) else np.nan

# —— 价量数据（用于动量与位阶） ——
try:
    px_df = get_px_vol_510300(start="2018-01-01")
    close = px_df['close']
except Exception:
    close = get_px_000300(start="2018-01-01")
    px_df = pd.DataFrame({'close': close})

# MA结构
ma50  = close.rolling(50).mean()
ma200 = close.rolling(200).mean()
above_ma = bool(close.iloc[-1] > ma200.iloc[-1]) if not np.isnan(ma200.iloc[-1]) else None

# 回撤指标
def drawdown_from_high(series: pd.Series) -> float:
    if series is None or len(series)==0:
        return np.nan
    s = pd.Series(series).dropna().astype(float)
    if s.empty: return np.nan
    return float((s / s.cummax() - 1.0).iloc[-1])

try:
    # 用 510300 调价近100日回撤
    dd_100 = drawdown_from_high(close.tail(100))
except Exception:
    dd_100 = np.nan

# ====== 价量动量 + 位阶 特征 ======

def _pct(x):
    x = float(x)
    return max(0.0, min(1.0, x))

# 价格动量（1~12月）：
ret_63  = (close / close.shift(63) - 1.0).iloc[-1] if len(close)>=63 else np.nan
ret_252 = (close / close.shift(252) - 1.0).iloc[-1] if len(close)>=252 else np.nan
slope50 = (ma50.iloc[-1] / ma50.iloc[-5] - 1.0) if len(ma50.dropna())>=5 else np.nan

# 位阶：距52周高、与MA200的乖离（用分位自动阈值）
roll_max_252 = close.rolling(252).max()
prox_hi = (close / roll_max_252).iloc[-1] if not np.isnan(roll_max_252.iloc[-1]) else np.nan  # 越接近1越高位
ma_gap = (close.iloc[-1] / ma200.iloc[-1] - 1.0) if not np.isnan(ma200.iloc[-1]) else np.nan

# 历史分布分位（5年窗口）
look = 252*5
if len(close) >= look:
    prox_hi_p = ( (close/close.rolling(252).max()).dropna().rank(pct=True).iloc[-1] )
    ma_gap_p  = ( (close/close.rolling(200).mean()-1.0).dropna().rank(pct=True).iloc[-1] )
else:
    prox_hi_p, ma_gap_p = np.nan, np.nan

# 添加计算量能比的函数
def compute_volume_ratio(px_df: pd.DataFrame, short=20, long=120) -> float:
    """
    返回 量能比 = SMA(short)/SMA(long)
    优先使用 volume；volume 不足则用 amount 作为 proxy
    不足样本返回 np.nan
    """
    if px_df is None or px_df.empty:
        return np.nan

    df = px_df.copy()
    # 优先用 volume
    if "volume" in df and df["volume"].notna().sum() >= long:
        v = df["volume"].astype(float)
        r = v.rolling(short).mean().iloc[-1] / v.rolling(long).mean().iloc[-1]
        return float(r) if np.isfinite(r) else np.nan

    # 回退用 amount
    if "amount" in df and df["amount"].notna().sum() >= long:
        a = df["amount"].astype(float)
        r = a.rolling(short).mean().iloc[-1] / a.rolling(long).mean().iloc[-1]
        return float(r) if np.isfinite(r) else np.nan

    return np.nan

# 计算量能比
vol_ratio = compute_volume_ratio(px_df, short=20, long=120)

# ====== 组合净值用于月度回撤 / 年化波动 ======

def _hist_series_for_code(code, start="2022-01-01"):
    try:
        if code in {"510300","510500","588000","513500","513800","518880","513100"}:
            df = ak.fund_etf_hist_em(symbol=code, period="daily",
                                     start_date=start.replace("-",""),
                                     end_date=dt.date.today().strftime("%Y%m%d"),
                                     adjust="")
            s = pd.to_numeric(df["收盘"], errors="coerce")
            return s
        elif code in {"161119","164824","009051","001917","007360"}:
            try:
                df = ak.fund_lof_hist_em(symbol=code)
                s  = pd.to_numeric(df["收盘价"], errors="coerce")
            except Exception:
                df = ak.fund_open_fund_info_em(code=code, indicator="单位净值走势")
                s  = pd.to_numeric(df["单位净值"], errors="coerce")
            return s
    except Exception:
        pass
    return pd.Series(dtype=float)


def portfolio_daily_series(weights: dict, start="2022-01-01"):
    cols = []
    for c, w in weights.items():
        s = _hist_series_for_code(c, start=start)
        if s is None or s.empty or w<=0:
            continue
        s = s.dropna()
        if s.empty:
            continue
        s = s / s.iloc[0]
        cols.append(s.to_frame(name=c).pct_change().add(1).cumprod())
    if not cols:
        return pd.Series(dtype=float)
    df = pd.concat(cols, axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    w_series = pd.Series(weights)
    w_series = w_series / w_series.sum()
    port = (df * w_series.reindex(df.columns).fillna(0)).sum(axis=1)
    return port


def portfolio_dd_and_vol(weights: dict, start="2022-01-01"):
    nav = portfolio_daily_series(weights, start=start)
    if nav is None or nav.empty:
        return np.nan, np.nan, np.nan
    ret = nav.pct_change().dropna()
    if ret.empty: 
        return np.nan, np.nan, np.nan
    def _dd(x):
        roll_max = x.cummax(); return (x/roll_max - 1.0).iloc[-1]
    dd_m = _dd(nav.tail(22))   if len(nav)>=22  else np.nan
    dd_y = _dd(nav.tail(252))  if len(nav)>=252 else np.nan
    vol  = ret.tail(60).std() * np.sqrt(252) if len(ret)>=60 else ret.std() * np.sqrt(252)
    return float(dd_m), float(dd_y), float(vol)

# ====== 目标感知版：权益上限 + 强度 ======
EQUITY_CODES = {"510300","510500","588000","001917","513500","513100","513800","009051","164824"}
TARGET_EQUITY = float(sum(w for c, w in targets.items() if c in EQUITY_CODES))


def portfolio_drawdown_control(dd_m, dd_y, vix=vix_now):
    """以目标权益为锚的权益上限与强度缩放。"""
    te = TARGET_EQUITY if TARGET_EQUITY>0 else 0.55
    base_cap = min(max(te + 0.05, 0.40), 0.75)
    cap = base_cap; intensity_scale = 1.0
    bad_1 = (not np.isnan(dd_m) and dd_m < -0.05) or (not np.isnan(vix) and vix >= 25)
    bad_2 = (not np.isnan(dd_m) and dd_m < -0.06) or (not np.isnan(vix) and vix >= 28)
    if bad_2:
        cap = max(te - 0.10, 0.40); intensity_scale = 0.6
    elif bad_1:
        cap = max(te - 0.05, 0.45); intensity_scale = 0.8
    cap = max(cap, te - 0.10)
    return {"equity_ratio_cap": float(cap), "intensity_scale": float(intensity_scale)}


def vol_target_adjustment(vol_annual, target_vol=0.10, floor=0.6, cap=1.1):
    """波动率目标：强度按目标波动缩放，上限降到1.1（更保守）"""
    if np.isnan(vol_annual) or vol_annual<=0:
        return 1.0
    scale = target_vol / vol_annual
    return float(max(floor, min(cap, scale)))


def decide_intensity_base_erp_trend():
    """原框架：收益率缺口+趋势+VIX → 1/0.5/0"""
    if not USE_INTENSITY:
        return 1.0
    base = 0.5
    if (((not np.isnan(yield_gap) and yield_gap<=0) or (above_ma is False))
         and (not np.isnan(vix_now) and vix_now>25)):
        base = 0.0
    elif ((not np.isnan(yield_gap) and yield_gap>0.02) and (above_ma is True)
           and (not np.isnan(vix_now) and vix_now<22)):
        base = 1.0
    elif ((not np.isnan(yield_gap) and yield_gap>0)
           and ((not np.isnan(div_gap) and div_gap>0) or (not np.isnan(vix_now) and 15<=vix_now<=25))):
        base = 0.5
    else:
        base = 0.5
    return base


def decide_intensity_price_volume_level():
    """V2：在 base 的基础上叠加 估值分位/回撤、价量动量、位阶因子（自动阈值）。"""
    base = decide_intensity_base_erp_trend()

    # —— 估值分位：更早降速 ——
    if not np.isnan(pe_pct):
        if pe_pct >= 0.80: base *= 0.70
        elif pe_pct >= 0.70: base *= 0.85
        elif pe_pct <= 0.10: base *= 1.30
        elif pe_pct <= 0.20: base *= 1.15

    # —— 回撤：逢跌稍加速 ——
    if not np.isnan(dd_100):
        if dd_100 <= -0.20: base *= 1.30
        elif dd_100 <= -0.10: base *= 1.15

    # —— 价量动量 ——
    # MA状态 + 3月/12月动量 + MA50斜率 + 量能配合
    if above_ma is False or (not np.isnan(ret_63) and ret_63 < 0):
        base *= 0.80
    else:
        base *= 1.00
    if not np.isnan(ret_252) and ret_252 > 0:
        base *= 1.05
    if not np.isnan(slope50) and slope50 < 0:
        base *= 0.95
    if not np.isnan(vol_ratio):
        if vol_ratio >= 1.10 and above_ma is True:
            base *= 1.03
        elif vol_ratio <= 0.90:
            base *= 0.97

    # —— 位阶因子（自动阈值：历史分位） ——
    # 位阶惩罚函数
    def level_penalty(q_level: float, start_q: float = 0.75, floor: float = 0.75) -> float:
        """
        q_level: 位阶分位 ∈ [0,1]（取 52周高分位 与 MA200乖离分位 的较大值）
        start_q: 从该分位开始线性降速（更稳：0.75）
        floor:   最低惩罚倍数（更稳：0.75）
        """
        if np.isnan(q_level):
            return 1.0
        if q_level <= start_q:
            return 1.0
        cut = (q_level - start_q) / max(1e-9, (1.0 - start_q))  # 0→1
        pen = 1.0 - cut * (1.0 - floor)
        return float(max(floor, min(1.0, pen)))
    
    # 取两个位阶分位中较大者
    level_q = np.nanmax([prox_hi_p, ma_gap_p]) if not (np.isnan(prox_hi_p) and np.isnan(ma_gap_p)) else np.nan
    penalty_level = level_penalty(level_q, start_q=0.75, floor=0.75)
    base *= penalty_level

    # —— 组合风险维度：月度回撤 + 波动率目标 ——
    weights = {c: current_positions.get(c,0)/max(1.0, sum(current_positions.values())) for c in targets}
    dd_m, dd_y, vol = portfolio_dd_and_vol(weights, start="2022-01-01")

    # 固定计算顺序：base × (估值/动量/量能) × 位阶惩罚 × 风险挡板 × 波动率缩放
    base_with_factors = base  # 已包含估值分位/回撤/动量/量能
    with_level = base_with_factors * penalty_level  # 位阶惩罚
    
    dd_ctl = portfolio_drawdown_control(dd_m, dd_y)
    with_risk = with_level * dd_ctl["intensity_scale"]  # 风险挡板
    
    vol_ctl = vol_target_adjustment(vol, target_vol=0.10)
    raw_intensity = with_risk * vol_ctl  # 波动率缩放
    
    # 最后一步再截断到 [0,1]
    enhanced = float(max(0.0, min(1.0, raw_intensity)))
    
    # 存储中间值便于打印/调试
    globals()["_EQUITY_CAP_FROM_RISK"] = dd_ctl["equity_ratio_cap"]
    globals()["_DD_M"], globals()["_DD_Y"], globals()["_VOL_ANNUAL"] = dd_m, dd_y, vol
    globals()["_LEVEL_PENALTY"] = penalty_level
    globals()["_LEVEL_Q"] = level_q
    globals()["_RAW_INTENSITY"] = raw_intensity
    globals()["_MOMO_FIELDS"] = {"ret_63":ret_63, "ret_252":ret_252, "slope50":slope50, "vol_ratio":vol_ratio,
                                  "prox_hi_p":prox_hi_p, "ma_gap_p":ma_gap_p}
    return enhanced

# 使用 V2 强度
after_intensity = decide_intensity_price_volume_level()
intensity = after_intensity

# ====== 动态分批额度 ======
invested_now = float(sum(current_positions.values()))
outstanding  = TOTAL_TARGET - invested_now


def auto_adjust_weekly_amount(outstanding, invested_now):
    if abs(outstanding) < MIN_TRADE_AMOUNT:
        return 0.0
    total_assets  = invested_now + max(outstanding, 0.0)
    max_trade_by_pct = total_assets * MAX_TRADE_PCT
    cash_reserve  = total_assets * CASH_RESERVE_PCT
    available_cash = max(0.0, outstanding - cash_reserve)
    deviation_pct = abs(outstanding) / TOTAL_TARGET
    deviation_factor = min(1.5, max(0.5, 1.0 + deviation_pct))
    base_amount = min(WEEKLY_LIMIT, max_trade_by_pct)
    if outstanding > 0:   # 买入
        return min(base_amount * deviation_factor, outstanding, available_cash)
    else:                 # 减持
        return min(base_amount * deviation_factor, abs(outstanding))

if AUTO_ADJUST:
    gross_weekly = auto_adjust_weekly_amount(outstanding, invested_now)
else:
    gross_weekly = min(WEEKLY_LIMIT, abs(outstanding)/max(STAGING_WEEKS,1))

planned_buy  = gross_weekly * intensity if outstanding>0 else 0.0
planned_sell = gross_weekly * intensity if outstanding<0 else 0.0

# ====== 与目标的差额 ======

target_amounts = {c: TOTAL_TARGET*w for c, w in targets.items()}
deficits = {c: max(target_amounts[c] - current_positions.get(c, 0.0), 0.0) for c in targets}
excesses = {c: max(current_positions.get(c, 0.0) - target_amounts[c], 0.0) for c in targets}


def apply_rebalance_band(deficits: dict, band_pp_abs=BAND_PP_ABS, band_rel=BAND_REL) -> dict:
    filt = {}
    for c, need_amt in deficits.items():
        cur_amt = current_positions.get(c, 0.0)
        cur_pct = cur_amt / TOTAL_TARGET if TOTAL_TARGET > 0 else 0.0
        tgt_pct = targets[c]
        under_pp = max((tgt_pct - cur_pct) * 100.0, 0.0)
        rel_pp = tgt_pct * band_rel * 100.0
        thresh = max(band_pp_abs, rel_pp)
        filt[c] = need_amt if under_pp >= thresh else 0.0
    return filt

deficits_active = apply_rebalance_band(deficits, BAND_PP_ABS, BAND_REL)

# ====== 触发器 / 整手 / 分配 ======

def triggers_for_buy(plan: dict) -> dict:
    plan = plan.copy()
    # 500估值高 → 减半，回流债/红利
    if pe_500 and pe_500 > 30:
        cut = plan.get("510500", 0.0) * 0.5
        plan["510500"] = plan.get("510500", 0.0) - cut
        plan["161119"] = plan.get("161119", 0.0) + cut * 0.6
        plan["009051"] = plan.get("009051", 0.0) + cut * 0.4
    # 黄金当批不追 → 回流债/红利
    if plan.get("518880", 0.0) > 0:
        extra = plan["518880"]; plan["518880"] = 0.0
        plan["161119"] = plan.get("161119", 0.0) + extra * 0.5
        plan["009051"] = plan.get("009051", 0.0) + extra * 0.5
    # 纳指限额50% → 砍掉部分回流
    if plan.get("513100", 0.0) > 0:
        original = plan["513100"]; kept = original * 0.5; cut = original - kept
        plan["513100"] = kept
        plan["161119"] = plan.get("161119", 0.0) + cut * 0.5
        plan["009051"] = plan.get("009051", 0.0) + cut * 0.5
    return plan


def apply_board_lot(plan: dict) -> dict:
    extra = 0.0
    adj = plan.copy()
    for code in list(BOARD_LOT_CODES):
        amt = adj.get(code, 0.0)
        if amt <= 0: continue
        price = float(spot.loc[code, 'price']) if (code in spot.index) else np.nan
        if np.isnan(price) or price <= 0:
            extra += amt; adj[code] = 0.0; continue
        lots = int(amt // (price * 100))
        if lots <= 0:
            extra += amt; adj[code] = 0.0
        else:
            adj_amt = lots * 100 * price
            adj[code] = adj_amt
            extra += (amt - adj_amt)
    if extra > 0:
        adj["161119"] = adj.get("161119", 0.0) + extra * 0.5
        adj["009051"] = adj.get("009051", 0.0) + extra * 0.5
    return adj


def alloc_buy(amount: float, deficits: dict, constrained: bool) -> dict:
    if amount <= 0:
        return {c:0.0 for c in targets}
    need_total = sum(deficits.values())
    if need_total == 0:
        return {c:0.0 for c in targets}
    plan = {c: (deficits[c]/need_total) * amount for c in targets}
    plan = triggers_for_buy(plan)
    if constrained:
        plan = apply_board_lot(plan)
    return plan


def alloc_sell(amount: float, excesses: dict, constrained: bool) -> dict:
    if amount <= 0:
        return {}
    if constrained:
        base_set = [c for c in targets if c in SELLABLE_CODES]
    else:
        base_set = list(targets.keys())
    exc_total = sum(excesses[c] for c in base_set)
    if exc_total <= 0:
        return {}
    plan = {}
    for c in base_set:
        alloc = amount * (excesses[c]/exc_total) if exc_total>0 else 0.0
        max_allow = min(excesses[c], current_positions.get(c, 0.0))
        plan[c] = float(min(alloc, max_allow))
    return plan


def make_df(plan: dict, buy: bool=True):
    rows, s = [], 0.0
    col = "建议买入金额(¥)" if buy else "建议减持金额(¥)"
    for code, amt in plan.items():
        if amt <= 0: continue
        rows.append([code, get_name(code), float(round(amt)), get_price(code)])
        s += float(round(amt))
    df = pd.DataFrame(rows, columns=["代码","名称", col, "参考价格"]).sort_values(col, ascending=False)
    if not df.empty:
        df[col] = pd.to_numeric(df[col], errors='coerce').round(0)
        if "参考价格" in df.columns:
            df["参考价格"] = pd.to_numeric(df["参考价格"], errors='coerce').round(3)
    return df, s

# ====== 持仓表 ======

def compute_after_positions(curr_pos: dict, buy_plan: dict, sell_plan: dict) -> dict:
    all_codes = set(list(targets.keys()) + list(curr_pos.keys()) + list(buy_plan.keys()) + list(sell_plan.keys()))
    new_pos = {}
    for c in all_codes:
        base = float(curr_pos.get(c, 0.0))
        inc  = float(buy_plan.get(c, 0.0))
        dec  = float(sell_plan.get(c, 0.0))
        new_pos[c] = max(base + inc - dec, 0.0)
    return new_pos


def make_holdings_table(positions: dict) -> pd.DataFrame:
    rows = []
    total_current = sum(positions.values())
    for c, w in targets.items():
        amt = float(positions.get(c, 0.0))
        target_amt = TOTAL_TARGET * w
        pct_of_current = 100.0 * amt / total_current if total_current > 0 else 0.0
        rows.append([c, get_name(c), round(100.0*w, 2), round(amt, 0), 
                     round(pct_of_current, 2), round(target_amt, 0)])
    df = pd.DataFrame(rows, columns=["代码","名称","目标权重%","持仓(¥)","当前配比%","目标金额(¥)"])
    return df.sort_values("目标权重%", ascending=False)


def make_targets_table() -> pd.DataFrame:
    rows = []
    for c, w in targets.items():
        rows.append([c, get_name(c), round(w*100,2), round(TOTAL_TARGET*w,0)])
    return pd.DataFrame(rows, columns=["代码","名称","目标权重%","目标金额(¥)"]).sort_values("目标权重%", ascending=False)

# ====== 回流策略 ======

def should_reallocate_to_defense() -> bool:
    dd_m = globals().get("_DD_M", np.nan)
    vix = globals().get("vix_now", np.nan)
    inten = globals().get("intensity", 1.0)
    am = globals().get("above_ma", None)
    cny = globals().get("cn10y", np.nan)
    dgap = globals().get("div_gap", np.nan)
    bad = ((not np.isnan(dd_m) and dd_m <= -0.05) or (not np.isnan(vix) and vix >= 25) or (inten <= 0.5) or (am is False))
    if bad: return False
    value_ok = ((not np.isnan(cny) and cny >= 2.0) or (not np.isnan(dgap) and dgap >= 0.008))
    momentum_ok = ((inten >= 0.75) and (not np.isnan(vix) and vix <= 22))
    return bool(value_ok or momentum_ok)


def enforce_equity_cap(buy_plan: dict, sell_plan: dict, cap: float) -> dict:
    if not cap or cap >= 0.99:
        return buy_plan
    total_now = sum(current_positions.values())
    eq_now = sum(current_positions.get(c, 0.0) for c in EQUITY_CODES)
    buy_eq_orig = sum(buy_plan.get(c, 0.0) for c in EQUITY_CODES)
    sell_eq = sum(sell_plan.get(c, 0.0) for c in EQUITY_CODES)
    total_after_orig = total_now + sum(buy_plan.values()) - sum(sell_plan.values())
    eq_after_orig    = eq_now + buy_eq_orig - sell_eq
    if eq_after_orig / max(1.0, total_after_orig) <= cap:
        return buy_plan
    # 缩放权益买单
    target_eq_after = cap * total_after_orig
    need_cut_eq = max(eq_after_orig - target_eq_after, 0.0)
    if need_cut_eq <= 0 or buy_eq_orig <= 0:
        return buy_plan
    scale = max(0.0, min(1.0, (buy_eq_orig - need_cut_eq) / buy_eq_orig))
    adj = buy_plan.copy(); cut_total = 0.0
    for c in EQUITY_CODES:
        if c in adj:
            old_amt = adj[c]; new_amt = old_amt * scale
            cut_total += (old_amt - new_amt)
            adj[c] = new_amt
    # 回流策略
    policy = globals().get("REALLOCATE_POLICY", "auto")
    do_realloc = (policy=="defense") or (policy=="auto" and should_reallocate_to_defense())
    if do_realloc and cut_total > 0:
        adj.setdefault("161119", 0.0); adj.setdefault("009051", 0.0)
        adj["161119"] += cut_total * 0.5; adj["009051"] += cut_total * 0.5
    # 安全检查
    total_after = total_now + sum(adj.values()) - sum(sell_plan.values())
    eq_after    = eq_now + sum(adj.get(c, 0.0) for c in EQUITY_CODES) - sell_eq
    if eq_after / max(1.0, total_after) > cap:
        over = eq_after - cap * total_after
        if over > 0 and sum(adj.get(c, 0.0) for c in EQUITY_CODES) > 0:
            shrink = min(1.0, over / max(1e-9, sum(adj.get(c, 0.0) for c in EQUITY_CODES)))
            for c in EQUITY_CODES:
                if c in adj:
                    cut = adj[c] * shrink; adj[c] -= cut
                    if do_realloc:
                        adj.setdefault("161119", 0.0); adj.setdefault("009051", 0.0)
                        adj["161119"] += cut * 0.5; adj["009051"] += cut * 0.5
    globals()["_REALLOCATE_POLICY_APPLIED"] = f"{policy}:{'on' if do_realloc else 'off'}"
    return adj

# ====== 生成两套方案 ======
buy_constrained  = alloc_buy(planned_buy,  deficits_active, constrained=True)
sell_constrained = alloc_sell(planned_sell, excesses,      constrained=True)
buy_constrained = enforce_equity_cap(buy_constrained, sell_constrained, globals().get("_EQUITY_CAP_FROM_RISK", 0.45))

buy_fundonly  = alloc_buy(planned_buy,  deficits_active, constrained=False)
sell_fundonly = alloc_sell(planned_sell, excesses,       constrained=False)
buy_fundonly = enforce_equity_cap(buy_fundonly, sell_fundonly, globals().get("_EQUITY_CAP_FROM_RISK", 0.45))

# ====== 打印 ======

def _fmt(v): 
    return "NaN" if v is None or (isinstance(v,float) and np.isnan(v)) else v

print("\n=== 关键指标（简要） ===")
print(f"CSI300 PE/E-P: {_fmt(round(pe_300,2))} / {_fmt(round((1.0/pe_300) if pe_300 and pe_300>0 else np.nan,4))}；中证红利DY: {_fmt(round(dy_div*100,2))}%")
print(f"CN10Y/美IG1-3Y: {_fmt(round(cn10y,2))}% / {_fmt(round(us_ig_1_3y,2))}%；收益率缺口: {_fmt(round((yield_gap*100) if not np.isnan(yield_gap) else np.nan,2))}pp")
print(f"红利-国债缺口: {_fmt(round((div_gap*100) if not np.isnan(div_gap) else np.nan,2))}pp；MA200上方: {above_ma}；VIX: {_fmt(round(vix_now,2))}")
print(f"PE分位(10y): {_fmt(round(pe_pct*100,1)) if not np.isnan(pe_pct) else 'NaN'}%；近100日回撤: {_fmt(round(dd_100*100,1)) if not np.isnan(dd_100) else 'NaN'}%")

# 价量/位阶调节信息
momo = globals().get("_MOMO_FIELDS", {})
print(f"动量(3/12月): {_fmt(round(momo.get('ret_63',np.nan)*100,1)) if momo.get('ret_63')==momo.get('ret_63') else 'NaN'}% / {_fmt(round(momo.get('ret_252',np.nan)*100,1)) if momo.get('ret_252')==momo.get('ret_252') else 'NaN'}%",
      f"；MA50斜率: {_fmt(round(momo.get('slope50',np.nan)*100,2)) if momo.get('slope50')==momo.get('slope50') else 'NaN'}%",
      f"；量能比(20/120): {_fmt(round(momo.get('vol_ratio',np.nan),2)) if momo.get('vol_ratio')==momo.get('vol_ratio') else 'NaN'} ")
print(f"位阶分位：综合 {_fmt(round(globals().get('_LEVEL_Q',np.nan)*100,1))}%",
      f"（52周高 {_fmt(round(momo.get('prox_hi_p',np.nan)*100,1)) if momo.get('prox_hi_p')==momo.get('prox_hi_p') else 'NaN'}%",
      f"/ MA200乖离 {_fmt(round(momo.get('ma_gap_p',np.nan)*100,1)) if momo.get('ma_gap_p')==momo.get('ma_gap_p') else 'NaN'}%）",
      f"；位阶惩罚: {_fmt(round(globals().get('_LEVEL_PENALTY',1.0),2))}")

# 组合风险
dd_m, dd_y, vol = globals().get("_DD_M", np.nan), globals().get("_DD_Y", np.nan), globals().get("_VOL_ANNUAL", np.nan)
eq_cap = globals().get("_EQUITY_CAP_FROM_RISK", 0.45)
print(f"组合月度回撤: {_fmt(round(dd_m*100,1)) if not np.isnan(dd_m) else 'NaN'}%；年化波动: {_fmt(round(vol*100,1)) if not np.isnan(vol) else 'NaN'}%")
print(f"权益上限: {_fmt(round(eq_cap*100,1))}%")
raw_intensity = globals().get("_RAW_INTENSITY", intensity)
print(f"强度(未截断/最终): {_fmt(round(raw_intensity,2))}/{_fmt(round(intensity,2))}")

if "_REALLOCATE_POLICY_APPLIED" in globals():
    policy_info = globals().get("_REALLOCATE_POLICY_APPLIED", "auto:off")
    policy, status = policy_info.split(":")
    print(f"权益重分配策略: {policy} ({status})")

print("\n=== 资金进度 & 交易参数 ===")
print(f"当前已投入: ¥{invested_now:.0f} | 目标: ¥{TOTAL_TARGET:.0f} | 与目标差额: ¥{outstanding:.0f}")

if AUTO_ADJUST:
    total_assets = invested_now + max(outstanding, 0.0)
    max_trade_by_pct = total_assets * MAX_TRADE_PCT
    deviation_pct = abs(outstanding) / TOTAL_TARGET
    deviation_factor = min(1.5, max(0.5, 1.0 + deviation_pct))
    print(f"自动交易模式：最小交易额 ¥{MIN_TRADE_AMOUNT:.0f} | 最大单次比例 {MAX_TRADE_PCT*100:.1f}% | 现金储备 {CASH_RESERVE_PCT*100:.1f}%")
    print(f"总资产: ¥{total_assets:.0f} | 偏离度: {deviation_pct*100:.1f}% | 调整系数: {deviation_factor:.2f}")
    print(f"基础周额度: ¥{min(WEEKLY_LIMIT, max_trade_by_pct):.0f}")
else:
    print(f"固定分批模式：分摊周数 {STAGING_WEEKS} | 每周上限: ¥{WEEKLY_LIMIT:.0f}")

print(f"→ 本周计划买入: ¥{planned_buy:.0f} | 本周计划减持: ¥{planned_sell:.0f}")

# ====== 导出 ======

def export_plan(label: str, buy_plan: dict, sell_plan: dict, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    today = dt.date.today().strftime("%Y%m%d")
    df_buy, sum_buy = make_df(buy_plan,  buy=True)
    df_sell, sum_sell = make_df(sell_plan, buy=False)
    if not df_buy.empty:
        df_buy.to_csv(os.path.join(outdir, f"buy_orders_{label}_{today}.csv"), index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["代码","名称","建议买入金额(¥)","参考价格"]).to_csv(
            os.path.join(outdir, f"buy_orders_{label}_{today}.csv"), index=False, encoding="utf-8-sig")
    if not df_sell.empty:
        df_sell.to_csv(os.path.join(outdir, f"sell_orders_{label}_{today}.csv"), index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["代码","名称","建议减持金额(¥)","参考价格"]).to_csv(
            os.path.join(outdir, f"sell_orders_{label}_{today}.csv"), index=False, encoding="utf-8-sig")

    before_df = make_holdings_table(current_positions)
    after_positions = compute_after_positions(current_positions, buy_plan, sell_plan)
    after_df  = make_holdings_table(after_positions)
    diff_df = before_df.merge(after_df, on=["代码","名称","目标权重%","目标金额(¥)"], suffixes=("_更新前","_更新后"))
    diff_df["金额变化(¥)"] = (diff_df["持仓(¥)_更新后"] - diff_df["持仓(¥)_更新前"]).round(0)
    keep_cols = [
        "代码","名称","目标权重%",
        "持仓(¥)_更新前","当前配比%_更新前",
        "持仓(¥)_更新后","当前配比%_更新后",
        "目标金额(¥)","金额变化(¥)"
    ]
    outdir = outdir
    before_df.to_csv(os.path.join(outdir, f"holdings_before_{label}_{today}.csv"), index=False, encoding="utf-8-sig")
    after_df.to_csv( os.path.join(outdir, f"holdings_after_{label}_{today}.csv"),  index=False, encoding="utf-8-sig")
    diff_df[keep_cols].to_csv(os.path.join(outdir, f"holdings_diff_{label}_{today}.csv"), index=False, encoding="utf-8-sig")

    tgt_df = make_targets_table()
    tgt_df.to_csv(os.path.join(outdir, f"targets_{label}_{today}.csv"), index=False, encoding="utf-8-sig")

    print(f"\n[{label}] 导出完成：")
    print(f"  buy_orders_{label}_{today}.csv（合计买入 ¥{sum_buy:.0f}）")
    print(f"  sell_orders_{label}_{today}.csv（合计减持 ¥{sum_sell:.0f}）")
    print(f"  holdings_before_{label}_{today}.csv / holdings_after_{label}_{today}.csv / holdings_diff_{label}_{today}.csv")
    print(f"  targets_{label}_{today}.csv（含中文名称与目标金额）")

# ====== CLI ======
parser = argparse.ArgumentParser(description="Rebalance helper V2")
parser.add_argument("--plan", choices=["A", "B", "both"], default="both")
parser.add_argument("--no-export", action="store_true")
parser.add_argument("--outdir", default="exports")
parser.add_argument("--no-auto", action="store_true")
parser.add_argument("--min-trade", type=float, default=MIN_TRADE_AMOUNT)
parser.add_argument("--max-trade-pct", type=float, default=MAX_TRADE_PCT)
parser.add_argument("--cash-reserve-pct", type=float, default=CASH_RESERVE_PCT)
parser.add_argument("--realloc", choices=["auto", "defense", "none"], default=REALLOCATE_POLICY)
args = parser.parse_args()
EXPORT = (not args.no_export)
AUTO_ADJUST = (not args.no_auto)
MIN_TRADE_AMOUNT = args.min_trade
MAX_TRADE_PCT = args.max_trade_pct
CASH_RESERVE_PCT = args.cash_reserve_pct
REALLOCATE_POLICY = args.realloc

# ====== 执行 ======
if EXPORT:
    if args.plan in ["A", "both"]:
        export_plan("A", alloc_buy(planned_buy, deficits_active, True), alloc_sell(planned_sell, excesses, True), args.outdir)
    if args.plan in ["B", "both"]:
        export_plan("B", alloc_buy(planned_buy, deficits_active, False), alloc_sell(planned_sell, excesses, False), args.outdir)
else:
    if args.plan in ["A", "both"]:
        df_b1, sum_b1 = make_df( enforce_equity_cap(alloc_buy(planned_buy, deficits_active, True), alloc_sell(planned_sell, excesses, True), globals().get("_EQUITY_CAP_FROM_RISK", 0.45)),  buy=True)
        df_s1, sum_s1 = make_df( alloc_sell(planned_sell, excesses, True), buy=False)
        before_df = make_holdings_table(current_positions)
        after_positions = compute_after_positions(current_positions, dict(df_b1[["代码","建议买入金额(¥)"]].set_index("代码")["建议买入金额(¥)"]), dict(df_s1[["代码","建议减持金额(¥)"]].set_index("代码")["建议减持金额(¥)"]))
        after_df  = make_holdings_table(after_positions)
        simp = before_df.merge(after_df[["代码","当前配比%"]], on="代码", suffixes=("","_更新后"))
        simp.rename(columns={"当前配比%":"当前配比%","当前配比%_更新后":"更新后配比%"}, inplace=True)
        print("\n=== 方案A：买入清单（摘要） ==="); print(df_b1.head(10).to_string(index=False) if not df_b1.empty else "（无）")
        print(f"合计买入: ¥{sum_b1:.0f}")
        print("\n=== 方案A：减持清单（摘要） ==="); print(df_s1.head(10).to_string(index=False) if not df_s1.empty else "（无）")
        print(f"合计减持: ¥{sum_s1:.0f}")
        print("\n=== 方案A：持仓配比表 ==="); print(simp[["代码","名称","目标权重%","持仓(¥)","当前配比%","更新后配比%"]].to_string(index=False))
    if args.plan in ["B", "both"]:
        df_b2, sum_b2 = make_df( enforce_equity_cap(alloc_buy(planned_buy, deficits_active, False), alloc_sell(planned_sell, excesses, False), globals().get("_EQUITY_CAP_FROM_RISK", 0.45)),  buy=True)
        df_s2, sum_s2 = make_df( alloc_sell(planned_sell, excesses, False), buy=False)
        before_df = make_holdings_table(current_positions)
        after_positions = compute_after_positions(current_positions, dict(df_b2[["代码","建议买入金额(¥)"]].set_index("代码")["建议买入金额(¥)"]), dict(df_s2[["代码","建议减持金额(¥)"]].set_index("代码")["建议减持金额(¥)"]))
        after_df  = make_holdings_table(after_positions)
        simp = before_df.merge(after_df[["代码","当前配比%"]], on="代码", suffixes=("","_更新后"))
        simp.rename(columns={"当前配比%":"当前配比%","当前配比%_更新后":"更新后配比%"}, inplace=True)
        print("\n=== 方案B：买入清单（摘要） ==="); print(df_b2.head(10).to_string(index=False) if not df_b2.empty else "（无）")
        print(f"合计买入: ¥{sum_b2:.0f}")
        print("\n=== 方案B：减持清单（摘要） ==="); print(df_s2.head(10).to_string(index=False) if not df_s2.empty else "（无）")
        print(f"合计减持: ¥{sum_s2:.0f}")
        print("\n=== 方案B：持仓配比表 ==="); print(simp[["代码","名称","目标权重%","持仓(¥)","当前配比%","更新后配比%"]].to_string(index=False))

print("\n提示：开放式/LOF 使用的是“估算净值”，与收盘净值存在偏差；下单以当时成交价/估值为准。导出的 CSV 建议在表格软件里查看与执行。")
