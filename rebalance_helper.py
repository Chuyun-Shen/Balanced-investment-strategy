# -*- coding: utf-8 -*-
"""
自动化资产再平衡助手（双结果打印 + 默认导出 CSV）：
- 自动判断：根据当前持仓和资金状况，智能确定每周交易金额
- 平滑到位：把与目标的差额按 STAGING_WEEKS 周分摊，再乘以强度系数 (0/0.5/1)
- 方案A（交易约束）：ETF 按100股整手、减持仅动开放式/LOF
- 方案B（纯基金）：全部按基金处理，任意金额买卖（便于按权重铺开）
- 强度增强：估值分位 + 回撤加权（贵时减速，跌时稍加速）
- 再平衡带宽：未明显偏离目标不动，降低高位小额补仓的概率
- 风险控制：组合级回撤挡板 + 波动率目标 + 权益上限约束

依赖：pip install akshare pandas numpy yfinance pandas-datareader
"""
import argparse
import os, warnings, datetime as dt
import pandas as pd
import numpy as np
import re

warnings.filterwarnings("ignore")

# 设置 pandas 显示选项，改善终端表格对齐
pd.set_option('display.unicode.east_asian_width', True)  # 让中文对齐
pd.set_option('display.max_columns', 20)
pd.set_option('display.precision', 2)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.width', 150)  # 增加显示宽度以改善对齐
pd.set_option('display.unicode.ambiguous_as_wide', True)  # 处理模糊宽度字符

# ====== 全局参数（按需修改） ======
TOTAL_TARGET   = 40000.0   # 目标总额
STAGING_WEEKS  = 6         # 原4 → 6，更平滑
WEEKLY_LIMIT   = 3000.0    # 原4000 → 3000，降低冲击
USE_INTENSITY  = True      # 是否启用强度系数 (0/0.5/1)
AUTO_ADJUST    = True      # 是否启用自动调整每周交易金额

# 自动调整参数
MIN_TRADE_AMOUNT = 500.0   # 最小交易金额（低于此值不交易）
MAX_TRADE_PCT    = 0.05    # 最大单次交易占总资产比例
CASH_RESERVE_PCT = 0.02    # 现金储备比例（占总资产）

# 带宽：绝对带宽 + 相对带宽（对目标的±20%）
BAND_PP_ABS = 1.5     # 百分点，原1.0 → 1.5
BAND_REL    = 0.20    # 相对偏离阈值：目标权重的±20%

# 场内ETF 需要按100股整手成交的代码（方案A生效）
BOARD_LOT_CODES = {"510300","510500","588000","513500","513800"}

# 减持时，默认只动这些“可任意金额”的开放式/LOF（避免ETF整手卖出的麻烦）
SELLABLE_CODES = {"161119","007360","001917","164824","009051"}

# （可选）中文名称表（用于打印更清晰；行情接口也会补充）
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

# ====== 目标权重（保守优化版） ======
targets = {
    # 防守资产 35%
    "161119": 0.25,  # 易方达中债新综合债券指数LOF（人民币债，久期基石）
    "007360": 0.10,  # 易方达中短期美元债(QDII)A（短久期IG，低波票息）

    # 核心权益 40%
    "510300": 0.15,  # 沪深300（核心宽基，性价比、流动性）
    "009051": 0.15,  # 中证红利ETF/联接（股息/质量，回撤缓冲）
    "510500": 0.06,  # 中证500（中盘补充）
    "001917": 0.04,  # 招商量化精选A（阿尔法补充）

    # 海外权益 15%
    "513500": 0.10,  # 标普500ETF（核心海外敞口）
    "513100": 0.02,  # 纳指100ETF（成长卫星，估值克制）
    "513800": 0.03,  # 日本东证ETF（治理红利，降权）

    # 商品+替代 10%
    "518880": 0.06,  # 黄金ETF（通胀与极端尾部对冲）
    "588000": 0.02,  # 科创50（本土成长卫星，谨慎）
    "164824": 0.02,  # 工银印度LOF（新兴市场卫星，降权）
}

# 债券地板（避免为追收益把防守砍太低）
DEBT_FLOOR = 0.35  # = 161119 + 007360 的合计权重下限

def apply_debt_floor(targets: dict, debt_floor=DEBT_FLOOR) -> dict:
    t = targets.copy()
    debt_sum = t.get("161119",0)+t.get("007360",0)
    if debt_sum >= debt_floor: 
        return t
    # 不足则从权益合集中按权重比例等比扣，补到 161119
    need = debt_floor - debt_sum
    eq_codes = [c for c in t if c not in ("161119","007360")]
    eq_sum = sum(t[c] for c in eq_codes)
    if eq_sum <= 0: 
        t["161119"] += need; return t
    for c in eq_codes:
        t[c] = max(t[c] - need * (t[c]/eq_sum), 0)
    t["161119"] += need
    # 归一化到 1
    s = sum(t.values())
    for c in t: t[c] = t[c]/s
    return t

targets = apply_debt_floor(targets, DEBT_FLOOR)

# ====== 当前持仓（从配置文件加载）======
from config import load_positions
current_positions = load_positions()

# ====== 外部数据抓取 ======
import akshare as ak
import yfinance as yf
from pandas_datareader import data as pdr

def get_etf_spot():
    df = ak.fund_etf_spot_em()
    df = df[['代码','名称','最新价','涨跌幅','成交额','成交量']].rename(
        columns={'代码':'code','名称':'name','最新价':'price'})
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    return df.set_index('code')

def get_lof_spot():
    try:
        df = ak.fund_lof_spot_em()
        df = df[['代码','名称','最新价']].rename(columns={'代码':'code','名称':'name','最新价':'price'})
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        return df.set_index('code')
    except Exception:
        return pd.DataFrame(columns=['name','price'])

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

def get_csindex_values(index_code):
    try:
        val = ak.stock_zh_index_value_csindex(symbol=index_code)
        last = val.tail(1).copy()
        pe = pd.to_numeric(last['市盈率2'], errors='coerce').iloc[0]
        dy = pd.to_numeric(last['股息率2'], errors='coerce').iloc[0]  # %
        return pe, dy/100.0
    except Exception:
        return np.nan, np.nan

def get_cn_yield_10y():
    try:
        ydf = ak.bond_china_yield()
        y = pd.to_numeric(ydf['10年']).dropna().iloc[-1]
        return float(y)
    except Exception:
        try:
            ydf = ak.bond_zh_us_rate(start_date="19901219")
            y = pd.to_numeric(ydf['中国国债收益率10年']).dropna().iloc[-1]
            return float(y)
        except Exception:
            try:
                import tradingeconomics as te
                te.login(os.getenv("TE_API_KEY","guest:guest"))
                cn10y = te.fetchMarkets(symbols="CN10Y", output_type='df')
                return float(cn10y['Close'].iloc[-1])
            except Exception:
                return np.nan

def get_fred_us_ig_1_3y():
    try:
        series = pdr.DataReader('BAMLC1A0C13YEY', 'fred')
        return float(series.dropna().iloc[-1])
    except Exception:
        return np.nan

def get_vix():
    try:
        vix = yf.Ticker("^VIX").history(period="1mo", interval="1d")["Close"].dropna()
        return float(vix.iloc[-1])
    except Exception:
        return np.nan

def get_index_trend(symbol="000300"):
    try:
        idx_hist = ak.index_zh_a_hist(
            symbol=symbol, period="daily",
            start_date="2018-01-01",
            end_date=dt.date.today().isoformat(),
            adjust=""
        )
        idx_hist['收盘'] = pd.to_numeric(idx_hist['收盘'], errors='coerce')
        idx_hist = idx_hist.dropna(subset=['收盘'])
        window = 200 if len(idx_hist) >= 220 else 120
        idx_hist['MA'] = idx_hist['收盘'].rolling(window).mean()
        last_close = float(idx_hist['收盘'].iloc[-1])
        ma = float(idx_hist['MA'].iloc[-1]) if not np.isnan(idx_hist['MA'].iloc[-1]) else np.nan
        above_ma = (last_close > ma) if not np.isnan(ma) else None
        return above_ma, last_close, ma, window
    except Exception:
        return None, np.nan, np.nan, None

def get_ma_via_510300():
    try:
        h = ak.fund_etf_hist_em(
            symbol="510300", period="daily",
            start_date="20180101",
            end_date=dt.date.today().strftime("%Y%m%d"),
            adjust=""
        )
        h['收盘'] = pd.to_numeric(h['收盘'], errors='coerce')
        h = h.dropna(subset=['收盘'])
        window = 200 if len(h) >= 220 else 120
        h['MA'] = h['收盘'].rolling(window).mean()
        last_close = float(h['收盘'].iloc[-1])
        ma = float(h['MA'].iloc[-1]) if not np.isnan(h['MA'].iloc[-1]) else np.nan
        above_ma = (last_close > ma) if not np.isnan(ma) else None
        return above_ma, last_close, ma, window
    except Exception:
        return None, np.nan, np.nan, None

# ====== 抓 RAW 行情 ======
spot = get_etf_spot()
lof_spot = get_lof_spot()
fund_est = get_fund_estimation(list(current_positions.keys()))

# 名称合并：行情里的name优先，其次手工 CODE_NAMES（备用）
NAME_MAP = {}
for df in (spot, lof_spot, fund_est):
    if not df.empty:
        NAME_MAP.update(df['name'].to_dict())
for k, v in CODE_NAMES.items():
    NAME_MAP.setdefault(k, v)

# ====== 价格聚合器（多源，最大化减少 NaN）======
def _normalize_code(x: str) -> str:
    # 兼容 "sz164824"/"sh510300"/"164824" 等，统一返回 6 位数字
    m = re.search(r'(\d{6})', str(x))
    return m.group(1) if m else str(x)

def load_price_book(codes):
    """
    返回 (price_book, name_book)
    优先级：
      1) 东财 ETF 实时: fund_etf_spot_em
      2) 东财 LOF 实时: fund_lof_spot_em
      3) 新浪 ETF/LOF 实时: fund_etf_category_sina(symbol="ETF基金"/"LOF基金")
      4) 基金“估算净值”: fund_em_value_estimation
      5) 开放式基金当日净值(16:00后更新): fund_open_fund_daily_em
    """
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
                price_book.setdefault(c, float(r["price"]))  # 优先级：只在没写过时写入
                if isinstance(r["name"], str) and r["name"]:
                    name_book.setdefault(c, r["name"])

    # 1) ETF 实时（东财）
    try:
        _ingest_df(ak.fund_etf_spot_em(), "代码", "名称", "最新价")
    except Exception:
        pass

    # 2) LOF 实时（东财）
    try:
        _ingest_df(ak.fund_lof_spot_em(), "代码", "名称", "最新价")
    except Exception:
        pass

    # 3) 新浪 ETF/LOF 实时
    try:
        _ingest_df(ak.fund_etf_category_sina(symbol="ETF基金"), "代码", "名称", "最新价")
    except Exception:
        pass
    try:
        _ingest_df(ak.fund_etf_category_sina(symbol="LOF基金"), "代码", "名称", "最新价")
    except Exception:
        pass

    # 4) 基金估算净值（东财）
    try:
        est = ak.fund_em_value_estimation()
        if est is not None and not est.empty:
            est = est[["基金代码", "基金简称", "估算净值"]].rename(
                columns={"基金代码": "代码", "基金简称": "名称", "估算净值": "最新价"}
            )
            _ingest_df(est, "代码", "名称", "最新价")
    except Exception:
        pass

    # 5) 开放式基金-当日净值（东财，交易日16:00-23:00更新）
    try:
        ofd = ak.fund_open_fund_daily_em()
        if ofd is not None and not ofd.empty:
            ofd = ofd[["基金代码", "基金简称", "单位净值"]].rename(
                columns={"基金代码": "代码", "基金简称": "名称", "单位净值": "最新价"}
            )
            _ingest_df(ofd, "代码", "名称", "最新价")
    except Exception:
        pass

    # 兜底名称：把你手工的 CODE_NAMES 回填
    for c, n in CODE_NAMES.items():
        name_book.setdefault(c, n)
    return price_book, name_book

# —— 加载一次价格与名称簿 —— #
PRICE_BOOK, NAME_BOOK = load_price_book(list(set(list(targets.keys()) + list(current_positions.keys()))))

def get_name(code: str) -> str:
    # 优先 PRICE/NAME_BOOK，再用 NAME_MAP，再用 CODE_NAMES
    return NAME_BOOK.get(code) or NAME_MAP.get(code) or CODE_NAMES.get(code, code)

def get_price(code: str) -> float:
    return PRICE_BOOK.get(code, np.nan)

# ====== 指标计算 ======
pe_300, dy_300 = get_csindex_values("000300")
pe_div, dy_div = get_csindex_values("000922")
pe_500, _      = get_csindex_values("000905")
pe_kc50, _     = get_csindex_values("000688")

cn10y = get_cn_yield_10y()
us_ig_1_3y = get_fred_us_ig_1_3y()
vix_now = get_vix()

ep_300 = (1.0/pe_300) if pe_300 and pe_300>0 else np.nan
yield_gap = (ep_300 - cn10y/100.0) if (not np.isnan(ep_300) and not np.isnan(cn10y)) else np.nan
div_gap = (dy_div - cn10y/100.0) if (not np.isnan(dy_div) and not np.isnan(cn10y)) else np.nan

above_ma, _, _, ma_win = get_index_trend("000300")
if above_ma is None:
    above_ma, _, _, ma_win = get_ma_via_510300()

# ====== 估值分位 & 近回撤（温和调节强度）======
def pe_percentile(symbol="000300", lookback_years=10):
    """沪深300 PE 的历史分位（0~1），用中证估值'市盈率2'近 lookback_years 数据"""
    try:
        df = ak.stock_zh_index_value_csindex(symbol=symbol)
        # 注意：akshare返回的数据行数有限，可能不足以计算10年分位
        # 如果数据量不足，使用所有可用数据
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        # 使用所有可用数据，不再按年份过滤
        # cutoff = pd.Timestamp.today() - pd.DateOffset(years=lookback_years)
        # df = df[df['日期'] >= cutoff]
        pe = pd.to_numeric(df['市盈率2'], errors='coerce').dropna()
        if len(pe) < 10:  # 降低最低要求，只要有10个数据点就计算
            return np.nan
        cur = pe.iloc[-1]
        pct = (pe <= cur).mean()  # 分位数
        return float(pct)
    except Exception as e:
        print(f"PE分位计算错误: {e}")  # 添加错误日志
        return np.nan

def drawdown_from_high(series: pd.Series) -> float:
    """给定价格序列的当前回撤（负数），如 -0.12 表示回撤12%"""
    if series is None or len(series)==0:
        return np.nan
    s = pd.Series(series).dropna().astype(float)
    if s.empty: 
        return np.nan
    cummax = s.cummax()
    dd = s / cummax - 1.0
    return float(dd.iloc[-1])

# 近100日回撤（优先用沪深300指数，退而求其次用510300）
try:
    _px = ak.index_zh_a_hist(symbol="000300", period="daily", start_date="2016-01-01",
                             end_date=dt.date.today().isoformat(), adjust="")
    px_ser = pd.to_numeric(_px['收盘'], errors='coerce')
except Exception:
    try:
        _px = ak.fund_etf_hist_em(symbol="510300", period="daily",
                                  start_date="20160101", end_date=dt.date.today().strftime("%Y%m%d"), adjust="")
        px_ser = pd.to_numeric(_px['收盘'], errors='coerce')
    except Exception:
        px_ser = pd.Series(dtype=float)

dd_100 = drawdown_from_high(px_ser.tail(100))
pe_pct = pe_percentile("000300")

def _hist_series_for_code(code, start="2022-01-01"):
    """抓取单只标的近端价格序列（尽量用 ETF/LOF；开放式基金用净值）"""
    try:
        if code in {"510300","510500","588000","513500","513800","518880"}:
            df = ak.fund_etf_hist_em(symbol=code, period="daily",
                                     start_date=start.replace("-",""),
                                     end_date=dt.date.today().strftime("%Y%m%d"),
                                     adjust="")
            s = pd.to_numeric(df["收盘"], errors="coerce")
            return s
        elif code in {"161119","164824","009051","001917","007360","513100"}:
            # 优先 LOF 历史（如有），否则开放式基金净值
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
    """合成组合净值序列：用各标的历史价格做加权（归一化基日=1）"""
    cols = []
    for c, w in weights.items():
        s = _hist_series_for_code(c, start=start)
        if s is None or s.empty or w<=0: 
            continue
        s = s.dropna()
        if s.empty: 
            continue
        s = s / s.iloc[0]  # 归一化
        cols.append(s.to_frame(name=c).pct_change().add(1).cumprod())  # 转净值路径
    if not cols:
        return pd.Series(dtype=float)
    df = pd.concat(cols, axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    # 加权
    w_series = pd.Series(weights)
    w_series = w_series / w_series.sum()
    port = (df * w_series.reindex(df.columns).fillna(0)).sum(axis=1)
    return port

def portfolio_dd_and_vol(weights: dict, start="2022-01-01"):
    """
    返回(月度回撤, 年度回撤, 年化波动)
    月度回撤=最近22个交易日内从局部峰值的跌幅；年度回撤=近252日同理
    年化波动=近60日日收益的 std * sqrt(252)
    """
    nav = portfolio_daily_series(weights, start=start)
    if nav is None or nav.empty:
        return np.nan, np.nan, np.nan

    ret = nav.pct_change().dropna()
    if ret.empty: 
        return np.nan, np.nan, np.nan

    def _dd(x):
        roll_max = x.cummax()
        dd = x/roll_max - 1.0
        return dd.iloc[-1]

    dd_m = _dd(nav.tail(22))   if len(nav)>=22  else np.nan
    dd_y = _dd(nav.tail(252))  if len(nav)>=252 else np.nan
    vol  = ret.tail(60).std() * np.sqrt(252) if len(ret)>=60 else ret.std() * np.sqrt(252)
    return float(dd_m), float(dd_y), float(vol)

def decide_intensity_base():
    """
    基础规则 + 温和因子（估值分位/回撤）：
      - 基础：yield_gap>0.02 且 MA200上方 且 VIX<22 => 1；若 (gap<=0 或 MA200下) & VIX>25 => 0；其余 0.5
      - 估值分位：pe_pct>=0.9 => *0.5; >=0.8 => *0.75; <=0.2 => *1.25; <=0.1 => *1.5
      - 回撤：dd_100<=-0.1 => *1.2; dd_100<=-0.2 => *1.4
      - 最终裁剪到 [0,1]
    """
    if not USE_INTENSITY:
        base = 1.0
    else:
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

    # 估值分位微调
    if not np.isnan(pe_pct):
        if pe_pct >= 0.90: base *= 0.50
        elif pe_pct >= 0.80: base *= 0.75
        elif pe_pct <= 0.10: base *= 1.50
        elif pe_pct <= 0.20: base *= 1.25

    # 回撤微调（逢跌稍加速）
    if not np.isnan(dd_100):
        if dd_100 <= -0.20: base *= 1.40
        elif dd_100 <= -0.10: base *= 1.20

    return float(max(0.0, min(1.0, base)))

def portfolio_drawdown_control(dd_m, dd_y):
    """
    回撤挡板（不清仓，只限高 + 降强度）
    - dd_m < -0.06 → 上限 0.25, 强度*0.3
    - dd_m < -0.05 → 上限 0.35, 强度*0.5
    - 否则         → 上限 0.45, 强度*1.0
    """
    if not np.isnan(dd_m) and dd_m < -0.06:
        return {"equity_ratio_cap": 0.25, "intensity_scale": 0.3}
    elif not np.isnan(dd_m) and dd_m < -0.05:
        return {"equity_ratio_cap": 0.35, "intensity_scale": 0.5}
    else:
        return {"equity_ratio_cap": 0.45, "intensity_scale": 1.0}

def vol_target_adjustment(vol_annual, target_vol=0.10, floor=0.6, cap=1.2):
    """波动率目标：强度按目标波动缩放，并夹在 [0.6,1.2]"""
    if np.isnan(vol_annual) or vol_annual<=0:
        return 1.0
    scale = target_vol / vol_annual
    return float(max(floor, min(cap, scale)))

def decide_intensity_enhanced():
    # 用"当前持仓权重"估算组合路径与风险（也可用目标权重：将 weights 改成 targets）
    weights = {c: current_positions.get(c,0)/max(1.0, sum(current_positions.values())) for c in targets}
    dd_m, dd_y, vol = portfolio_dd_and_vol(weights, start="2022-01-01")

    base = decide_intensity_base()
    dd_ctl = portfolio_drawdown_control(dd_m, dd_y)      # 强度乘数 + 权益上限
    vol_ctl = vol_target_adjustment(vol, target_vol=0.10)

    enhanced = float(max(0.0, min(1.0, base * dd_ctl["intensity_scale"] * vol_ctl)))
    # 将"权益上限"放到全局（供后面'买入分配'阶段使用）
    globals()["_EQUITY_CAP_FROM_RISK"] = dd_ctl["equity_ratio_cap"]
    # 也把这些指标记下来，方便打印/导出
    globals()["_DD_M"], globals()["_DD_Y"], globals()["_VOL_ANNUAL"] = dd_m, dd_y, vol
    return enhanced

# 使用增强后的强度
intensity = decide_intensity_enhanced()

# ====== 动态分批额度（多次加/减仓） ======
invested_now = float(sum(current_positions.values()))
outstanding  = TOTAL_TARGET - invested_now  # >0 需买入；<0 需减持

def auto_adjust_weekly_amount(outstanding, invested_now):
    """
    根据当前持仓和资金状况，智能调整每周交易金额
    策略:
    1. 如果差额小于最小交易金额，不交易
    2. 根据总资产规模动态调整最大单次交易金额
    3. 考虑现金储备需求
    4. 根据偏离程度调整交易力度
    """
    if abs(outstanding) < MIN_TRADE_AMOUNT:
        return 0.0  # 差额太小，不值得交易
    
    # 计算总资产（当前持仓 + 现金）
    total_assets = invested_now + max(0, -outstanding)  # 如果outstanding<0，表示超配，有现金
    
    # 基于总资产的最大单次交易限额
    max_trade_by_pct = total_assets * MAX_TRADE_PCT
    
    # 考虑现金储备需求（仅对买入有影响）
    cash_reserve = total_assets * CASH_RESERVE_PCT
    available_cash = max(0, -outstanding - cash_reserve)  # 可用于买入的现金
    
    # 计算偏离度系数（偏离越大，调整力度越大）
    deviation_pct = abs(outstanding) / TOTAL_TARGET
    deviation_factor = min(1.5, max(0.5, 1.0 + deviation_pct))
    
    if outstanding > 0:  # 需要买入
        # 基础额度：标准每周额度与最大单次交易额的较小值
        base_amount = min(WEEKLY_LIMIT, max_trade_by_pct)
        # 考虑可用现金限制（如果有现金信息）
        if -outstanding > 0:  # 有现金信息
            return min(base_amount * deviation_factor, outstanding, available_cash)
        else:
            return min(base_amount * deviation_factor, outstanding)
    else:  # 需要减持
        # 基础额度：标准每周额度与最大单次交易额的较小值
        base_amount = min(WEEKLY_LIMIT, max_trade_by_pct)
        return min(base_amount * deviation_factor, abs(outstanding))

# 计算本周交易金额
if AUTO_ADJUST:
    gross_weekly = auto_adjust_weekly_amount(outstanding, invested_now)
else:
    gross_weekly = min(WEEKLY_LIMIT, abs(outstanding)/max(STAGING_WEEKS,1))

planned_buy  = gross_weekly * intensity if outstanding>0 else 0.0
planned_sell = gross_weekly * intensity if outstanding<0 else 0.0

# ====== 与目标的差额（用于分配） ======
target_amounts = {c: TOTAL_TARGET*w for c, w in targets.items()}
deficits = {c: max(target_amounts[c] - current_positions.get(c, 0.0), 0.0) for c in targets}  # 需要补多少
excesses = {c: max(current_positions.get(c, 0.0) - target_amounts[c], 0.0) for c in targets}  # 超配多少

# 再平衡带宽：将“低配但未超过带宽”的标的置零（不买）
def apply_rebalance_band(deficits: dict, band_pp_abs=BAND_PP_ABS, band_rel=BAND_REL) -> dict:
    """
    双阈值：绝对带宽(百分点) + 相对带宽(目标的±比例)
    未超过任一阈值 -> 本批不买该标的（降低高位小额补的频率）
    """
    filt = {}
    for c, need_amt in deficits.items():
        cur_amt = current_positions.get(c, 0.0)
        cur_pct = cur_amt / TOTAL_TARGET if TOTAL_TARGET > 0 else 0.0
        tgt_pct = targets[c]
        under_pp = max((tgt_pct - cur_pct) * 100.0, 0.0)  # 只看低配侧
        # 计算相对阈值对应的"百分点"
        rel_pp = tgt_pct * band_rel * 100.0
        thresh = max(band_pp_abs, rel_pp)
        filt[c] = need_amt if under_pp >= thresh else 0.0
    return filt

deficits_active = apply_rebalance_band(deficits, BAND_PP_ABS, BAND_REL)

# ====== 公共函数（触发器 / 整手 / 分配）======
def triggers_for_buy(plan: dict) -> dict:
    plan = plan.copy()
    # 中证500估值高 -> 减半，回流债/红利
    if pe_500 and pe_500 > 30:
        cut = plan.get("510500", 0.0) * 0.5
        plan["510500"] = plan.get("510500", 0.0) - cut
        plan["161119"] = plan.get("161119", 0.0) + cut * 0.6
        plan["009051"] = plan.get("009051", 0.0) + cut * 0.4
    # 黄金当批不追：回流债/红利
    if plan.get("518880", 0.0) > 0:
        extra = plan["518880"]
        plan["518880"] = 0.0
        plan["161119"] = plan.get("161119", 0.0) + extra * 0.5
        plan["009051"] = plan.get("009051", 0.0) + extra * 0.5
    # 纳指限额50%
    plan["513100"] = plan.get("513100", 0.0) * 0.5
    return plan

def apply_board_lot(plan: dict) -> dict:
    """ETF 按100股整手，不足回流债/红利"""
    extra = 0.0
    adj = plan.copy()
    for code in list(BOARD_LOT_CODES):
        amt = adj.get(code, 0.0)
        if amt <= 0: 
            continue
        price = float(spot.loc[code, 'price']) if (code in spot.index) else np.nan
        if np.isnan(price) or price <= 0:
            extra += amt; adj[code] = 0.0; continue
        lot_cost = 100 * price
        if amt < lot_cost:
            extra += amt; adj[code] = 0.0
        else:
            lots = int(amt // lot_cost)
            adj[code] = lots * lot_cost
            extra += (amt - adj[code])
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
        # 仅在 SELLABLE_CODES 间按“超配占比”分配，不超过各自超配和现有持仓
        base_set = [c for c in targets if c in SELLABLE_CODES]
    else:
        # 纯基金模式：所有标的都可以卖
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
        if amt <= 0: 
            continue
        rows.append([code, get_name(code), float(round(amt)), get_price(code)])
        s += float(round(amt))
    df = pd.DataFrame(rows, columns=["代码","名称", col, "参考价格"]).sort_values(col, ascending=False)
    # 确保金额列格式统一
    if not df.empty:
        df[col] = pd.to_numeric(df[col], errors='coerce').round(0)
        if "参考价格" in df.columns:
            df["参考价格"] = pd.to_numeric(df["参考价格"], errors='coerce').round(3)
    return df, s

# ====== 持仓表生成：更新前 vs 更新后 ======
def compute_after_positions(curr_pos: dict, buy_plan: dict, sell_plan: dict) -> dict:
    """根据本周买/卖计划，计算更新后的持仓金额（不允许负数）"""
    all_codes = set(list(targets.keys()) + list(curr_pos.keys()) + list(buy_plan.keys()) + list(sell_plan.keys()))
    new_pos = {}
    for c in all_codes:
        base = float(curr_pos.get(c, 0.0))
        inc  = float(buy_plan.get(c, 0.0))
        dec  = float(sell_plan.get(c, 0.0))
        new_pos[c] = max(base + inc - dec, 0.0)
    return new_pos

def make_holdings_table(positions: dict) -> pd.DataFrame:
    """生成含：代码 / 名称 / 目标权重% / 持仓(¥) / 占当前% / 目标金额(¥) 的表"""
    rows = []
    total_current = sum(positions.values())
    for c, w in targets.items():
        amt = float(positions.get(c, 0.0))
        target_amt = TOTAL_TARGET * w
        pct_of_current = 100.0 * amt / total_current if total_current > 0 else 0.0
        rows.append([
            c, get_name(c), round(100.0*w, 2), round(amt, 0), 
            round(pct_of_current, 2), round(target_amt, 0)
        ])
    df = pd.DataFrame(rows, columns=["代码","名称","目标权重%","持仓(¥)","当前配比%","目标金额(¥)"])
    return df.sort_values("目标权重%", ascending=False)

def make_targets_table() -> pd.DataFrame:
    """导出目标权重表（含中文名称与目标金额）"""
    rows = []
    for c, w in targets.items():
        rows.append([c, get_name(c), round(w*100,2), round(TOTAL_TARGET*w,0)])
    return pd.DataFrame(rows, columns=["代码","名称","目标权重%","目标金额(¥)"]).sort_values("目标权重%", ascending=False)

# ====== 命令行参数 ======
parser = argparse.ArgumentParser(description="Rebalance helper")
parser.add_argument("--plan", choices=["A", "B", "both"], default="both",
                    help="A=交易约束（ETF整手、卖只动基金/LOF）；B=纯基金；both=两套都导出（默认）")
parser.add_argument("--no-export", action="store_true",
                    help="不导出 CSV（默认会导出）")
parser.add_argument("--outdir", default="exports",
                    help="CSV 导出目录，默认 ./exports")
parser.add_argument("--no-auto", action="store_true",
                    help="禁用自动调整交易金额功能")
parser.add_argument("--min-trade", type=float, default=MIN_TRADE_AMOUNT,
                    help=f"最小交易金额，默认 {MIN_TRADE_AMOUNT}元")
parser.add_argument("--max-trade-pct", type=float, default=MAX_TRADE_PCT,
                    help=f"最大单次交易占总资产比例，默认 {MAX_TRADE_PCT*100}%")
parser.add_argument("--cash-reserve-pct", type=float, default=CASH_RESERVE_PCT,
                    help=f"现金储备比例，默认 {CASH_RESERVE_PCT*100}%")
args = parser.parse_args()
EXPORT = (not args.no_export)
AUTO_ADJUST = (not args.no_auto)
MIN_TRADE_AMOUNT = args.min_trade
MAX_TRADE_PCT = args.max_trade_pct
CASH_RESERVE_PCT = args.cash_reserve_pct

EQUITY_CODES = {"510300","510500","588000","001917","513500","513100","513800","009051"}  # 视为权益类
# （黄金/债不算权益；你也可以把 009051(红利)算成"半权益半防守"，这里简单视为权益）

def enforce_equity_cap(buy_plan: dict, sell_plan: dict, cap: float) -> dict:
    if not cap or cap >= 0.99:
        return buy_plan
    # 计算执行后权益占比
    total_now = sum(current_positions.values())
    eq_now = sum(current_positions.get(c,0) for c in EQUITY_CODES)
    buy_eq = sum(buy_plan.get(c,0) for c in EQUITY_CODES)
    sell_eq = sum(sell_plan.get(c,0) for c in EQUITY_CODES)

    total_after = total_now + sum(buy_plan.values()) - sum(sell_plan.values())
    eq_after    = eq_now + buy_eq - sell_eq
    eq_ratio_after = eq_after / max(1.0, total_after)

    if eq_ratio_after <= cap:
        return buy_plan

    # 需要把权益买单缩放到刚好不超 cap
    target_eq = cap * total_after
    need_cut  = max(eq_after - target_eq, 0.0)  # 要少买的权益金额
    if need_cut <= 0 or buy_eq <= 0:
        return buy_plan

    scale = max(0.0, min(1.0, (buy_eq - need_cut) / buy_eq))
    adj = buy_plan.copy()
    for c in EQUITY_CODES:
        if c in adj:
            adj[c] = adj[c] * scale
    return adj

# ====== 生成两套方案（买/卖计划）======
# 先用带宽过滤后的 deficits_active
buy_constrained  = alloc_buy(planned_buy,  deficits_active, constrained=True)
sell_constrained = alloc_sell(planned_sell, excesses,      constrained=True)
# 执行权益上限约束
buy_constrained = enforce_equity_cap(buy_constrained, sell_constrained, globals().get("_EQUITY_CAP_FROM_RISK", 0.45))

buy_fundonly  = alloc_buy(planned_buy,  deficits_active, constrained=False)
sell_fundonly = alloc_sell(planned_sell, excesses,       constrained=False)
# 执行权益上限约束
buy_fundonly = enforce_equity_cap(buy_fundonly, sell_fundonly, globals().get("_EQUITY_CAP_FROM_RISK", 0.45))

# ====== 控制台简要打印（以防你想快速看一下） ======
def _fmt(v): 
    return "NaN" if v is None or (isinstance(v,float) and np.isnan(v)) else v

print("\n=== 关键指标（简要） ===")
print(f"CSI300 PE/E-P: {_fmt(round(pe_300,2))} / {_fmt(round(ep_300,4))}；中证红利DY: {_fmt(round(dy_div*100,2))}%")
print(f"CN10Y/美IG1-3Y: {_fmt(round(cn10y,2))}% / {_fmt(round(us_ig_1_3y,2))}%；收益率缺口: {_fmt(round(yield_gap*100,2))}pct")
print(f"红利-国债缺口: {_fmt(round(div_gap*100,2))}pct；MA{ma_win}上方: {above_ma}；VIX: {_fmt(round(vix_now,2))}")
print(f"PE分位(10y): {_fmt(round(pe_pct*100,1)) if not np.isnan(pe_pct) else 'NaN'}%；近100日回撤: {_fmt(round(dd_100*100,1)) if not np.isnan(dd_100) else 'NaN'}%")

# 新增：组合风险指标
dd_m, dd_y, vol = globals().get("_DD_M", np.nan), globals().get("_DD_Y", np.nan), globals().get("_VOL_ANNUAL", np.nan)
eq_cap = globals().get("_EQUITY_CAP_FROM_RISK", 0.45)
print(f"组合月度回撤: {_fmt(round(dd_m*100,1)) if not np.isnan(dd_m) else 'NaN'}%；年化波动: {_fmt(round(vol*100,1)) if not np.isnan(vol) else 'NaN'}%")
print(f"权益上限: {_fmt(round(eq_cap*100,1))}%")

print("\n=== 资金进度 & 交易参数 ===")
print(f"当前已投入: ¥{invested_now:.0f} | 目标: ¥{TOTAL_TARGET:.0f} | 与目标差额: ¥{outstanding:.0f}")

# 显示交易金额计算方式
if AUTO_ADJUST:
    total_assets = invested_now + max(0, -outstanding)
    max_trade_by_pct = total_assets * MAX_TRADE_PCT
    deviation_pct = abs(outstanding) / TOTAL_TARGET
    deviation_factor = min(1.5, max(0.5, 1.0 + deviation_pct))
    
    print(f"自动交易模式：最小交易额 ¥{MIN_TRADE_AMOUNT:.0f} | 最大单次比例 {MAX_TRADE_PCT*100:.1f}% | 现金储备 {CASH_RESERVE_PCT*100:.1f}%")
    print(f"总资产: ¥{total_assets:.0f} | 偏离度: {deviation_pct*100:.1f}% | 调整系数: {deviation_factor:.2f}")
    print(f"基础周额度: ¥{min(WEEKLY_LIMIT, max_trade_by_pct):.0f} | 强度(综合): {intensity:.2f}")
else:
    print(f"固定分批模式：分摊周数 {STAGING_WEEKS} | 每周上限: ¥{WEEKLY_LIMIT:.0f} | 强度(综合): {intensity:.2f}")

print(f"→ 本周计划买入: ¥{planned_buy:.0f} | 本周计划减持: ¥{planned_sell:.0f}")

# ====== 导出函数 ======
def export_plan(label: str, buy_plan: dict, sell_plan: dict, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    today = dt.date.today().strftime("%Y%m%d")

    # 订单清单
    df_buy, sum_buy = make_df(buy_plan,  buy=True)
    df_sell, sum_sell = make_df(sell_plan, buy=False)

    if not df_buy.empty:
        df_buy.to_csv(os.path.join(outdir, f"buy_orders_{label}_{today}.csv"), index=False, encoding="utf-8-sig")
    else:
        # 也生成空文件，便于流水管理
        pd.DataFrame(columns=["代码","名称","建议买入金额(¥)","参考价格"]).to_csv(
            os.path.join(outdir, f"buy_orders_{label}_{today}.csv"), index=False, encoding="utf-8-sig"
        )
    if not df_sell.empty:
        df_sell.to_csv(os.path.join(outdir, f"sell_orders_{label}_{today}.csv"), index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["代码","名称","建议减持金额(¥)","参考价格"]).to_csv(
            os.path.join(outdir, f"sell_orders_{label}_{today}.csv"), index=False, encoding="utf-8-sig"
        )

    # 持仓前后 & 变化摘要
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
    before_df.to_csv(os.path.join(outdir, f"holdings_before_{label}_{today}.csv"), index=False, encoding="utf-8-sig")
    after_df.to_csv( os.path.join(outdir, f"holdings_after_{label}_{today}.csv"),  index=False, encoding="utf-8-sig")
    diff_df[keep_cols].to_csv(os.path.join(outdir, f"holdings_diff_{label}_{today}.csv"), index=False, encoding="utf-8-sig")

    # 目标权重表（带中文名）
    tgt_df = make_targets_table()
    tgt_df.to_csv(os.path.join(outdir, f"targets_{label}_{today}.csv"), index=False, encoding="utf-8-sig")

    # 控制台简报
    print(f"\n[{label}] 导出完成：")
    print(f"  buy_orders_{label}_{today}.csv（合计买入 ¥{sum_buy:.0f}）")
    print(f"  sell_orders_{label}_{today}.csv（合计减持 ¥{sum_sell:.0f}）")
    print(f"  holdings_before_{label}_{today}.csv / holdings_after_{label}_{today}.csv / holdings_diff_{label}_{today}.csv")
    print(f"  targets_{label}_{today}.csv（含中文名称与目标金额）")

# ====== 执行导出 ======
if EXPORT:
    if args.plan in ["A", "both"]:
        export_plan("A", buy_constrained, sell_constrained, args.outdir)
    if args.plan in ["B", "both"]:
        export_plan("B", buy_fundonly,  sell_fundonly,  args.outdir)
else:
    # 若禁用导出，仅做建议打印（优化终端对齐问题）
    if args.plan in ["A", "both"]:
        df_b1, sum_b1 = make_df(buy_constrained,  buy=True)
        df_s1, sum_s1 = make_df(sell_constrained, buy=False)
        
        # 生成持仓前后表格（用于展示）
        before_df_A = make_holdings_table(current_positions)
        after_positions_A = compute_after_positions(current_positions, buy_constrained, sell_constrained)
        after_df_A = make_holdings_table(after_positions_A)
        
        # 创建一个简化的差异表，只包含所需列
        simplified_df = before_df_A.merge(
            after_df_A[["代码", "当前配比%"]], 
            on="代码", 
            suffixes=("", "_更新后")
        )
        simplified_df.rename(columns={"当前配比%": "当前配比%", "当前配比%_更新后": "更新后配比%"}, inplace=True)
        
        print("\n=== 方案A：买入清单（摘要） ===")
        if not df_b1.empty:
            # 使用指定的列宽以改善对齐
            print(df_b1.head(10).to_string(index=False))
        else:
            print("（无）")
        print(f"合计买入: ¥{sum_b1:.0f}")
        
        print("\n=== 方案A：减持清单（摘要） ===")
        if not df_s1.empty:
            print(df_s1.head(10).to_string(index=False))
        else:
            print("（无）")
        print(f"合计减持: ¥{sum_s1:.0f}")
        
        print("\n=== 方案A：持仓配比表 ===")
        display_cols = ["代码", "名称", "目标权重%", "持仓(¥)", "当前配比%", "更新后配比%"]
        print(simplified_df[display_cols].to_string(index=False))
        
    if args.plan in ["B", "both"]:
        df_b2, sum_b2 = make_df(buy_fundonly,  buy=True)
        df_s2, sum_s2 = make_df(sell_fundonly, buy=False)
        
        # 生成持仓前后表格（用于展示）
        before_df_B = make_holdings_table(current_positions)
        after_positions_B = compute_after_positions(current_positions, buy_fundonly, sell_fundonly)
        after_df_B = make_holdings_table(after_positions_B)
        
        # 创建一个简化的差异表，只包含所需列
        simplified_df = before_df_B.merge(
            after_df_B[["代码", "当前配比%"]], 
            on="代码", 
            suffixes=("", "_更新后")
        )
        simplified_df.rename(columns={"当前配比%": "当前配比%", "当前配比%_更新后": "更新后配比%"}, inplace=True)
        
        print("\n=== 方案B：买入清单（摘要） ===")
        if not df_b2.empty:
            print(df_b2.head(10).to_string(index=False))
        else:
            print("（无）")
        print(f"合计买入: ¥{sum_b2:.0f}")
        
        print("\n=== 方案B：减持清单（摘要） ===")
        if not df_s2.empty:
            print(df_s2.head(10).to_string(index=False))
        else:
            print("（无）")
        print(f"合计减持: ¥{sum_s2:.0f}")
        
        print("\n=== 方案B：持仓配比表 ===")
        display_cols = ["代码", "名称", "目标权重%", "持仓(¥)", "当前配比%", "更新后配比%"]
        print(simplified_df[display_cols].to_string(index=False))

print("\n提示：开放式/LOF 使用的是“估算净值”，与收盘净值存在偏差；下单以当时成交价/估值为准。导出的 CSV 建议在表格软件里查看与执行。")
