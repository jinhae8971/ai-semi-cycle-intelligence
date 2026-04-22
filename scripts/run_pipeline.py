"""AI/Semiconductor Cycle Intelligence — Serverless Pipeline.

Runs on GitHub Actions, fetches free APIs (yfinance + FRED),
computes ASCS (AI Semiconductor Cycle Score), writes JSON for Pages,
sends Telegram daily report.

Architecture mirrors crypto-cycle-intelligence: Git-as-DB, GitHub Pages,
Actions cron. No DB, no servers, no local install.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import requests
import numpy as np
import pandas as pd
import yfinance as yf

# Silence yfinance's noisy info logs
import logging
logging.getLogger("yfinance").setLevel(logging.WARNING)


# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

LATEST_FILE = DATA_DIR / "latest.json"
HISTORY_FILE = DATA_DIR / "history.json"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(exist_ok=True)

MAX_HISTORY_DAYS = 730


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# ══════════════════════════════════════════════════════════════
# Universe — companies & indices we track
# ══════════════════════════════════════════════════════════════

# Bellwether AI/semi stocks (must-have for cycle reading)
SEMI_LEADERS = {
    "NVDA":  "Nvidia",
    "AVGO":  "Broadcom",
    "TSM":   "TSMC ADR",
    "AMD":   "AMD",
    "MU":    "Micron",
    "ASML":  "ASML",
    "AMAT":  "Applied Materials",
    "KLAC":  "KLA",
    "LRCX":  "Lam Research",
    "INTC":  "Intel",
}

# Hyperscalers (capex cycle drivers)
HYPERSCALERS = {
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "META": "Meta",
    "ORCL": "Oracle",
}

# ETFs / Indices (sector benchmarks)
INDICES = {
    "^SOX":  "PHLX Semiconductor (SOX)",
    "SOXX":  "iShares Semi ETF (SOXX)",
    "SMH":   "VanEck Semi ETF (SMH)",
    "^GSPC": "S&P 500",
    "^NDX":  "Nasdaq 100",
    "^VIX":  "VIX",
    "DX-Y.NYB": "DXY (USD Index)",
}

ALL_TICKERS = list(SEMI_LEADERS) + list(HYPERSCALERS) + list(INDICES)


# ══════════════════════════════════════════════════════════════
# Robust HTTP & yfinance wrappers
# ══════════════════════════════════════════════════════════════

def robust_get(url: str, params: dict | None = None, retries: int = 3) -> dict | list | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20,
                             headers={"User-Agent": "asci-serverless/1.0"})
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", "30"))
                time.sleep(wait)
                continue
            if 500 <= r.status_code < 600:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    return None


def silent_yf_call(fn, *args, **kwargs):
    """Call a yfinance function while silencing its stdout/stderr noise."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            return fn(*args, **kwargs)
    except Exception as e:
        print(f"  [yf err] {fn.__name__ if hasattr(fn, '__name__') else 'call'}: {str(e)[:120]}")
        return None


# ══════════════════════════════════════════════════════════════
# Data Fetchers
# ══════════════════════════════════════════════════════════════

def fetch_price_history(symbol: str, period: str = "5y") -> pd.DataFrame | None:
    """Daily OHLC + volume."""
    df = silent_yf_call(yf.Ticker(symbol).history, period=period, auto_adjust=True)
    if df is None or len(df) < 50:
        return None
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def fetch_ticker_info(symbol: str) -> dict:
    """Get fundamental ratios (P/E, P/S, etc) — best-effort."""
    try:
        info = silent_yf_call(lambda: yf.Ticker(symbol).info)
        if not info:
            return {}
        # Pick out the fields we care about
        return {
            "trailingPE":      info.get("trailingPE"),
            "forwardPE":       info.get("forwardPE"),
            "priceToSales":    info.get("priceToSalesTrailing12Months"),
            "priceToBook":     info.get("priceToBook"),
            "marketCap":       info.get("marketCap"),
            "enterpriseValue": info.get("enterpriseValue"),
            "evToRevenue":     info.get("enterpriseToRevenue"),
            "evToEbitda":      info.get("enterpriseToEbitda"),
            "profitMargins":   info.get("profitMargins"),
            "operatingMargins": info.get("operatingMargins"),
            "revenueGrowth":   info.get("revenueGrowth"),
            "earningsGrowth":  info.get("earningsGrowth"),
            "freeCashflow":    info.get("freeCashflow"),
            "operatingCashflow": info.get("operatingCashflow"),
            "totalCash":       info.get("totalCash"),
            "totalDebt":       info.get("totalDebt"),
            "shortRatio":      info.get("shortRatio"),
            "shortPercentOfFloat": info.get("shortPercentOfFloat"),
            "beta":            info.get("beta"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "fiftyDayAverage": info.get("fiftyDayAverage"),
            "twoHundredDayAverage": info.get("twoHundredDayAverage"),
            "averageVolume":   info.get("averageVolume"),
            "currentPrice":    info.get("currentPrice") or info.get("regularMarketPrice"),
            "currency":        info.get("currency"),
        }
    except Exception as e:
        print(f"  [info err] {symbol}: {str(e)[:120]}")
        return {}


def fetch_capex(symbol: str) -> dict:
    """Get capex from cashflow statement (most recent annual + YoY)."""
    try:
        ticker = yf.Ticker(symbol)
        cf = silent_yf_call(lambda: ticker.cashflow)
        if cf is None or cf.empty:
            return {}
        # Capex is usually negative (outflow) — try common names
        for key in ["Capital Expenditure", "Capital Expenditures", "CapitalExpenditure"]:
            if key in cf.index:
                row = cf.loc[key].dropna()
                if len(row) >= 1:
                    latest = abs(float(row.iloc[0]))
                    result = {"capex_latest": latest}
                    if len(row) >= 2:
                        prev = abs(float(row.iloc[1]))
                        if prev > 0:
                            result["capex_yoy_pct"] = (latest / prev - 1) * 100
                    return result
        return {}
    except Exception as e:
        print(f"  [capex err] {symbol}: {str(e)[:80]}")
        return {}


def fetch_fred_series(series_id: str, api_key: str) -> dict | None:
    """Fetch the latest value of a FRED economic series."""
    if not api_key:
        return None
    data = robust_get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id, "api_key": api_key, "file_type": "json",
            "sort_order": "desc", "limit": 13,  # last ~year of monthly data
        },
    )
    if not data or "observations" not in data:
        return None
    obs = data["observations"]
    valid = [(o["date"], float(o["value"])) for o in obs if o["value"] not in (".", "")]
    if not valid:
        return None
    latest_date, latest_val = valid[0]
    result = {"series_id": series_id, "latest": latest_val, "latest_date": latest_date}
    if len(valid) >= 2:
        # YoY change if monthly series
        if len(valid) >= 12:
            year_ago = valid[11][1]
            if year_ago != 0:
                result["yoy_pct"] = (latest_val / year_ago - 1) * 100
        result["mom_pct"] = (latest_val / valid[1][1] - 1) * 100 if valid[1][1] else None
    return result


# ══════════════════════════════════════════════════════════════
# Technical Indicators
# ══════════════════════════════════════════════════════════════

def compute_technicals(df: pd.DataFrame) -> dict:
    """Compute SMA, RSI, distance from 52w high/low, etc."""
    if df is None or len(df) < 50:
        return {}
    close = df["Close"]
    result = {"current": float(close.iloc[-1])}

    if len(close) >= 2:
        result["chg_1d_pct"] = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)

    if len(close) >= 5:
        result["chg_1w_pct"] = float((close.iloc[-1] / close.iloc[-5] - 1) * 100)

    if len(close) >= 21:
        result["chg_1m_pct"] = float((close.iloc[-1] / close.iloc[-21] - 1) * 100)

    if len(close) >= 200:
        sma200 = float(close.rolling(200).mean().iloc[-1])
        result["sma_200"] = sma200
        result["dist_from_sma200_pct"] = float((close.iloc[-1] / sma200 - 1) * 100)

    if len(close) >= 50:
        sma50 = float(close.rolling(50).mean().iloc[-1])
        result["sma_50"] = sma50
        result["dist_from_sma50_pct"] = float((close.iloc[-1] / sma50 - 1) * 100)

    # 52-week position (0 = at low, 100 = at high)
    if len(close) >= 252:
        last_year = close.tail(252)
        wk52_high = float(last_year.max())
        wk52_low  = float(last_year.min())
        if wk52_high > wk52_low:
            result["wk52_position_pct"] = float((close.iloc[-1] - wk52_low) / (wk52_high - wk52_low) * 100)
            result["wk52_high"] = wk52_high
            result["wk52_low"] = wk52_low

    # Daily RSI (Wilder's smoothing)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    daily_rsi = 100 - (100 / (1 + rs))
    if not np.isnan(daily_rsi.iloc[-1]):
        result["daily_rsi"] = float(daily_rsi.iloc[-1])

    # Weekly RSI
    weekly = close.resample("W").last()
    if len(weekly) >= 14:
        dw = weekly.diff()
        gw = dw.where(dw > 0, 0.0)
        lw = -dw.where(dw < 0, 0.0)
        agw = gw.ewm(alpha=1/14, adjust=False).mean()
        alw = lw.ewm(alpha=1/14, adjust=False).mean()
        rsw = agw / alw.replace(0, np.nan)
        wrsi = 100 - (100 / (1 + rsw))
        if not np.isnan(wrsi.iloc[-1]):
            result["weekly_rsi"] = float(wrsi.iloc[-1])

    # Realized volatility (21-day annualized)
    if len(close) >= 22:
        returns = close.pct_change().dropna().tail(21)
        result["realized_vol_21d"] = float(returns.std() * np.sqrt(252) * 100)

    return result


def compute_pe_percentile(symbol: str, current_pe: float | None) -> float | None:
    """Estimate where current P/E sits in 5-year history.

    yfinance doesn't give historical P/E, so we approximate using:
    historical price / trailing EPS (approximate via current EPS).
    """
    if current_pe is None or current_pe <= 0:
        return None
    try:
        df = silent_yf_call(yf.Ticker(symbol).history, period="5y", auto_adjust=True)
        if df is None or len(df) < 252:
            return None
        # Use current_pe to back out implied EPS, then build PE series
        current_price = float(df["Close"].iloc[-1])
        implied_eps = current_price / current_pe
        if implied_eps <= 0:
            return None
        pe_series = df["Close"] / implied_eps
        # Where does today rank within last 5y?
        rank = (pe_series < current_pe).sum() / len(pe_series) * 100
        return float(rank)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════
# ASCS Scoring Engine — 6 dimensions
# ══════════════════════════════════════════════════════════════

@dataclass
class ScoringRule:
    metric: str
    dimension: str
    bottom: float    # value at which score = 0 (capitulation)
    top: float       # value at which score = 100 (euphoria)
    weight: float = 1.0

    def score(self, value: float | None) -> float | None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if self.top == self.bottom:
            return 50.0
        pct = (value - self.bottom) / (self.top - self.bottom)
        return max(0.0, min(100.0, pct * 100))


DIMENSION_WEIGHTS = {
    "valuation":   0.25,
    "earnings":    0.20,
    "capital":     0.15,
    "sentiment":   0.15,
    "macro":       0.10,
    "technical":   0.15,
}

RULES: list[ScoringRule] = [
    # ── Valuation (25%) — high P/E = late cycle ──
    ScoringRule("nvda_forward_pe",      "valuation", 25, 70, 1.5),
    ScoringRule("nvda_price_to_sales",  "valuation", 8,  35, 1.0),
    ScoringRule("sox_avg_pe_pctl",      "valuation", 20, 90, 1.5),  # 5-yr percentile
    ScoringRule("top10_mcap_to_gdp_pct","valuation", 5, 18, 1.0),

    # ── Earnings momentum (20%) — overheated growth = late cycle ──
    ScoringRule("nvda_revenue_growth_pct", "earnings", -10, 100, 1.5),
    ScoringRule("semi_avg_revenue_growth_pct", "earnings", -15, 60, 1.0),
    ScoringRule("hyperscaler_capex_yoy_pct",   "earnings", -5, 50, 1.0),

    # ── Capital cycle (15%) — heavy capex = top, light = bottom ──
    ScoringRule("memory_capex_yoy_pct",   "capital", -30, 50, 1.0),
    ScoringRule("semi_capex_to_revenue_pct", "capital", 8, 30, 1.0),

    # ── Sentiment (15%) — low VIX + extended = euphoria ──
    ScoringRule("vix",                  "sentiment", 35, 11, 1.5),  # inverted: low VIX = top
    ScoringRule("sox_above_sma200_pct", "sentiment", -20, 35, 1.0),
    ScoringRule("nvda_short_pct",       "sentiment", 5, 0.5, 1.0),  # low short = top

    # ── Macro (10%) — easy money + low yields = supportive ──
    ScoringRule("us_10y_yield",     "macro", 5.5, 2.5, 1.0),  # inverted
    ScoringRule("dxy",              "macro", 110, 95, 0.8),    # inverted (strong $ = bad)
    ScoringRule("ism_pmi",          "macro", 42, 58, 1.0),
    ScoringRule("m2_yoy_pct",       "macro", -3, 8, 0.8),

    # ── Technical (15%) — momentum exhaustion ──
    ScoringRule("sox_weekly_rsi",   "technical", 30, 80, 1.5),
    ScoringRule("sox_daily_rsi",    "technical", 30, 75, 1.0),
    ScoringRule("nvda_wk52_pos_pct","technical", 10, 95, 1.5),
    ScoringRule("sox_dist_sma200_pct","technical", -25, 35, 1.0),
]

PHASE_RANGES = [
    (0, 20, "Capitulation",   "🧊", "#0a4d68"),
    (20, 40, "Recovery",      "🌱", "#3a8891"),
    (40, 60, "Expansion",     "📈", "#f6c945"),
    (60, 80, "Late Bull",     "🔥", "#e76f51"),
    (80, 101, "Euphoria",     "🚨", "#c1121f"),
]


def phase_for(score: float | None) -> tuple[str, str, str]:
    if score is None or np.isnan(score):
        return "Unknown", "❓", "#6b7280"
    for lo, hi, name, emoji, color in PHASE_RANGES:
        if lo <= score < hi:
            return name, emoji, color
    return "Euphoria", "🚨", "#c1121f"


def compute_ascs(raw: dict[str, float]) -> dict:
    by_dim: dict[str, list[tuple[float, float]]] = {d: [] for d in DIMENSION_WEIGHTS}
    indicators = []
    for rule in RULES:
        val = raw.get(rule.metric)
        s = rule.score(val)
        indicators.append({
            "metric": rule.metric, "raw": val, "score": s,
            "dimension": rule.dimension, "bottom": rule.bottom, "top": rule.top,
        })
        if s is not None:
            by_dim[rule.dimension].append((s, rule.weight))

    dim_scores: dict[str, float | None] = {}
    for dim, items in by_dim.items():
        if items:
            tw = sum(w for _, w in items)
            dim_scores[dim] = sum(s * w for s, w in items) / tw
        else:
            dim_scores[dim] = None

    present = {d: s for d, s in dim_scores.items() if s is not None}
    if not present:
        return {"composite": None, "phase": "Unknown", "emoji": "❓", "color": "#6b7280",
                "dimensions": dim_scores, "indicators": indicators}

    total_w = sum(DIMENSION_WEIGHTS[d] for d in present)
    ascs = sum(s * DIMENSION_WEIGHTS[d] for d, s in present.items()) / total_w
    name, emoji, color = phase_for(ascs)
    return {
        "composite": round(ascs, 2),
        "phase": name, "emoji": emoji, "color": color,
        "dimensions": {d: round(v, 2) if v is not None else None for d, v in dim_scores.items()},
        "indicators": indicators,
        "weights_used": {d: round(DIMENSION_WEIGHTS[d]/total_w, 3) for d in present},
    }


# ══════════════════════════════════════════════════════════════
# JSON Persistence
# ══════════════════════════════════════════════════════════════

def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "snapshots" in data:
            return data["snapshots"]
        return data if isinstance(data, list) else []
    except Exception:
        return []


def append_history(snapshot: dict) -> list[dict]:
    history = load_history()
    snapshot_date = snapshot["ts"][:10]
    history = [h for h in history if h.get("ts", "")[:10] != snapshot_date]
    history.append(snapshot)
    history.sort(key=lambda h: h.get("ts", ""))
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_HISTORY_DAYS)
    history = [
        h for h in history
        if datetime.fromisoformat(h["ts"].replace("Z", "+00:00")) >= cutoff
    ]
    return history


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False),
                    encoding="utf-8")
    print(f"  [saved] {path} ({path.stat().st_size:,} bytes)")


# ══════════════════════════════════════════════════════════════
# Telegram
# ══════════════════════════════════════════════════════════════

def send_telegram(message: str, token: str, chat_id: str) -> bool:
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": chat_id, "text": message,
            "parse_mode": "Markdown", "disable_web_page_preview": True,
        }, timeout=20)
        r.raise_for_status()
        print(f"  [ok] Telegram sent ({len(message)} chars)")
        return True
    except Exception as e:
        print(f"  [warn] Markdown failed: {e}, trying plain")
        try:
            r = requests.post(url, json={"chat_id": chat_id, "text": message[:4000]},
                              timeout=20)
            r.raise_for_status()
            return True
        except Exception as e2:
            print(f"  [err] Plain fallback: {e2}")
            return False


def format_telegram_report(ascs: dict, leaders: dict, sox_tech: dict,
                            indices: dict, raw: dict, pages_url: str = "") -> str:
    now_kst = datetime.now(timezone.utc).astimezone(
        timezone(timedelta(hours=9))
    ).strftime("%Y-%m-%d %H:%M KST")

    score = ascs.get("composite")
    phase = ascs.get("phase", "Unknown")
    emoji = ascs.get("emoji", "❓")
    score_str = f"*{score:.0f}* / 100" if score is not None else "N/A"

    lines = [
        "🤖 *AI Semiconductor Cycle Report*",
        f"_{now_kst}_",
        "",
    ]

    # Headline
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"{emoji} *{phase}*")
    lines.append(f"ASCS: {score_str}")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("")

    # Sector benchmarks
    sox = indices.get("^SOX", {})
    nvda = leaders.get("NVDA", {})
    vix = indices.get("^VIX", {})
    if sox or nvda:
        lines.append("*🎯 Sector Pulse*")
        if sox.get("current"):
            chg = sox.get("chg_1d_pct", 0)
            arrow = "🟢" if chg >= 0 else "🔴"
            lines.append(f"`SOX     ` {sox['current']:>9,.0f}  {arrow} {chg:+.2f}%")
        if nvda.get("current"):
            chg = nvda.get("chg_1d_pct", 0)
            arrow = "🟢" if chg >= 0 else "🔴"
            lines.append(f"`NVDA    ` ${nvda['current']:>8,.2f}  {arrow} {chg:+.2f}%")
        if vix.get("current"):
            lines.append(f"`VIX     ` {vix['current']:>9,.2f}")
        lines.append("")

    # Dimensions
    lines.append("*📊 6-Dimension Breakdown*")
    for dim in ["valuation", "earnings", "capital", "sentiment", "macro", "technical"]:
        v = ascs["dimensions"].get(dim)
        if v is None:
            lines.append(f"▫️ `{dim:<10}` — no data")
        else:
            bar_len = int(v / 10)
            bar = "█" * bar_len + "░" * (10 - bar_len)
            lines.append(f"▪️ `{dim:<10}` {bar} {v:.0f}")
    lines.append("")

    # Top 5 most extreme indicators
    scored = [i for i in ascs["indicators"] if i["score"] is not None]
    scored.sort(key=lambda i: abs(i["score"] - 50), reverse=True)
    if scored:
        lines.append("*🎯 Top 5 Extreme Indicators*")
        for i in scored[:5]:
            tag = "🔴" if i["score"] > 70 else ("🟢" if i["score"] < 30 else "🟡")
            raw_v = i["raw"]
            raw_str = f"{raw_v:.2f}" if abs(raw_v) < 1000 else f"{raw_v:,.0f}"
            metric = i["metric"].replace("_pct", "%").replace("_", " ")[:18]
            lines.append(f"{tag} `{metric:<18}` {raw_str:>9}  (→ {i['score']:.0f})")
        lines.append("")

    # Top movers in semis (1-day)
    movers = [(sym, d.get("chg_1d_pct", 0))
              for sym, d in leaders.items()
              if d.get("chg_1d_pct") is not None]
    if movers:
        movers.sort(key=lambda x: x[1], reverse=True)
        lines.append("*📈 Today's Movers*")
        for sym, chg in movers[:3]:
            lines.append(f"🟢 {sym:<6}  {chg:+.2f}%")
        for sym, chg in movers[-3:][::-1]:
            if chg < 0:
                lines.append(f"🔴 {sym:<6}  {chg:+.2f}%")
        lines.append("")

    # Alerts
    alerts = []
    if raw.get("nvda_forward_pe") is not None:
        if raw["nvda_forward_pe"] >= 60:
            alerts.append(f"🚨 NVDA Forward P/E = {raw['nvda_forward_pe']:.1f} (very rich)")
        elif raw["nvda_forward_pe"] <= 20:
            alerts.append(f"🌱 NVDA Forward P/E = {raw['nvda_forward_pe']:.1f} (cheap)")
    if raw.get("vix") is not None:
        if raw["vix"] >= 35:
            alerts.append(f"😱 VIX = {raw['vix']:.1f} (panic)")
        elif raw["vix"] <= 12:
            alerts.append(f"😴 VIX = {raw['vix']:.1f} (complacency)")
    if raw.get("nvda_wk52_pos_pct") is not None:
        if raw["nvda_wk52_pos_pct"] >= 90:
            alerts.append(f"🔥 NVDA at {raw['nvda_wk52_pos_pct']:.0f}% of 52w range")
    if alerts:
        lines.append("*⚠️ Alerts*")
        for a in alerts:
            lines.append(a)
        lines.append("")

    if pages_url:
        lines.append(f"🔗 [Dashboard]({pages_url})")
    lines.append("_ASCS v1.0 · Sources: Yahoo Finance + FRED_")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 64)
    print(f"  ASCI Pipeline · {datetime.now(timezone.utc).isoformat()}")
    print("=" * 64)

    cfg = {
        "telegram_token":   env("TELEGRAM_TOKEN"),
        "telegram_chat_id": env("TELEGRAM_CHAT_ID"),
        "fred_api_key":     env("FRED_API_KEY"),
        "pages_url":        env("PAGES_URL"),
    }

    # ── Phase 1: Stock data ──
    print("\n📥 Phase 1: Fetching stock prices & fundamentals...")

    leaders_data = {}
    for symbol, name in SEMI_LEADERS.items():
        df = fetch_price_history(symbol, period="2y")
        info = fetch_ticker_info(symbol)
        tech = compute_technicals(df) if df is not None else {}
        leaders_data[symbol] = {"name": name, **info, **tech}
        print(f"  · {symbol:<6} ({name:<22}) ${tech.get('current', 0):>9,.2f}  "
              f"P/E={info.get('forwardPE') or info.get('trailingPE') or 'n/a'}")

    print("\n📥 Phase 2: Fetching hyperscaler capex...")
    hyperscalers_data = {}
    for symbol, name in HYPERSCALERS.items():
        info = fetch_ticker_info(symbol)
        capex = fetch_capex(symbol)
        df = fetch_price_history(symbol, period="2y")
        tech = compute_technicals(df) if df is not None else {}
        hyperscalers_data[symbol] = {"name": name, **info, **capex, **tech}
        print(f"  · {symbol:<6} capex_yoy={capex.get('capex_yoy_pct', 'n/a')}")

    print("\n📥 Phase 3: Fetching indices...")
    indices_data = {}
    for symbol, name in INDICES.items():
        df = fetch_price_history(symbol, period="2y")
        tech = compute_technicals(df) if df is not None else {}
        indices_data[symbol] = {"name": name, **tech}
        print(f"  · {symbol:<10} ({name:<24}) {tech.get('current', 0):>9,.2f}  "
              f"1D={tech.get('chg_1d_pct', 0):+.2f}%")

    # ── Phase 4: FRED macro ──
    print("\n📥 Phase 4: Fetching FRED macro indicators...")
    macro = {}
    if cfg["fred_api_key"]:
        for sid, label in [
            ("DGS10",   "10Y Treasury"),
            ("DTWEXBGS","DXY broad"),
            ("M2SL",    "M2 Money Supply"),
            ("MANEMP",  "Manufacturing employment"),
            ("CPIAUCSL","CPI"),
            ("UNRATE",  "Unemployment"),
        ]:
            d = fetch_fred_series(sid, cfg["fred_api_key"])
            if d:
                macro[sid] = d
                print(f"  · {sid:<10} ({label:<25}) {d['latest']:.2f} ({d['latest_date']})")
        # ISM PMI is available as NAPM in FRED
        ism = fetch_fred_series("NAPM", cfg["fred_api_key"])
        if ism:
            macro["NAPM"] = ism
    else:
        print("  [skip] FRED_API_KEY not set — macro dimension will be empty")

    # ── Phase 5: Compute aggregate metrics ──
    print("\n⚙️  Phase 5: Computing aggregate metrics...")

    raw: dict[str, float] = {}

    # NVDA-specific (the bellwether)
    nvda = leaders_data.get("NVDA", {})
    if nvda.get("forwardPE"):
        raw["nvda_forward_pe"] = float(nvda["forwardPE"])
    if nvda.get("priceToSales"):
        raw["nvda_price_to_sales"] = float(nvda["priceToSales"])
    if nvda.get("revenueGrowth") is not None:
        raw["nvda_revenue_growth_pct"] = float(nvda["revenueGrowth"]) * 100
    if nvda.get("wk52_position_pct") is not None:
        raw["nvda_wk52_pos_pct"] = float(nvda["wk52_position_pct"])
    if nvda.get("shortPercentOfFloat") is not None:
        raw["nvda_short_pct"] = float(nvda["shortPercentOfFloat"]) * 100

    # SOX-specific
    sox = indices_data.get("^SOX", {})
    if sox.get("dist_from_sma200_pct") is not None:
        raw["sox_above_sma200_pct"] = sox["dist_from_sma200_pct"]
        raw["sox_dist_sma200_pct"] = sox["dist_from_sma200_pct"]
    if sox.get("weekly_rsi") is not None:
        raw["sox_weekly_rsi"] = sox["weekly_rsi"]
    if sox.get("daily_rsi") is not None:
        raw["sox_daily_rsi"] = sox["daily_rsi"]

    # Average semi P/E percentile (proxy: average of leaders' forward P/Es)
    pes = [d["forwardPE"] for d in leaders_data.values() if d.get("forwardPE")]
    if pes:
        avg_pe = float(np.mean(pes))
        raw["sox_avg_forward_pe"] = avg_pe
        # Approximate percentile: assume long-run semi P/E range 10-50
        raw["sox_avg_pe_pctl"] = max(0, min(100, (avg_pe - 12) / (45 - 12) * 100))

    # Top 10 chip mcap / GDP — approximation
    mcaps = [d.get("marketCap") for d in leaders_data.values() if d.get("marketCap")]
    if len(mcaps) >= 5:
        total_mcap_t = sum(mcaps) / 1e12
        # US nominal GDP 2026 ~= $30T
        us_gdp_t = 30.0
        raw["top10_mcap_to_gdp_pct"] = total_mcap_t / us_gdp_t * 100

    # Average revenue growth across semi leaders
    growths = [d["revenueGrowth"] * 100 for d in leaders_data.values()
               if d.get("revenueGrowth") is not None]
    if growths:
        raw["semi_avg_revenue_growth_pct"] = float(np.mean(growths))

    # Hyperscaler capex YoY
    capex_yoys = [d["capex_yoy_pct"] for d in hyperscalers_data.values()
                  if d.get("capex_yoy_pct") is not None]
    if capex_yoys:
        raw["hyperscaler_capex_yoy_pct"] = float(np.mean(capex_yoys))

    # Memory capex YoY (Micron + others if available)
    mu_capex = leaders_data.get("MU", {})
    mu_capex_data = fetch_capex("MU")
    if mu_capex_data.get("capex_yoy_pct") is not None:
        raw["memory_capex_yoy_pct"] = mu_capex_data["capex_yoy_pct"]

    # Semi capex / revenue ratio (for AMAT, LRCX, KLAC the capex IS the revenue;
    # for the equipment customers, ratio matters)
    semi_capex_ratios = []
    for sym in ["NVDA", "AMD", "TSM", "MU", "INTC"]:
        d = leaders_data.get(sym, {})
        cf_data = fetch_capex(sym)
        if cf_data.get("capex_latest") and d.get("marketCap"):
            # Use TTM revenue from EV/Revenue if available
            ev = d.get("enterpriseValue")
            ev_rev = d.get("evToRevenue")
            if ev and ev_rev and ev_rev > 0:
                ttm_rev = ev / ev_rev
                if ttm_rev > 0:
                    semi_capex_ratios.append(cf_data["capex_latest"] / ttm_rev * 100)
    if semi_capex_ratios:
        raw["semi_capex_to_revenue_pct"] = float(np.mean(semi_capex_ratios))

    # VIX
    vix = indices_data.get("^VIX", {})
    if vix.get("current"):
        raw["vix"] = vix["current"]

    # Macro
    if "DGS10" in macro:
        raw["us_10y_yield"] = macro["DGS10"]["latest"]
    if "DTWEXBGS" in macro:
        raw["dxy"] = macro["DTWEXBGS"]["latest"]
    if "M2SL" in macro and macro["M2SL"].get("yoy_pct") is not None:
        raw["m2_yoy_pct"] = macro["M2SL"]["yoy_pct"]
    if "NAPM" in macro:
        raw["ism_pmi"] = macro["NAPM"]["latest"]

    # ── Phase 6: Compute ASCS ──
    print(f"\n⚙️  Phase 6: Computing ASCS with {len(raw)} indicators...")
    ascs = compute_ascs(raw)
    print(f"  ASCS = {ascs['composite']} ({ascs['phase']})")

    # ── Phase 7: Persist JSON ──
    print("\n💾 Phase 7: Writing JSON artifacts...")
    now_iso = datetime.now(timezone.utc).isoformat()

    snapshot = {
        "ts":         now_iso,
        "ascs":       ascs["composite"],
        "phase":      ascs["phase"],
        "dimensions": ascs["dimensions"],
        "indicators_raw": raw,
        "nvda_price": nvda.get("current"),
        "sox":        sox.get("current"),
        "vix":        vix.get("current"),
    }

    # Build chart series for SOX last 90 days
    sox_df = fetch_price_history("^SOX", period="6mo")
    sox_90d = []
    if sox_df is not None:
        for ts, row in sox_df.tail(90).iterrows():
            sox_90d.append({
                "ts": ts.isoformat(),
                "close": float(row["Close"]),
            })

    nvda_df = fetch_price_history("NVDA", period="6mo")
    nvda_90d = []
    if nvda_df is not None:
        for ts, row in nvda_df.tail(90).iterrows():
            nvda_90d.append({
                "ts": ts.isoformat(),
                "close": float(row["Close"]),
            })

    latest_payload = {
        "generated_at": now_iso,
        "ascs":         ascs,
        "leaders":      leaders_data,
        "hyperscalers": hyperscalers_data,
        "indices":      indices_data,
        "macro":        macro,
        "sox_90d":      sox_90d,
        "nvda_90d":     nvda_90d,
    }
    save_json(LATEST_FILE, latest_payload)

    history = append_history(snapshot)
    save_json(HISTORY_FILE, history)
    print(f"  History: {len(history)} snapshots")

    snap_file = SNAPSHOTS_DIR / f"{now_iso[:10]}.json"
    save_json(snap_file, latest_payload)

    # ── Phase 8: Telegram ──
    print("\n📨 Phase 8: Sending Telegram report...")
    report = format_telegram_report(ascs, leaders_data, sox, indices_data,
                                     raw, cfg["pages_url"])
    sent = send_telegram(report, cfg["telegram_token"], cfg["telegram_chat_id"])

    print("\n" + "=" * 64)
    print(f"  ✅ Pipeline complete · ASCS={ascs['composite']} ({ascs['phase']})")
    print(f"  Telegram sent: {sent}")
    print("=" * 64)

    if env("GITHUB_STEP_SUMMARY"):
        with open(env("GITHUB_STEP_SUMMARY"), "a", encoding="utf-8") as f:
            f.write(f"## ASCI Pipeline Summary\n\n")
            f.write(f"- **ASCS**: {ascs['composite']} ({ascs['phase']})\n")
            f.write(f"- **NVDA**: ${nvda.get('current', 0):,.2f}\n")
            f.write(f"- **SOX**: {sox.get('current', 0):,.0f}\n")
            f.write(f"- **VIX**: {vix.get('current', 0):.2f}\n")
            f.write(f"- **Indicators used**: {len(raw)}\n")
            f.write(f"- **Telegram**: {sent}\n")

    return 0 if ascs["composite"] is not None else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ FATAL: {e}")
        traceback.print_exc()
        token = env("TELEGRAM_TOKEN")
        chat = env("TELEGRAM_CHAT_ID")
        if token and chat:
            send_telegram(
                f"❌ *ASCI pipeline failed*\n\n```\n{str(e)[:500]}\n```",
                token, chat,
            )
        sys.exit(1)
