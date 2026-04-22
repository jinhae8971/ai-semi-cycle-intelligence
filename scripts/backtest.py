"""ASCS Historical Backtest.

Replays the ASCS model against the past ~8 years of daily data, with
emphasis on three known cycle events:
  - 2018-Q4: US-China trade war crash (SOX -30%)
  - 2020-Mar: COVID crash + recovery
  - 2022: Inflation crash (SOX -40%)
  - 2023-2025: AI boom

For each historical day, we reconstruct the indicators that yfinance
gives us deep history for (price-based: RSI, SMA distance, 52w position,
P/E from EPS) and compute ASCS. Fundamentals (P/E, capex YoY) use
TTM-back-extrapolation where possible.

Output: data/backtest_results.json + console summary + matplotlib charts.
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logging.getLogger("yfinance").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Reuse the same scoring logic from production pipeline
sys.path.insert(0, str(Path(__file__).parent))
from run_pipeline import (
    ScoringRule, RULES, DIMENSION_WEIGHTS, PHASE_RANGES, phase_for, compute_ascs,
)


# ══════════════════════════════════════════════════════════════
# Historical data fetch (long range)
# ══════════════════════════════════════════════════════════════

def fetch_long_history(symbol: str, start: str, end: str | None = None) -> pd.DataFrame | None:
    try:
        df = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
        if len(df) < 100:
            return None
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    except Exception as e:
        print(f"  [err] {symbol}: {e}")
        return None


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ══════════════════════════════════════════════════════════════
# Historical indicators reconstruction
# ══════════════════════════════════════════════════════════════

def build_historical_panel(
    sox: pd.DataFrame, nvda: pd.DataFrame, vix: pd.DataFrame,
    samsung: pd.DataFrame | None, skhynix: pd.DataFrame | None,
) -> pd.DataFrame:
    """For each business day, compute every indicator we can derive from price."""
    print("  Building historical indicator panel...")

    # Use SOX dates as the master index
    idx = sox.index
    panel = pd.DataFrame(index=idx)

    # SOX-based
    sox_close = sox["Close"]
    panel["sox_close"] = sox_close
    panel["sox_above_sma200_pct"] = (sox_close / sox_close.rolling(200).mean() - 1) * 100
    panel["sox_dist_sma200_pct"] = panel["sox_above_sma200_pct"]
    panel["sox_daily_rsi"] = compute_rsi(sox_close, 14)
    sox_weekly = sox_close.resample("W").last()
    sox_weekly_rsi = compute_rsi(sox_weekly, 14)
    panel["sox_weekly_rsi"] = sox_weekly_rsi.reindex(idx, method="ffill")

    # NVDA-based — align to SOX dates
    nvda_aligned = nvda["Close"].reindex(idx, method="ffill")
    nvda_252 = nvda_aligned.rolling(252)
    panel["nvda_wk52_pos_pct"] = (
        (nvda_aligned - nvda_252.min()) / (nvda_252.max() - nvda_252.min()) * 100
    )

    # VIX
    vix_aligned = vix["Close"].reindex(idx, method="ffill")
    panel["vix"] = vix_aligned

    # Samsung 52-week position
    if samsung is not None:
        samsung_aligned = samsung["Close"].reindex(idx, method="ffill")
        s252 = samsung_aligned.rolling(252)
        panel["samsung_wk52_pos_pct"] = (
            (samsung_aligned - s252.min()) / (s252.max() - s252.min()) * 100
        )

    # SK Hynix 52-week position
    if skhynix is not None:
        skhynix_aligned = skhynix["Close"].reindex(idx, method="ffill")
        sk252 = skhynix_aligned.rolling(252)
        panel["skhynix_wk52_pos_pct"] = (
            (skhynix_aligned - sk252.min()) / (sk252.max() - sk252.min()) * 100
        )

    # KOSPI proxy (use samsung as proxy since they correlate >0.85)
    if samsung is not None:
        samsung_aligned = samsung["Close"].reindex(idx, method="ffill")
        panel["kospi_above_sma200_pct"] = (
            samsung_aligned / samsung_aligned.rolling(200).mean() - 1
        ) * 100
        kospi_weekly_rsi = compute_rsi(samsung_aligned.resample("W").last(), 14)
        panel["kospi_weekly_rsi"] = kospi_weekly_rsi.reindex(idx, method="ffill")

    return panel.dropna(how="all")


# ══════════════════════════════════════════════════════════════
# Score reconstruction (subset of rules that work historically)
# ══════════════════════════════════════════════════════════════

# Only price-derivable rules survive backtesting
HISTORICAL_RULES = [
    r for r in RULES
    if r.metric in {
        "sox_above_sma200_pct", "sox_dist_sma200_pct",
        "sox_weekly_rsi", "sox_daily_rsi",
        "nvda_wk52_pos_pct", "vix",
        "samsung_wk52_pos_pct", "skhynix_wk52_pos_pct",
        "kospi_above_sma200_pct", "kospi_weekly_rsi",
    }
]


def compute_historical_ascs(panel: pd.DataFrame) -> pd.DataFrame:
    """For each row, compute ASCS using only price-derivable indicators."""
    print(f"  Scoring {len(panel)} historical days using {len(HISTORICAL_RULES)} rules...")

    results = []
    for ts, row in panel.iterrows():
        raw = {col: row[col] for col in panel.columns
               if col in {r.metric for r in HISTORICAL_RULES}
               and not pd.isna(row[col])}

        # Reweight dimensions in a backtest-friendly way
        by_dim = {d: [] for d in DIMENSION_WEIGHTS}
        for rule in HISTORICAL_RULES:
            val = raw.get(rule.metric)
            s = rule.score(val)
            if s is not None:
                by_dim[rule.dimension].append((s, rule.weight))

        dim_scores = {}
        for dim, items in by_dim.items():
            if items:
                tw = sum(w for _, w in items)
                dim_scores[dim] = sum(s * w for s, w in items) / tw

        if not dim_scores:
            results.append({"ts": ts, "ascs": None})
            continue

        total_w = sum(DIMENSION_WEIGHTS[d] for d in dim_scores)
        ascs = sum(s * DIMENSION_WEIGHTS[d] for d, s in dim_scores.items()) / total_w

        # Phase
        name, _, _ = phase_for(ascs)

        results.append({
            "ts": ts, "ascs": ascs, "phase": name,
            "n_indicators": len(raw),
            **{f"dim_{d}": s for d, s in dim_scores.items()},
        })

    df = pd.DataFrame(results).set_index("ts")
    return df


# ══════════════════════════════════════════════════════════════
# Event analysis
# ══════════════════════════════════════════════════════════════

EVENTS = [
    {
        "name": "2018 Trade War Bottom",
        "date": "2018-12-24",
        "expected_phase": "Capitulation",
        "expected_score": "<25",
        "context": "SOX fell ~30% from Sep peak on US-China trade tensions. Bottom of cycle.",
    },
    {
        "name": "2020 COVID Crash",
        "date": "2020-03-23",
        "expected_phase": "Capitulation",
        "expected_score": "<20",
        "context": "Pandemic bottom. SOX -36% from Feb high in 4 weeks.",
    },
    {
        "name": "2021 Cycle Top",
        "date": "2021-12-27",
        "expected_phase": "Late Bull / Euphoria",
        "expected_score": ">70",
        "context": "Pre-2022 crash high. Memory boom + WFH chip shortage peak.",
    },
    {
        "name": "2022 Inflation Bottom",
        "date": "2022-10-13",
        "expected_phase": "Capitulation",
        "expected_score": "<25",
        "context": "Fed hiking + memory glut + crypto contagion. SOX -40% from peak.",
    },
    {
        "name": "ChatGPT Launch (Nov 2022)",
        "date": "2022-11-30",
        "expected_phase": "Recovery",
        "expected_score": "20-40",
        "context": "ChatGPT release sparks AI infrastructure cycle.",
    },
    {
        "name": "2024 Mid-Year",
        "date": "2024-06-28",
        "expected_phase": "Late Bull",
        "expected_score": "60-80",
        "context": "NVDA hit $140 on AI hype. Hindsight: top approaching.",
    },
    {
        "name": "2025 Q4 ATH",
        "date": "2025-10-06",
        "expected_phase": "Late Bull / Euphoria",
        "expected_score": ">75",
        "context": "Cycle peak before 2026 Q1 correction.",
    },
    {
        "name": "Today (Live)",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "expected_phase": "Late Bull",
        "expected_score": "70-80",
        "context": "Current pipeline live result for comparison.",
    },
]


def evaluate_event(scores: pd.DataFrame, event: dict) -> dict:
    target_date = pd.Timestamp(event["date"], tz="UTC")
    # Find nearest available date in scores
    nearest_idx = scores.index.get_indexer([target_date], method="nearest")[0]
    if nearest_idx < 0 or nearest_idx >= len(scores):
        return {**event, "actual_score": None, "actual_phase": None, "match": "no_data"}

    actual_ts = scores.index[nearest_idx]
    actual_score = scores.iloc[nearest_idx]["ascs"]
    actual_phase = scores.iloc[nearest_idx]["phase"]

    # Loose match check with ±5 point tolerance for boundary cases
    expected = event["expected_score"]
    match = "❌"
    TOL = 5  # tolerance points
    if pd.notna(actual_score):
        if "<" in expected:
            threshold = float(expected.replace("<", ""))
            match = "✅" if actual_score < threshold + TOL else "❌"
        elif ">" in expected:
            threshold = float(expected.replace(">", ""))
            match = "✅" if actual_score > threshold - TOL else "❌"
        elif "-" in expected:
            lo, hi = map(float, expected.split("-"))
            match = "✅" if (lo - TOL) <= actual_score <= (hi + TOL) else "❌"

    return {
        "name":           event["name"],
        "target_date":    event["date"],
        "actual_date":    actual_ts.strftime("%Y-%m-%d"),
        "expected_phase": event["expected_phase"],
        "expected_score": expected,
        "actual_score":   round(actual_score, 1) if pd.notna(actual_score) else None,
        "actual_phase":   actual_phase,
        "match":          match,
        "context":        event["context"],
    }


# ══════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════

def print_event_table(results: list[dict]):
    print("\n" + "═" * 90)
    print("  ASCS Backtest Validation — Event-Based")
    print("═" * 90)
    print(f"  {'Event':<32} {'Target':<12} {'Actual':<12} {'Expected':<14} {'Got':<10} {'Match':<5}")
    print("─" * 90)
    for r in results:
        actual = f"{r['actual_score']:.1f}" if r['actual_score'] is not None else "n/a"
        actual_phase = r['actual_phase'] or "—"
        print(f"  {r['name']:<32} {r['target_date']:<12} {r['actual_date']:<12} "
              f"{r['expected_score']:<14} {actual:<5} {actual_phase[:8]:<10} {r['match']}")
    print("─" * 90)
    correct = sum(1 for r in results if r['match'] == '✅')
    total = sum(1 for r in results if r['match'] in ('✅', '❌'))
    print(f"  Validation: {correct}/{total} events matched expected phase")
    print("═" * 90)


def save_charts(scores: pd.DataFrame, sox: pd.DataFrame, results: list[dict],
                output_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("  [skip] matplotlib not installed — no charts")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: SOX price (log scale)
    ax = axes[0]
    ax.semilogy(sox.index, sox["Close"], color="#f6c945", linewidth=1.2)
    ax.set_ylabel("SOX (log scale)")
    ax.set_title("SOX Index vs ASCS Backtest")
    ax.grid(True, alpha=0.3)

    # Mark events
    for r in results:
        if r["actual_score"] is None:
            continue
        ts = pd.Timestamp(r["actual_date"], tz="UTC")
        color = "green" if r["match"] == "✅" else ("red" if r["match"] == "❌" else "gray")
        ax.axvline(ts, color=color, linestyle="--", alpha=0.4, linewidth=0.8)
        try:
            y = sox["Close"].asof(ts)
            if pd.notna(y):
                ax.annotate(
                    r["name"][:20], xy=(ts, y),
                    xytext=(5, 10), textcoords="offset points",
                    fontsize=7, color=color, rotation=15,
                )
        except Exception:
            pass

    # Bottom: ASCS scores
    ax2 = axes[1]
    ax2.plot(scores.index, scores["ascs"], color="#3a8891", linewidth=1.2, label="ASCS")
    ax2.fill_between(scores.index, 0, 20, color="#0a4d68", alpha=0.15, label="Capitulation")
    ax2.fill_between(scores.index, 20, 40, color="#3a8891", alpha=0.15, label="Recovery")
    ax2.fill_between(scores.index, 40, 60, color="#f6c945", alpha=0.15, label="Expansion")
    ax2.fill_between(scores.index, 60, 80, color="#e76f51", alpha=0.15, label="Late Bull")
    ax2.fill_between(scores.index, 80, 100, color="#c1121f", alpha=0.15, label="Euphoria")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("ASCS")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left", fontsize=8, ncol=5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "asci_backtest.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {out_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    START_DATE = "2017-01-01"

    print("═" * 70)
    print(f"  ASCS Backtest · {START_DATE} → present")
    print("═" * 70)

    print("\n📥 Fetching long historical data...")
    sox = fetch_long_history("^SOX", START_DATE)
    nvda = fetch_long_history("NVDA", START_DATE)
    vix = fetch_long_history("^VIX", START_DATE)
    samsung = fetch_long_history("005930.KS", START_DATE)
    skhynix = fetch_long_history("000660.KS", START_DATE)

    if sox is None or nvda is None or vix is None:
        print("❌ Could not fetch core data")
        return 1

    print(f"  SOX:     {len(sox):,} days  ({sox.index.min().date()} → {sox.index.max().date()})")
    print(f"  NVDA:    {len(nvda):,} days")
    print(f"  VIX:     {len(vix):,} days")
    print(f"  Samsung: {len(samsung) if samsung is not None else 0:,} days")
    print(f"  SKHynix: {len(skhynix) if skhynix is not None else 0:,} days")

    print("\n⚙️  Building indicator panel...")
    panel = build_historical_panel(sox, nvda, vix, samsung, skhynix)
    print(f"  Panel: {len(panel):,} days × {len(panel.columns)} indicators")

    print("\n⚙️  Computing historical ASCS...")
    scores = compute_historical_ascs(panel)
    valid_scores = scores.dropna(subset=["ascs"])
    print(f"  Scored: {len(valid_scores):,} days")
    print(f"  ASCS range: {valid_scores['ascs'].min():.1f} → {valid_scores['ascs'].max():.1f}")
    print(f"  ASCS mean:  {valid_scores['ascs'].mean():.1f}")
    print(f"  ASCS std:   {valid_scores['ascs'].std():.1f}")

    print("\n🔍 Evaluating against historical events...")
    event_results = [evaluate_event(scores, e) for e in EVENTS]
    print_event_table(event_results)

    # Phase distribution
    print("\n📊 Phase distribution over backtest period:")
    phase_counts = valid_scores["phase"].value_counts(normalize=True) * 100
    for phase in ["Capitulation", "Recovery", "Expansion", "Late Bull", "Euphoria"]:
        pct = phase_counts.get(phase, 0)
        bar = "█" * int(pct / 2)
        print(f"  {phase:<14} {bar:<25} {pct:.1f}%")

    print("\n💾 Saving results...")

    # JSON output
    json_out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backtest_window": {
            "start": str(scores.index.min().date()),
            "end":   str(scores.index.max().date()),
            "n_days": len(valid_scores),
        },
        "indicators_used": [r.metric for r in HISTORICAL_RULES],
        "event_results": event_results,
        "ascs_stats": {
            "min":    float(valid_scores["ascs"].min()),
            "max":    float(valid_scores["ascs"].max()),
            "mean":   float(valid_scores["ascs"].mean()),
            "median": float(valid_scores["ascs"].median()),
            "std":    float(valid_scores["ascs"].std()),
        },
        "phase_distribution_pct": {p: float(phase_counts.get(p, 0))
                                    for p in ["Capitulation", "Recovery", "Expansion", "Late Bull", "Euphoria"]},
    }
    out_json = output_dir / "backtest_results.json"
    out_json.write_text(json.dumps(json_out, indent=2, default=str), encoding="utf-8")
    print(f"  [saved] {out_json}")

    # Time series CSV
    valid_scores[["ascs", "phase", "n_indicators"]].reset_index().rename(
        columns={"ts": "date"}
    ).to_csv(output_dir / "backtest_timeseries.csv", index=False)
    print(f"  [saved] {output_dir}/backtest_timeseries.csv")

    # Charts
    save_charts(scores, sox, event_results, output_dir)

    print("\n✅ Backtest complete\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
