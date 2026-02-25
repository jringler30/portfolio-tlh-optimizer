#!/usr/bin/env python3
"""
Portfolio Returns Calculation Engine — Streamlit App
====================================================
Run:  streamlit run portfolio_returns_engine.py

Dependencies: streamlit, pandas, numpy, gdown, scipy

CHANGELOG:
  - Original: buy-and-hold engine with daily series
  - V2: Added daily rebalancing strategy
  - V3: Unified rebalancing engine: Daily / Weekly / Monthly / Quarterly
  - V4 (CURRENT):
      * Threshold (drift-band) rebalancing with absolute/relative drift modes
      * Per-asset tolerance bands with advanced per-ticker overrides
      * Full / Partial rebalance action modes
      * Calendar + Threshold combination with event logging
      * Cooldown option for threshold triggers
      * Enhanced metrics: Skewness, Kurtosis, Avg Drawdown, Tracking Error, Info Ratio
      * Drift diagnostics section with per-ticker histograms
      * Universal page-level tax parameters
      * Internal event log DataFrame for future CSV export
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Tuple, Any, Optional, Set
from scipy import stats as sp_stats
import warnings

# -- MSBA v1 Optimizer Import --
try:
    from optimizer_msba_v1_engine import run_optimizer_simulation
    _OPTIMIZER_AVAILABLE = True
except ImportError:
    _OPTIMIZER_AVAILABLE = False

st.set_page_config(
    page_title="Portfolio Returns Calculator",
    page_icon="\U0001f4ca",
    layout="wide",
)

try:
    from ui_style import inject_site_css, render_hero
    inject_site_css()
    _STYLE_LOADED = True
except ImportError:
    _STYLE_LOADED = False

import gdown
from pathlib import Path

DATA_PATH = Path("/tmp/price_data.parquet")
FILE_ID = "1pMQ817V05j4RK0vqJcVkBMmOBK5zRrug"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"


@st.cache_data(show_spinner=True)
def ensure_data():
    if not DATA_PATH.exists():
        out = gdown.download(GDRIVE_URL, str(DATA_PATH), quiet=False, fuzzy=True)
        if out is None or not DATA_PATH.exists():
            st.error("Google Drive download failed (permissions/quota/bad link).")
            st.stop()


ensure_data()


# ================================================================
#  CORE ENGINE FUNCTIONS
# ================================================================


def validate_weights(tickers: List[str], weights: List[float],
                     tolerance: float = 0.05) -> Tuple[List[str], List[float]]:
    if len(tickers) != len(weights):
        raise ValueError(f"Length mismatch: {len(tickers)} tickers vs {len(weights)} weights.")
    if any(w < 0 for w in weights):
        raise ValueError("Negative weights are not allowed.")
    combined: Dict[str, float] = {}
    for t, w in zip(tickers, weights):
        t_upper = t.strip().upper()
        combined[t_upper] = combined.get(t_upper, 0.0) + w
    tickers_out = list(combined.keys())
    weights_out = list(combined.values())
    total = sum(weights_out)
    if total == 0:
        raise ValueError("Total weight is zero.")
    if abs(total - 1.0) > tolerance:
        raise ValueError(
            f"Weights sum to {total:.4f}, which deviates from 1.0 by more than "
            f"tolerance ({tolerance}). Please fix your weights."
        )
    weights_out = [w / total for w in weights_out]
    return tickers_out, weights_out


def prepare_price_data(df: pd.DataFrame, price_field: str = "PRICECLOSE") -> pd.DataFrame:
    df = df.copy()
    df["PRICEDATE"] = pd.to_datetime(df["PRICEDATE"], errors="coerce")
    df = df.dropna(subset=["PRICEDATE"])
    if "TRADINGITEMSTATUSID" in df.columns:
        df = df[df["TRADINGITEMSTATUSID"].isin([1, 15])].copy()
    if price_field not in df.columns:
        raise ValueError(f"Price field '{price_field}' not found in dataset.")
    df[price_field] = pd.to_numeric(df[price_field], errors="coerce")
    df = df.dropna(subset=[price_field])
    df["TICKERSYMBOL"] = df["TICKERSYMBOL"].astype(str).str.strip().str.upper()
    df = df.sort_values(["TICKERSYMBOL", "PRICEDATE"]).reset_index(drop=True)
    return df


def get_ticker_prices(ticker_df, ticker, start_date, end_date, price_field):
    flags = []
    on_or_after = ticker_df[ticker_df["PRICEDATE"] >= start_date]
    if on_or_after.empty:
        return {"error": f"No data for {ticker} on/after {start_date.date()}."}
    start_row = on_or_after.iloc[0]
    start_date_used = start_row["PRICEDATE"]
    start_price = float(start_row[price_field])
    if start_date_used != start_date:
        flags.append(f"start shifted {start_date.date()}->{start_date_used.date()}")
    on_or_before = ticker_df[ticker_df["PRICEDATE"] <= end_date]
    if on_or_before.empty:
        return {"error": f"No data for {ticker} on/before {end_date.date()}."}
    end_row = on_or_before.iloc[-1]
    end_date_used = end_row["PRICEDATE"]
    end_price = float(end_row[price_field])
    if end_date_used != end_date:
        flags.append(f"end shifted {end_date.date()}->{end_date_used.date()}")
    if start_date_used > end_date_used:
        return {"error": f"Adjusted start after end for {ticker}."}
    return {
        "start_date_used": start_date_used, "end_date_used": end_date_used,
        "start_price": start_price, "end_price": end_price, "flags": flags,
    }


def calculate_portfolio_returns(
    df, tickers, weights, start_date, end_date,
    initial_capital=100_000.0, price_field="PRICECLOSE",
    allow_cash_residual=False,
):
    if price_field not in ("PRICECLOSE", "PRICEMID"):
        raise ValueError(f"price_field must be 'PRICECLOSE' or 'PRICEMID'.")
    tickers, weights = validate_weights(tickers, weights)
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    if start_dt >= end_dt:
        raise ValueError("start_date must be before end_date.")
    clean = df
    available = set(clean["TICKERSYMBOL"].unique())
    missing = [t for t in tickers if t not in available]
    if missing:
        raise ValueError(f"Tickers not found in dataset: {missing}")
    rows, dropped = [], []
    for ticker, weight in zip(tickers, weights):
        result = get_ticker_prices(
            clean[clean["TICKERSYMBOL"] == ticker], ticker, start_dt, end_dt, price_field
        )
        if "error" in result:
            dropped.append((ticker, weight, result["error"]))
            continue
        rows.append({"ticker": ticker, "weight": weight, **result})
    if dropped and not rows:
        raise ValueError("All tickers dropped -- insufficient data.")
    if dropped:
        total_w = sum(r["weight"] for r in rows)
        for r in rows:
            r["weight"] /= total_w
    holdings_data = []
    total_cash_residual = 0.0
    for r in rows:
        alloc = initial_capital * r["weight"]
        if allow_cash_residual:
            shares = int(alloc // r["start_price"])
            total_cash_residual += alloc - shares * r["start_price"]
        else:
            shares = alloc / r["start_price"]
        end_value = shares * r["end_price"]
        cost = shares * r["start_price"]
        holdings_data.append({
            "Ticker": r["ticker"], "Weight": r["weight"],
            "Start Date": r["start_date_used"].strftime("%Y-%m-%d"),
            "End Date": r["end_date_used"].strftime("%Y-%m-%d"),
            "Start Price": round(r["start_price"], 2),
            "End Price": round(r["end_price"], 2),
            "Shares": round(shares, 4),
            "Start Value": round(cost, 2),
            "End Value": round(end_value, 2),
            "Return": round((r["end_price"] / r["start_price"]) - 1, 6),
            "Gain ($)": round(end_value - cost, 2),
            "Gain (%)": round((end_value - cost) / cost, 6) if cost else 0,
            "Flags": "; ".join(r["flags"]) if r["flags"] else "OK",
        })
    holdings_df = pd.DataFrame(holdings_data)
    port_end = holdings_df["End Value"].sum() + total_cash_residual
    summary = {
        "portfolio_start_value": initial_capital,
        "portfolio_end_value": round(port_end, 2),
        "portfolio_total_return": round(port_end / initial_capital - 1, 6),
        "total_unrealized_gain_dollars": round(port_end - initial_capital, 2),
        "total_unrealized_gain_pct": round((port_end - initial_capital) / initial_capital, 6),
        "cash_residual": round(total_cash_residual, 2),
        "tickers_dropped": len(dropped),
        "dropped_details": dropped,
    }
    return summary, holdings_df


def build_daily_series(df, holdings, initial_capital, price_field="PRICECLOSE"):
    """Buy-and-hold daily series (original logic, unchanged)."""
    clean = df.copy()
    clean["PRICEDATE"] = pd.to_datetime(clean["PRICEDATE"], errors="coerce")
    clean["TICKERSYMBOL"] = clean["TICKERSYMBOL"].astype(str).str.strip().str.upper()
    clean[price_field] = pd.to_numeric(clean[price_field], errors="coerce")
    all_start = pd.to_datetime(holdings["Start Date"]).min()
    all_end = pd.to_datetime(holdings["End Date"]).max()
    clean = clean[(clean["PRICEDATE"] >= all_start) & (clean["PRICEDATE"] <= all_end)]
    tickers = holdings["Ticker"].tolist()
    clean = clean[clean["TICKERSYMBOL"].isin(tickers)]
    frames = []
    for _, row in holdings.iterrows():
        tk = row["Ticker"]
        shares = row["Shares"]
        tk_prices = (
            clean[clean["TICKERSYMBOL"] == tk][["PRICEDATE", price_field]]
            .drop_duplicates(subset="PRICEDATE")
            .set_index("PRICEDATE").sort_index()
            .rename(columns={price_field: tk})
        )
        tk_prices[tk] = tk_prices[tk] * shares
        frames.append(tk_prices)
    daily = frames[0]
    for f in frames[1:]:
        daily = daily.join(f, how="outer")
    daily = daily.sort_index().ffill().bfill()
    daily["Portfolio Value"] = daily[tickers].sum(axis=1)
    daily["Cost Basis"] = initial_capital
    for tk in tickers:
        start_val = daily[tk].iloc[0]
        daily[f"{tk} Return (%)"] = (daily[tk] / start_val - 1) * 100
    return daily


# ================================================================
# V3: Calendar rebalancing engine (UNCHANGED)
# ================================================================

def build_prices_wide(df, tickers, start_date, end_date, price_field="PRICECLOSE"):
    """Build a (date x ticker) wide price matrix."""
    mask = (
        df["TICKERSYMBOL"].isin(tickers)
        & (df["PRICEDATE"] >= pd.Timestamp(start_date))
        & (df["PRICEDATE"] <= pd.Timestamp(end_date))
    )
    subset = df.loc[mask, ["TICKERSYMBOL", "PRICEDATE", price_field]].copy()
    subset = subset.drop_duplicates(subset=["TICKERSYMBOL", "PRICEDATE"])
    wide = subset.pivot(index="PRICEDATE", columns="TICKERSYMBOL", values=price_field)
    wide = wide.sort_index().ffill().bfill()
    missing_cols = [t for t in tickers if t not in wide.columns]
    if missing_cols:
        raise ValueError(f"Tickers missing from price data after filtering: {missing_cols}")
    wide = wide[tickers]
    return wide


def _get_rebalance_dates(trading_dates, freq):
    """Return the SET of dates on which calendar rebalancing should occur."""
    dates = pd.DatetimeIndex(trading_dates)
    if len(dates) < 2:
        return set()
    if freq == "Daily":
        return set(dates[1:])
    rebal_set = set()
    if freq == "Weekly":
        prev_week = dates[0].isocalendar()[1]
        prev_year = dates[0].year
        for dt in dates[1:]:
            iso = dt.isocalendar()
            if iso[1] != prev_week or dt.year != prev_year:
                rebal_set.add(dt)
                prev_week = iso[1]
                prev_year = dt.year
    elif freq == "Monthly":
        prev_month = dates[0].month
        prev_year = dates[0].year
        for dt in dates[1:]:
            if dt.month != prev_month or dt.year != prev_year:
                rebal_set.add(dt)
                prev_month = dt.month
                prev_year = dt.year
    elif freq == "Quarterly":
        quarter_months = {1, 4, 7, 10}
        prev_month = dates[0].month
        prev_year = dates[0].year
        for dt in dates[1:]:
            if dt.month in quarter_months and (dt.month != prev_month or dt.year != prev_year):
                rebal_set.add(dt)
            if dt.month != prev_month or dt.year != prev_year:
                prev_month = dt.month
                prev_year = dt.year
    else:
        raise ValueError(f"Unknown rebalance frequency: {freq}")
    return rebal_set


def build_rebalanced_series(prices_wide, target_weights, initial_capital, rebalance_freq):
    """Unified rebalancing engine (ORIGINAL -- calendar-only, unchanged)."""
    tickers = list(target_weights.keys())
    dates = prices_wide.index.tolist()
    n_days = len(dates)
    if n_days == 0:
        raise ValueError("No trading dates in the filtered price data.")
    rebal_dates = _get_rebalance_dates(dates, rebalance_freq)
    shares = {}
    for tk in tickers:
        alloc = initial_capital * target_weights[tk]
        shares[tk] = alloc / prices_wide.loc[dates[0], tk]
    portfolio_values = np.empty(n_days, dtype=np.float64)
    ticker_values_arr = {tk: np.empty(n_days, dtype=np.float64) for tk in tickers}
    rebalance_count = 0
    total_turnover_dollars = 0.0
    for i, dt in enumerate(dates):
        total_value = 0.0
        tv = {}
        for tk in tickers:
            val = shares[tk] * prices_wide.loc[dt, tk]
            tv[tk] = val
            total_value += val
        portfolio_values[i] = total_value
        for tk in tickers:
            ticker_values_arr[tk][i] = tv[tk]
        if dt in rebal_dates and total_value > 0:
            day_turnover = 0.0
            needs_rebalance = False
            for tk in tickers:
                current_weight = tv[tk] / total_value
                if abs(current_weight - target_weights[tk]) > 1e-10:
                    needs_rebalance = True
                    break
            if needs_rebalance:
                rebalance_count += 1
                for tk in tickers:
                    target_value = target_weights[tk] * total_value
                    new_shares = target_value / prices_wide.loc[dt, tk]
                    trade_shares = new_shares - shares[tk]
                    trade_dollars = abs(trade_shares * prices_wide.loc[dt, tk])
                    day_turnover += trade_dollars
                    shares[tk] = new_shares
                total_turnover_dollars += day_turnover
    avg_port_value = np.mean(portfolio_values)
    turnover_proxy = (total_turnover_dollars / avg_port_value) if avg_port_value > 0 else 0.0
    rebal_daily = pd.DataFrame(index=dates)
    rebal_daily.index.name = "PRICEDATE"
    for tk in tickers:
        rebal_daily[f"{tk} (Rebal)"] = ticker_values_arr[tk]
    rebal_daily["Portfolio Value"] = portfolio_values
    rebal_stats = {
        "rebalance_count": rebalance_count,
        "turnover_proxy": round(turnover_proxy, 4),
        "final_value": round(portfolio_values[-1], 2),
        "total_return": round(portfolio_values[-1] / initial_capital - 1, 6),
    }
    return rebal_daily, rebal_stats


# ================================================================
# V4 ADDITION: THRESHOLD (DRIFT-BAND) REBALANCING ENGINE
# ================================================================

### THRESHOLD REBALANCE ADDITIONS -- Helper Functions


def compute_weights(shares: Dict[str, float], prices: Dict[str, float]) -> Dict[str, float]:
    """Compute current portfolio weights from shares and prices."""
    values = {tk: shares[tk] * prices[tk] for tk in shares}
    total = sum(values.values())
    if total <= 0:
        return {tk: 0.0 for tk in shares}
    return {tk: values[tk] / total for tk in shares}


def compute_drift(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    drift_mode: str = "Absolute",
) -> Dict[str, float]:
    """
    Compute per-asset drift between current and target weights.
    Absolute: abs(w_i - target_i)
    Relative: abs(w_i / target_i - 1), safe for target_i == 0
    """
    drift = {}
    for tk in target_weights:
        w_cur = current_weights.get(tk, 0.0)
        w_tgt = target_weights[tk]
        if drift_mode == "Relative":
            if w_tgt < 1e-12:
                drift[tk] = abs(w_cur)
            else:
                drift[tk] = abs(w_cur / w_tgt - 1.0)
        else:
            drift[tk] = abs(w_cur - w_tgt)
    return drift


def find_threshold_triggers(
    drift: Dict[str, float],
    tolerances: Dict[str, float],
) -> List[str]:
    """Return list of tickers whose drift exceeds their per-asset tolerance."""
    breached = []
    for tk, d in drift.items():
        tol = tolerances.get(tk, 0.05)
        if d > tol + 1e-12:
            breached.append(tk)
    return breached


def apply_rebalance_full(
    shares: Dict[str, float],
    target_weights: Dict[str, float],
    prices: Dict[str, float],
    total_value: float,
    whole_shares: bool = False,
) -> Tuple[Dict[str, float], float]:
    """Full rebalance: set ALL assets to exact target weights."""
    turnover = 0.0
    new_shares = {}
    for tk in target_weights:
        target_val = target_weights[tk] * total_value
        if whole_shares:
            ns = int(target_val // prices[tk]) if prices[tk] > 0 else 0
        else:
            ns = target_val / prices[tk] if prices[tk] > 0 else 0.0
        trade_dollars = abs(ns - shares[tk]) * prices[tk]
        turnover += trade_dollars
        new_shares[tk] = ns
    return new_shares, turnover


def apply_rebalance_partial(
    shares: Dict[str, float],
    target_weights: Dict[str, float],
    tolerances: Dict[str, float],
    breached_tickers: List[str],
    prices: Dict[str, float],
    total_value: float,
    whole_shares: bool = False,
) -> Tuple[Dict[str, float], float]:
    """
    Partial rebalance: bring breached assets back to target weight.
    Scale non-breached assets proportionally so weights sum to 1.
    """
    tickers = list(target_weights.keys())
    current_weights = compute_weights(shares, prices)
    breached_set = set(breached_tickers)
    breached_target_sum = sum(target_weights[tk] for tk in breached_set)
    remaining_budget = 1.0 - breached_target_sum
    non_breached_current_sum = sum(
        current_weights.get(tk, 0.0) for tk in tickers if tk not in breached_set
    )
    desired_weights = {}
    for tk in tickers:
        if tk in breached_set:
            desired_weights[tk] = target_weights[tk]
        else:
            if non_breached_current_sum > 1e-12:
                desired_weights[tk] = (current_weights.get(tk, 0.0) / non_breached_current_sum) * remaining_budget
            else:
                n_non = len(tickers) - len(breached_set)
                desired_weights[tk] = remaining_budget / n_non if n_non > 0 else 0.0
    turnover = 0.0
    new_shares = {}
    for tk in tickers:
        target_val = desired_weights[tk] * total_value
        if whole_shares:
            ns = int(target_val // prices[tk]) if prices[tk] > 0 else 0
        else:
            ns = target_val / prices[tk] if prices[tk] > 0 else 0.0
        trade_dollars = abs(ns - shares[tk]) * prices[tk]
        turnover += trade_dollars
        new_shares[tk] = ns
    return new_shares, turnover


def build_threshold_rebalanced_series(
    prices_wide,
    target_weights,
    initial_capital,
    tolerances,
    drift_mode="Absolute",
    rebalance_action="Full",
    cooldown_days=0,
    calendar_freq=None,
    enable_calendar=False,
    enable_threshold=True,
    whole_shares=False,
):
    """
    Combined calendar + threshold (drift-band) rebalancing engine.

    Returns: (rebal_daily, rebal_stats, event_log_df, drift_history)
    """
    tickers = list(target_weights.keys())
    dates = prices_wide.index.tolist()
    n_days = len(dates)
    if n_days == 0:
        raise ValueError("No trading dates in the filtered price data.")

    # Calendar rebalance dates
    calendar_dates = set()
    if enable_calendar and calendar_freq and calendar_freq != "None":
        calendar_dates = _get_rebalance_dates(dates, calendar_freq)

    # Initialize shares at Day 0
    shares = {}
    for tk in tickers:
        alloc = initial_capital * target_weights[tk]
        p0 = prices_wide.loc[dates[0], tk]
        if whole_shares:
            shares[tk] = int(alloc // p0) if p0 > 0 else 0
        else:
            shares[tk] = alloc / p0 if p0 > 0 else 0.0

    portfolio_values = np.empty(n_days, dtype=np.float64)
    ticker_values_arr = {tk: np.empty(n_days, dtype=np.float64) for tk in tickers}
    drift_history = {tk: [] for tk in tickers}
    event_log = []

    rebalance_count = 0
    calendar_rebal_count = 0
    threshold_rebal_count = 0
    total_turnover_dollars = 0.0
    cooldown_remaining = 0
    pending_threshold_breach = False
    pending_breached_tickers = []
    pending_max_drift = 0.0

    for i, dt in enumerate(dates):
        prices_today = {tk: float(prices_wide.loc[dt, tk]) for tk in tickers}
        total_value = sum(shares[tk] * prices_today[tk] for tk in tickers)
        portfolio_values[i] = total_value
        for tk in tickers:
            ticker_values_arr[tk][i] = shares[tk] * prices_today[tk]

        # Compute drift for diagnostics (every day)
        current_weights = compute_weights(shares, prices_today)
        drift = compute_drift(current_weights, target_weights, drift_mode)
        for tk in tickers:
            drift_history[tk].append(drift.get(tk, 0.0))

        did_rebalance_today = False
        rebal_reasons = []

        # 1. Execute pending threshold rebalance (breach detected yesterday)
        if pending_threshold_breach and enable_threshold and i > 0:
            if cooldown_remaining <= 0 and total_value > 0:
                if rebalance_action == "Full":
                    new_shares, turnover = apply_rebalance_full(
                        shares, target_weights, prices_today, total_value, whole_shares)
                else:
                    new_shares, turnover = apply_rebalance_partial(
                        shares, target_weights, tolerances, pending_breached_tickers,
                        prices_today, total_value, whole_shares)
                shares = new_shares
                total_turnover_dollars += turnover
                rebalance_count += 1
                threshold_rebal_count += 1
                did_rebalance_today = True
                rebal_reasons.append("threshold")
                cooldown_remaining = cooldown_days
                total_value = sum(shares[tk] * prices_today[tk] for tk in tickers)
                portfolio_values[i] = total_value
                for tk in tickers:
                    ticker_values_arr[tk][i] = shares[tk] * prices_today[tk]
                event_log.append({
                    "date": dt, "reason": "threshold",
                    "breached_tickers": ", ".join(pending_breached_tickers),
                    "max_drift": round(pending_max_drift, 6),
                    "turnover_dollars": round(turnover, 2),
                })
            pending_threshold_breach = False
            pending_breached_tickers = []
            pending_max_drift = 0.0

        # 2. Calendar rebalance
        if enable_calendar and dt in calendar_dates and total_value > 0:
            cw_now = compute_weights(shares, prices_today)
            needs_rebal = any(abs(cw_now.get(tk, 0) - target_weights[tk]) > 1e-10 for tk in tickers)
            if needs_rebal:
                new_shares, turnover = apply_rebalance_full(
                    shares, target_weights, prices_today, total_value, whole_shares)
                shares = new_shares
                total_turnover_dollars += turnover
                if not did_rebalance_today:
                    rebalance_count += 1
                calendar_rebal_count += 1
                rebal_reasons.append("calendar")
                total_value = sum(shares[tk] * prices_today[tk] for tk in tickers)
                portfolio_values[i] = total_value
                for tk in tickers:
                    ticker_values_arr[tk][i] = shares[tk] * prices_today[tk]
                reason_str = "+".join(rebal_reasons) if len(rebal_reasons) > 1 else "calendar"
                event_log.append({
                    "date": dt, "reason": reason_str,
                    "breached_tickers": "",
                    "max_drift": round(max(drift.values()) if drift else 0, 6),
                    "turnover_dollars": round(turnover, 2),
                })

        # 3. Check for threshold breach at end of day -> schedule for next trading day
        if enable_threshold and i < n_days - 1:
            cw_post = compute_weights(shares, prices_today)
            drift_post = compute_drift(cw_post, target_weights, drift_mode)
            breached = find_threshold_triggers(drift_post, tolerances)
            if breached and cooldown_remaining <= 0:
                pending_threshold_breach = True
                pending_breached_tickers = breached
                pending_max_drift = max(drift_post[tk] for tk in breached)

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

    # Build output
    avg_port_value = np.mean(portfolio_values)
    turnover_proxy = (total_turnover_dollars / avg_port_value) if avg_port_value > 0 else 0.0
    rebal_daily = pd.DataFrame(index=dates)
    rebal_daily.index.name = "PRICEDATE"
    for tk in tickers:
        rebal_daily[f"{tk} (Thresh)"] = ticker_values_arr[tk]
    rebal_daily["Portfolio Value"] = portfolio_values
    rebal_stats = {
        "rebalance_count": rebalance_count,
        "calendar_rebal_count": calendar_rebal_count,
        "threshold_rebal_count": threshold_rebal_count,
        "turnover_proxy": round(turnover_proxy, 4),
        "final_value": round(portfolio_values[-1], 2),
        "total_return": round(portfolio_values[-1] / initial_capital - 1, 6),
    }
    if event_log:
        event_log_df = pd.DataFrame(event_log)
    else:
        event_log_df = pd.DataFrame(columns=["date", "reason", "breached_tickers", "max_drift", "turnover_dollars"])
    return rebal_daily, rebal_stats, event_log_df, drift_history


# ================================================================
# V4 ADDITION: Enhanced Performance Metrics
# ================================================================

def compute_strategy_metrics(daily_values, initial_capital, benchmark_values=None):
    """
    Compute performance metrics from a daily portfolio value series.
    V4 adds: skewness, kurtosis, avg_drawdown, tracking_error, information_ratio.
    """
    n = len(daily_values)
    if n < 2:
        return {
            "total_return": 0.0, "cagr": 0.0, "annualized_vol": 0.0,
            "sharpe": 0.0, "max_drawdown": 0.0,
            "skewness": 0.0, "kurtosis": 0.0, "avg_drawdown": 0.0,
            "tracking_error": 0.0, "information_ratio": 0.0,
        }
    final = daily_values[-1]
    total_return = final / initial_capital - 1
    years = n / 252.0
    if years > 0 and final > 0 and initial_capital > 0:
        cagr = (final / initial_capital) ** (1 / years) - 1
    else:
        cagr = 0.0
    daily_rets = np.diff(daily_values) / daily_values[:-1]
    ann_vol = np.std(daily_rets, ddof=1) * np.sqrt(252) if len(daily_rets) > 1 else 0.0
    sharpe = (cagr / ann_vol) if ann_vol > 0 else 0.0
    running_max = np.maximum.accumulate(daily_values)
    drawdowns = (daily_values - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    # V4 additions
    skewness = float(sp_stats.skew(daily_rets)) if len(daily_rets) > 2 else 0.0
    kurtosis = float(sp_stats.kurtosis(daily_rets, fisher=True)) if len(daily_rets) > 3 else 0.0
    avg_drawdown = float(np.mean(drawdowns))

    tracking_error = 0.0
    information_ratio = 0.0
    if benchmark_values is not None and len(benchmark_values) == n:
        bm_rets = np.diff(benchmark_values) / benchmark_values[:-1]
        active_rets = daily_rets - bm_rets
        tracking_error = float(np.std(active_rets, ddof=1) * np.sqrt(252)) if len(active_rets) > 1 else 0.0
        if tracking_error > 1e-12:
            ann_active_mean = float(np.mean(active_rets) * 252)
            information_ratio = ann_active_mean / tracking_error

    return {
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6),
        "annualized_vol": round(ann_vol, 6),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 6),
        "skewness": round(skewness, 4),
        "kurtosis": round(kurtosis, 4),
        "avg_drawdown": round(avg_drawdown, 6),
        "tracking_error": round(tracking_error, 6),
        "information_ratio": round(information_ratio, 4),
    }


# ================================================================
# Chart column-name sanitizer (Altair/Vega cannot parse &, $, (), :)
# ================================================================

def _safe_chart_cols(chart_df):
    """Return a copy of chart_df with Vega-safe column names."""
    out = chart_df.copy()
    out.columns = [
        c.replace(" ", "_")
         .replace("&", "and")
         .replace("(", "")
         .replace(")", "")
         .replace("$", "USD")
         .replace(":", "")
         .replace("/", "_")
        for c in out.columns
    ]
    return out


# ================================================================
#  LOAD DATA
# ================================================================

_REQUIRED_COLS = ["TRADINGITEMID", "TICKERSYMBOL", "PRICEDATE", "PRICECLOSE", "PRICEMID", "TRADINGITEMSTATUSID"]


@st.cache_data(show_spinner=True)
def load_data():
    ensure_data()
    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        st.stop()
    if DATA_PATH.stat().st_size < 10 * 1024 * 1024:
        st.error("Downloaded file looks too small (likely corrupt). Reboot app to retry.")
        st.stop()
    try:
        df = pd.read_parquet(DATA_PATH, columns=_REQUIRED_COLS)
    except Exception as e:
        st.error(f"Failed to read parquet: {e}")
        st.stop()
    df = prepare_price_data(df, price_field="PRICECLOSE")
    return df


df = load_data()
available_tickers = sorted(df["TICKERSYMBOL"].astype(str).str.strip().str.upper().unique())

# ================================================================
#  SIDEBAR -- USER INPUTS
# ================================================================

st.sidebar.title("\u2699\ufe0f Portfolio Settings")

st.sidebar.markdown("### Holdings")
st.sidebar.caption("Add tickers and their portfolio weights (must sum to ~1.0).")

num_holdings = st.sidebar.number_input(
    "Number of holdings", min_value=1, max_value=20, value=3, step=1
)

ticker_inputs = []
weight_inputs = []

defaults = [
    ("SPY", 0.50), ("AGG", 0.30), ("QQQ", 0.20),
    ("AAPL", 0.00), ("BND", 0.00),
]

for i in range(int(num_holdings)):
    cols = st.sidebar.columns([2, 1])
    default_tk = defaults[i][0] if i < len(defaults) else ""
    default_wt = defaults[i][1] if i < len(defaults) else 0.0
    default_idx = available_tickers.index(default_tk) if default_tk in available_tickers else 0
    tk = cols[0].selectbox(
        f"Ticker {i+1}", options=available_tickers,
        index=default_idx, key=f"tk_{i}",
    )
    wt = cols[1].number_input(
        f"Weight", min_value=0.0, max_value=1.0, value=default_wt,
        step=0.05, key=f"wt_{i}", format="%.2f",
    )
    ticker_inputs.append(tk)
    weight_inputs.append(wt)

st.sidebar.markdown("---")
st.sidebar.markdown("### Parameters")

date_cols = st.sidebar.columns(2)
df_dates = pd.to_datetime(df["PRICEDATE"], errors="coerce").dropna()
min_date = df_dates.min().date()
max_date = df_dates.max().date()
default_end = max_date
default_start = max(min_date, (max_date - timedelta(days=365)))

start_date = date_cols[0].date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
end_date = date_cols[1].date_input("End Date", value=default_end, min_value=min_date, max_value=max_date)

initial_capital = st.sidebar.number_input(
    "Initial Capital ($)", min_value=1_000, max_value=100_000_000,
    value=100_000, step=10_000, format="%d",
)
price_field = st.sidebar.selectbox("Price Field", ["PRICECLOSE", "PRICEMID"])
allow_cash = st.sidebar.checkbox("Whole shares only (cash residual)", value=False)

# ================================================================
# V4: Universal Page-Level Tax Parameters
# ================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### \U0001f3db\ufe0f Tax Parameters")
st.sidebar.caption("Universal tax rates applied across all strategies.")

global_st_rate = st.sidebar.number_input(
    "Short-Term Tax Rate (%)", min_value=0.0, max_value=60.0,
    value=35.0, step=1.0, format="%.1f", key="global_st_rate"
) / 100.0
global_lt_rate = st.sidebar.number_input(
    "Long-Term Tax Rate (%)", min_value=0.0, max_value=40.0,
    value=20.0, step=1.0, format="%.1f", key="global_lt_rate"
) / 100.0
global_tax_rates = {"st_rate": global_st_rate, "lt_rate": global_lt_rate}

# ================================================================
# Rebalancing controls
# ================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### Rebalancing")

enable_rebalancing = st.sidebar.checkbox("Enable Rebalancing Comparison", value=True)

if enable_rebalancing:
    enable_calendar_rebal = st.sidebar.checkbox("Enable Calendar Rebalancing", value=True)
else:
    enable_calendar_rebal = False

REBAL_FREQS = ["Daily", "Weekly", "Monthly", "Quarterly"]
selected_freq = st.sidebar.selectbox(
    "Rebalance Frequency",
    options=REBAL_FREQS,
    index=2,
    disabled=not (enable_rebalancing and enable_calendar_rebal),
)

show_all_strategies = st.sidebar.checkbox(
    "Show all calendar strategies (slower)",
    value=False,
    disabled=not (enable_rebalancing and enable_calendar_rebal),
    help="Compute & compare Buy-and-Hold + all 4 calendar rebalance frequencies at once.",
)

### THRESHOLD REBALANCE ADDITIONS -- Sidebar Controls ###
if enable_rebalancing:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### \U0001f4cf Threshold Rebalancing")
    enable_threshold_rebal = st.sidebar.checkbox(
        "Enable Threshold (Drift-Band) Rebalancing", value=False,
        help="Trigger rebalance when any asset drifts beyond its tolerance band."
    )
else:
    enable_threshold_rebal = False

if enable_threshold_rebal:
    drift_mode = st.sidebar.selectbox(
        "Drift Mode", ["Absolute", "Relative"],
        help="Absolute: |current_weight - target_weight|. Relative: |current_weight / target_weight - 1|.",
    )
    rebalance_action = st.sidebar.selectbox(
        "Rebalance Action", ["Full", "Partial"],
        help="Full: rebalance ALL assets to target. Partial: only trade breached assets, scale others.",
    )
    default_tolerance_pct = st.sidebar.slider(
        "Default Drift Tolerance (%)", min_value=0.5, max_value=20.0,
        value=5.0, step=0.5, key="thresh_tol",
        help="Default tolerance for all assets. Override per-asset in the main panel.",
    )
    cooldown_days = st.sidebar.number_input(
        "Cooldown (trading days)", min_value=0, max_value=60, value=0, step=1,
        help="Suppress additional threshold triggers for N days after a threshold rebalance.",
    )
else:
    drift_mode = "Absolute"
    rebalance_action = "Full"
    default_tolerance_pct = 5.0
    cooldown_days = 0

# ================================================================
# MSBA v1 OPTIMIZER SIDEBAR
# ================================================================
if _OPTIMIZER_AVAILABLE:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### \U0001f9e0 Optimizer MSBA v1")
    enable_optimizer = st.sidebar.toggle("Enable Optimizer MSBA v1", value=False)
    if enable_optimizer:
        opt_tlh_threshold = st.sidebar.number_input(
            "TLH Loss Threshold (%)", min_value=0.0, max_value=50.0,
            value=5.0, step=0.5, format="%.1f", key="opt_tlh",
            help="Harvest tax lots that are down by at least this %"
        ) / 100.0
        opt_div_handling = st.sidebar.selectbox(
            "Dividend Handling",
            ["Reinvest dividends", "Keep dividends as cash"],
            key="opt_div",
        )
    else:
        opt_tlh_threshold = 0.05
        opt_div_handling = "Reinvest dividends"
else:
    enable_optimizer = False

run_btn = st.sidebar.button("\U0001f680 Calculate Returns", use_container_width=True, type="primary")


# ================================================================
#  MAIN PAGE
# ================================================================

if _STYLE_LOADED:
    render_hero(
        eyebrow="UTexas MSBA // VISE",
        title='\U0001f4ca Portfolio Returns<br><em>Calculator</em>',
        subtitle="Price-based returns engine with tax-aware optimizer.",
        formula='Portfolio Value &nbsp;=&nbsp; <span>(Shares \u00d7 Price)</span> &nbsp;+&nbsp; Cash',
    )
else:
    st.title("\U0001f4ca Portfolio Returns Calculator")
    st.caption("Price-based returns engine")

if not run_btn:
    st.info("\U0001f448 Configure your portfolio in the sidebar and press **Calculate Returns**.")
    st.stop()

weight_sum = sum(weight_inputs)
if weight_sum == 0:
    st.error("All weights are zero. Please assign weights to at least one ticker.")
    st.stop()

try:
    summary, holdings = calculate_portfolio_returns(
        df=df, tickers=ticker_inputs, weights=weight_inputs,
        start_date=str(start_date), end_date=str(end_date),
        initial_capital=float(initial_capital), price_field=price_field,
        allow_cash_residual=allow_cash,
    )
except ValueError as e:
    st.error(f"**Error:** {e}")
    st.stop()

if summary["tickers_dropped"] > 0:
    for tk, w, reason in summary["dropped_details"]:
        st.warning(f"\u26a0\ufe0f Dropped **{tk}** (weight {w:.2%}): {reason}")

# KPI Cards (Buy-and-Hold)
total_return = summary["portfolio_total_return"]
gain_dollars = summary["total_unrealized_gain_dollars"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Starting Value", f"${summary['portfolio_start_value']:,.0f}")
col2.metric("Ending Value", f"${summary['portfolio_end_value']:,.0f}")
col3.metric("Total Return", f"{total_return:+.2%}", delta=f"${gain_dollars:+,.0f}")
col4.metric("Unrealized Gain", f"${gain_dollars:+,.0f}", delta=f"{summary['total_unrealized_gain_pct']:+.2%}")

st.markdown("---")

# Charts (Buy-and-Hold)
daily = build_daily_series(df, holdings, float(initial_capital), price_field)
tickers_used = holdings["Ticker"].tolist()

st.subheader("Portfolio Value vs Cost Basis")
st.caption("Green shading = unrealized gain \u00b7 Red shading = unrealized loss")
chart1_df = daily[["Portfolio Value", "Cost Basis"]].copy()
st.line_chart(chart1_df, color=["#1a73e8", "#888888"], use_container_width=True, height=420)

gain_area = daily[["Portfolio Value", "Cost Basis"]].copy()
gain_area["Unrealized Gain/Loss ($)"] = gain_area["Portfolio Value"] - gain_area["Cost Basis"]
st.area_chart(
    _safe_chart_cols(gain_area[["Unrealized Gain/Loss ($)"]]),
    color=["#34a853"] if gain_dollars >= 0 else ["#ea4335"],
    use_container_width=True, height=200,
)

st.subheader("Per-Ticker Cumulative Return (%)")
return_cols = [f"{tk} Return (%)" for tk in tickers_used]
st.line_chart(_safe_chart_cols(daily[return_cols]), use_container_width=True, height=350)

st.markdown("---")


# ================================================================
# REBALANCING COMPARISON SECTION (Calendar + Threshold)
# ================================================================

if enable_rebalancing:
    st.subheader("\U0001f504 Rebalancing Strategy Comparison")
    target_weights = {row["Ticker"]: row["Weight"] for _, row in holdings.iterrows()}

    try:
        all_start = pd.to_datetime(holdings["Start Date"]).min()
        all_end = pd.to_datetime(holdings["End Date"]).max()
        prices_wide = build_prices_wide(df, tickers_used, all_start, all_end, price_field)
    except ValueError as e:
        st.error(f"**Error building price matrix:** {e}")
        st.stop()

    ### Per-asset tolerance UI ###
    tolerances = {tk: default_tolerance_pct / 100.0 for tk in tickers_used}
    if enable_threshold_rebal:
        with st.expander("\U0001f4cf Advanced: Per-Asset Drift Tolerances"):
            st.caption("Override the default tolerance for individual tickers.")
            tol_cols = st.columns(min(len(tickers_used), 4))
            for idx, tk in enumerate(tickers_used):
                col_idx = idx % min(len(tickers_used), 4)
                tol_val = tol_cols[col_idx].number_input(
                    f"{tk} tol (%)", min_value=0.5, max_value=50.0,
                    value=default_tolerance_pct, step=0.5,
                    key=f"tol_{tk}", format="%.1f",
                )
                tolerances[tk] = tol_val / 100.0

    # Compute strategies
    strategy_results = {}
    event_logs = {}
    drift_histories = {}

    # Calendar strategies
    if enable_calendar_rebal:
        if show_all_strategies:
            freqs_to_run = REBAL_FREQS
            st.caption("Computing: Buy & Hold + all calendar frequencies" +
                       (" + Threshold" if enable_threshold_rebal else "") + "...")
        else:
            freqs_to_run = [selected_freq]
            st.caption(f"Comparing Buy & Hold vs **{selected_freq}** rebalancing" +
                       (" + Threshold" if enable_threshold_rebal else "") + ".")
        for freq in freqs_to_run:
            try:
                rd, rs = build_rebalanced_series(prices_wide, target_weights, float(initial_capital), freq)
                strategy_results[f"Rebal: {freq}"] = (rd, rs)
            except ValueError as e:
                st.warning(f"\u26a0\ufe0f Could not compute {freq} rebalancing: {e}")
    else:
        freqs_to_run = []
        if enable_threshold_rebal:
            st.caption("Comparing Buy & Hold vs **Threshold** rebalancing (no calendar).")

    # Threshold strategy
    if enable_threshold_rebal:
        try:
            thresh_rd, thresh_rs, thresh_log, thresh_drift = build_threshold_rebalanced_series(
                prices_wide=prices_wide, target_weights=target_weights,
                initial_capital=float(initial_capital), tolerances=tolerances,
                drift_mode=drift_mode, rebalance_action=rebalance_action,
                cooldown_days=cooldown_days,
                calendar_freq=selected_freq if enable_calendar_rebal else None,
                enable_calendar=enable_calendar_rebal, enable_threshold=True,
                whole_shares=allow_cash,
            )
            combo_label = "Threshold" if not enable_calendar_rebal else f"Cal({selected_freq})+Thresh"
            strategy_results[combo_label] = (thresh_rd, thresh_rs)
            event_logs[combo_label] = thresh_log
            drift_histories[combo_label] = thresh_drift
        except ValueError as e:
            st.warning(f"\u26a0\ufe0f Could not compute threshold rebalancing: {e}")

    if not strategy_results and not enable_threshold_rebal and not enable_calendar_rebal:
        st.info("Enable at least one rebalancing strategy to see comparison results.")
    elif strategy_results:
        # Build comparison DataFrame
        comparison_df = pd.DataFrame(index=prices_wide.index)
        comparison_df.index.name = "PRICEDATE"
        bh_values = daily["Portfolio Value"].reindex(comparison_df.index)
        comparison_df["Buy & Hold"] = bh_values
        for label, (rd, rs) in strategy_results.items():
            comparison_df[label] = rd["Portfolio Value"].reindex(comparison_df.index)
        comparison_df = comparison_df.dropna()

        # Enhanced Metrics table
        bh_vals_arr = comparison_df["Buy & Hold"].values
        bh_metrics = compute_strategy_metrics(bh_vals_arr, float(initial_capital), benchmark_values=None)
        bh_metrics["rebalance_count"] = 0
        bh_metrics["turnover_proxy"] = 0.0

        metrics_rows = [{
            "Strategy": "Buy & Hold",
            "Final Value ($)": f"${bh_vals_arr[-1]:,.0f}",
            "Total Return": f"{bh_metrics['total_return']:+.2%}",
            "CAGR": f"{bh_metrics['cagr']:+.2%}",
            "Ann. Vol": f"{bh_metrics['annualized_vol']:.2%}",
            "Sharpe": f"{bh_metrics['sharpe']:.3f}",
            "Max DD": f"{bh_metrics['max_drawdown']:.2%}",
            "Avg DD": f"{bh_metrics['avg_drawdown']:.2%}",
            "Skew": f"{bh_metrics['skewness']:.3f}",
            "Kurt": f"{bh_metrics['kurtosis']:.3f}",
            "TE": "\u2014",
            "IR": "\u2014",
            "Turnover": "0.00",
            "Events": 0,
        }]

        for label in strategy_results:
            rd, rs = strategy_results[label]
            vals = comparison_df[label].values
            m = compute_strategy_metrics(vals, float(initial_capital), benchmark_values=bh_vals_arr)
            metrics_rows.append({
                "Strategy": label,
                "Final Value ($)": f"${rs['final_value']:,.0f}",
                "Total Return": f"{m['total_return']:+.2%}",
                "CAGR": f"{m['cagr']:+.2%}",
                "Ann. Vol": f"{m['annualized_vol']:.2%}",
                "Sharpe": f"{m['sharpe']:.3f}",
                "Max DD": f"{m['max_drawdown']:.2%}",
                "Avg DD": f"{m['avg_drawdown']:.2%}",
                "Skew": f"{m['skewness']:.3f}",
                "Kurt": f"{m['kurtosis']:.3f}",
                "TE": f"{m['tracking_error']:.4f}",
                "IR": f"{m['information_ratio']:.3f}",
                "Turnover": f"{rs['turnover_proxy']:.2f}",
                "Events": rs["rebalance_count"],
            })

        st.markdown("#### Performance Metrics")
        metrics_df = pd.DataFrame(metrics_rows)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # KPI cards
        primary_label = list(strategy_results.keys())[0]
        _, primary_stats = strategy_results[primary_label]
        bh_final = bh_vals_arr[-1]
        rb_final = primary_stats["final_value"]
        rb_return = primary_stats["total_return"]
        bh_ret = bh_metrics["total_return"]

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric(f"{primary_label} Final", f"${rb_final:,.0f}", delta=f"{rb_return:+.2%}")
        rc2.metric("Buy-and-Hold Final", f"${bh_final:,.0f}", delta=f"{bh_ret:+.2%}")
        rc3.metric("Strategy Advantage", f"${rb_final - bh_final:+,.0f}", delta=f"{(rb_return - bh_ret):+.4%}")
        rc4.metric("Rebalance Events", f"{primary_stats['rebalance_count']:,}", delta=f"Turnover: {primary_stats['turnover_proxy']:.2f}")

        # Chart: Portfolio value over time — with strategy toggle (Issue 4)
        st.markdown("#### Portfolio Value Over Time")
        freq_color_map = {"Daily": "#e8710a", "Weekly": "#34a853", "Monthly": "#9c27b0", "Quarterly": "#ea4335"}
        threshold_color = "#ffab00"

        # Build color map for all columns
        _all_strat_labels = list(comparison_df.columns)  # "Buy & Hold" + strategy labels
        _color_map = {}
        _color_map["Buy & Hold"] = "#1a73e8"
        for label in strategy_results:
            if label.startswith("Rebal:"):
                freq_key = label.replace("Rebal: ", "")
                _color_map[label] = freq_color_map.get(freq_key, "#666666")
            else:
                _color_map[label] = threshold_color

        # Multiselect for visible strategies
        selected_strats = st.multiselect(
            "Strategies to display",
            options=_all_strat_labels,
            default=_all_strat_labels,
            key="value_chart_strats",
        )

        if selected_strats:
            _chart_df = comparison_df[selected_strats]
            _chart_colors = [_color_map.get(s, "#666666") for s in selected_strats]
            st.line_chart(_safe_chart_cols(_chart_df), color=_chart_colors, use_container_width=True, height=420)
        else:
            st.info("Select at least one strategy to display.")

        # Also build full strategy_colors list for drawdown chart (uses all strategies)
        strategy_colors = [_color_map.get(c, "#666666") for c in comparison_df.columns]

        # Drawdown chart
        show_drawdown = st.checkbox("Show Drawdown Chart", value=False)
        if show_drawdown:
            st.markdown("#### Drawdown Over Time")
            dd_df = pd.DataFrame(index=comparison_df.index)
            for col in comparison_df.columns:
                vals = comparison_df[col].values
                rm = np.maximum.accumulate(vals)
                dd_df[col] = ((vals - rm) / rm) * 100
            st.area_chart(_safe_chart_cols(dd_df), color=strategy_colors, use_container_width=True, height=300)

        # Difference chart
        if primary_label in comparison_df.columns:
            diff_series = comparison_df[primary_label] - comparison_df["Buy & Hold"]
            diff_chart = pd.DataFrame({f"{primary_label} vs B&H ($)": diff_series})
            advantage_color = "#34a853" if diff_series.iloc[-1] >= 0 else "#ea4335"
            st.area_chart(_safe_chart_cols(diff_chart), color=[advantage_color], use_container_width=True, height=200)

        ### DRIFT DIAGNOSTICS ###
        if drift_histories:
            st.markdown("---")
            st.markdown("#### \U0001f4ca Drift Diagnostics")
            drift_strategy_options = list(drift_histories.keys())

            # Issue 2: persist selections via session_state
            if "drift_strat_idx" not in st.session_state:
                st.session_state["drift_strat_idx"] = 0
            if "drift_ticker_idx" not in st.session_state:
                st.session_state["drift_ticker_idx"] = 0

            # Clamp indices to valid range (strategies/tickers may change between runs)
            _strat_idx = min(st.session_state["drift_strat_idx"], len(drift_strategy_options) - 1)
            _ticker_idx = min(st.session_state["drift_ticker_idx"], len(tickers_used) - 1)

            selected_drift_strategy = st.selectbox(
                "Select strategy for drift analysis",
                drift_strategy_options,
                index=_strat_idx,
                key="drift_strat_select",
            )
            st.session_state["drift_strat_idx"] = drift_strategy_options.index(selected_drift_strategy)

            dh = drift_histories[selected_drift_strategy]

            drift_ticker_select = st.selectbox(
                "Select ticker for drift distribution",
                tickers_used,
                index=_ticker_idx,
                key="drift_ticker_select",
            )
            st.session_state["drift_ticker_idx"] = tickers_used.index(drift_ticker_select)

            drift_values = np.array(dh[drift_ticker_select])
            if len(drift_values) > 0:
                # Issue 1: show stats in percent
                drift_pct = drift_values * 100.0
                tol_for_tk = tolerances.get(drift_ticker_select, default_tolerance_pct / 100.0)
                breach_pct = np.mean(drift_values > tol_for_tk) * 100

                ds1, ds2, ds3, ds4 = st.columns(4)
                ds1.metric(f"Mean Drift ({drift_mode})", f"{np.mean(drift_pct):.2f}%")
                ds2.metric("P95 Drift", f"{np.percentile(drift_pct, 95):.2f}%")
                ds3.metric("Max Drift", f"{np.max(drift_pct):.2f}%")
                ds4.metric("Days Breached (%)", f"{breach_pct:.1f}%")

                # Issue 1: proper histogram with bins in percent
                n_bins = min(30, max(15, len(drift_pct) // 15))
                counts, bin_edges = np.histogram(drift_pct, bins=n_bins)
                # Build labels as bin ranges: "0.00-0.25"
                bin_labels = [
                    f"{bin_edges[j]:.2f}-{bin_edges[j+1]:.2f}"
                    for j in range(len(counts))
                ]
                hist_df = pd.DataFrame({
                    "Drift_pct": bin_labels,
                    "Days_count": counts,
                }).set_index("Drift_pct")
                st.bar_chart(hist_df, use_container_width=True, height=250)
                st.caption(
                    f"Distribution of daily {drift_mode.lower()} drift (%) for **{drift_ticker_select}** "
                    f"under **{selected_drift_strategy}**. Tolerance = {tol_for_tk:.2%}."
                )

            show_drift_ts = st.checkbox("Show drift time series (all tickers)", value=False)
            if show_drift_ts:
                # Convert to percent for display
                drift_ts_data = {tk: np.array(vals) * 100.0 for tk, vals in dh.items()}
                drift_ts_df = pd.DataFrame(drift_ts_data, index=prices_wide.index[:len(list(dh.values())[0])])
                drift_ts_df.index.name = "PRICEDATE"
                st.line_chart(drift_ts_df, use_container_width=True, height=300)
                st.caption(f"Daily {drift_mode.lower()} drift (%) per ticker under **{selected_drift_strategy}**.")

        ### EVENT LOG ###
        if event_logs:
            with st.expander("\U0001f4cb Rebalance Event Log"):
                for label, log_df in event_logs.items():
                    st.markdown(f"**{label}**")
                    if not log_df.empty:
                        st.dataframe(log_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No rebalance events triggered.")

        # Methodology expander
        with st.expander("\u2139\ufe0f Rebalancing Methodology"):
            st.markdown("""
**Calendar Rebalancing:** At each scheduled date, portfolio is valued and rebalanced to target weights using closing prices.

**Calendar Schedules:** Daily (every trading day), Weekly (first day of ISO week), Monthly (first day of month), Quarterly (first day of Jan/Apr/Jul/Oct).

**Threshold (Drift-Band) Rebalancing:** At end of each day, per-asset drift is computed. If ANY asset breaches its tolerance, rebalance executes on the NEXT trading day.
- **Absolute drift:** |current_weight - target_weight|
- **Relative drift:** |current_weight / target_weight - 1|
- **Full rebalance:** all assets return to exact targets
- **Partial rebalance:** only breached assets return to target; others scale proportionally
- **Cooldown:** suppresses further threshold triggers for N days after a threshold rebalance

**Calendar + Threshold Combination:** Both active simultaneously. Neither suppresses the other.

**Enhanced Metrics:** Skewness, Kurtosis, Avg Drawdown, Tracking Error vs B&H, Information Ratio vs B&H.

**Limitations:** No transaction costs/slippage. Tax parameters threaded through but applied only in MSBA v1 optimizer currently.
            """)

    st.markdown("---")


# ================================================================
# MSBA v1 OPTIMIZER SECTION
# ================================================================
if enable_optimizer:
    st.subheader("\U0001f9e0 Optimizer MSBA v1 \u2014 Tax-Aware Simulation")

    _div_df = None
    try:
        import os
        _div_path = os.path.join(os.path.dirname(__file__), "dividend_data.csv")
        if os.path.exists(_div_path):
            _div_df = pd.read_csv(_div_path)
            _div_df["PAYDATE"] = pd.to_datetime(_div_df["PAYDATE"], errors="coerce")
            _div_df["EXDATE"] = pd.to_datetime(_div_df["EXDATE"], errors="coerce")
            if "TICKERSYMBOL" not in _div_df.columns:
                if "TRADINGITEMID" in _div_df.columns and "TRADINGITEMID" in df.columns:
                    _ticker_map = (
                        df[["TRADINGITEMID", "TICKERSYMBOL"]]
                        .drop_duplicates()
                        .set_index("TRADINGITEMID")["TICKERSYMBOL"]
                        .to_dict()
                    )
                    _div_df["TICKERSYMBOL"] = _div_df["TRADINGITEMID"].map(_ticker_map)
                    _div_df = _div_df.dropna(subset=["TICKERSYMBOL"])
    except Exception:
        _div_df = None

    _opt_tax_rates = global_tax_rates  # V4: use universal tax rates
    _opt_reinvest = opt_div_handling == "Reinvest dividends"
    _opt_tickers = holdings["Ticker"].tolist()
    _opt_weights = holdings["Weight"].tolist()
    _opt_rebal_freq = selected_freq if enable_calendar_rebal else "None"

    with st.spinner("Running MSBA v1 Static simulation..."):
        try:
            static_result = run_optimizer_simulation(
                prices_df=df, dividends_df=_div_df,
                tickers=_opt_tickers, weights=_opt_weights,
                start_date=str(start_date), end_date=str(end_date),
                rebalance_frequency=_opt_rebal_freq,
                tax_rates=_opt_tax_rates, tlh_threshold=opt_tlh_threshold,
                reinvest_dividends=_opt_reinvest,
                initial_capital=float(initial_capital),
                price_field=price_field, static=True,
            )
        except Exception as e:
            st.error(f"MSBA v1 Static simulation failed: {e}")
            static_result = None

    with st.spinner("Running MSBA v1 Optimized simulation..."):
        try:
            opt_result = run_optimizer_simulation(
                prices_df=df, dividends_df=_div_df,
                tickers=_opt_tickers, weights=_opt_weights,
                start_date=str(start_date), end_date=str(end_date),
                rebalance_frequency=_opt_rebal_freq,
                tax_rates=_opt_tax_rates, tlh_threshold=opt_tlh_threshold,
                reinvest_dividends=_opt_reinvest,
                initial_capital=float(initial_capital),
                price_field=price_field, static=False,
            )
        except Exception as e:
            st.error(f"MSBA v1 Optimized simulation failed: {e}")
            opt_result = None

    if static_result and opt_result:
        s_nav = static_result["nav_series"]
        o_nav = opt_result["nav_series"]
        s_final = s_nav.iloc[-1]
        o_final = o_nav.iloc[-1]
        cap = float(initial_capital)

        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric("Static Final NAV", f"${s_final:,.0f}", delta=f"{(s_final/cap - 1):+.2%}")
        kc2.metric("Optimized Final NAV", f"${o_final:,.0f}", delta=f"{(o_final/cap - 1):+.2%}")
        kc3.metric("Optimizer Advantage", f"${o_final - s_final:+,.0f}", delta=f"{((o_final - s_final)/cap):+.4%}")
        kc4.metric("Total Tax Paid (Opt)", f"${opt_result['tax_paid_total']:,.0f}", delta=f"Static: ${static_result['tax_paid_total']:,.0f}")

        st.markdown("#### MSBA v1 \u2014 Portfolio NAV Over Time")
        opt_chart = pd.DataFrame({"Static (TLH only)": s_nav, "Optimized (Rebal + TLH)": o_nav}).dropna()
        st.line_chart(_safe_chart_cols(opt_chart), color=["#888888", "#e8710a"], use_container_width=True, height=400)

        if len(opt_chart) > 0:
            diff = opt_chart["Optimized (Rebal + TLH)"] - opt_chart["Static (TLH only)"]
            adv_color = "#34a853" if diff.iloc[-1] >= 0 else "#ea4335"
            st.area_chart(_safe_chart_cols(pd.DataFrame({"Optimizer Advantage ($)": diff})), color=[adv_color], use_container_width=True, height=200)

        with st.expander("\U0001f4cb Optimized Portfolio \u2014 Trade Log"):
            _tdf = opt_result["trades_df"]
            if not _tdf.empty:
                st.dataframe(_tdf, use_container_width=True, hide_index=True)
            else:
                st.info("No trades recorded.")

        with st.expander("\U0001f4cb Optimized Portfolio \u2014 Realized Gains"):
            _rdf = opt_result["realized_df"]
            if not _rdf.empty:
                st.dataframe(_rdf, use_container_width=True, hide_index=True)
            else:
                st.info("No realized gains/losses.")

        with st.expander("\u2139\ufe0f MSBA v1 Methodology"):
            st.markdown(f"""
**Optimizer MSBA v1** runs two parallel tax-aware simulations:

- **Static Portfolio**: Buy at start, dividends handled per settings, TLH active, no rebalancing.
- **Optimized Portfolio**: Scheduled rebalancing ({_opt_rebal_freq}) + TLH + tax-aware lot disposal.

**Tax-Loss Harvesting**: Any lot down >= {opt_tlh_threshold:.1%} is sold and immediately repurchased.

**Tax Rates**: ST = {global_st_rate:.0%} / LT = {global_lt_rate:.0%} (universal page-level parameters)

**Lot Disposal**: TAX_OPTIMAL -- sells loss lots first (largest ST loss first), then smallest gains.

**Dividends**: {'Reinvested (DRIP)' if _opt_reinvest else 'Kept as cash'}.
            """)

    st.markdown("---")

# ================================================================
#  HOLDINGS TABLE
# ================================================================

st.subheader("Per-Holding Detail")

display_df = holdings.copy()
display_df["Weight"] = display_df["Weight"].apply(lambda x: f"{x:.1%}")
display_df["Return"] = display_df["Return"].apply(lambda x: f"{x:+.2%}")
display_df["Gain (%)"] = display_df["Gain (%)"].apply(lambda x: f"{x:+.2%}")
display_df["Gain ($)"] = display_df["Gain ($)"].apply(lambda x: f"${x:+,.2f}")
display_df["Start Value"] = display_df["Start Value"].apply(lambda x: f"${x:,.2f}")
display_df["End Value"] = display_df["End Value"].apply(lambda x: f"${x:,.2f}")
display_df["Start Price"] = display_df["Start Price"].apply(lambda x: f"${x:.2f}")
display_df["End Price"] = display_df["End Price"].apply(lambda x: f"${x:.2f}")

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ================================================================
#  ASSUMPTIONS EXPANDER
# ================================================================

with st.expander("\u2139\ufe0f Assumptions & Methodology"):
    st.markdown("""
**TRADINGITEMSTATUSID**: Keeps rows where status is `1` or `15`.

**Date Shifting**: Start -> first trading day on/after; End -> last trading day on/before.

**Dividends / Splits**: Not implemented in base engine. Price-based returns only.

**Duplicate Tickers**: Weights are automatically summed.

**Fractional Shares**: Allowed by default. Toggle "Whole shares only" for integer shares.

**Tax Parameters**: Universal page-level ST/LT tax rates threaded through all strategies. Currently applied in MSBA v1 optimizer; placeholder for future calendar/threshold integration.
    """)