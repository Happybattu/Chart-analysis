"""Support & resistance level detection using multiple methods."""

import numpy as np
import pandas as pd


def swing_levels(df: pd.DataFrame, lookback: int = 5, n_levels: int = 4):
    highs, lows = [], []
    h, l = df["High"].values, df["Low"].values
    for i in range(lookback, len(df) - lookback):
        window_h = h[i - lookback:i + lookback + 1]
        window_l = l[i - lookback:i + lookback + 1]
        if h[i] == window_h.max():
            highs.append(h[i])
        if l[i] == window_l.min():
            lows.append(l[i])

    def cluster(levels, tol=0.02):
        levels = sorted(levels)
        clusters = []
        for lvl in levels:
            if clusters and abs(lvl - clusters[-1][-1]) / clusters[-1][-1] < tol:
                clusters[-1].append(lvl)
            else:
                clusters.append([lvl])
        return [round(float(np.mean(c)), 2) for c in clusters]

    res = sorted(cluster(highs), reverse=True)[:n_levels]
    sup = sorted(cluster(lows), reverse=True)[:n_levels]
    return sorted(res), sorted(sup, reverse=True)


def classic_pivot_points(df: pd.DataFrame) -> dict:
    """Classic floor-trader pivots from the most recently completed period."""
    prev = df.iloc[-2]
    h, l, c = prev["High"], prev["Low"], prev["Close"]
    pivot = (h + l + c) / 3
    r1 = 2 * pivot - l
    s1 = 2 * pivot - h
    r2 = pivot + (h - l)
    s2 = pivot - (h - l)
    r3 = h + 2 * (pivot - l)
    s3 = l - 2 * (h - pivot)
    return {k: round(v, 2) for k, v in
            {"P": pivot, "R1": r1, "R2": r2, "R3": r3, "S1": s1, "S2": s2, "S3": s3}.items()}


def fibonacci_retracement(df: pd.DataFrame, lookback: int = 120) -> dict:
    window = df.tail(lookback)
    swing_high = float(window["High"].max())
    swing_low = float(window["Low"].min())
    diff = swing_high - swing_low
    levels = {
        "0.0% (High)": swing_high,
        "23.6%": swing_high - 0.236 * diff,
        "38.2%": swing_high - 0.382 * diff,
        "50.0%": swing_high - 0.5 * diff,
        "61.8%": swing_high - 0.618 * diff,
        "78.6%": swing_high - 0.786 * diff,
        "100.0% (Low)": swing_low,
    }
    return {k: round(v, 2) for k, v in levels.items()}


def prev_period_high_low(df: pd.DataFrame) -> dict:
    weekly = df.resample("W").agg({"High": "max", "Low": "min"})
    monthly = df.resample("ME").agg({"High": "max", "Low": "min"})
    out = {}
    if len(weekly) >= 2:
        out["Prev Week High"] = round(float(weekly["High"].iloc[-2]), 2)
        out["Prev Week Low"] = round(float(weekly["Low"].iloc[-2]), 2)
    if len(monthly) >= 2:
        out["Prev Month High"] = round(float(monthly["High"].iloc[-2]), 2)
        out["Prev Month Low"] = round(float(monthly["Low"].iloc[-2]), 2)
    return out


def volume_profile(df: pd.DataFrame, lookback: int = 120, bins: int = 24, top_n: int = 3):
    """Approximate High Volume Nodes by bucketing (High+Low)/2 into price bins weighted by volume."""
    window = df.tail(lookback)
    typical = (window["High"] + window["Low"] + window["Close"]) / 3
    price_min, price_max = typical.min(), typical.max()
    if price_max == price_min:
        return []
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_idx = np.digitize(typical, bin_edges) - 1
    vol_by_bin = pd.Series(0.0, index=range(bins))
    for idx, vol in zip(bin_idx, window["Volume"].values):
        idx = min(max(idx, 0), bins - 1)
        vol_by_bin[idx] += vol
    top_bins = vol_by_bin.sort_values(ascending=False).head(top_n).index
    hvn = [round(float((bin_edges[i] + bin_edges[i + 1]) / 2), 2) for i in top_bins]
    return sorted(hvn)


def gap_levels(df: pd.DataFrame, lookback: int = 90, min_gap_pct: float = 1.5):
    """Detect unfilled price gaps (potential support/resistance)."""
    window = df.tail(lookback).reset_index()
    gaps = []
    for i in range(1, len(window)):
        prev_close = window.loc[i - 1, "Close"]
        today_open = window.loc[i, "Open"]
        gap_pct = (today_open - prev_close) / prev_close * 100
        if abs(gap_pct) >= min_gap_pct:
            gaps.append({
                "date": window.loc[i, window.columns[0]],
                "type": "Gap Up" if gap_pct > 0 else "Gap Down",
                "from": round(float(prev_close), 2),
                "to": round(float(today_open), 2),
            })
    return gaps[-5:]


def anchored_vwap(df: pd.DataFrame, anchor_idx=None) -> pd.Series:
    """VWAP anchored from a specific bar (defaults to lowest low in last 120 bars)."""
    if anchor_idx is None:
        anchor_idx = df["Low"].tail(120).idxmin()
    sub = df.loc[anchor_idx:]
    typical = (sub["High"] + sub["Low"] + sub["Close"]) / 3
    cum_vp = (typical * sub["Volume"]).cumsum()
    cum_vol = sub["Volume"].cumsum()
    avwap = cum_vp / cum_vol
    return avwap.reindex(df.index)
