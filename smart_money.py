"""
Smart Money Concepts (SMC) — simplified retail-friendly implementations.
These follow the common community definitions (not an institutional
data feed), so treat them as visual context, not certainty.
"""

import pandas as pd


def find_fair_value_gaps(df: pd.DataFrame, lookback: int = 60, min_gap_pct: float = 0.3):
    """3-candle imbalance: gap between candle[i-1].high/low and candle[i+1].low/high."""
    window = df.tail(lookback)
    fvgs = []
    idx = window.index
    for i in range(1, len(window) - 1):
        c0, c2 = window.iloc[i - 1], window.iloc[i + 1]
        mid_date = idx[i]
        # Bullish FVG: candle[i-1].high < candle[i+1].low
        if c2["Low"] > c0["High"]:
            gap_pct = (c2["Low"] - c0["High"]) / c0["High"] * 100
            if gap_pct >= min_gap_pct:
                fvgs.append({"date": mid_date, "type": "Bullish FVG",
                              "top": round(float(c2["Low"]), 2), "bottom": round(float(c0["High"]), 2)})
        # Bearish FVG: candle[i-1].low > candle[i+1].high
        if c0["Low"] > c2["High"]:
            gap_pct = (c0["Low"] - c2["High"]) / c2["High"] * 100
            if gap_pct >= min_gap_pct:
                fvgs.append({"date": mid_date, "type": "Bearish FVG",
                              "top": round(float(c0["Low"]), 2), "bottom": round(float(c2["High"]), 2)})
    return fvgs[-5:]


def find_order_blocks(df: pd.DataFrame, lookback: int = 60):
    """
    Simplified order block: last down-candle before a strong up-move (bullish OB),
    or last up-candle before a strong down-move (bearish OB).
    """
    window = df.tail(lookback)
    obs = []
    avg_range = (window["High"] - window["Low"]).mean()
    for i in range(1, len(window) - 1):
        cur = window.iloc[i]
        nxt = window.iloc[i + 1]
        move = nxt["Close"] - cur["Close"]
        if cur["Close"] < cur["Open"] and move > 1.5 * avg_range:
            obs.append({"date": window.index[i], "type": "Bullish Order Block",
                        "top": round(float(cur["Open"]), 2), "bottom": round(float(cur["Low"]), 2)})
        if cur["Close"] > cur["Open"] and -move > 1.5 * avg_range:
            obs.append({"date": window.index[i], "type": "Bearish Order Block",
                        "top": round(float(cur["High"]), 2), "bottom": round(float(cur["Open"]), 2)})
    return obs[-4:]


def find_equal_highs_lows(df: pd.DataFrame, lookback: int = 90, tol: float = 0.003):
    window = df.tail(lookback)
    highs, lows = [], []
    h, l = window["High"].values, window["Low"].values
    for i in range(2, len(window) - 2):
        if h[i] == h[i - 2:i + 3].max():
            highs.append((window.index[i], h[i]))
        if l[i] == l[i - 2:i + 3].min():
            lows.append((window.index[i], l[i]))

    eqh, eql = [], []
    for a, b in zip(highs, highs[1:]):
        if abs(a[1] - b[1]) / a[1] < tol:
            eqh.append(round((a[1] + b[1]) / 2, 2))
    for a, b in zip(lows, lows[1:]):
        if abs(a[1] - b[1]) / a[1] < tol:
            eql.append(round((a[1] + b[1]) / 2, 2))
    return sorted(set(eqh)), sorted(set(eql))


def premium_discount_zone(df: pd.DataFrame, lookback: int = 60) -> dict:
    window = df.tail(lookback)
    high, low = float(window["High"].max()), float(window["Low"].min())
    mid = (high + low) / 2
    close = float(df["Close"].iloc[-1])
    zone = "Premium (sell zone)" if close > mid else "Discount (buy zone)"
    return {"range_high": round(high, 2), "range_low": round(low, 2),
            "equilibrium": round(mid, 2), "zone": zone}
