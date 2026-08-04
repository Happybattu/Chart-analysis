"""
Heuristic chart pattern recognition.
These are geometric approximations based on swing-pivot sequences —
not a trained classifier. Treat flagged patterns as "worth a manual
look on the chart," not confirmed signals. Cup & Handle, Flag/Pennant
and Wedge require more context (volume shape, duration) than pure
OHLC geometry reliably gives, so they're intentionally left out
rather than guessed at with low confidence.
"""

import numpy as np
import pandas as pd


def _get_pivots(df: pd.DataFrame, lookback: int = 5):
    h, l = df["High"].values, df["Low"].values
    piv_highs, piv_lows = [], []
    for i in range(lookback, len(df) - lookback):
        wh = h[i - lookback:i + lookback + 1]
        wl = l[i - lookback:i + lookback + 1]
        if h[i] == wh.max():
            piv_highs.append((i, h[i]))
        if l[i] == wl.min():
            piv_lows.append((i, l[i]))
    return piv_highs, piv_lows


def detect_double_top_bottom(df: pd.DataFrame, tol: float = 0.02):
    piv_highs, piv_lows = _get_pivots(df)
    patterns = []

    for a, b in zip(piv_highs, piv_highs[1:]):
        if abs(a[1] - b[1]) / a[1] < tol and (b[0] - a[0]) > 5:
            patterns.append({
                "pattern": "Double Top", "direction": "Bearish",
                "points": [a, b], "level": round((a[1] + b[1]) / 2, 2),
            })
    for a, b in zip(piv_lows, piv_lows[1:]):
        if abs(a[1] - b[1]) / a[1] < tol and (b[0] - a[0]) > 5:
            patterns.append({
                "pattern": "Double Bottom", "direction": "Bullish",
                "points": [a, b], "level": round((a[1] + b[1]) / 2, 2),
            })
    return patterns[-2:]  # most recent


def detect_head_shoulders(df: pd.DataFrame, tol: float = 0.03):
    piv_highs, _ = _get_pivots(df)
    patterns = []
    for i in range(len(piv_highs) - 2):
        l_sh, head, r_sh = piv_highs[i], piv_highs[i + 1], piv_highs[i + 2]
        if head[1] > l_sh[1] and head[1] > r_sh[1] and abs(l_sh[1] - r_sh[1]) / l_sh[1] < tol:
            patterns.append({
                "pattern": "Head & Shoulders", "direction": "Bearish",
                "points": [l_sh, head, r_sh],
                "neckline": round(min(l_sh[1], r_sh[1]) * 0.97, 2),
            })
    _, piv_lows = _get_pivots(df)
    for i in range(len(piv_lows) - 2):
        l_sh, head, r_sh = piv_lows[i], piv_lows[i + 1], piv_lows[i + 2]
        if head[1] < l_sh[1] and head[1] < r_sh[1] and abs(l_sh[1] - r_sh[1]) / l_sh[1] < tol:
            patterns.append({
                "pattern": "Inverse Head & Shoulders", "direction": "Bullish",
                "points": [l_sh, head, r_sh],
                "neckline": round(max(l_sh[1], r_sh[1]) * 1.03, 2),
            })
    return patterns[-2:]


def detect_triangle(df: pd.DataFrame, lookback: int = 40, slope_flat_tol: float = 0.001):
    """Fit simple linear trendlines to recent swing highs and lows."""
    window = df.tail(lookback).reset_index()
    piv_highs, piv_lows = _get_pivots(window, lookback=3)
    if len(piv_highs) < 2 or len(piv_lows) < 2:
        return None

    xs_h = np.array([p[0] for p in piv_highs])
    ys_h = np.array([p[1] for p in piv_highs])
    xs_l = np.array([p[0] for p in piv_lows])
    ys_l = np.array([p[1] for p in piv_lows])

    slope_h = np.polyfit(xs_h, ys_h, 1)[0] if len(xs_h) >= 2 else 0
    slope_l = np.polyfit(xs_l, ys_l, 1)[0] if len(xs_l) >= 2 else 0

    avg_price = window["Close"].mean()
    norm_h = slope_h / avg_price
    norm_l = slope_l / avg_price

    flat = slope_flat_tol
    if abs(norm_h) < flat and norm_l > flat:
        return {"pattern": "Ascending Triangle", "direction": "Bullish"}
    if norm_h < -flat and abs(norm_l) < flat:
        return {"pattern": "Descending Triangle", "direction": "Bearish"}
    if norm_h < -flat and norm_l > flat:
        return {"pattern": "Symmetrical Triangle", "direction": "Neutral (breakout pending)"}
    if abs(norm_h) < flat and abs(norm_l) < flat:
        return {"pattern": "Rectangle / Consolidation", "direction": "Neutral"}
    return None


def detect_all(df: pd.DataFrame) -> list:
    found = []
    found += detect_double_top_bottom(df)
    found += detect_head_shoulders(df)
    tri = detect_triangle(df)
    if tri:
        found.append(tri)
    return found
