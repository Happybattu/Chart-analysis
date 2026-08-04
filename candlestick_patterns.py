"""
Rule-based candlestick pattern detection.
No TA-Lib dependency (not installable in this sandbox) — patterns are
identified from OHLC geometry using standard textbook ratios.
Reliability scores are heuristic confidence estimates based on how
cleanly the candle(s) match the textbook definition, NOT backtested
statistics — treat them as a relative signal, not a probability.
"""

import pandas as pd


def _body(row):
    return abs(row["Close"] - row["Open"])


def _range(row):
    return row["High"] - row["Low"] or 1e-9


def _upper_wick(row):
    return row["High"] - max(row["Open"], row["Close"])


def _lower_wick(row):
    return min(row["Open"], row["Close"]) - row["Low"]


def _is_bull(row):
    return row["Close"] > row["Open"]


def detect_patterns(df: pd.DataFrame, lookback: int = 5) -> list:
    """Scan the last `lookback` candles and return detected patterns."""
    results = []
    n = len(df)
    start = max(2, n - lookback)

    for i in range(start, n):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        prev2 = df.iloc[i - 2] if i >= 2 else None
        rng = _range(row)
        body = _body(row)
        upper = _upper_wick(row)
        lower = _lower_wick(row)
        date = df.index[i]

        # Doji
        if body <= 0.1 * rng:
            results.append((date, "Doji", "Neutral", 55))

        # Hammer (small body near top, long lower wick, downtrend context)
        if lower >= 2 * body and upper <= 0.3 * body and body > 0:
            results.append((date, "Hammer", "Bullish", 72))

        # Shooting Star (small body near bottom, long upper wick)
        if upper >= 2 * body and lower <= 0.3 * body and body > 0:
            results.append((date, "Shooting Star", "Bearish", 70))

        # Marubozu (almost no wicks, large body)
        if body >= 0.9 * rng:
            results.append((date, "Marubozu", "Bullish" if _is_bull(row) else "Bearish", 68))

        # Bullish / Bearish Engulfing
        if _body(prev) > 0:
            if _is_bull(row) and not _is_bull(prev) and row["Close"] > prev["Open"] and row["Open"] < prev["Close"]:
                results.append((date, "Bullish Engulfing", "Bullish", 80))
            if not _is_bull(row) and _is_bull(prev) and row["Open"] > prev["Close"] and row["Close"] < prev["Open"]:
                results.append((date, "Bearish Engulfing", "Bearish", 78))

        # Harami (small body inside prior large body)
        if _body(prev) > 0 and body < 0.5 * _body(prev):
            if max(row["Open"], row["Close"]) < max(prev["Open"], prev["Close"]) and \
               min(row["Open"], row["Close"]) > min(prev["Open"], prev["Close"]):
                results.append((date, "Harami", "Bullish" if _is_bull(row) else "Bearish", 60))

        # Morning Star / Evening Star (3-candle)
        if prev2 is not None:
            first, mid, last = prev2, prev, row
            if (not _is_bull(first)) and _body(mid) < 0.3 * _range(mid) and _is_bull(last) and \
               last["Close"] > (first["Open"] + first["Close"]) / 2:
                results.append((date, "Morning Star", "Bullish", 82))
            if _is_bull(first) and _body(mid) < 0.3 * _range(mid) and (not _is_bull(last)) and \
               last["Close"] < (first["Open"] + first["Close"]) / 2:
                results.append((date, "Evening Star", "Bearish", 80))

            # Three White Soldiers / Three Black Crows
            if _is_bull(first) and _is_bull(mid) and _is_bull(last) and \
               mid["Close"] > first["Close"] and last["Close"] > mid["Close"] and \
               _body(first) > 0.5 * _range(first) and _body(mid) > 0.5 * _range(mid) and _body(last) > 0.5 * _range(last):
                results.append((date, "Three White Soldiers", "Bullish", 76))
            if (not _is_bull(first)) and (not _is_bull(mid)) and (not _is_bull(last)) and \
               mid["Close"] < first["Close"] and last["Close"] < mid["Close"] and \
               _body(first) > 0.5 * _range(first) and _body(mid) > 0.5 * _range(mid) and _body(last) > 0.5 * _range(last):
                results.append((date, "Three Black Crows", "Bearish", 76))

    # keep only most recent instance of each pattern
    latest = {}
    for date, name, direction, rel in results:
        latest[name] = (date, name, direction, rel)
    return sorted(latest.values(), key=lambda x: x[0])
