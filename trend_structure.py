"""Market structure — Higher High/Higher Low sequencing, BOS, CHOCH."""

import pandas as pd


def _get_pivots(df: pd.DataFrame, lookback: int = 5):
    h, l = df["High"].values, df["Low"].values
    piv = []
    for i in range(lookback, len(df) - lookback):
        wh = h[i - lookback:i + lookback + 1]
        wl = l[i - lookback:i + lookback + 1]
        if h[i] == wh.max():
            piv.append((df.index[i], "H", h[i]))
        if l[i] == wl.min():
            piv.append((df.index[i], "L", l[i]))
    piv.sort(key=lambda x: x[0])
    return piv


def analyze_structure(df: pd.DataFrame, lookback: int = 5) -> dict:
    piv = _get_pivots(df, lookback)
    highs = [p for p in piv if p[1] == "H"]
    lows = [p for p in piv if p[1] == "L"]

    def sequence_tag(points):
        tags = []
        for a, b in zip(points, points[1:]):
            tags.append("Higher" if b[2] > a[2] else "Lower")
        return tags

    high_seq = sequence_tag(highs)
    low_seq = sequence_tag(lows)

    hh = high_seq[-3:].count("Higher") if high_seq else 0
    hl = low_seq[-3:].count("Higher") if low_seq else 0
    lh = high_seq[-3:].count("Lower") if high_seq else 0
    ll = low_seq[-3:].count("Lower") if low_seq else 0

    if hh >= 2 and hl >= 2:
        structure = "Strong Bullish (HH + HL)"
        strength = min(9, 5 + hh + hl)
    elif lh >= 2 and ll >= 2:
        structure = "Strong Bearish (LH + LL)"
        strength = min(9, 5 + lh + ll)
    else:
        structure = "Choppy / Transitional"
        strength = 5

    # Break of Structure: close beyond the most recent opposite-side pivot
    bos, choch = None, None
    if highs and df["Close"].iloc[-1] > highs[-1][2]:
        bos = f"Bullish BOS above {highs[-1][2]:.2f} ({highs[-1][0].date()})"
    if lows and df["Close"].iloc[-1] < lows[-1][2]:
        bos = f"Bearish BOS below {lows[-1][2]:.2f} ({lows[-1][0].date()})"

    # CHOCH: prior structure was bearish (LH/LL) but price just broke above a lower-high -> shift
    if len(highs) >= 2 and low_seq[-2:] == ["Lower", "Lower"] and df["Close"].iloc[-1] > highs[-1][2]:
        choch = "Possible bullish CHOCH — trend may be shifting up"
    if len(lows) >= 2 and high_seq[-2:] == ["Higher", "Higher"] and df["Close"].iloc[-1] < lows[-1][2]:
        choch = "Possible bearish CHOCH — trend may be shifting down"

    return {
        "structure": structure,
        "strength": strength,
        "bos": bos,
        "choch": choch,
        "recent_highs": highs[-3:],
        "recent_lows": lows[-3:],
    }
