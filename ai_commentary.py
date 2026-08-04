"""
Generates dynamic natural-language commentary from computed indicators.
This is template-based (not an LLM call) so it stays fast and free —
but the sentences are assembled conditionally from real numbers, so
it reads dynamic rather than fixed boilerplate.
"""


def generate_commentary(last_row, prev_close, avg_vol, structure, candle_patterns,
                         chart_patterns, smc_zone, score_result) -> list:
    lines = []
    close = float(last_row["Close"])
    vol = float(last_row["Volume"])
    rsi = last_row.get("RSI")
    adx = last_row.get("ADX")
    macd_hist = last_row.get("MACD_Hist")

    # Volume
    if avg_vol and vol > avg_vol * 2:
        lines.append(f"Volume is {vol / avg_vol:.1f}x the 20-day average — strong participation.")
    elif avg_vol and vol > avg_vol * 1.3:
        lines.append(f"Volume is running {vol / avg_vol:.1f}x above average.")

    # EMA stack
    if close > last_row.get("EMA20", close) > last_row.get("EMA50", close) > last_row.get("EMA200", close):
        lines.append("Price is above all major EMAs (20/50/200) — bullish alignment.")
    elif close < last_row.get("EMA200", close):
        lines.append("Price is below the 200 EMA — long-term trend is not yet bullish.")

    # Structure
    lines.append(f"Market structure: {structure['structure']} (strength {structure['strength']}/10).")
    if structure.get("bos"):
        lines.append(structure["bos"] + ".")
    if structure.get("choch"):
        lines.append(structure["choch"] + ".")

    # Momentum
    if rsi is not None:
        if rsi > 70:
            lines.append(f"RSI at {rsi:.0f} — overbought, watch for a pause or pullback.")
        elif rsi > 55:
            lines.append(f"RSI at {rsi:.0f} — healthy bullish momentum.")
        elif rsi < 35:
            lines.append(f"RSI at {rsi:.0f} — oversold zone.")

    if adx is not None:
        if adx > 25:
            lines.append(f"ADX at {adx:.0f} confirms a trending (not choppy) market.")
        else:
            lines.append(f"ADX at {adx:.0f} suggests a weak/range-bound trend currently.")

    if macd_hist is not None:
        lines.append("MACD histogram is " + ("expanding above zero — bullish momentum building."
                     if macd_hist > 0 else "below zero — momentum still negative."))

    # Patterns
    if candle_patterns:
        d, name, direction, rel = candle_patterns[-1]
        lines.append(f"Most recent candle pattern: {name} ({direction}), reliability ~{rel}%.")
    if chart_patterns:
        cp = chart_patterns[-1]
        lines.append(f"Chart pattern flagged: {cp['pattern']} ({cp.get('direction', '')}).")

    # SMC
    lines.append(f"Price is in the {smc_zone['zone']} of its recent range "
                 f"({smc_zone['range_low']}–{smc_zone['range_high']}).")

    # Wrap-up probability line (heuristic, tied to score)
    prob = min(95, max(20, score_result["score"] + 10))
    lines.append(f"Composite technical score: {score_result['score']}/100 "
                 f"({score_result['verdict']}). Heuristic continuation probability ~{prob}%.")

    return lines
