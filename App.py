"""
Takdeer Stock Analyzer — Full Technical Analysis & Trading Plan Generator
Run: streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd

from indicators import add_all_indicators
from support_resistance import (
    swing_levels, classic_pivot_points, fibonacci_retracement,
    prev_period_high_low, volume_profile, gap_levels,
)
from candlestick_patterns import detect_patterns as detect_candle_patterns
from chart_patterns import detect_all as detect_chart_patterns
from trend_structure import analyze_structure
from smart_money import (
    find_fair_value_gaps, find_order_blocks, find_equal_highs_lows, premium_discount_zone,
)
from ai_commentary import generate_commentary
from chart import build_chart

st.set_page_config(page_title="NSE Technical Analyzer", layout="wide")


# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------
@st.cache_data(ttl=900)
def load_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


# ----------------------------------------------------------------------
# SCORING
# ----------------------------------------------------------------------
def score_stock(row: pd.Series, avg_vol: float, structure: dict) -> dict:
    close = row["Close"]
    score, notes = 0, []

    if close > row.get("EMA20", close):
        score += 8; notes.append("Price above EMA20")
    if close > row.get("EMA50", close):
        score += 8; notes.append("Price above EMA50")
    if close > row.get("EMA200", close):
        score += 15; notes.append("Price above EMA200 (long-term uptrend)")
    if row.get("RSI", 0) > 55:
        score += 8; notes.append(f"RSI strong ({row['RSI']:.0f})")
    if avg_vol and row["Volume"] > avg_vol * 1.5:
        score += 12; notes.append("Volume surge vs 20-day avg")
    if row.get("EMA20", 0) > row.get("EMA50", 0) > row.get("EMA200", 0):
        score += 10; notes.append("EMA stack bullish (20>50>200)")
    if row.get("MACD_Hist", 0) > 0:
        score += 8; notes.append("MACD histogram positive")
    if row.get("ADX", 0) > 25:
        score += 8; notes.append(f"ADX confirms trend ({row['ADX']:.0f})")
    if row.get("ST_Direction", -1) == 1:
        score += 8; notes.append("Supertrend bullish")
    if row.get("CMF", 0) > 0:
        score += 5; notes.append("Chaikin Money Flow positive (accumulation)")
    if "Strong Bullish" in structure["structure"]:
        score += 10; notes.append("Market structure: Higher Highs + Higher Lows")

    score = min(score, 100)
    if score >= 70:
        stars, verdict = "\u2605\u2605\u2605\u2605\u2605", "Bullish"
    elif score >= 50:
        stars, verdict = "\u2605\u2605\u2605\u2605\u2606", "Moderately Bullish"
    elif score >= 30:
        stars, verdict = "\u2605\u2605\u2605\u2606\u2606", "Neutral"
    else:
        stars, verdict = "\u2605\u2605\u2606\u2606\u2606", "Weak"

    return {"score": score, "stars": stars, "verdict": verdict, "notes": notes}


# ----------------------------------------------------------------------
# TRADING PLAN
# ----------------------------------------------------------------------
def build_trading_plan(close: float, resistances: list, supports: list, atr: float):
    nearest_res = min([r for r in resistances if r > close], default=close * 1.1)
    nearest_sup = max([s for s in supports if s < close], default=close * 0.9)

    aggressive_entry = round(nearest_res * 1.005, 1)
    stop_loss_agg = round(max(nearest_sup, aggressive_entry - 3 * atr), 1) if atr else round(nearest_sup, 1)
    conservative_entry_low = round(close * 0.94, 1)
    conservative_entry_high = round(close * 0.97, 1)
    stop_loss_cons = round(conservative_entry_low - 2 * atr, 1) if atr else round(conservative_entry_low * 0.96, 1)

    span = max(nearest_res - close, atr * 3 if atr else nearest_res - close)
    targets = [round(nearest_res, 1), round(nearest_res + span, 1), round(nearest_res + span * 2, 1)]

    risk = aggressive_entry - stop_loss_agg
    reward = targets[0] - aggressive_entry
    rr = round(reward / risk, 2) if risk > 0 else None

    return {
        "aggressive_entry": aggressive_entry, "stop_loss_aggressive": stop_loss_agg,
        "conservative_entry": (conservative_entry_low, conservative_entry_high),
        "stop_loss_conservative": stop_loss_cons, "targets": targets,
        "major_resistance": nearest_res, "major_support": nearest_sup, "risk_reward": rr,
    }


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("\U0001F4CA NSE Technical Analyzer — Full Suite")

col_a, col_b, col_c = st.columns([2, 1, 1])
with col_a:
    ticker_input = st.text_input("NSE Symbol (e.g. FSL, RELIANCE, TCS)", "FSL")
with col_b:
    period = st.selectbox("History", ["1y", "2y", "3y", "5y"], index=2)
with col_c:
    interval = st.selectbox("Interval", ["1d", "1wk"], index=0)

with st.expander("Chart display options"):
    o1, o2 = st.columns(2)
    show_bb = o1.checkbox("Show Bollinger Bands", value=True)
    show_st = o2.checkbox("Show Supertrend", value=True)

symbol = f"{ticker_input.strip().upper()}.NS"
run = st.button("\U0001F50D Analyze", type="primary")

if run:
    with st.spinner(f"Fetching {symbol} and running full analysis..."):
        raw = load_data(symbol, period, interval)
        if raw.empty or len(raw) < 60:
            st.error("No/insufficient data returned — check the symbol.")
            st.stop()

        df = add_all_indicators(raw)
        resistances, supports = swing_levels(df)
        pivots = classic_pivot_points(df)
        fibs = fibonacci_retracement(df)
        prev_hl = prev_period_high_low(df)
        hvn = volume_profile(df)
        gaps = gap_levels(df)

        candle_patterns = detect_candle_patterns(df)
        chart_pats = detect_chart_patterns(df)
        structure = analyze_structure(df)
        fvgs = find_fair_value_gaps(df)
        obs = find_order_blocks(df)
        eqh, eql = find_equal_highs_lows(df)
        smc_zone = premium_discount_zone(df)

        last = df.iloc[-1]
        avg_vol = df["VolAvg20"].iloc[-1]
        atr = float(last["ATR"]) if pd.notna(last["ATR"]) else 0.0
        result = score_stock(last, avg_vol, structure)
        plan = build_trading_plan(float(last["Close"]), resistances, supports, atr)
        commentary = generate_commentary(last, df["Close"].iloc[-2], avg_vol, structure,
                                          candle_patterns, chart_pats, smc_zone, result)

    close = float(last["Close"])
    chg = close - float(df["Close"].iloc[-2])
    chg_pct = chg / float(df["Close"].iloc[-2]) * 100

    st.subheader(f"{ticker_input.upper()} — \u20b9{close:.2f}  "
                 f"({'+' if chg >= 0 else ''}{chg:.2f}, {chg_pct:+.2f}%)")

    st.plotly_chart(
        build_chart(df, symbol, plan, resistances, supports, fib_levels=fibs,
                    candle_patterns=candle_patterns, chart_patterns=chart_pats,
                    smc=smc_zone, show_bb=show_bb, show_supertrend=show_st),
        use_container_width=True,
    )

    # ---- AI Commentary ----
    st.markdown("### \U0001F9E0 AI Commentary")
    st.info("\n\n".join(commentary))

    # ---- Score / Structure / Levels ----
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### \u2b50 Rating")
        st.metric("Score", f"{result['score']}/100")
        st.write(f"**Trend:** {result['stars']} — {result['verdict']}")
        st.write(f"**Structure:** {structure['structure']}")
        st.write(f"**Structure strength:** {structure['strength']}/10")
        if structure["bos"]:
            st.write(f"**BOS:** {structure['bos']}")
        if structure["choch"]:
            st.write(f"**CHOCH:** {structure['choch']}")

    with c2:
        st.markdown("### \U0001F3AF Key Levels")
        st.write(f"**Swing Resistance:** {', '.join(str(r) for r in resistances) or '—'}")
        st.write(f"**Swing Support:** {', '.join(str(s) for s in supports) or '—'}")
        st.write(f"**Pivot (classic):** P {pivots['P']} | R1 {pivots['R1']} | S1 {pivots['S1']}")
        if hvn:
            st.write(f"**Volume Profile HVN:** {', '.join(str(v) for v in hvn)}")
        for k, v in prev_hl.items():
            st.write(f"**{k}:** {v}")

    with c3:
        st.markdown("### \U0001F4C8 Momentum & Volatility")
        st.write(f"RSI: {last['RSI']:.1f} | StochRSI K: {last['StochRSI_K']:.1f}")
        st.write(f"ADX: {last['ADX']:.1f} (DI+ {last['DI+']:.1f} / DI- {last['DI-']:.1f})")
        st.write(f"ATR: {last['ATR']:.2f} | CMF: {last['CMF']:.3f} | MFI: {last['MFI']:.1f}")
        st.write(f"Supertrend: {'Bullish' if last['ST_Direction'] == 1 else 'Bearish'}")

    st.markdown("---")

    # ---- Patterns ----
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("### \U0001F56F\uFE0F Candlestick Patterns (recent)")
        if candle_patterns:
            for date, name, direction, rel in candle_patterns[-6:]:
                emoji = "\U0001F7E2" if direction == "Bullish" else ("\U0001F534" if direction == "Bearish" else "\u26AA")
                st.write(f"{emoji} **{name}** ({direction}) — {date.date()} — reliability ~{rel}%")
        else:
            st.write("No clear patterns in the recent window.")

    with p2:
        st.markdown("### \U0001F4D0 Chart Patterns")
        if chart_pats:
            for cp in chart_pats:
                st.write(f"- **{cp['pattern']}** ({cp.get('direction', '')}) "
                         f"{'@ ' + str(cp['level']) if cp.get('level') else ''}"
                         f"{'@ neckline ' + str(cp['neckline']) if cp.get('neckline') else ''}")
        else:
            st.write("No clean geometric pattern detected currently.")

    st.markdown("---")

    # ---- Smart Money Concepts ----
    st.markdown("### \U0001F4B0 Smart Money Concepts")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.write("**Fair Value Gaps**")
        for g in fvgs[-4:]:
            st.write(f"- {g['type']}: {g['bottom']}–{g['top']} ({g['date'].date()})")
        if not fvgs:
            st.write("None detected recently.")
    with s2:
        st.write("**Order Blocks**")
        for o in obs:
            st.write(f"- {o['type']}: {o['bottom']}–{o['top']} ({o['date'].date()})")
        if not obs:
            st.write("None detected recently.")
    with s3:
        st.write("**Equal Highs/Lows & Zone**")
        st.write(f"Equal Highs: {eqh or '—'}")
        st.write(f"Equal Lows: {eql or '—'}")
        st.write(f"Current zone: **{smc_zone['zone']}**")
        st.write(f"Range: {smc_zone['range_low']} – {smc_zone['range_high']} "
                 f"(EQ {smc_zone['equilibrium']})")

    if gaps:
        st.markdown("**Unfilled Price Gaps**")
        for g in gaps:
            st.write(f"- {g['type']}: {g['from']} → {g['to']} ({g['date'].date() if hasattr(g['date'], 'date') else g['date']})")

    st.markdown("---")

    # ---- Trading Plan ----
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("### \U0001F7E2 Aggressive Entry")
        st.write(f"- Buy above **\u20b9{plan['aggressive_entry']}** with strong volume")
        st.write(f"- Stop Loss: **\u20b9{plan['stop_loss_aggressive']}**")
        st.write(f"- Targets: {', '.join('\u20b9' + str(t) for t in plan['targets'])}")
        if plan["risk_reward"]:
            st.write(f"- Risk:Reward (to T1): **1:{plan['risk_reward']}**")
    with t2:
        st.markdown("### \U0001F7E1 Conservative Entry")
        lo, hi = plan["conservative_entry"]
        st.write(f"- Buy in pullback zone **\u20b9{lo} – \u20b9{hi}**")
        st.write(f"- Stop Loss: **\u20b9{plan['stop_loss_conservative']}**")
        st.write(f"- Targets: {', '.join('\u20b9' + str(t) for t in plan['targets'])}")

    st.markdown("### Fibonacci Retracement (last ~120 bars)")
    st.write(" | ".join(f"{k}: {v}" for k, v in fibs.items()))

    st.caption("Educational purposes only. Not a buy/sell recommendation. Patterns and "
               "\"probabilities\" above are rule-based heuristics, not statistically "
               "backtested or guaranteed. Do your own research or consult a financial advisor.")
else:
    st.info("Enter an NSE symbol and click Analyze. Example: FSL, RELIANCE, INFY, SONACOMS")
    
