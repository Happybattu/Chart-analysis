"""
Takdeer Stock Analyzer — Technical Analysis & Trading Plan Generator
Run: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator

st.set_page_config(page_title="NSE Technical Analyzer", layout="wide")

# ----------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------
@st.cache_data(ttl=900)
def load_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA20"] = EMAIndicator(df["Close"], 20).ema_indicator()
    df["EMA50"] = EMAIndicator(df["Close"], 50).ema_indicator()
    df["EMA200"] = EMAIndicator(df["Close"], 200).ema_indicator()
    df["RSI"] = RSIIndicator(df["Close"]).rsi()
    df["OBV"] = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    return df


# ----------------------------------------------------------------------
# 2. SUPPORT / RESISTANCE (simple swing-pivot detection)
# ----------------------------------------------------------------------
def find_swing_levels(df: pd.DataFrame, lookback: int = 5, n_levels: int = 3):
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
        return [round(np.mean(c), 2) for c in clusters]

    res = sorted(cluster(highs), reverse=True)[:n_levels]
    sup = sorted(cluster(lows), reverse=True)[:n_levels]
    return sorted(res), sorted(sup, reverse=True)


# ----------------------------------------------------------------------
# 3. SCORING
# ----------------------------------------------------------------------
def score_stock(row: pd.Series, avg_vol: float) -> dict:
    close, ema20, ema50, ema200 = row["Close"], row["EMA20"], row["EMA50"], row["EMA200"]
    rsi, vol = row["RSI"], row["Volume"]

    score = 0
    notes = []

    if close > ema20:
        score += 10; notes.append("Price above EMA20")
    if close > ema50:
        score += 10; notes.append("Price above EMA50")
    if close > ema200:
        score += 20; notes.append("Price above EMA200 (long-term uptrend)")
    if rsi and rsi > 55:
        score += 10; notes.append(f"RSI strong ({rsi:.0f})")
    elif rsi and rsi < 40:
        notes.append(f"RSI weak ({rsi:.0f})")
    if avg_vol and vol > avg_vol * 1.5:
        score += 15; notes.append("Volume surge vs 20-day avg")
    if ema20 > ema50 > ema200:
        score += 15; notes.append("EMA stack bullish (20>50>200)")

    score = min(score, 100)

    if score >= 70:
        trend_stars, verdict = "★★★★★", "Bullish"
    elif score >= 50:
        trend_stars, verdict = "★★★★☆", "Moderately Bullish"
    elif score >= 30:
        trend_stars, verdict = "★★★☆☆", "Neutral"
    else:
        trend_stars, verdict = "★★☆☆☆", "Weak"

    return {"score": score, "stars": trend_stars, "verdict": verdict, "notes": notes}


# ----------------------------------------------------------------------
# 4. TRADING PLAN
# ----------------------------------------------------------------------
def build_trading_plan(close: float, resistances: list, supports: list):
    nearest_res = min([r for r in resistances if r > close], default=close * 1.1)
    nearest_sup = max([s for s in supports if s < close], default=close * 0.9)

    aggressive_entry = round(nearest_res * 1.01, 1)
    stop_loss_agg = round(nearest_sup, 1)
    conservative_entry_low = round(close * 0.94, 1)
    conservative_entry_high = round(close * 0.97, 1)
    stop_loss_cons = round(conservative_entry_low * 0.96, 1)

    span = nearest_res - close
    targets = [round(nearest_res, 1),
               round(nearest_res + span, 1),
               round(nearest_res + span * 2, 1)]

    return {
        "aggressive_entry": aggressive_entry,
        "stop_loss_aggressive": stop_loss_agg,
        "conservative_entry": (conservative_entry_low, conservative_entry_high),
        "stop_loss_conservative": stop_loss_cons,
        "targets": targets,
        "major_resistance": nearest_res,
        "major_support": nearest_sup,
    }


# ----------------------------------------------------------------------
# 5. CHART (TradingView-style dark theme)
# ----------------------------------------------------------------------
def make_chart(df: pd.DataFrame, symbol: str, plan: dict, resistances: list, supports: list):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False, row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
        subplot_titles=(f"{symbol} — Price", "Volume"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ), row=1, col=1)

    for col, color in [("EMA20", "white"), ("EMA50", "#2196f3"), ("EMA200", "#ef5350")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], line=dict(color=color, width=1),
                                  name=col), row=1, col=1)

    vol_colors = np.where(df["Close"] >= df["Open"], "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, name="Volume"),
                  row=2, col=1)

    for r in resistances:
        fig.add_hline(y=r, line_dash="dash", line_color="#ef5350", opacity=0.6,
                       annotation_text=f"R {r}", annotation_position="right", row=1, col=1)
    for s in supports:
        fig.add_hline(y=s, line_dash="dot", line_color="#26a69a", opacity=0.6,
                       annotation_text=f"S {s}", annotation_position="right", row=1, col=1)

    last_date = df.index[-1]
    fig.add_annotation(x=last_date, y=plan["aggressive_entry"], text="BUY ▲",
                        showarrow=True, arrowhead=2, font=dict(color="#26a69a"), row=1, col=1)
    for i, t in enumerate(plan["targets"], 1):
        fig.add_hline(y=t, line_dash="dashdot", line_color="#4caf50", opacity=0.5, row=1, col=1)

    fig.update_layout(
        template="plotly_dark", height=750, showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=60, t=40, b=10),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    )
    return fig


# ----------------------------------------------------------------------
# 6. STREAMLIT UI
# ----------------------------------------------------------------------
st.title("📊 NSE Technical Analyzer & Trading Plan")

col_a, col_b, col_c = st.columns([2, 1, 1])
with col_a:
    ticker_input = st.text_input("NSE Symbol (e.g. FSL, RELIANCE, TCS)", "FSL")
with col_b:
    period = st.selectbox("History", ["1y", "2y", "3y", "5y"], index=2)
with col_c:
    interval = st.selectbox("Interval", ["1d", "1wk"], index=0)

symbol = f"{ticker_input.strip().upper()}.NS"
run = st.button("🔍 Analyze", type="primary")

if run:
    with st.spinner(f"Fetching {symbol} and computing indicators..."):
        raw = load_data(symbol, period, interval)
        if raw.empty:
            st.error("No data returned — check the symbol (NSE tickers need .NS, auto-added here).")
            st.stop()
        df = add_indicators(raw)
        resistances, supports = find_swing_levels(df)
        last = df.iloc[-1]
        avg_vol = df["VolAvg20"].iloc[-1]
        result = score_stock(last, avg_vol)
        plan = build_trading_plan(float(last["Close"]), resistances, supports)

    close = float(last["Close"])
    chg = close - float(df["Close"].iloc[-2])
    chg_pct = chg / float(df["Close"].iloc[-2]) * 100

    st.subheader(f"{ticker_input.upper()} — ₹{close:.2f}  "
                 f"({'+' if chg >= 0 else ''}{chg:.2f}, {chg_pct:+.2f}%)")

    st.plotly_chart(make_chart(df, symbol, plan, resistances, supports), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### ✅ Key Takeaways")
        for n in result["notes"]:
            st.write(f"- {n}")
        st.write(f"- RSI: {last['RSI']:.1f}")

    with c2:
        st.markdown("### ⭐ Rating")
        st.metric("Score", f"{result['score']}/100")
        st.write(f"**Trend:** {result['stars']} — {result['verdict']}")

    with c3:
        st.markdown("### 🎯 Key Levels")
        st.write(f"**Resistance:** {', '.join(str(r) for r in resistances) or '—'}")
        st.write(f"**Support:** {', '.join(str(s) for s in supports) or '—'}")

    st.markdown("---")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("### 🟢 Aggressive Entry")
        st.write(f"- Buy above **₹{plan['aggressive_entry']}** with strong volume")
        st.write(f"- Stop Loss: **₹{plan['stop_loss_aggressive']}**")
        st.write(f"- Targets: {', '.join('₹' + str(t) for t in plan['targets'])}")
    with p2:
        st.markdown("### 🟡 Conservative Entry")
        lo, hi = plan["conservative_entry"]
        st.write(f"- Buy in pullback zone **₹{lo} – ₹{hi}**")
        st.write(f"- Stop Loss: **₹{plan['stop_loss_conservative']}**")
        st.write(f"- Targets: {', '.join('₹' + str(t) for t in plan['targets'])}")

    st.caption("Educational purposes only. Not a buy/sell recommendation. "
               "Do your own research or consult a financial advisor before investing.")
else:
    st.info("Enter an NSE symbol and click Analyze. Example: FSL, RELIANCE, INFY, SONACOMS")
