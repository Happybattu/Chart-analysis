"""Builds the multi-subplot TradingView-style chart."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_chart(df: pd.DataFrame, symbol: str, plan: dict, resistances: list, supports: list,
                 fib_levels: dict = None, candle_patterns: list = None,
                 chart_patterns: list = None, smc: dict = None, show_bb=True, show_supertrend=True):

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.15, 0.17, 0.18],
        vertical_spacing=0.02,
        subplot_titles=(f"{symbol} — Price", "Volume", "RSI / Stoch RSI", "MACD / ADX"),
    )

    # ---- Price ----
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ), row=1, col=1)

    for col, color in [("EMA20", "#e0e0e0"), ("EMA50", "#2196f3"), ("EMA200", "#ef5350"), ("VWAP", "#ffb300")]:
        if col in df:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], line=dict(color=color, width=1),
                                      name=col), row=1, col=1)

    if show_bb and "BB_Upper" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], line=dict(color="rgba(150,150,255,0.4)", width=1),
                                  name="BB Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], line=dict(color="rgba(150,150,255,0.4)", width=1),
                                  name="BB Lower", fill="tonexty", fillcolor="rgba(150,150,255,0.06)"), row=1, col=1)

    if show_supertrend and "Supertrend" in df:
        up = df["Supertrend"].where(df["ST_Direction"] == 1)
        down = df["Supertrend"].where(df["ST_Direction"] == -1)
        fig.add_trace(go.Scatter(x=df.index, y=up, line=dict(color="#26a69a", width=2),
                                  name="Supertrend (Up)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=down, line=dict(color="#ef5350", width=2),
                                  name="Supertrend (Down)"), row=1, col=1)

    # ---- S/R levels ----
    for r in resistances:
        fig.add_hline(y=r, line_dash="dash", line_color="#ef5350", opacity=0.6,
                       annotation_text=f"R {r}", annotation_position="right", row=1, col=1)
    for s in supports:
        fig.add_hline(y=s, line_dash="dot", line_color="#26a69a", opacity=0.6,
                       annotation_text=f"S {s}", annotation_position="right", row=1, col=1)

    # ---- Fibonacci ----
    if fib_levels:
        for label, level in fib_levels.items():
            fig.add_hline(y=level, line_dash="dashdot", line_color="rgba(255,193,7,0.5)",
                          annotation_text=f"Fib {label}", annotation_position="left",
                          row=1, col=1)

    # ---- SMC premium/discount zone ----
    if smc:
        fig.add_hrect(y0=smc["equilibrium"], y1=smc["range_high"], fillcolor="rgba(239,83,80,0.06)",
                      line_width=0, row=1, col=1)
        fig.add_hrect(y0=smc["range_low"], y1=smc["equilibrium"], fillcolor="rgba(38,166,154,0.06)",
                      line_width=0, row=1, col=1)

    # ---- Candlestick pattern markers ----
    if candle_patterns:
        for date, name, direction, rel in candle_patterns:
            if date not in df.index:
                continue
            y = df.loc[date, "High"] * 1.01 if direction == "Bullish" else df.loc[date, "Low"] * 0.99
            fig.add_annotation(x=date, y=y, text=name, showarrow=True, arrowhead=1,
                                font=dict(size=9, color="#26a69a" if direction == "Bullish" else "#ef5350"),
                                row=1, col=1)

    # ---- Chart pattern labels ----
    if chart_patterns:
        for cp in chart_patterns:
            label = cp["pattern"]
            level = cp.get("level") or cp.get("neckline")
            if level:
                fig.add_hline(y=level, line_dash="longdash", line_color="#ba68c8", opacity=0.5,
                              annotation_text=label, annotation_position="left", row=1, col=1)

    # ---- Trade plan markers ----
    last_date = df.index[-1]
    fig.add_annotation(x=last_date, y=plan["aggressive_entry"], text="BUY \u25b2",
                        showarrow=True, arrowhead=2, font=dict(color="#26a69a"), row=1, col=1)
    fig.add_annotation(x=last_date, y=plan["stop_loss_aggressive"], text="SL \u25bc",
                        showarrow=True, arrowhead=2, font=dict(color="#ef5350"), row=1, col=1)
    for t in plan["targets"]:
        fig.add_hline(y=t, line_dash="dashdot", line_color="#4caf50", opacity=0.5, row=1, col=1)

    # ---- Risk/Reward box ----
    entry = plan["aggressive_entry"]
    sl = plan["stop_loss_aggressive"]
    t1 = plan["targets"][0]
    fig.add_shape(type="rect", x0=last_date, x1=df.index[-1], y0=sl, y1=entry,
                  fillcolor="rgba(239,83,80,0.15)", line_width=0, row=1, col=1)
    fig.add_shape(type="rect", x0=last_date, x1=df.index[-1], y0=entry, y1=t1,
                  fillcolor="rgba(38,166,154,0.15)", line_width=0, row=1, col=1)

    # ---- Volume ----
    vol_colors = np.where(df["Close"] >= df["Open"], "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, name="Volume"), row=2, col=1)
    if "VolAvg20" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["VolAvg20"], line=dict(color="#ffb300", width=1),
                                  name="Vol Avg 20"), row=2, col=1)

    # ---- RSI / StochRSI ----
    if "RSI" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#ab47bc", width=1.3),
                                  name="RSI"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#ef5350", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#26a69a", opacity=0.5, row=3, col=1)
    if "StochRSI_K" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["StochRSI_K"], line=dict(color="#42a5f5", width=1),
                                  name="StochRSI K"), row=3, col=1)

    # ---- MACD / ADX ----
    if "MACD" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color="#42a5f5", width=1.2),
                                  name="MACD"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], line=dict(color="#ffb300", width=1),
                                  name="Signal"), row=4, col=1)
        hist_colors = np.where(df["MACD_Hist"] >= 0, "#26a69a", "#ef5350")
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], marker_color=hist_colors, name="MACD Hist"),
                      row=4, col=1)
    if "ADX" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["ADX"], line=dict(color="#ff7043", width=1.3, dash="dot"),
                                  name="ADX"), row=4, col=1)

    fig.update_layout(
        template="plotly_dark", height=1050, showlegend=True,
        xaxis4_rangeslider_visible=False,
        margin=dict(l=10, r=60, t=40, b=10),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    for i in range(1, 5):
        fig.update_xaxes(rangeslider_visible=False, row=i, col=1)

    return fig
