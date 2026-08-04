"""Technical indicators — wraps `ta` library + custom Supertrend."""

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, MACD, ADXIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volume import (
    OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator,
    VolumeWeightedAveragePrice,
)
from ta.volatility import (
    AverageTrueRange, BollingerBands, KeltnerChannel, DonchianChannel,
)


def _supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Custom Supertrend (ta library has no native implementation)."""
    hl2 = (df["High"] + df["Low"]) / 2
    atr = AverageTrueRange(df["High"], df["Low"], df["Close"], window=period).average_true_range()

    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):
        if upper_basic.iloc[i] < upper_band.iloc[i - 1] or df["Close"].iloc[i - 1] > upper_band.iloc[i - 1]:
            upper_band.iloc[i] = upper_basic.iloc[i]
        else:
            upper_band.iloc[i] = upper_band.iloc[i - 1]

        if lower_basic.iloc[i] > lower_band.iloc[i - 1] or df["Close"].iloc[i - 1] < lower_band.iloc[i - 1]:
            lower_band.iloc[i] = lower_basic.iloc[i]
        else:
            lower_band.iloc[i] = lower_band.iloc[i - 1]

    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
            continue
        if supertrend.iloc[i - 1] == upper_band.iloc[i - 1]:
            if df["Close"].iloc[i] <= upper_band.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
        else:
            if df["Close"].iloc[i] >= lower_band.iloc[i]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1

    return pd.DataFrame({"Supertrend": supertrend, "ST_Direction": direction})


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Trend
    df["EMA20"] = EMAIndicator(df["Close"], 20).ema_indicator()
    df["EMA50"] = EMAIndicator(df["Close"], 50).ema_indicator()
    df["EMA200"] = EMAIndicator(df["Close"], 200).ema_indicator()

    macd = MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    adx = ADXIndicator(df["High"], df["Low"], df["Close"])
    df["ADX"] = adx.adx()
    df["DI+"] = adx.adx_pos()
    df["DI-"] = adx.adx_neg()

    ichi = IchimokuIndicator(df["High"], df["Low"])
    df["Ichimoku_A"] = ichi.ichimoku_a()
    df["Ichimoku_B"] = ichi.ichimoku_b()
    df["Ichimoku_Base"] = ichi.ichimoku_base_line()
    df["Ichimoku_Conv"] = ichi.ichimoku_conversion_line()

    st = _supertrend(df)
    df["Supertrend"] = st["Supertrend"]
    df["ST_Direction"] = st["ST_Direction"]

    # Momentum
    df["RSI"] = RSIIndicator(df["Close"]).rsi()
    stoch_rsi = StochRSIIndicator(df["Close"])
    df["StochRSI_K"] = stoch_rsi.stochrsi_k() * 100
    df["StochRSI_D"] = stoch_rsi.stochrsi_d() * 100

    # Volatility
    df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    bb = BollingerBands(df["Close"])
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Mid"] = bb.bollinger_mavg()
    kc = KeltnerChannel(df["High"], df["Low"], df["Close"])
    df["KC_Upper"] = kc.keltner_channel_hband()
    df["KC_Lower"] = kc.keltner_channel_lband()
    dc = DonchianChannel(df["High"], df["Low"], df["Close"])
    df["DC_Upper"] = dc.donchian_channel_hband()
    df["DC_Lower"] = dc.donchian_channel_lband()

    # Volume
    df["OBV"] = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    df["CMF"] = ChaikinMoneyFlowIndicator(df["High"], df["Low"], df["Close"], df["Volume"]).chaikin_money_flow()
    df["MFI"] = MFIIndicator(df["High"], df["Low"], df["Close"], df["Volume"]).money_flow_index()
    df["VWAP"] = VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()
    df["VolAvg20"] = df["Volume"].rolling(20).mean()

    return df
