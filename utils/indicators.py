"""
Custom indicators for BTC day trading strategies.
Implements swing detection, liquidity zones, volume profile, and market structure.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import talib.abstract as ta


def calculate_swing_points(
    dataframe: pd.DataFrame,
    lookback: int = 5,
    column_high: str = 'high',
    column_low: str = 'low'
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect swing highs and swing lows.

    Args:
        dataframe: OHLCV dataframe
        lookback: Number of candles to look back/forward for swing detection

    Returns:
        Tuple of (swing_high, swing_low) boolean series
    """
    highs = dataframe[column_high]
    lows = dataframe[column_low]

    swing_high = pd.Series(False, index=dataframe.index)
    swing_low = pd.Series(False, index=dataframe.index)

    for i in range(lookback, len(dataframe) - lookback):
        # Swing High: current high is highest in window
        if highs.iloc[i] == highs.iloc[i-lookback:i+lookback+1].max():
            swing_high.iloc[i] = True

        # Swing Low: current low is lowest in window
        if lows.iloc[i] == lows.iloc[i-lookback:i+lookback+1].min():
            swing_low.iloc[i] = True

    return swing_high, swing_low


def detect_liquidity_zones(
    dataframe: pd.DataFrame,
    swing_lookback: int = 5,
    zone_tolerance: float = 0.001
) -> pd.DataFrame:
    """
    Detect liquidity zones based on swing points.
    Liquidity typically accumulates above swing highs and below swing lows.

    Args:
        dataframe: OHLCV dataframe
        swing_lookback: Lookback period for swing detection
        zone_tolerance: Tolerance for zone detection (as percentage)

    Returns:
        DataFrame with liquidity zone columns
    """
    df = dataframe.copy()

    swing_high, swing_low = calculate_swing_points(df, swing_lookback)

    # Store swing levels
    df['swing_high_level'] = np.where(swing_high, df['high'], np.nan)
    df['swing_low_level'] = np.where(swing_low, df['low'], np.nan)

    # Forward fill to maintain last known swing levels
    df['swing_high_level'] = df['swing_high_level'].ffill()
    df['swing_low_level'] = df['swing_low_level'].ffill()

    # Detect liquidity sweep (price briefly exceeds swing level then reverses)
    df['liquidity_sweep_high'] = (
        (df['high'] > df['swing_high_level'].shift(1)) &
        (df['close'] < df['swing_high_level'].shift(1))
    )

    df['liquidity_sweep_low'] = (
        (df['low'] < df['swing_low_level'].shift(1)) &
        (df['close'] > df['swing_low_level'].shift(1))
    )

    return df


def calculate_volume_profile(
    dataframe: pd.DataFrame,
    period: int = 20,
    num_bins: int = 10
) -> pd.DataFrame:
    """
    Calculate a simplified rolling volume profile.

    Args:
        dataframe: OHLCV dataframe
        period: Rolling window period
        num_bins: Number of price bins

    Returns:
        DataFrame with POC (Point of Control) and Value Area
    """
    df = dataframe.copy()

    poc = pd.Series(index=df.index, dtype=float)
    value_area_high = pd.Series(index=df.index, dtype=float)
    value_area_low = pd.Series(index=df.index, dtype=float)

    for i in range(period, len(df)):
        window = df.iloc[i-period:i]

        price_range = window['high'].max() - window['low'].min()
        if price_range == 0:
            continue

        bin_size = price_range / num_bins

        # Create price bins
        bins = np.linspace(window['low'].min(), window['high'].max(), num_bins + 1)

        # Assign volume to bins based on typical price
        typical_price = (window['high'] + window['low'] + window['close']) / 3
        volume_by_bin = np.zeros(num_bins)

        for j, (tp, vol) in enumerate(zip(typical_price, window['volume'])):
            bin_idx = min(int((tp - window['low'].min()) / bin_size), num_bins - 1)
            if bin_idx >= 0:
                volume_by_bin[bin_idx] += vol

        # POC is the bin with highest volume
        poc_bin = np.argmax(volume_by_bin)
        poc.iloc[i] = (bins[poc_bin] + bins[poc_bin + 1]) / 2

        # Value Area (70% of volume)
        total_volume = volume_by_bin.sum()
        target_volume = total_volume * 0.7

        cumulative = 0
        va_bins = [poc_bin]

        lower = poc_bin - 1
        upper = poc_bin + 1

        while cumulative < target_volume and (lower >= 0 or upper < num_bins):
            add_lower = lower >= 0 and (upper >= num_bins or volume_by_bin[lower] >= volume_by_bin[upper])

            if add_lower and lower >= 0:
                va_bins.append(lower)
                cumulative += volume_by_bin[lower]
                lower -= 1
            elif upper < num_bins:
                va_bins.append(upper)
                cumulative += volume_by_bin[upper]
                upper += 1
            else:
                break

        value_area_high.iloc[i] = bins[max(va_bins) + 1]
        value_area_low.iloc[i] = bins[min(va_bins)]

    df['poc'] = poc
    df['value_area_high'] = value_area_high
    df['value_area_low'] = value_area_low

    return df


def detect_market_structure(
    dataframe: pd.DataFrame,
    lookback: int = 5
) -> pd.DataFrame:
    """
    Detect market structure: HH/HL (uptrend), LH/LL (downtrend).
    Also detects Break of Structure (BOS) and Change of Character (CHoCH).

    Args:
        dataframe: OHLCV dataframe
        lookback: Swing detection lookback

    Returns:
        DataFrame with structure columns
    """
    df = dataframe.copy()

    swing_high, swing_low = calculate_swing_points(df, lookback)

    # Track swing high/low values
    last_swing_high = np.nan
    last_swing_low = np.nan
    prev_swing_high = np.nan
    prev_swing_low = np.nan

    structure = []  # 1 = bullish, -1 = bearish, 0 = neutral
    bos = []  # Break of Structure
    choch = []  # Change of Character

    current_structure = 0

    for i in range(len(df)):
        is_bos = False
        is_choch = False

        if swing_high.iloc[i]:
            prev_swing_high = last_swing_high
            last_swing_high = df['high'].iloc[i]

            if not np.isnan(prev_swing_high):
                if last_swing_high > prev_swing_high:  # Higher High
                    if current_structure != 1:
                        is_choch = True
                    current_structure = 1
                else:  # Lower High
                    is_bos = current_structure == 1

        if swing_low.iloc[i]:
            prev_swing_low = last_swing_low
            last_swing_low = df['low'].iloc[i]

            if not np.isnan(prev_swing_low):
                if last_swing_low < prev_swing_low:  # Lower Low
                    if current_structure != -1:
                        is_choch = True
                    current_structure = -1
                else:  # Higher Low
                    is_bos = current_structure == -1

        structure.append(current_structure)
        bos.append(is_bos)
        choch.append(is_choch)

    df['market_structure'] = structure
    df['bos'] = bos
    df['choch'] = choch

    return df


def calculate_atr_bands(
    dataframe: pd.DataFrame,
    period: int = 14,
    multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Calculate ATR-based bands for dynamic support/resistance.

    Args:
        dataframe: OHLCV dataframe
        period: ATR period
        multiplier: Band multiplier

    Returns:
        DataFrame with ATR bands
    """
    df = dataframe.copy()

    df['atr'] = ta.ATR(df, timeperiod=period)

    # Typical price as middle
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

    # ATR Bands
    df['atr_upper'] = df['typical_price'] + (df['atr'] * multiplier)
    df['atr_lower'] = df['typical_price'] - (df['atr'] * multiplier)

    return df


def detect_engulfing_pattern(dataframe: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Detect bullish and bearish engulfing patterns.

    Returns:
        Tuple of (bullish_engulfing, bearish_engulfing) boolean series
    """
    df = dataframe.copy()

    # Previous candle
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    prev_body = abs(prev_close - prev_open)

    # Current candle
    curr_body = abs(df['close'] - df['open'])

    # Bullish Engulfing: Previous bearish, current bullish and engulfs
    bullish_engulfing = (
        (prev_close < prev_open) &  # Previous bearish
        (df['close'] > df['open']) &  # Current bullish
        (df['close'] > prev_open) &  # Current close > prev open
        (df['open'] < prev_close) &  # Current open < prev close
        (curr_body > prev_body)  # Current body larger
    )

    # Bearish Engulfing: Previous bullish, current bearish and engulfs
    bearish_engulfing = (
        (prev_close > prev_open) &  # Previous bullish
        (df['close'] < df['open']) &  # Current bearish
        (df['close'] < prev_open) &  # Current close < prev open
        (df['open'] > prev_close) &  # Current open > prev close
        (curr_body > prev_body)  # Current body larger
    )

    return bullish_engulfing, bearish_engulfing


def detect_rejection_candle(
    dataframe: pd.DataFrame,
    wick_ratio: float = 2.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect rejection candles (pin bars) based on wick to body ratio.

    Args:
        dataframe: OHLCV dataframe
        wick_ratio: Minimum wick to body ratio for rejection

    Returns:
        Tuple of (bullish_rejection, bearish_rejection) boolean series
    """
    df = dataframe.copy()

    body = abs(df['close'] - df['open'])
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']

    # Prevent division by zero
    body = body.replace(0, 0.0001)

    # Bullish Rejection: Long lower wick
    bullish_rejection = (lower_wick / body > wick_ratio) & (upper_wick < lower_wick)

    # Bearish Rejection: Long upper wick
    bearish_rejection = (upper_wick / body > wick_ratio) & (lower_wick < upper_wick)

    return bullish_rejection, bearish_rejection
