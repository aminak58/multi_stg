"""
Order Flow Indicators for BTC Trading Strategies
=================================================

This module implements order flow analysis indicators:
1. CVD (Cumulative Volume Delta)
2. Volume Imbalance
3. Absorption Detection
4. Exhaustion Detection
5. Order Pressure Ratio
6. Real-time Order Book Analysis

Note: Without tick data, we use OHLCV approximations.
For live trading, order book data provides additional accuracy.

References:
- https://www.luxalgo.com/blog/cumulative-volume-delta-explained/
- https://bookmap.com/blog/how-cumulative-volume-delta-transform-your-trading-strategy
- https://trading-strategies.academy/archives/1160
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from enum import Enum
import talib.abstract as ta


class OrderFlowSignal(Enum):
    """Order flow signal types."""
    STRONG_BUY = "strong_buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    STRONG_SELL = "strong_sell"
    ABSORPTION_BUY = "absorption_buy"
    ABSORPTION_SELL = "absorption_sell"
    EXHAUSTION_BUY = "exhaustion_buy"
    EXHAUSTION_SELL = "exhaustion_sell"


def calculate_buy_sell_volume(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate buy and sell volume from OHLCV data.

    Method 1: Simple - Based on candle direction
    Method 2: Proportional - Based on close position within range

    Args:
        dataframe: OHLCV dataframe

    Returns:
        DataFrame with buy_volume, sell_volume columns
    """
    df = dataframe.copy()

    # Calculate candle metrics
    df['body'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

    # Prevent division by zero
    df['range'] = df['range'].replace(0, 0.0001)

    # Method 1: Simple direction-based
    df['buy_volume_simple'] = np.where(df['close'] > df['open'], df['volume'], 0)
    df['sell_volume_simple'] = np.where(df['close'] < df['open'], df['volume'], 0)

    # Method 2: Proportional based on close position
    # Close position ratio: 0 = closed at low, 1 = closed at high
    df['close_position'] = (df['close'] - df['low']) / df['range']

    # Buy volume proportional to how high the close is
    df['buy_volume'] = df['volume'] * df['close_position']
    df['sell_volume'] = df['volume'] * (1 - df['close_position'])

    # Method 3: Wicks indicate rejection (advanced)
    # Large lower wick = buy pressure absorbed selling
    # Large upper wick = sell pressure absorbed buying
    wick_factor = 0.3  # Weight for wick contribution

    df['wick_buy_volume'] = df['volume'] * (df['lower_wick'] / df['range']) * wick_factor
    df['wick_sell_volume'] = df['volume'] * (df['upper_wick'] / df['range']) * wick_factor

    # Combined buy/sell volume
    df['buy_volume'] = df['buy_volume'] + df['wick_buy_volume']
    df['sell_volume'] = df['sell_volume'] + df['wick_sell_volume']

    return df


def calculate_delta(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Volume Delta (difference between buy and sell volume).

    Delta = Buy Volume - Sell Volume

    Args:
        dataframe: OHLCV dataframe

    Returns:
        DataFrame with delta columns
    """
    df = calculate_buy_sell_volume(dataframe)

    # Volume Delta
    df['delta'] = df['buy_volume'] - df['sell_volume']

    # Delta percentage (normalized)
    df['delta_pct'] = df['delta'] / df['volume'].replace(0, 1)

    # Delta strength (absolute)
    df['delta_strength'] = abs(df['delta']) / df['volume'].replace(0, 1)

    # Delta direction
    df['delta_positive'] = df['delta'] > 0
    df['delta_negative'] = df['delta'] < 0

    return df


def calculate_cvd(
    dataframe: pd.DataFrame,
    reset_period: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate Cumulative Volume Delta (CVD).

    CVD = Running sum of Delta

    Args:
        dataframe: OHLCV dataframe
        reset_period: Optional period to reset CVD (e.g., daily)

    Returns:
        DataFrame with CVD columns
    """
    df = calculate_delta(dataframe)

    # Simple CVD (cumulative sum)
    df['cvd'] = df['delta'].cumsum()

    # Rolling CVD (windowed)
    for period in [14, 50, 100]:
        df[f'cvd_{period}'] = df['delta'].rolling(window=period).sum()

    # CVD rate of change
    df['cvd_roc'] = df['cvd'].diff(5)

    # CVD vs Price divergence
    df['price_roc'] = df['close'].pct_change(5) * 100

    # Divergence: Price up but CVD down = bearish divergence
    df['cvd_divergence_bearish'] = (
        (df['price_roc'] > 0) &
        (df['cvd_roc'] < 0) &
        (abs(df['price_roc']) > 0.5)  # Meaningful price move
    )

    # Divergence: Price down but CVD up = bullish divergence
    df['cvd_divergence_bullish'] = (
        (df['price_roc'] < 0) &
        (df['cvd_roc'] > 0) &
        (abs(df['price_roc']) > 0.5)
    )

    # CVD trend
    df['cvd_ema_fast'] = ta.EMA(df['cvd'], timeperiod=10)
    df['cvd_ema_slow'] = ta.EMA(df['cvd'], timeperiod=30)
    df['cvd_trend_up'] = df['cvd_ema_fast'] > df['cvd_ema_slow']
    df['cvd_trend_down'] = df['cvd_ema_fast'] < df['cvd_ema_slow']

    return df


def calculate_volume_imbalance(
    dataframe: pd.DataFrame,
    threshold: float = 1.5,
    strong_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Calculate Volume Imbalance Ratio.

    Imbalance = max(buy, sell) / min(buy, sell)

    Args:
        dataframe: OHLCV dataframe
        threshold: Minimum ratio for imbalance (default 1.5 = 150%)
        strong_threshold: Ratio for strong imbalance (default 3.0 = 300%)

    Returns:
        DataFrame with imbalance columns
    """
    df = calculate_buy_sell_volume(dataframe)

    # Calculate imbalance ratio
    min_volume = df[['buy_volume', 'sell_volume']].min(axis=1).replace(0, 0.0001)
    max_volume = df[['buy_volume', 'sell_volume']].max(axis=1)

    df['imbalance_ratio'] = max_volume / min_volume

    # Imbalance direction
    df['buy_imbalance'] = (
        (df['buy_volume'] > df['sell_volume']) &
        (df['imbalance_ratio'] >= threshold)
    )

    df['sell_imbalance'] = (
        (df['sell_volume'] > df['buy_volume']) &
        (df['imbalance_ratio'] >= threshold)
    )

    # Strong imbalance
    df['strong_buy_imbalance'] = (
        (df['buy_volume'] > df['sell_volume']) &
        (df['imbalance_ratio'] >= strong_threshold)
    )

    df['strong_sell_imbalance'] = (
        (df['sell_volume'] > df['buy_volume']) &
        (df['imbalance_ratio'] >= strong_threshold)
    )

    # Stacked imbalances (consecutive)
    df['stacked_buy_imbalance'] = (
        df['buy_imbalance'] &
        df['buy_imbalance'].shift(1)
    )

    df['stacked_sell_imbalance'] = (
        df['sell_imbalance'] &
        df['sell_imbalance'].shift(1)
    )

    return df


def detect_absorption(
    dataframe: pd.DataFrame,
    volume_threshold: float = 1.5,
    price_threshold: float = 0.002
) -> pd.DataFrame:
    """
    Detect absorption patterns.

    Absorption: High volume but price doesn't move much
    - Buy Absorption: High sell volume but price doesn't drop
    - Sell Absorption: High buy volume but price doesn't rise

    Args:
        dataframe: OHLCV dataframe
        volume_threshold: Volume multiplier for high volume
        price_threshold: Maximum price change for absorption

    Returns:
        DataFrame with absorption columns
    """
    df = calculate_delta(dataframe)

    # Volume metrics
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['high_volume'] = df['volume'] > (df['volume_sma'] * volume_threshold)

    # Price change
    df['price_change_pct'] = abs(df['close'] - df['open']) / df['open']
    df['small_price_move'] = df['price_change_pct'] < price_threshold

    # ATR for context
    df['atr'] = ta.ATR(df, timeperiod=14)
    df['small_range'] = (df['high'] - df['low']) < df['atr'] * 0.5

    # Buy Absorption: Selling pressure absorbed
    # High volume, mostly sell, but price doesn't drop (or rises)
    df['absorption_buy'] = (
        df['high_volume'] &
        (df['sell_volume'] > df['buy_volume']) &
        (df['close'] >= df['open']) &  # Price didn't drop
        df['small_range']
    )

    # Sell Absorption: Buying pressure absorbed
    # High volume, mostly buy, but price doesn't rise (or drops)
    df['absorption_sell'] = (
        df['high_volume'] &
        (df['buy_volume'] > df['sell_volume']) &
        (df['close'] <= df['open']) &  # Price didn't rise
        df['small_range']
    )

    # Absorption strength
    df['absorption_strength'] = np.where(
        df['absorption_buy'] | df['absorption_sell'],
        df['volume'] / df['volume_sma'],
        0
    )

    # Consecutive absorption (stronger signal)
    df['absorption_buy_consecutive'] = (
        df['absorption_buy'] &
        (df['absorption_buy'].shift(1) | df['absorption_buy'].shift(2))
    )

    df['absorption_sell_consecutive'] = (
        df['absorption_sell'] &
        (df['absorption_sell'].shift(1) | df['absorption_sell'].shift(2))
    )

    return df


def detect_exhaustion(
    dataframe: pd.DataFrame,
    volume_threshold: float = 2.0,
    momentum_threshold: float = 0.3
) -> pd.DataFrame:
    """
    Detect exhaustion patterns.

    Exhaustion: Very high volume but momentum is fading
    Indicates potential trend reversal

    Args:
        dataframe: OHLCV dataframe
        volume_threshold: Volume spike multiplier
        momentum_threshold: Momentum fade threshold

    Returns:
        DataFrame with exhaustion columns
    """
    df = calculate_delta(dataframe)

    # Volume spike
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = df['volume'] > (df['volume_sma'] * volume_threshold)

    # Momentum indicators
    df['rsi'] = ta.RSI(df, timeperiod=14)
    df['mom'] = ta.MOM(df, timeperiod=10)
    df['mom_sma'] = df['mom'].rolling(window=5).mean()

    # Momentum fading (slowing down)
    df['momentum_fading_up'] = (
        (df['mom'] > 0) &
        (df['mom'] < df['mom'].shift(1)) &
        (df['mom'].shift(1) < df['mom'].shift(2))
    )

    df['momentum_fading_down'] = (
        (df['mom'] < 0) &
        (df['mom'] > df['mom'].shift(1)) &
        (df['mom'].shift(1) > df['mom'].shift(2))
    )

    # Price making new highs/lows with exhaustion
    df['new_high'] = df['high'] == df['high'].rolling(window=20).max()
    df['new_low'] = df['low'] == df['low'].rolling(window=20).min()

    # Buy Exhaustion: Price at high, volume spike, momentum fading
    df['exhaustion_buy'] = (
        df['volume_spike'] &
        (df['rsi'] > 70) &
        df['momentum_fading_up'] &
        (df['close'] > df['open'])  # Still bullish candle
    )

    # Sell Exhaustion: Price at low, volume spike, momentum fading
    df['exhaustion_sell'] = (
        df['volume_spike'] &
        (df['rsi'] < 30) &
        df['momentum_fading_down'] &
        (df['close'] < df['open'])  # Still bearish candle
    )

    # Exhaustion with divergence (strongest signal)
    df['exhaustion_buy_divergence'] = (
        df['exhaustion_buy'] &
        df['new_high'] &
        (df['delta'] < df['delta'].shift(1))  # Delta not confirming
    )

    df['exhaustion_sell_divergence'] = (
        df['exhaustion_sell'] &
        df['new_low'] &
        (df['delta'] > df['delta'].shift(1))  # Delta not confirming
    )

    return df


def calculate_order_pressure(
    dataframe: pd.DataFrame,
    lookback: int = 10
) -> pd.DataFrame:
    """
    Calculate Order Pressure - the cumulative strength of buyers vs sellers.

    Args:
        dataframe: OHLCV dataframe
        lookback: Period for pressure calculation

    Returns:
        DataFrame with order pressure columns
    """
    df = calculate_delta(dataframe)

    # Cumulative delta over lookback period
    df['cum_buy'] = df['buy_volume'].rolling(window=lookback).sum()
    df['cum_sell'] = df['sell_volume'].rolling(window=lookback).sum()

    # Order Pressure Ratio
    total_volume = df['cum_buy'] + df['cum_sell']
    total_volume = total_volume.replace(0, 1)

    df['buy_pressure'] = df['cum_buy'] / total_volume
    df['sell_pressure'] = df['cum_sell'] / total_volume

    # Pressure imbalance (-1 to +1 scale)
    df['pressure_imbalance'] = df['buy_pressure'] - df['sell_pressure']

    # Pressure trend
    df['pressure_trend'] = df['pressure_imbalance'].rolling(window=5).mean()
    df['pressure_increasing'] = df['pressure_imbalance'] > df['pressure_imbalance'].shift(1)

    # Extreme pressure
    df['extreme_buy_pressure'] = df['pressure_imbalance'] > 0.3
    df['extreme_sell_pressure'] = df['pressure_imbalance'] < -0.3

    return df


def analyze_order_book(order_book: dict, levels: int = 10) -> dict:
    """
    Analyze real-time order book data (for live trading).

    Args:
        order_book: Order book dict from exchange {'bids': [...], 'asks': [...]}
        levels: Number of levels to analyze

    Returns:
        Dictionary with order book analysis
    """
    bids = order_book.get('bids', [])[:levels]
    asks = order_book.get('asks', [])[:levels]

    if not bids or not asks:
        return {}

    # Extract prices and volumes
    bid_prices = [float(b[0]) for b in bids]
    bid_volumes = [float(b[1]) for b in bids]
    ask_prices = [float(a[0]) for a in asks]
    ask_volumes = [float(a[1]) for a in asks]

    # Total volumes
    total_bid_volume = sum(bid_volumes)
    total_ask_volume = sum(ask_volumes)

    # Best bid/ask
    best_bid = bid_prices[0] if bid_prices else 0
    best_ask = ask_prices[0] if ask_prices else 0
    spread = best_ask - best_bid if best_bid and best_ask else 0
    spread_pct = (spread / best_bid * 100) if best_bid else 0

    # Volume imbalance
    total_volume = total_bid_volume + total_ask_volume
    bid_ratio = total_bid_volume / total_volume if total_volume else 0.5
    ask_ratio = total_ask_volume / total_volume if total_volume else 0.5

    # Imbalance signal
    if bid_ratio > 0.6:
        signal = OrderFlowSignal.STRONG_BUY
    elif bid_ratio > 0.55:
        signal = OrderFlowSignal.WEAK_BUY
    elif ask_ratio > 0.6:
        signal = OrderFlowSignal.STRONG_SELL
    elif ask_ratio > 0.55:
        signal = OrderFlowSignal.WEAK_SELL
    else:
        signal = OrderFlowSignal.NEUTRAL

    # Large orders detection
    avg_bid_size = total_bid_volume / levels if levels else 0
    avg_ask_size = total_ask_volume / levels if levels else 0

    large_bid_walls = sum(1 for v in bid_volumes if v > avg_bid_size * 3)
    large_ask_walls = sum(1 for v in ask_volumes if v > avg_ask_size * 3)

    return {
        'best_bid': best_bid,
        'best_ask': best_ask,
        'spread': spread,
        'spread_pct': spread_pct,
        'total_bid_volume': total_bid_volume,
        'total_ask_volume': total_ask_volume,
        'bid_ratio': bid_ratio,
        'ask_ratio': ask_ratio,
        'imbalance': bid_ratio - ask_ratio,
        'signal': signal,
        'large_bid_walls': large_bid_walls,
        'large_ask_walls': large_ask_walls
    }


def get_order_flow_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive order flow signals.

    Combines all order flow indicators into actionable signals.

    Args:
        dataframe: OHLCV dataframe

    Returns:
        DataFrame with all order flow signals
    """
    # Calculate all indicators
    df = calculate_cvd(dataframe)
    df = calculate_volume_imbalance(df)
    df = detect_absorption(df)
    df = detect_exhaustion(df)
    df = calculate_order_pressure(df)

    # Composite Buy Signal
    df['of_buy_signal'] = (
        # CVD confirmation
        (df['cvd_trend_up']) &
        # No exhaustion
        (~df['exhaustion_buy']) &
        # Either imbalance or absorption
        (
            (df['buy_imbalance']) |
            (df['absorption_buy']) |
            (df['cvd_divergence_bullish'])
        ) &
        # Pressure confirming
        (df['pressure_imbalance'] > 0)
    )

    # Composite Sell Signal
    df['of_sell_signal'] = (
        # CVD confirmation
        (df['cvd_trend_down']) &
        # No exhaustion
        (~df['exhaustion_sell']) &
        # Either imbalance or absorption
        (
            (df['sell_imbalance']) |
            (df['absorption_sell']) |
            (df['cvd_divergence_bearish'])
        ) &
        # Pressure confirming
        (df['pressure_imbalance'] < 0)
    )

    # Strong signals (multiple confirmations)
    df['of_strong_buy_signal'] = (
        df['of_buy_signal'] &
        (df['strong_buy_imbalance'] | df['stacked_buy_imbalance']) &
        (df['pressure_imbalance'] > 0.2)
    )

    df['of_strong_sell_signal'] = (
        df['of_sell_signal'] &
        (df['strong_sell_imbalance'] | df['stacked_sell_imbalance']) &
        (df['pressure_imbalance'] < -0.2)
    )

    # Reversal signals (exhaustion + divergence)
    df['of_reversal_buy'] = (
        df['exhaustion_sell'] |
        df['exhaustion_sell_divergence'] |
        df['absorption_buy_consecutive']
    )

    df['of_reversal_sell'] = (
        df['exhaustion_buy'] |
        df['exhaustion_buy_divergence'] |
        df['absorption_sell_consecutive']
    )

    # Signal quality score (0-100)
    df['of_buy_quality'] = (
        (df['cvd_trend_up'].astype(int) * 20) +
        (df['buy_imbalance'].astype(int) * 20) +
        (df['absorption_buy'].astype(int) * 25) +
        ((df['pressure_imbalance'] > 0).astype(int) * 15) +
        ((~df['exhaustion_buy']).astype(int) * 20)
    )

    df['of_sell_quality'] = (
        (df['cvd_trend_down'].astype(int) * 20) +
        (df['sell_imbalance'].astype(int) * 20) +
        (df['absorption_sell'].astype(int) * 25) +
        ((df['pressure_imbalance'] < 0).astype(int) * 15) +
        ((~df['exhaustion_sell']).astype(int) * 20)
    )

    return df
