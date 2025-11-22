"""
Market Regime Detector for adaptive strategy selection.
Determines if market is trending, ranging, or volatile.
"""

import numpy as np
import pandas as pd
import talib.abstract as ta
from enum import Enum
from typing import Tuple


class MarketRegime(Enum):
    """Market regime classifications."""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"
    WEAK_DOWNTREND = "weak_downtrend"
    RANGING = "ranging"
    VOLATILE = "volatile"


class MarketRegimeDetector:
    """
    Detects current market regime to select appropriate strategy.

    Regime Detection Logic:
    - Strong Trend: ADX > 25, clear EMA alignment
    - Weak Trend: ADX 20-25, partial EMA alignment
    - Ranging: ADX < 20, price within Bollinger Bands
    - Volatile: High ATR relative to historical average
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_threshold_strong: float = 25,
        adx_threshold_weak: float = 20,
        ema_fast: int = 20,
        ema_slow: int = 50,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        volatility_multiplier: float = 1.5
    ):
        self.adx_period = adx_period
        self.adx_threshold_strong = adx_threshold_strong
        self.adx_threshold_weak = adx_threshold_weak
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.volatility_multiplier = volatility_multiplier

    def analyze(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze dataframe and add regime detection columns.

        Args:
            dataframe: OHLCV dataframe

        Returns:
            DataFrame with regime columns
        """
        df = dataframe.copy()

        # Calculate indicators
        df['ema_fast'] = ta.EMA(df, timeperiod=self.ema_fast)
        df['ema_slow'] = ta.EMA(df, timeperiod=self.ema_slow)
        df['adx'] = ta.ADX(df, timeperiod=self.adx_period)
        df['plus_di'] = ta.PLUS_DI(df, timeperiod=self.adx_period)
        df['minus_di'] = ta.MINUS_DI(df, timeperiod=self.adx_period)
        df['atr'] = ta.ATR(df, timeperiod=self.atr_period)

        # Bollinger Bands
        bb = ta.BBANDS(df, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std)
        df['bb_upper'] = bb['upperband']
        df['bb_middle'] = bb['middleband']
        df['bb_lower'] = bb['lowerband']

        # BB Width for ranging detection
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # ATR percentile for volatility
        df['atr_sma'] = df['atr'].rolling(window=50).mean()
        df['atr_ratio'] = df['atr'] / df['atr_sma']

        # Detect regime
        df['regime'] = df.apply(lambda row: self._classify_regime(row), axis=1)

        # Numeric regime for easier usage
        regime_map = {
            MarketRegime.STRONG_UPTREND: 2,
            MarketRegime.WEAK_UPTREND: 1,
            MarketRegime.RANGING: 0,
            MarketRegime.WEAK_DOWNTREND: -1,
            MarketRegime.STRONG_DOWNTREND: -2,
            MarketRegime.VOLATILE: 3
        }
        df['regime_numeric'] = df['regime'].map(regime_map)

        # Strategy recommendations
        df['use_pullback'] = df['regime'].isin([
            MarketRegime.STRONG_UPTREND,
            MarketRegime.WEAK_UPTREND,
            MarketRegime.STRONG_DOWNTREND,
            MarketRegime.WEAK_DOWNTREND
        ])

        df['use_liquidity'] = df['regime'].isin([
            MarketRegime.RANGING,
            MarketRegime.VOLATILE
        ])

        df['use_breakout'] = df['regime'].isin([
            MarketRegime.RANGING,
            MarketRegime.VOLATILE
        ]) | (df['bb_width'] < df['bb_width'].rolling(20).mean())

        return df

    def _classify_regime(self, row: pd.Series) -> MarketRegime:
        """Classify single row into market regime."""
        adx = row.get('adx', 0)
        plus_di = row.get('plus_di', 0)
        minus_di = row.get('minus_di', 0)
        ema_fast = row.get('ema_fast', 0)
        ema_slow = row.get('ema_slow', 0)
        atr_ratio = row.get('atr_ratio', 1)

        # Check for high volatility first
        if atr_ratio > self.volatility_multiplier:
            return MarketRegime.VOLATILE

        # Check trend strength
        if adx > self.adx_threshold_strong:
            if plus_di > minus_di and ema_fast > ema_slow:
                return MarketRegime.STRONG_UPTREND
            elif minus_di > plus_di and ema_fast < ema_slow:
                return MarketRegime.STRONG_DOWNTREND

        if adx > self.adx_threshold_weak:
            if plus_di > minus_di:
                return MarketRegime.WEAK_UPTREND
            else:
                return MarketRegime.WEAK_DOWNTREND

        return MarketRegime.RANGING

    def get_recommended_strategy(self, regime: MarketRegime) -> str:
        """
        Get recommended strategy based on regime.

        Args:
            regime: Current market regime

        Returns:
            Strategy name recommendation
        """
        strategy_map = {
            MarketRegime.STRONG_UPTREND: "trend_pullback",
            MarketRegime.WEAK_UPTREND: "trend_pullback",
            MarketRegime.STRONG_DOWNTREND: "trend_pullback",
            MarketRegime.WEAK_DOWNTREND: "trend_pullback",
            MarketRegime.RANGING: "liquidity_sweep",
            MarketRegime.VOLATILE: "breakout_volume"
        }
        return strategy_map.get(regime, "trend_pullback")


def calculate_regime_metrics(dataframe: pd.DataFrame) -> dict:
    """
    Calculate summary metrics for current regime.

    Args:
        dataframe: DataFrame with regime analysis

    Returns:
        Dictionary of regime metrics
    """
    if len(dataframe) == 0:
        return {}

    latest = dataframe.iloc[-1]
    lookback_20 = dataframe.tail(20)

    return {
        'current_regime': str(latest.get('regime', 'unknown')),
        'adx': float(latest.get('adx', 0)),
        'trend_direction': 'up' if latest.get('ema_fast', 0) > latest.get('ema_slow', 0) else 'down',
        'volatility_ratio': float(latest.get('atr_ratio', 1)),
        'bb_width': float(latest.get('bb_width', 0)),
        'regime_consistency': (lookback_20['regime'] == latest['regime']).sum() / len(lookback_20),
        'recommended_strategy': latest.get('use_pullback', False) and 'pullback' or
                              latest.get('use_liquidity', False) and 'liquidity' or 'breakout'
    }
