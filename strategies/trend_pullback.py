"""
Trend-Pullback Strategy for BTC Futures
========================================

Strategy Logic:
- Trend Detection: EMA20 > EMA50 (bullish) or EMA20 < EMA50 (bearish)
- Entry on pullback to EMA zone with confirmation
- Confirmation: Engulfing pattern, rejection candle, or volume spike

Timeframes:
- Trend: 1H or 4H
- Entry: 5m or 15m

Features:
- Supports LONG and SHORT positions (Futures)
- ATR-based dynamic stop loss
- Volume confirmation filter
- Maximum 3 trades per day limit
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.persistence import Trade
import talib.abstract as ta

# Import custom utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.indicators import (
    detect_engulfing_pattern,
    detect_rejection_candle,
    calculate_atr_bands
)


class TrendPullbackStrategy(IStrategy):
    """
    Trend-Pullback Strategy optimized for BTC Futures.
    Designed for 1-3 high-quality trades per day.
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Futures specific settings
    can_short = True
    trading_direction = 'both'  # 'long', 'short', or 'both'

    # ROI table - Conservative for quality trades
    minimal_roi = {
        "0": 0.05,    # 5% target
        "30": 0.03,   # 3% after 30 mins
        "60": 0.02,   # 2% after 1 hour
        "120": 0.01   # 1% after 2 hours
    }

    # Stoploss
    stoploss = -0.02  # 2% max loss
    use_custom_stoploss = True
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Timeframe
    timeframe = '15m'
    informative_timeframe = '1h'

    # Process only new candles
    process_only_new_candles = True

    # Startup candle count
    startup_candle_count = 100

    # Order types for futures
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

    # Leverage settings
    leverage_default = 5
    leverage_max = 10

    # Strategy parameters (optimizable)
    ema_fast = IntParameter(10, 30, default=20, space='buy')
    ema_slow = IntParameter(40, 60, default=50, space='buy')
    pullback_threshold = DecimalParameter(0.001, 0.01, default=0.005, space='buy')
    volume_confirm_mult = DecimalParameter(1.0, 2.0, default=1.3, space='buy')
    atr_sl_mult = DecimalParameter(1.0, 3.0, default=1.5, space='sell')

    # Trade limiting
    max_trades_per_day = 3

    # Custom variables
    custom_trade_count = {}

    def informative_pairs(self):
        """Define informative pairs for multi-timeframe analysis."""
        return [
            (f"BTC/USDT:USDT", self.informative_timeframe)
        ]

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Calculate all technical indicators."""

        # EMAs for trend
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)

        # Trend direction
        dataframe['uptrend'] = dataframe['ema_fast'] > dataframe['ema_slow']
        dataframe['downtrend'] = dataframe['ema_fast'] < dataframe['ema_slow']

        # Distance from EMAs (for pullback detection)
        dataframe['dist_to_fast_ema'] = (dataframe['close'] - dataframe['ema_fast']) / dataframe['close']
        dataframe['dist_to_slow_ema'] = (dataframe['close'] - dataframe['ema_slow']) / dataframe['close']

        # Pullback zone (between EMAs or close to fast EMA)
        dataframe['in_pullback_zone_long'] = (
            (dataframe['close'] > dataframe['ema_slow']) &
            (dataframe['close'] <= dataframe['ema_fast'] * 1.005) &
            (dataframe['low'] <= dataframe['ema_fast'])
        )

        dataframe['in_pullback_zone_short'] = (
            (dataframe['close'] < dataframe['ema_slow']) &
            (dataframe['close'] >= dataframe['ema_fast'] * 0.995) &
            (dataframe['high'] >= dataframe['ema_fast'])
        )

        # ATR for dynamic SL
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # Volume analysis
        dataframe['volume_sma'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_spike'] = dataframe['volume'] > (dataframe['volume_sma'] * self.volume_confirm_mult.value)

        # Volume decreasing in pullback (healthy pullback sign)
        dataframe['volume_decreasing'] = (
            (dataframe['volume'] < dataframe['volume'].shift(1)) &
            (dataframe['volume'].shift(1) < dataframe['volume'].shift(2))
        )

        # Candlestick patterns
        bullish_engulf, bearish_engulf = detect_engulfing_pattern(dataframe)
        dataframe['bullish_engulfing'] = bullish_engulf
        dataframe['bearish_engulfing'] = bearish_engulf

        bullish_reject, bearish_reject = detect_rejection_candle(dataframe)
        dataframe['bullish_rejection'] = bullish_reject
        dataframe['bearish_rejection'] = bearish_reject

        # RSI for confirmation
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Structure: Higher Highs / Lower Lows
        dataframe['higher_low'] = (
            (dataframe['low'] > dataframe['low'].shift(1)) &
            (dataframe['low'].shift(1) > dataframe['low'].shift(2))
        )

        dataframe['lower_high'] = (
            (dataframe['high'] < dataframe['high'].shift(1)) &
            (dataframe['high'].shift(1) < dataframe['high'].shift(2))
        )

        # ADX for trend strength
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Define entry signals for LONG and SHORT positions."""

        # LONG Entry Conditions
        dataframe.loc[
            (
                # Trend: Bullish
                (dataframe['uptrend']) &
                (dataframe['adx'] > 20) &

                # Pullback to EMA zone
                (dataframe['in_pullback_zone_long']) &

                # Confirmation (at least one)
                (
                    (dataframe['bullish_engulfing']) |
                    (dataframe['bullish_rejection']) |
                    (dataframe['higher_low'] & dataframe['volume_spike'])
                ) &

                # Volume confirmation
                (
                    (dataframe['volume_decreasing'].shift(1)) |  # Weak volume in pullback
                    (dataframe['volume_spike'])  # Or spike on confirmation candle
                ) &

                # RSI not overbought
                (dataframe['rsi'] < 70) &

                # Volume exists
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1

        # SHORT Entry Conditions
        dataframe.loc[
            (
                # Trend: Bearish
                (dataframe['downtrend']) &
                (dataframe['adx'] > 20) &

                # Pullback to EMA zone
                (dataframe['in_pullback_zone_short']) &

                # Confirmation (at least one)
                (
                    (dataframe['bearish_engulfing']) |
                    (dataframe['bearish_rejection']) |
                    (dataframe['lower_high'] & dataframe['volume_spike'])
                ) &

                # Volume confirmation
                (
                    (dataframe['volume_decreasing'].shift(1)) |
                    (dataframe['volume_spike'])
                ) &

                # RSI not oversold
                (dataframe['rsi'] > 30) &

                # Volume exists
                (dataframe['volume'] > 0)
            ),
            'enter_short'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Define exit signals."""

        # Exit LONG when trend reverses or target reached
        dataframe.loc[
            (
                (dataframe['downtrend']) |  # Trend reversal
                (dataframe['rsi'] > 75) |   # Overbought
                (dataframe['bearish_engulfing'])  # Reversal pattern
            ),
            'exit_long'
        ] = 1

        # Exit SHORT when trend reverses or target reached
        dataframe.loc[
            (
                (dataframe['uptrend']) |  # Trend reversal
                (dataframe['rsi'] < 25) |   # Oversold
                (dataframe['bullish_engulfing'])  # Reversal pattern
            ),
            'exit_short'
        ] = 1

        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs
    ) -> Optional[float]:
        """
        ATR-based dynamic stoploss.
        Returns stoploss as negative decimal (e.g., -0.02 = 2% stoploss).
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) < 1:
            return None

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get('atr', 0)

        if atr == 0:
            return None

        # Calculate ATR-based stoploss distance
        atr_stoploss = (atr * self.atr_sl_mult.value) / current_rate

        # Ensure minimum stoploss
        return max(-atr_stoploss, -0.03)  # Max 3% loss

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        """
        Dynamic leverage based on ATR volatility.
        Higher volatility = Lower leverage.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) < 1:
            return self.leverage_default

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get('atr', 0)
        close = last_candle.get('close', 1)

        # ATR as percentage of price
        atr_pct = (atr / close) * 100

        # Adjust leverage based on volatility
        if atr_pct > 3:  # High volatility
            leverage = 3
        elif atr_pct > 2:  # Medium volatility
            leverage = 5
        else:  # Low volatility
            leverage = 7

        return min(leverage, max_leverage, self.leverage_max)

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> bool:
        """
        Confirm trade entry - limit to max trades per day.
        """
        today = current_time.date()

        if today not in self.custom_trade_count:
            self.custom_trade_count[today] = 0

        if self.custom_trade_count[today] >= self.max_trades_per_day:
            return False

        self.custom_trade_count[today] += 1
        return True

    def custom_entry_price(
        self,
        pair: str,
        trade: Optional[Trade],
        current_time: datetime,
        proposed_rate: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        """
        Custom entry price - slightly better than market for limit orders.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) < 1:
            return proposed_rate

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get('atr', 0)

        # Place limit order slightly inside the spread
        adjustment = atr * 0.1

        if side == 'long':
            return proposed_rate - adjustment  # Buy lower
        else:
            return proposed_rate + adjustment  # Sell higher
