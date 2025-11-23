"""
Breakout with Volume Filter Strategy for BTC Futures
=====================================================

Strategy Logic:
- Detect consolidation ranges (Bollinger Band squeeze, low ATR)
- Wait for breakout with volume confirmation
- Enter on pullback to breakout level
- Target: Measured move (range height) from breakout

Key Features:
- Volume Profile POC (Point of Control) detection
- Value Area breakout confirmation
- ATR-based range detection
- Pullback entry for better R:R

Features:
- Supports LONG and SHORT positions (Futures)
- Explosive moves in volatile conditions
- Works best in range -> breakout market phases
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.persistence import Trade
import talib.abstract as ta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.indicators import calculate_volume_profile, calculate_atr_bands


class BreakoutVolumeStrategy(IStrategy):
    """
    Breakout Strategy with Volume Confirmation for BTC Futures.
    Best for consolidation -> explosion phases.
    Designed for 1-3 powerful trades per day.
    """

    INTERFACE_VERSION = 3

    # Futures settings
    can_short = True
    trading_direction = 'both'

    # ROI - Aggressive targets for breakout moves
    minimal_roi = {
        "0": 0.10,    # 10% target
        "30": 0.06,   # 6% after 30 mins
        "60": 0.04,   # 4% after 1 hour
        "120": 0.02   # 2% after 2 hours
    }

    # Stoploss
    stoploss = -0.02  # 2% max
    use_custom_stoploss = True
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # Timeframe
    timeframe = '15m'
    informative_timeframe = '1h'

    process_only_new_candles = True
    startup_candle_count = 100

    # Order types
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

    # Leverage
    leverage_default = 5
    leverage_max = 10

    # Strategy parameters
    bb_period = IntParameter(15, 25, default=20, space='buy')
    bb_std = DecimalParameter(1.5, 2.5, default=2.0, space='buy')
    squeeze_threshold = DecimalParameter(0.01, 0.05, default=0.03, space='buy')
    volume_breakout_mult = DecimalParameter(1.3, 2.5, default=1.5, space='buy')
    pullback_tolerance = DecimalParameter(0.002, 0.01, default=0.005, space='buy')
    range_period = IntParameter(10, 30, default=20, space='buy')

    # Trade limiting
    max_trades_per_day = 3
    custom_trade_count = {}

    def informative_pairs(self):
        return [(f"BTC/USDT:USDT", self.informative_timeframe)]

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Calculate breakout and volume indicators."""

        # Bollinger Bands
        bb = ta.BBANDS(
            dataframe,
            timeperiod=self.bb_period.value,
            nbdevup=self.bb_std.value,
            nbdevdn=self.bb_std.value
        )
        dataframe['bb_upper'] = bb['upperband']
        dataframe['bb_middle'] = bb['middleband']
        dataframe['bb_lower'] = bb['lowerband']

        # BB Width (squeeze detection)
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        dataframe['bb_width_sma'] = dataframe['bb_width'].rolling(window=20).mean()

        # Squeeze: BB width below threshold
        dataframe['bb_squeeze'] = dataframe['bb_width'] < self.squeeze_threshold.value

        # Expanding: BB width increasing from squeeze
        dataframe['bb_expanding'] = (
            (dataframe['bb_width'] > dataframe['bb_width'].shift(1)) &
            (dataframe['bb_width'].shift(1) < dataframe['bb_width_sma'].shift(1))
        )

        # ATR for volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_sma'] = dataframe['atr'].rolling(window=20).mean()
        dataframe['low_volatility'] = dataframe['atr'] < dataframe['atr_sma'] * 0.8

        # Range detection
        dataframe['range_high'] = dataframe['high'].rolling(window=self.range_period.value).max()
        dataframe['range_low'] = dataframe['low'].rolling(window=self.range_period.value).min()
        dataframe['range_size'] = dataframe['range_high'] - dataframe['range_low']
        dataframe['range_mid'] = (dataframe['range_high'] + dataframe['range_low']) / 2

        # Range as percentage of price
        dataframe['range_pct'] = dataframe['range_size'] / dataframe['close']

        # Consolidation: Small range relative to ATR
        dataframe['consolidation'] = (
            (dataframe['range_pct'] < 0.03) |  # Range < 3% of price
            (dataframe['bb_squeeze'])
        )

        # Volume analysis
        dataframe['volume_sma'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_mult'] = dataframe['volume'] / dataframe['volume_sma']
        dataframe['volume_breakout'] = dataframe['volume'] > (
            dataframe['volume_sma'] * self.volume_breakout_mult.value
        )

        # Breakout detection
        # Bullish breakout: Close above range high with volume
        dataframe['breakout_up'] = (
            (dataframe['close'] > dataframe['range_high'].shift(1)) &
            (dataframe['close'].shift(1) <= dataframe['range_high'].shift(2)) &
            (dataframe['volume_breakout'])
        )

        # Bearish breakout: Close below range low with volume
        dataframe['breakout_down'] = (
            (dataframe['close'] < dataframe['range_low'].shift(1)) &
            (dataframe['close'].shift(1) >= dataframe['range_low'].shift(2)) &
            (dataframe['volume_breakout'])
        )

        # Track breakout levels for pullback entry
        dataframe['breakout_level_up'] = np.where(
            dataframe['breakout_up'],
            dataframe['range_high'].shift(1),
            np.nan
        )
        dataframe['breakout_level_up'] = dataframe['breakout_level_up'].ffill(limit=10)

        dataframe['breakout_level_down'] = np.where(
            dataframe['breakout_down'],
            dataframe['range_low'].shift(1),
            np.nan
        )
        dataframe['breakout_level_down'] = dataframe['breakout_level_down'].ffill(limit=10)

        # Pullback to breakout level
        dataframe['pullback_to_breakout_up'] = (
            (dataframe['breakout_level_up'].notna()) &
            (dataframe['low'] <= dataframe['breakout_level_up'] * (1 + self.pullback_tolerance.value)) &
            (dataframe['close'] > dataframe['breakout_level_up'])
        )

        dataframe['pullback_to_breakout_down'] = (
            (dataframe['breakout_level_down'].notna()) &
            (dataframe['high'] >= dataframe['breakout_level_down'] * (1 - self.pullback_tolerance.value)) &
            (dataframe['close'] < dataframe['breakout_level_down'])
        )

        # Strong breakout candle (not doji)
        dataframe['body'] = abs(dataframe['close'] - dataframe['open'])
        dataframe['candle_range'] = dataframe['high'] - dataframe['low']
        min_range = dataframe['candle_range'].replace(0, 0.0001)
        dataframe['strong_candle'] = dataframe['body'] / min_range > 0.5

        # Momentum confirmation
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['macd'], dataframe['macd_signal'], dataframe['macd_hist'] = ta.MACD(
            dataframe,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )

        # ADX for trend strength after breakout
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        # Volume Profile approximation
        df_vp = calculate_volume_profile(dataframe, period=self.range_period.value)
        dataframe['poc'] = df_vp['poc']
        dataframe['va_high'] = df_vp['value_area_high']
        dataframe['va_low'] = df_vp['value_area_low']

        # Breakout from Value Area
        dataframe['va_breakout_up'] = (
            (dataframe['close'] > dataframe['va_high'].shift(1)) &
            (dataframe['close'].shift(1) <= dataframe['va_high'].shift(2)) &
            (dataframe['volume_breakout'])
        )

        dataframe['va_breakout_down'] = (
            (dataframe['close'] < dataframe['va_low'].shift(1)) &
            (dataframe['close'].shift(1) >= dataframe['va_low'].shift(2)) &
            (dataframe['volume_breakout'])
        )

        # Measured move target
        dataframe['target_up'] = dataframe['range_high'] + dataframe['range_size']
        dataframe['target_down'] = dataframe['range_low'] - dataframe['range_size']

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Define entry signals for breakouts."""

        # LONG Entry: Breakout up with confirmation OR pullback to breakout level
        dataframe.loc[
            (
                # Recent consolidation
                (dataframe['consolidation'].shift(3) | dataframe['consolidation'].shift(5)) &

                # Breakout conditions (either direct or pullback)
                (
                    # Direct breakout with strong volume
                    (
                        (dataframe['breakout_up']) &
                        (dataframe['strong_candle']) &
                        (dataframe['volume_mult'] > 1.5)
                    ) |
                    # OR pullback to breakout level
                    (
                        (dataframe['pullback_to_breakout_up']) &
                        (dataframe['volume'] > dataframe['volume_sma'] * 0.8)
                    ) |
                    # OR Value Area breakout
                    (
                        (dataframe['va_breakout_up']) &
                        (dataframe['strong_candle'])
                    )
                ) &

                # Momentum confirmation
                (dataframe['rsi'] > 50) &
                (dataframe['macd_hist'] > 0) &

                # Volume exists
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1

        # SHORT Entry: Breakout down with confirmation
        dataframe.loc[
            (
                # Recent consolidation
                (dataframe['consolidation'].shift(3) | dataframe['consolidation'].shift(5)) &

                # Breakout conditions
                (
                    # Direct breakout
                    (
                        (dataframe['breakout_down']) &
                        (dataframe['strong_candle']) &
                        (dataframe['volume_mult'] > 1.5)
                    ) |
                    # OR pullback to breakout level
                    (
                        (dataframe['pullback_to_breakout_down']) &
                        (dataframe['volume'] > dataframe['volume_sma'] * 0.8)
                    ) |
                    # OR Value Area breakout
                    (
                        (dataframe['va_breakout_down']) &
                        (dataframe['strong_candle'])
                    )
                ) &

                # Momentum confirmation
                (dataframe['rsi'] < 50) &
                (dataframe['macd_hist'] < 0) &

                # Volume exists
                (dataframe['volume'] > 0)
            ),
            'enter_short'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Define exit signals."""

        # Exit LONG: Failed breakout or momentum loss
        dataframe.loc[
            (
                # Price back inside range
                (dataframe['close'] < dataframe['bb_middle']) |
                # Momentum reversal
                (dataframe['macd_hist'] < 0) & (dataframe['macd_hist'].shift(1) > 0) |
                # Volume drying up
                (dataframe['volume'] < dataframe['volume_sma'] * 0.5)
            ),
            'exit_long'
        ] = 1

        # Exit SHORT
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_middle']) |
                (dataframe['macd_hist'] > 0) & (dataframe['macd_hist'].shift(1) < 0) |
                (dataframe['volume'] < dataframe['volume_sma'] * 0.5)
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
        Dynamic stoploss based on breakout level.
        SL placed inside the range (failed breakout invalidates trade).
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) < 1:
            return None

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get('atr', 0)

        if atr == 0:
            return None

        if trade.is_short:
            # For short, SL above breakout level
            breakout_level = last_candle.get('breakout_level_down', 0)
            if breakout_level and breakout_level > 0:
                sl_distance = (breakout_level * 1.005 - current_rate) / current_rate
                if sl_distance > 0:
                    return -sl_distance
        else:
            # For long, SL below breakout level
            breakout_level = last_candle.get('breakout_level_up', 0)
            if breakout_level and breakout_level > 0:
                sl_distance = (current_rate - breakout_level * 0.995) / current_rate
                if sl_distance > 0:
                    return -sl_distance

        # Fallback to ATR-based SL
        return max(-(atr * 1.5) / current_rate, -0.025)

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
        Leverage based on breakout strength.
        Stronger breakouts (higher volume) = more confidence.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) < 1:
            return self.leverage_default

        last_candle = dataframe.iloc[-1]
        volume_mult = last_candle.get('volume_mult', 1)
        atr = last_candle.get('atr', 0)
        close = last_candle.get('close', 1)

        atr_pct = (atr / close) * 100

        # Higher volume = more confidence = can use more leverage
        if volume_mult > 2.0 and atr_pct < 2:
            leverage = 7
        elif volume_mult > 1.5:
            leverage = 5
        else:
            leverage = 4

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
        """Limit trades per day."""
        today = current_time.date()

        if today not in self.custom_trade_count:
            self.custom_trade_count[today] = 0

        if self.custom_trade_count[today] >= self.max_trades_per_day:
            return False

        self.custom_trade_count[today] += 1
        return True

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> Optional[str]:
        """
        Exit at measured move target.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) < 1:
            return None

        last_candle = dataframe.iloc[-1]

        # Check measured move targets
        if not trade.is_short:
            target = last_candle.get('target_up', float('inf'))
            if current_rate >= target * 0.99:
                return 'measured_move_target'
        else:
            target = last_candle.get('target_down', 0)
            if target > 0 and current_rate <= target * 1.01:
                return 'measured_move_target'

        # Breakout failure - price back in range
        range_mid = last_candle.get('range_mid', 0)
        if range_mid > 0:
            if not trade.is_short and current_rate < range_mid:
                return 'breakout_failure'
            elif trade.is_short and current_rate > range_mid:
                return 'breakout_failure'

        return None
