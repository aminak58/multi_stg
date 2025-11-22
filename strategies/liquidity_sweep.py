"""
Liquidity Sweep + Reversal Strategy for BTC Futures
====================================================

Strategy Logic (ICT/SMC Based):
- Identify liquidity zones (swing highs/lows, session levels)
- Wait for price to sweep liquidity (wick through level)
- Enter on reversal after Break of Market Structure (BMS)
- Small SL behind sweep, targeting opposite liquidity

Key Concepts:
- Liquidity Sweep: Price spikes through a key level, taking out stops
- BMS (Break of Market Structure): Confirmation of reversal
- Order Block: Zone where price originated the sweep

Features:
- Supports LONG and SHORT positions (Futures)
- Session-based liquidity (London, NY opens)
- ATR validation for sweep authenticity
- High R:R trades (typically 1:2 to 1:4)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from typing import Optional, Dict, List
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.persistence import Trade
import talib.abstract as ta

import sys
sys.path.append('/home/user/multi_stg')
from utils.indicators import (
    calculate_swing_points,
    detect_liquidity_zones,
    detect_market_structure,
    calculate_atr_bands
)


class LiquiditySweepStrategy(IStrategy):
    """
    Liquidity Sweep Strategy for BTC Futures.
    Based on ICT/SMC concepts - hunting stops and reversing.
    Designed for 1-3 high R:R trades per day.
    """

    INTERFACE_VERSION = 3

    # Futures settings
    can_short = True
    trading_direction = 'both'

    # ROI - Higher targets due to high R:R nature
    minimal_roi = {
        "0": 0.08,    # 8% target
        "60": 0.05,   # 5% after 1 hour
        "120": 0.03,  # 3% after 2 hours
        "240": 0.02   # 2% after 4 hours
    }

    # Tight stoploss (key feature of this strategy)
    stoploss = -0.015  # 1.5% max
    use_custom_stoploss = True
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    # Timeframe
    timeframe = '5m'
    informative_timeframe = '1h'

    process_only_new_candles = True
    startup_candle_count = 200

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
    swing_lookback = IntParameter(3, 10, default=5, space='buy')
    atr_sweep_mult = DecimalParameter(0.3, 1.0, default=0.5, space='buy')
    bms_confirmation_bars = IntParameter(1, 5, default=2, space='buy')
    min_sweep_wick_ratio = DecimalParameter(1.5, 3.0, default=2.0, space='buy')

    # Session times (UTC) - Key liquidity periods
    london_open = time(8, 0)
    london_close = time(16, 0)
    ny_open = time(13, 0)
    ny_close = time(21, 0)

    # Trade limiting
    max_trades_per_day = 3
    custom_trade_count = {}

    def informative_pairs(self):
        return [(f"BTC/USDT:USDT", self.informative_timeframe)]

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Calculate liquidity and structure indicators."""

        # Swing points for liquidity zones
        swing_high, swing_low = calculate_swing_points(
            dataframe,
            lookback=self.swing_lookback.value
        )
        dataframe['swing_high'] = swing_high
        dataframe['swing_low'] = swing_low

        # Track swing levels
        dataframe['swing_high_level'] = np.where(swing_high, dataframe['high'], np.nan)
        dataframe['swing_low_level'] = np.where(swing_low, dataframe['low'], np.nan)
        dataframe['swing_high_level'] = dataframe['swing_high_level'].ffill()
        dataframe['swing_low_level'] = dataframe['swing_low_level'].ffill()

        # ATR for validation
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # Session-based levels (rolling 24-period high/low for approximate session)
        dataframe['session_high'] = dataframe['high'].rolling(window=24).max()
        dataframe['session_low'] = dataframe['low'].rolling(window=24).min()

        # Detect liquidity sweeps
        # Sweep High: Wick above swing high but close below
        dataframe['sweep_high'] = (
            (dataframe['high'] > dataframe['swing_high_level'].shift(1)) &
            (dataframe['close'] < dataframe['swing_high_level'].shift(1)) &
            (dataframe['high'] - dataframe[['open', 'close']].max(axis=1) >
             dataframe['atr'] * self.atr_sweep_mult.value)
        )

        # Sweep Low: Wick below swing low but close above
        dataframe['sweep_low'] = (
            (dataframe['low'] < dataframe['swing_low_level'].shift(1)) &
            (dataframe['close'] > dataframe['swing_low_level'].shift(1)) &
            (dataframe[['open', 'close']].min(axis=1) - dataframe['low'] >
             dataframe['atr'] * self.atr_sweep_mult.value)
        )

        # Market structure detection
        df_structure = detect_market_structure(dataframe, lookback=self.swing_lookback.value)
        dataframe['market_structure'] = df_structure['market_structure']
        dataframe['bos'] = df_structure['bos']
        dataframe['choch'] = df_structure['choch']

        # Break of Market Structure (BMS) after sweep
        # Bearish BMS after sweep high
        dataframe['bms_bearish'] = (
            dataframe['sweep_high'].shift(1) &
            (dataframe['close'] < dataframe['low'].shift(1))
        )

        # Bullish BMS after sweep low
        dataframe['bms_bullish'] = (
            dataframe['sweep_low'].shift(1) &
            (dataframe['close'] > dataframe['high'].shift(1))
        )

        # Volume analysis
        dataframe['volume_sma'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['high_volume'] = dataframe['volume'] > dataframe['volume_sma'] * 1.5

        # Candle body analysis
        dataframe['body'] = abs(dataframe['close'] - dataframe['open'])
        dataframe['upper_wick'] = dataframe['high'] - dataframe[['open', 'close']].max(axis=1)
        dataframe['lower_wick'] = dataframe[['open', 'close']].min(axis=1) - dataframe['low']

        # Strong rejection (long wick)
        min_body = dataframe['body'].replace(0, 0.0001)
        dataframe['bearish_rejection'] = (
            (dataframe['upper_wick'] / min_body > self.min_sweep_wick_ratio.value) &
            (dataframe['close'] < dataframe['open'])
        )

        dataframe['bullish_rejection'] = (
            (dataframe['lower_wick'] / min_body > self.min_sweep_wick_ratio.value) &
            (dataframe['close'] > dataframe['open'])
        )

        # Order Block detection (last bullish candle before bearish move, and vice versa)
        dataframe['bullish_ob'] = (
            (dataframe['close'].shift(1) > dataframe['open'].shift(1)) &
            (dataframe['close'] < dataframe['open']) &
            (dataframe['low'] < dataframe['low'].shift(1))
        )

        dataframe['bearish_ob'] = (
            (dataframe['close'].shift(1) < dataframe['open'].shift(1)) &
            (dataframe['close'] > dataframe['open']) &
            (dataframe['high'] > dataframe['high'].shift(1))
        )

        # RSI for confluence
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Time-based filter (session activity)
        if 'date' in dataframe.columns:
            dataframe['hour'] = pd.to_datetime(dataframe['date']).dt.hour
            dataframe['active_session'] = (
                ((dataframe['hour'] >= 8) & (dataframe['hour'] <= 16)) |  # London
                ((dataframe['hour'] >= 13) & (dataframe['hour'] <= 21))   # NY
            )
        else:
            dataframe['active_session'] = True

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Define entry signals based on liquidity sweeps."""

        # LONG Entry: After sweep low + bullish BMS
        dataframe.loc[
            (
                # Liquidity sweep occurred
                (
                    (dataframe['sweep_low'].shift(1)) |
                    (dataframe['sweep_low'].shift(2))
                ) &

                # Structure break confirmation
                (
                    (dataframe['bms_bullish']) |
                    (dataframe['bullish_rejection'] & dataframe['high_volume'])
                ) &

                # Close above sweep candle's body
                (dataframe['close'] > dataframe['open'].shift(1)) &

                # Volume confirmation
                (dataframe['volume'] > 0) &

                # RSI not extreme
                (dataframe['rsi'] < 70) &

                # Active session preferred
                (dataframe['active_session'])
            ),
            'enter_long'
        ] = 1

        # SHORT Entry: After sweep high + bearish BMS
        dataframe.loc[
            (
                # Liquidity sweep occurred
                (
                    (dataframe['sweep_high'].shift(1)) |
                    (dataframe['sweep_high'].shift(2))
                ) &

                # Structure break confirmation
                (
                    (dataframe['bms_bearish']) |
                    (dataframe['bearish_rejection'] & dataframe['high_volume'])
                ) &

                # Close below sweep candle's body
                (dataframe['close'] < dataframe['open'].shift(1)) &

                # Volume confirmation
                (dataframe['volume'] > 0) &

                # RSI not extreme
                (dataframe['rsi'] > 30) &

                # Active session preferred
                (dataframe['active_session'])
            ),
            'enter_short'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Define exit signals."""

        # Exit LONG at opposite liquidity or structure break
        dataframe.loc[
            (
                (dataframe['sweep_high']) |  # Reached opposite liquidity
                (dataframe['bms_bearish']) |  # Structure break against
                (dataframe['rsi'] > 75)  # Overbought
            ),
            'exit_long'
        ] = 1

        # Exit SHORT at opposite liquidity or structure break
        dataframe.loc[
            (
                (dataframe['sweep_low']) |  # Reached opposite liquidity
                (dataframe['bms_bullish']) |  # Structure break against
                (dataframe['rsi'] < 25)  # Oversold
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
        Dynamic stoploss placed behind the liquidity sweep.
        This is the key to small SL / high R:R.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) < 1:
            return None

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get('atr', 0)

        if atr == 0:
            return None

        # Find the sweep level for SL placement
        if trade.is_short:
            # For short, SL above the sweep high
            sweep_level = dataframe.loc[dataframe['sweep_high']].tail(3)
            if len(sweep_level) > 0:
                sl_price = sweep_level['high'].max()
                sl_distance = (sl_price - current_rate) / current_rate
                return min(-sl_distance - 0.002, -0.005)  # Min 0.5% above sweep
        else:
            # For long, SL below the sweep low
            sweep_level = dataframe.loc[dataframe['sweep_low']].tail(3)
            if len(sweep_level) > 0:
                sl_price = sweep_level['low'].min()
                sl_distance = (current_rate - sl_price) / current_rate
                return min(-sl_distance - 0.002, -0.005)  # Min 0.5% below sweep

        # Default ATR-based SL
        return max(-(atr * 1.5) / current_rate, -0.02)

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
        Leverage based on expected R:R.
        Higher R:R potential = can use slightly higher leverage.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) < 1:
            return self.leverage_default

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get('atr', 0)
        close = last_candle.get('close', 1)

        atr_pct = (atr / close) * 100

        # Conservative leverage due to tight stops
        if atr_pct > 2.5:
            leverage = 3
        elif atr_pct > 1.5:
            leverage = 5
        else:
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
        Custom exit for reaching opposite liquidity target.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) < 1:
            return None

        last_candle = dataframe.iloc[-1]

        # For long trades, exit at swing high (opposite liquidity)
        if not trade.is_short:
            swing_high = last_candle.get('swing_high_level', 0)
            if current_rate >= swing_high * 0.998:  # Within 0.2% of target
                return 'liquidity_target_reached'

        # For short trades, exit at swing low
        else:
            swing_low = last_candle.get('swing_low_level', float('inf'))
            if current_rate <= swing_low * 1.002:
                return 'liquidity_target_reached'

        # Time-based exit if trade is stagnant
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600
        if trade_duration > 4 and abs(current_profit) < 0.005:
            return 'time_exit_stagnant'

        return None
