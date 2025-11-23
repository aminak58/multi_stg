"""
Hybrid BTC Day Trading Strategy for Futures (v2.0 - Order Flow Enhanced)
=========================================================================

This is the MAIN strategy that combines all three approaches:
1. Trend-Pullback: For trending markets (ADX > 25)
2. Liquidity Sweep: For ranging/choppy markets
3. Breakout Volume: For consolidation -> explosion phases

NEW in v2.0 - Order Flow Filters:
- CVD (Cumulative Volume Delta) confirmation
- Volume Imbalance detection
- Absorption pattern recognition
- Exhaustion detection for reversal avoidance
- Order Pressure analysis

The strategy automatically detects market regime and applies
the most appropriate entry logic WITH order flow confirmation.

Key Philosophy:
- Trending days -> Use Pullback entries (CVD must confirm)
- Ranging/Choppy days -> Hunt Liquidity Sweeps (Absorption confirmation)
- Squeeze/Consolidation -> Trade Breakouts (Imbalance confirmation)

Target: 1-3 high-quality trades per day with optimal R:R
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
from utils.indicators import (
    calculate_swing_points,
    detect_liquidity_zones,
    detect_market_structure,
    calculate_volume_profile,
    detect_engulfing_pattern,
    detect_rejection_candle
)
from utils.market_regime import MarketRegimeDetector, MarketRegime
from utils.orderflow import (
    get_order_flow_signals,
    calculate_cvd,
    calculate_volume_imbalance,
    detect_absorption,
    detect_exhaustion,
    calculate_order_pressure,
    analyze_order_book
)


class HybridBTCStrategy(IStrategy):
    """
    Adaptive Hybrid Strategy for BTC Futures.
    Automatically switches between three sub-strategies based on market conditions.
    """

    INTERFACE_VERSION = 3

    # Futures settings
    can_short = True
    trading_direction = 'both'

    # ROI - Balanced for all strategies
    minimal_roi = {
        "0": 0.06,    # 6% target
        "30": 0.04,   # 4% after 30 mins
        "60": 0.025,  # 2.5% after 1 hour
        "120": 0.015  # 1.5% after 2 hours
    }

    # Stoploss
    stoploss = -0.02  # 2% max
    use_custom_stoploss = True
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Timeframe
    timeframe = '15m'
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

    # ===== TREND-PULLBACK PARAMETERS =====
    ema_fast = IntParameter(15, 25, default=20, space='buy', load=True)
    ema_slow = IntParameter(40, 60, default=50, space='buy', load=True)

    # ===== LIQUIDITY SWEEP PARAMETERS =====
    swing_lookback = IntParameter(3, 8, default=5, space='buy', load=True)
    atr_sweep_mult = DecimalParameter(0.3, 0.8, default=0.5, space='buy', load=True)

    # ===== BREAKOUT PARAMETERS =====
    bb_period = IntParameter(15, 25, default=20, space='buy', load=True)
    squeeze_threshold = DecimalParameter(0.02, 0.05, default=0.03, space='buy', load=True)
    volume_breakout_mult = DecimalParameter(1.3, 2.0, default=1.5, space='buy', load=True)

    # ===== REGIME DETECTION =====
    adx_trend_threshold = IntParameter(20, 30, default=25, space='buy', load=True)

    # ===== ORDER FLOW PARAMETERS =====
    use_orderflow_filter = True  # Master switch for order flow filters
    of_imbalance_threshold = DecimalParameter(1.3, 2.0, default=1.5, space='buy', load=True)
    of_strong_imbalance = DecimalParameter(2.5, 4.0, default=3.0, space='buy', load=True)
    of_min_quality_score = IntParameter(40, 80, default=60, space='buy', load=True)

    # Trade limiting
    max_trades_per_day = 3
    custom_trade_count = {}
    custom_entry_tags = {}

    # Regime detector
    regime_detector = MarketRegimeDetector()

    def informative_pairs(self):
        return [(f"BTC/USDT:USDT", self.informative_timeframe)]

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Calculate ALL indicators for all three strategies."""

        # ========================================
        # MARKET REGIME DETECTION
        # ========================================
        df_regime = self.regime_detector.analyze(dataframe)
        dataframe['regime'] = df_regime['regime']
        dataframe['regime_numeric'] = df_regime['regime_numeric']
        dataframe['use_pullback'] = df_regime['use_pullback']
        dataframe['use_liquidity'] = df_regime['use_liquidity']
        dataframe['use_breakout'] = df_regime['use_breakout']

        # ========================================
        # TREND-PULLBACK INDICATORS
        # ========================================
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)

        dataframe['uptrend'] = dataframe['ema_fast'] > dataframe['ema_slow']
        dataframe['downtrend'] = dataframe['ema_fast'] < dataframe['ema_slow']

        # Pullback zones
        dataframe['pullback_zone_long'] = (
            (dataframe['close'] > dataframe['ema_slow']) &
            (dataframe['low'] <= dataframe['ema_fast'] * 1.005)
        )

        dataframe['pullback_zone_short'] = (
            (dataframe['close'] < dataframe['ema_slow']) &
            (dataframe['high'] >= dataframe['ema_fast'] * 0.995)
        )

        # ========================================
        # LIQUIDITY SWEEP INDICATORS
        # ========================================
        swing_high, swing_low = calculate_swing_points(
            dataframe, lookback=self.swing_lookback.value
        )
        dataframe['swing_high'] = swing_high
        dataframe['swing_low'] = swing_low

        # Swing levels
        dataframe['swing_high_level'] = np.where(swing_high, dataframe['high'], np.nan)
        dataframe['swing_low_level'] = np.where(swing_low, dataframe['low'], np.nan)
        dataframe['swing_high_level'] = dataframe['swing_high_level'].ffill()
        dataframe['swing_low_level'] = dataframe['swing_low_level'].ffill()

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # Liquidity sweeps
        dataframe['sweep_high'] = (
            (dataframe['high'] > dataframe['swing_high_level'].shift(1)) &
            (dataframe['close'] < dataframe['swing_high_level'].shift(1)) &
            (dataframe['high'] - dataframe[['open', 'close']].max(axis=1) >
             dataframe['atr'] * self.atr_sweep_mult.value)
        )

        dataframe['sweep_low'] = (
            (dataframe['low'] < dataframe['swing_low_level'].shift(1)) &
            (dataframe['close'] > dataframe['swing_low_level'].shift(1)) &
            (dataframe[['open', 'close']].min(axis=1) - dataframe['low'] >
             dataframe['atr'] * self.atr_sweep_mult.value)
        )

        # Market structure
        df_structure = detect_market_structure(dataframe, lookback=self.swing_lookback.value)
        dataframe['market_structure'] = df_structure['market_structure']
        dataframe['bos'] = df_structure['bos']
        dataframe['choch'] = df_structure['choch']

        # BMS after sweep
        dataframe['bms_bearish'] = (
            dataframe['sweep_high'].shift(1) &
            (dataframe['close'] < dataframe['low'].shift(1))
        )

        dataframe['bms_bullish'] = (
            dataframe['sweep_low'].shift(1) &
            (dataframe['close'] > dataframe['high'].shift(1))
        )

        # ========================================
        # BREAKOUT INDICATORS
        # ========================================
        bb = ta.BBANDS(dataframe, timeperiod=self.bb_period.value, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upper'] = bb['upperband']
        dataframe['bb_middle'] = bb['middleband']
        dataframe['bb_lower'] = bb['lowerband']

        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        dataframe['bb_squeeze'] = dataframe['bb_width'] < self.squeeze_threshold.value

        # Range detection
        dataframe['range_high'] = dataframe['high'].rolling(window=20).max()
        dataframe['range_low'] = dataframe['low'].rolling(window=20).min()
        dataframe['range_mid'] = (dataframe['range_high'] + dataframe['range_low']) / 2

        # Volume
        dataframe['volume_sma'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_spike'] = dataframe['volume'] > (
            dataframe['volume_sma'] * self.volume_breakout_mult.value
        )

        # Breakouts
        dataframe['breakout_up'] = (
            (dataframe['close'] > dataframe['range_high'].shift(1)) &
            (dataframe['volume_spike']) &
            (dataframe['bb_squeeze'].shift(3) | dataframe['bb_squeeze'].shift(5))
        )

        dataframe['breakout_down'] = (
            (dataframe['close'] < dataframe['range_low'].shift(1)) &
            (dataframe['volume_spike']) &
            (dataframe['bb_squeeze'].shift(3) | dataframe['bb_squeeze'].shift(5))
        )

        # ========================================
        # COMMON INDICATORS
        # ========================================
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        # Candlestick patterns
        bullish_engulf, bearish_engulf = detect_engulfing_pattern(dataframe)
        dataframe['bullish_engulfing'] = bullish_engulf
        dataframe['bearish_engulfing'] = bearish_engulf

        bullish_reject, bearish_reject = detect_rejection_candle(dataframe)
        dataframe['bullish_rejection'] = bullish_reject
        dataframe['bearish_rejection'] = bearish_reject

        # Higher Lows / Lower Highs
        dataframe['higher_low'] = (
            (dataframe['low'] > dataframe['low'].shift(1)) &
            (dataframe['low'].shift(1) > dataframe['low'].shift(2))
        )

        dataframe['lower_high'] = (
            (dataframe['high'] < dataframe['high'].shift(1)) &
            (dataframe['high'].shift(1) < dataframe['high'].shift(2))
        )

        # Volume decreasing (healthy pullback)
        dataframe['volume_decreasing'] = (
            (dataframe['volume'] < dataframe['volume'].shift(1)) &
            (dataframe['volume'].shift(1) < dataframe['volume'].shift(2))
        )

        # ========================================
        # ORDER FLOW INDICATORS (NEW)
        # ========================================
        if self.use_orderflow_filter:
            # Get comprehensive order flow signals
            df_of = get_order_flow_signals(dataframe)

            # CVD (Cumulative Volume Delta)
            dataframe['delta'] = df_of['delta']
            dataframe['cvd'] = df_of['cvd']
            dataframe['cvd_trend_up'] = df_of['cvd_trend_up']
            dataframe['cvd_trend_down'] = df_of['cvd_trend_down']
            dataframe['cvd_divergence_bullish'] = df_of['cvd_divergence_bullish']
            dataframe['cvd_divergence_bearish'] = df_of['cvd_divergence_bearish']

            # Volume Imbalance
            dataframe['buy_imbalance'] = df_of['buy_imbalance']
            dataframe['sell_imbalance'] = df_of['sell_imbalance']
            dataframe['strong_buy_imbalance'] = df_of['strong_buy_imbalance']
            dataframe['strong_sell_imbalance'] = df_of['strong_sell_imbalance']
            dataframe['stacked_buy_imbalance'] = df_of['stacked_buy_imbalance']
            dataframe['stacked_sell_imbalance'] = df_of['stacked_sell_imbalance']

            # Absorption
            dataframe['absorption_buy'] = df_of['absorption_buy']
            dataframe['absorption_sell'] = df_of['absorption_sell']
            dataframe['absorption_buy_consecutive'] = df_of['absorption_buy_consecutive']
            dataframe['absorption_sell_consecutive'] = df_of['absorption_sell_consecutive']

            # Exhaustion (reversal warning)
            dataframe['exhaustion_buy'] = df_of['exhaustion_buy']
            dataframe['exhaustion_sell'] = df_of['exhaustion_sell']
            dataframe['exhaustion_buy_divergence'] = df_of['exhaustion_buy_divergence']
            dataframe['exhaustion_sell_divergence'] = df_of['exhaustion_sell_divergence']

            # Order Pressure
            dataframe['pressure_imbalance'] = df_of['pressure_imbalance']
            dataframe['extreme_buy_pressure'] = df_of['extreme_buy_pressure']
            dataframe['extreme_sell_pressure'] = df_of['extreme_sell_pressure']

            # Composite signals
            dataframe['of_buy_signal'] = df_of['of_buy_signal']
            dataframe['of_sell_signal'] = df_of['of_sell_signal']
            dataframe['of_strong_buy_signal'] = df_of['of_strong_buy_signal']
            dataframe['of_strong_sell_signal'] = df_of['of_strong_sell_signal']
            dataframe['of_reversal_buy'] = df_of['of_reversal_buy']
            dataframe['of_reversal_sell'] = df_of['of_reversal_sell']

            # Quality scores
            dataframe['of_buy_quality'] = df_of['of_buy_quality']
            dataframe['of_sell_quality'] = df_of['of_sell_quality']
        else:
            # Default values when order flow is disabled
            dataframe['of_buy_signal'] = True
            dataframe['of_sell_signal'] = True
            dataframe['exhaustion_buy'] = False
            dataframe['exhaustion_sell'] = False
            dataframe['of_buy_quality'] = 100
            dataframe['of_sell_quality'] = 100

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        ADAPTIVE ENTRY LOGIC WITH ORDER FLOW FILTERS
        Selects entry method based on market regime + order flow confirmation.
        """

        # ========================================
        # ORDER FLOW FILTER CONDITIONS
        # ========================================
        # These filters ensure we only enter when order flow confirms

        # For LONG trades: CVD trending up, no exhaustion, quality score OK
        of_long_filter = (
            (~dataframe['exhaustion_buy']) &  # No buy exhaustion (avoid tops)
            (dataframe['of_buy_quality'] >= self.of_min_quality_score.value)
        )

        # For SHORT trades: CVD trending down, no exhaustion, quality score OK
        of_short_filter = (
            (~dataframe['exhaustion_sell']) &  # No sell exhaustion (avoid bottoms)
            (dataframe['of_sell_quality'] >= self.of_min_quality_score.value)
        )

        # ========================================
        # LONG ENTRIES
        # ========================================

        # 1. TREND-PULLBACK LONG (for trending markets)
        # Order Flow: CVD must confirm uptrend
        pullback_long = (
            (dataframe['use_pullback']) &
            (dataframe['uptrend']) &
            (dataframe['adx'] > self.adx_trend_threshold.value) &
            (dataframe['pullback_zone_long']) &
            (
                (dataframe['bullish_engulfing']) |
                (dataframe['bullish_rejection']) |
                (dataframe['higher_low'] & dataframe['volume_spike'])
            ) &
            (dataframe['rsi'] < 70) &
            (dataframe['volume'] > 0) &
            # ORDER FLOW FILTERS
            (of_long_filter) &
            (dataframe['cvd_trend_up'] | dataframe['buy_imbalance'])  # CVD or imbalance confirms
        )

        # 2. LIQUIDITY SWEEP LONG (for ranging/choppy markets)
        # Order Flow: Absorption pattern confirms (smart money absorbing sells)
        liquidity_long = (
            (dataframe['use_liquidity']) &
            (
                (dataframe['sweep_low'].shift(1)) |
                (dataframe['sweep_low'].shift(2))
            ) &
            (
                (dataframe['bms_bullish']) |
                (dataframe['bullish_rejection'] & dataframe['volume_spike'])
            ) &
            (dataframe['close'] > dataframe['open'].shift(1)) &
            (dataframe['rsi'] < 70) &
            (dataframe['volume'] > 0) &
            # ORDER FLOW FILTERS
            (of_long_filter) &
            (
                (dataframe['absorption_buy']) |  # Absorption confirms
                (dataframe['of_reversal_buy']) |  # Reversal signal
                (dataframe['stacked_buy_imbalance'])  # Strong imbalance
            )
        )

        # 3. BREAKOUT LONG (for consolidation phases)
        # Order Flow: Strong imbalance confirms breakout is real
        breakout_long = (
            (dataframe['use_breakout']) &
            (dataframe['breakout_up']) &
            (dataframe['rsi'] > 50) &
            (dataframe['volume'] > 0) &
            # ORDER FLOW FILTERS
            (of_long_filter) &
            (
                (dataframe['strong_buy_imbalance']) |  # Strong imbalance
                (dataframe['extreme_buy_pressure']) |  # Extreme pressure
                (dataframe['delta'] > 0)  # Positive delta on breakout
            )
        )

        # Combine all LONG conditions
        dataframe.loc[
            pullback_long | liquidity_long | breakout_long,
            'enter_long'
        ] = 1

        # Store entry tags for tracking
        dataframe['entry_tag_long'] = np.where(
            pullback_long, 'pullback',
            np.where(liquidity_long, 'liquidity',
            np.where(breakout_long, 'breakout', ''))
        )

        # ========================================
        # SHORT ENTRIES
        # ========================================

        # 1. TREND-PULLBACK SHORT
        # Order Flow: CVD must confirm downtrend
        pullback_short = (
            (dataframe['use_pullback']) &
            (dataframe['downtrend']) &
            (dataframe['adx'] > self.adx_trend_threshold.value) &
            (dataframe['pullback_zone_short']) &
            (
                (dataframe['bearish_engulfing']) |
                (dataframe['bearish_rejection']) |
                (dataframe['lower_high'] & dataframe['volume_spike'])
            ) &
            (dataframe['rsi'] > 30) &
            (dataframe['volume'] > 0) &
            # ORDER FLOW FILTERS
            (of_short_filter) &
            (dataframe['cvd_trend_down'] | dataframe['sell_imbalance'])  # CVD or imbalance confirms
        )

        # 2. LIQUIDITY SWEEP SHORT
        # Order Flow: Absorption pattern confirms (smart money absorbing buys)
        liquidity_short = (
            (dataframe['use_liquidity']) &
            (
                (dataframe['sweep_high'].shift(1)) |
                (dataframe['sweep_high'].shift(2))
            ) &
            (
                (dataframe['bms_bearish']) |
                (dataframe['bearish_rejection'] & dataframe['volume_spike'])
            ) &
            (dataframe['close'] < dataframe['open'].shift(1)) &
            (dataframe['rsi'] > 30) &
            (dataframe['volume'] > 0) &
            # ORDER FLOW FILTERS
            (of_short_filter) &
            (
                (dataframe['absorption_sell']) |  # Absorption confirms
                (dataframe['of_reversal_sell']) |  # Reversal signal
                (dataframe['stacked_sell_imbalance'])  # Strong imbalance
            )
        )

        # 3. BREAKOUT SHORT
        # Order Flow: Strong imbalance confirms breakout is real
        breakout_short = (
            (dataframe['use_breakout']) &
            (dataframe['breakout_down']) &
            (dataframe['rsi'] < 50) &
            (dataframe['volume'] > 0) &
            # ORDER FLOW FILTERS
            (of_short_filter) &
            (
                (dataframe['strong_sell_imbalance']) |  # Strong imbalance
                (dataframe['extreme_sell_pressure']) |  # Extreme pressure
                (dataframe['delta'] < 0)  # Negative delta on breakout
            )
        )

        # Combine all SHORT conditions
        dataframe.loc[
            pullback_short | liquidity_short | breakout_short,
            'enter_short'
        ] = 1

        dataframe['entry_tag_short'] = np.where(
            pullback_short, 'pullback',
            np.where(liquidity_short, 'liquidity',
            np.where(breakout_short, 'breakout', ''))
        )

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Define exit signals based on entry type + order flow."""

        # Exit LONG - Enhanced with Order Flow
        dataframe.loc[
            (
                # Trend reversal
                (dataframe['downtrend']) |
                # RSI overbought
                (dataframe['rsi'] > 78) |
                # Bearish reversal pattern
                (dataframe['bearish_engulfing']) |
                # Opposite sweep (liquidity target)
                (dataframe['sweep_high']) |
                # Failed breakout
                (dataframe['close'] < dataframe['range_mid']) |
                # ORDER FLOW EXIT SIGNALS
                (dataframe['exhaustion_buy_divergence']) |  # Exhaustion at top
                (dataframe['of_reversal_sell']) |  # Reversal signal
                (dataframe['cvd_divergence_bearish'])  # CVD diverging
            ),
            'exit_long'
        ] = 1

        # Exit SHORT - Enhanced with Order Flow
        dataframe.loc[
            (
                (dataframe['uptrend']) |
                (dataframe['rsi'] < 22) |
                (dataframe['bullish_engulfing']) |
                (dataframe['sweep_low']) |
                (dataframe['close'] > dataframe['range_mid']) |
                # ORDER FLOW EXIT SIGNALS
                (dataframe['exhaustion_sell_divergence']) |  # Exhaustion at bottom
                (dataframe['of_reversal_buy']) |  # Reversal signal
                (dataframe['cvd_divergence_bullish'])  # CVD diverging
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
        Adaptive stoploss based on entry type.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) < 1:
            return None

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get('atr', 0)

        if atr == 0:
            return None

        entry_tag = trade.enter_tag or ''

        # Different SL logic based on entry type
        if 'liquidity' in entry_tag:
            # Tight SL for liquidity trades (behind sweep)
            if trade.is_short:
                sweep_highs = dataframe.loc[dataframe['sweep_high']].tail(3)
                if len(sweep_highs) > 0:
                    sl_level = sweep_highs['high'].max()
                    return -((sl_level * 1.003 - current_rate) / current_rate)
            else:
                sweep_lows = dataframe.loc[dataframe['sweep_low']].tail(3)
                if len(sweep_lows) > 0:
                    sl_level = sweep_lows['low'].min()
                    return -((current_rate - sl_level * 0.997) / current_rate)

            return max(-(atr * 1.2) / current_rate, -0.015)

        elif 'breakout' in entry_tag:
            # SL inside the range (failed breakout)
            if trade.is_short:
                range_high = last_candle.get('range_high', current_rate * 1.02)
                return -((range_high - current_rate) / current_rate)
            else:
                range_low = last_candle.get('range_low', current_rate * 0.98)
                return -((current_rate - range_low) / current_rate)

        else:  # Pullback or default
            # ATR-based SL
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
        Leverage based on strategy type and market conditions.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) < 1:
            return self.leverage_default

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get('atr', 0)
        close = last_candle.get('close', 1)
        adx = last_candle.get('adx', 0)

        atr_pct = (atr / close) * 100

        # Base leverage on volatility
        if atr_pct > 3:
            base_leverage = 3
        elif atr_pct > 2:
            base_leverage = 5
        else:
            base_leverage = 7

        # Adjust based on entry type
        entry_tag = entry_tag or ''
        if 'liquidity' in entry_tag:
            # Liquidity trades have tight SL, can use more leverage
            leverage = min(base_leverage + 1, 8)
        elif 'breakout' in entry_tag:
            # Breakouts are explosive but risky
            leverage = base_leverage
        else:
            # Pullbacks are conservative
            leverage = max(base_leverage - 1, 3)

        # Higher ADX = more confident in trend
        if adx > 30 and 'pullback' in entry_tag:
            leverage = min(leverage + 1, 8)

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
        """Limit trades per day and validate entry."""
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
        Custom exit logic based on entry type.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) < 1:
            return None

        last_candle = dataframe.iloc[-1]
        entry_tag = trade.enter_tag or ''

        # Time-based stagnant exit
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600
        if trade_duration > 4 and abs(current_profit) < 0.005:
            return 'time_exit_stagnant'

        # Liquidity trades: exit at opposite liquidity
        if 'liquidity' in entry_tag:
            if not trade.is_short:
                swing_high = last_candle.get('swing_high_level', float('inf'))
                if current_rate >= swing_high * 0.998:
                    return 'liquidity_target'
            else:
                swing_low = last_candle.get('swing_low_level', 0)
                if swing_low > 0 and current_rate <= swing_low * 1.002:
                    return 'liquidity_target'

        # Breakout trades: exit on measured move
        if 'breakout' in entry_tag:
            range_size = last_candle.get('range_high', 0) - last_candle.get('range_low', 0)
            if not trade.is_short:
                target = last_candle.get('range_high', 0) + range_size
                if target > 0 and current_rate >= target * 0.99:
                    return 'measured_move'
            else:
                target = last_candle.get('range_low', float('inf')) - range_size
                if target > 0 and current_rate <= target * 1.01:
                    return 'measured_move'

        return None

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
        Slightly better entry price using limit orders.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) < 1:
            return proposed_rate

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get('atr', 0)

        # Small adjustment for better fill
        adjustment = atr * 0.05

        if side == 'long':
            return proposed_rate - adjustment
        else:
            return proposed_rate + adjustment
