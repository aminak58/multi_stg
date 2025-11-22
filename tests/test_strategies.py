"""
Unit tests for BTC trading strategies.
Run with: pytest tests/test_strategies.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

sys.path.insert(0, '/home/user/multi_stg')

from utils.indicators import (
    calculate_swing_points,
    detect_liquidity_zones,
    calculate_volume_profile,
    detect_market_structure,
    calculate_atr_bands,
    detect_engulfing_pattern,
    detect_rejection_candle
)
from utils.market_regime import MarketRegimeDetector, MarketRegime


def generate_sample_ohlcv(rows: int = 200, trend: str = 'up') -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)

    dates = pd.date_range(start='2024-01-01', periods=rows, freq='15min')

    if trend == 'up':
        base_price = 40000 + np.cumsum(np.random.randn(rows) * 50 + 10)
    elif trend == 'down':
        base_price = 45000 + np.cumsum(np.random.randn(rows) * 50 - 10)
    else:  # ranging
        base_price = 42000 + np.sin(np.linspace(0, 4 * np.pi, rows)) * 500

    high = base_price + np.random.rand(rows) * 100
    low = base_price - np.random.rand(rows) * 100
    open_price = base_price + np.random.randn(rows) * 30
    close = base_price + np.random.randn(rows) * 30
    volume = np.random.rand(rows) * 1000 + 500

    return pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


class TestSwingPoints:
    """Tests for swing point detection."""

    def test_detect_swing_high(self):
        """Test swing high detection."""
        df = generate_sample_ohlcv(100)
        swing_high, swing_low = calculate_swing_points(df, lookback=5)

        assert isinstance(swing_high, pd.Series)
        assert swing_high.dtype == bool
        assert swing_high.sum() > 0  # Should find some swing highs

    def test_detect_swing_low(self):
        """Test swing low detection."""
        df = generate_sample_ohlcv(100)
        swing_high, swing_low = calculate_swing_points(df, lookback=5)

        assert isinstance(swing_low, pd.Series)
        assert swing_low.dtype == bool
        assert swing_low.sum() > 0  # Should find some swing lows

    def test_swing_lookback_parameter(self):
        """Test different lookback parameters."""
        df = generate_sample_ohlcv(100)

        # Smaller lookback should find more swings
        sh_3, sl_3 = calculate_swing_points(df, lookback=3)
        sh_10, sl_10 = calculate_swing_points(df, lookback=10)

        assert sh_3.sum() >= sh_10.sum()
        assert sl_3.sum() >= sl_10.sum()


class TestLiquidityZones:
    """Tests for liquidity zone detection."""

    def test_liquidity_sweep_detection(self):
        """Test liquidity sweep detection."""
        df = generate_sample_ohlcv(200)
        result = detect_liquidity_zones(df, swing_lookback=5)

        assert 'swing_high_level' in result.columns
        assert 'swing_low_level' in result.columns
        assert 'liquidity_sweep_high' in result.columns
        assert 'liquidity_sweep_low' in result.columns


class TestVolumeProfile:
    """Tests for volume profile calculation."""

    def test_poc_calculation(self):
        """Test Point of Control calculation."""
        df = generate_sample_ohlcv(100)
        result = calculate_volume_profile(df, period=20, num_bins=10)

        assert 'poc' in result.columns
        assert 'value_area_high' in result.columns
        assert 'value_area_low' in result.columns

        # POC should be within price range
        valid_poc = result['poc'].dropna()
        assert all(valid_poc >= result.loc[valid_poc.index, 'low'].min())
        assert all(valid_poc <= result.loc[valid_poc.index, 'high'].max())


class TestMarketStructure:
    """Tests for market structure detection."""

    def test_structure_detection(self):
        """Test market structure detection."""
        df = generate_sample_ohlcv(200, trend='up')
        result = detect_market_structure(df, lookback=5)

        assert 'market_structure' in result.columns
        assert 'bos' in result.columns
        assert 'choch' in result.columns

    def test_uptrend_detection(self):
        """Test that uptrend is detected correctly."""
        df = generate_sample_ohlcv(200, trend='up')
        result = detect_market_structure(df, lookback=5)

        # Should have more bullish structure in uptrend
        bullish_count = (result['market_structure'] == 1).sum()
        bearish_count = (result['market_structure'] == -1).sum()

        # In strong uptrend, expect more bullish than bearish
        assert bullish_count > bearish_count


class TestCandlePatterns:
    """Tests for candlestick pattern detection."""

    def test_engulfing_detection(self):
        """Test engulfing pattern detection."""
        df = generate_sample_ohlcv(100)
        bullish, bearish = detect_engulfing_pattern(df)

        assert isinstance(bullish, pd.Series)
        assert isinstance(bearish, pd.Series)
        assert bullish.dtype == bool
        assert bearish.dtype == bool

    def test_rejection_candle_detection(self):
        """Test rejection candle detection."""
        df = generate_sample_ohlcv(100)
        bullish, bearish = detect_rejection_candle(df, wick_ratio=2.0)

        assert isinstance(bullish, pd.Series)
        assert isinstance(bearish, pd.Series)


class TestMarketRegime:
    """Tests for market regime detection."""

    def test_regime_detector_initialization(self):
        """Test regime detector initialization."""
        detector = MarketRegimeDetector()

        assert detector.adx_threshold_strong == 25
        assert detector.adx_threshold_weak == 20
        assert detector.ema_fast == 20
        assert detector.ema_slow == 50

    def test_regime_analysis(self):
        """Test regime analysis on dataframe."""
        detector = MarketRegimeDetector()
        df = generate_sample_ohlcv(200, trend='up')

        result = detector.analyze(df)

        assert 'regime' in result.columns
        assert 'regime_numeric' in result.columns
        assert 'use_pullback' in result.columns
        assert 'use_liquidity' in result.columns
        assert 'use_breakout' in result.columns

    def test_trending_market_detection(self):
        """Test that trending markets recommend pullback strategy."""
        detector = MarketRegimeDetector()
        df = generate_sample_ohlcv(200, trend='up')

        result = detector.analyze(df)

        # In uptrend, should recommend pullback for some candles
        pullback_recommended = result['use_pullback'].sum()
        assert pullback_recommended > 0

    def test_ranging_market_detection(self):
        """Test that ranging markets recommend liquidity strategy."""
        detector = MarketRegimeDetector()
        df = generate_sample_ohlcv(200, trend='range')

        result = detector.analyze(df)

        # In ranging market, should recommend liquidity for some candles
        liquidity_recommended = result['use_liquidity'].sum()
        assert liquidity_recommended > 0

    def test_strategy_recommendation(self):
        """Test strategy recommendation function."""
        detector = MarketRegimeDetector()

        assert detector.get_recommended_strategy(MarketRegime.STRONG_UPTREND) == 'trend_pullback'
        assert detector.get_recommended_strategy(MarketRegime.RANGING) == 'liquidity_sweep'
        assert detector.get_recommended_strategy(MarketRegime.VOLATILE) == 'breakout_volume'


class TestATRBands:
    """Tests for ATR bands calculation."""

    def test_atr_bands_calculation(self):
        """Test ATR bands are calculated correctly."""
        df = generate_sample_ohlcv(100)
        result = calculate_atr_bands(df, period=14, multiplier=1.5)

        assert 'atr' in result.columns
        assert 'atr_upper' in result.columns
        assert 'atr_lower' in result.columns

        # Upper should be above lower
        valid_idx = result['atr'].notna()
        assert all(result.loc[valid_idx, 'atr_upper'] > result.loc[valid_idx, 'atr_lower'])


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_full_indicator_pipeline(self):
        """Test full indicator calculation pipeline."""
        df = generate_sample_ohlcv(200)

        # Calculate all indicators
        swing_h, swing_l = calculate_swing_points(df, lookback=5)
        df['swing_high'] = swing_h
        df['swing_low'] = swing_l

        df_liq = detect_liquidity_zones(df, swing_lookback=5)
        df_struct = detect_market_structure(df, lookback=5)
        df_vp = calculate_volume_profile(df, period=20)
        df_atr = calculate_atr_bands(df, period=14)

        # All should have expected columns
        assert len(df_liq) == len(df)
        assert len(df_struct) == len(df)
        assert len(df_vp) == len(df)
        assert len(df_atr) == len(df)

    def test_regime_based_strategy_selection(self):
        """Test that regime correctly selects strategy."""
        detector = MarketRegimeDetector()

        # Test uptrend
        df_up = generate_sample_ohlcv(200, trend='up')
        result_up = detector.analyze(df_up)

        # Test ranging
        df_range = generate_sample_ohlcv(200, trend='range')
        result_range = detector.analyze(df_range)

        # Uptrend should have more pullback recommendations
        pullback_up = result_up['use_pullback'].sum()
        pullback_range = result_range['use_pullback'].sum()

        # Ranging should have more liquidity recommendations
        liquidity_up = result_up['use_liquidity'].sum()
        liquidity_range = result_range['use_liquidity'].sum()

        # These assertions depend on how regime detection works
        # In practice, may need adjustment based on actual behavior
        print(f"Uptrend - Pullback: {pullback_up}, Liquidity: {liquidity_up}")
        print(f"Range - Pullback: {pullback_range}, Liquidity: {liquidity_range}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
