# BTC Futures Day Trading - Implementation Roadmap

## Overview

This project implements three professional-grade Bitcoin day trading strategies for Freqtrade futures trading, designed for 1-3 high-quality trades per day.

## Architecture

```
multi_stg/
├── strategies/
│   ├── __init__.py
│   ├── trend_pullback.py      # Strategy 1: EMA Pullback
│   ├── liquidity_sweep.py     # Strategy 2: ICT/SMC Liquidity
│   ├── breakout_volume.py     # Strategy 3: Range Breakout
│   └── hybrid_btc_strategy.py # Main: Adaptive Hybrid
├── utils/
│   ├── __init__.py
│   ├── indicators.py          # Custom technical indicators
│   └── market_regime.py       # Market regime detection
├── configs/
│   ├── config_futures.json    # Live/Dry-run config
│   ├── config_backtest.json   # Backtesting config
│   └── hyperopt_config.json   # Optimization config
├── tests/
│   └── test_strategies.py     # Unit tests
└── docs/
    └── ROADMAP.md             # This file
```

---

## Strategy Details

### 1. Trend-Pullback Strategy (`trend_pullback.py`)

**Best For:** Trending market days (ADX > 25)

**Logic:**
- Detect trend using EMA20/EMA50 crossover
- Wait for price to pullback to EMA zone
- Enter on confirmation (engulfing, rejection, structure)

**Entry Conditions (LONG):**
```python
- EMA20 > EMA50 (uptrend)
- ADX > 20 (trend strength)
- Price pulls back to EMA zone
- Bullish engulfing OR rejection candle OR Higher Low + volume spike
- RSI < 70
```

**Entry Conditions (SHORT):**
```python
- EMA20 < EMA50 (downtrend)
- ADX > 20
- Price pulls back to EMA zone
- Bearish engulfing OR rejection OR Lower High + volume
- RSI > 30
```

**Risk Management:**
- ATR-based dynamic stoploss (1.5x ATR)
- Trailing stop: 1% positive, 2% offset
- Max 3 trades per day

---

### 2. Liquidity Sweep Strategy (`liquidity_sweep.py`)

**Best For:** Ranging/choppy market days

**Logic (ICT/SMC):**
- Identify swing highs/lows (liquidity zones)
- Wait for price to "sweep" liquidity (wick through level)
- Enter on reversal after Break of Market Structure (BMS)

**Entry Conditions (LONG):**
```python
- Sweep low detected (wick below swing low, close above)
- BMS bullish (close > previous high) OR bullish rejection + volume
- Active trading session (London/NY)
- RSI < 70
```

**Entry Conditions (SHORT):**
```python
- Sweep high detected (wick above swing high, close below)
- BMS bearish (close < previous low) OR bearish rejection + volume
- Active trading session
- RSI > 30
```

**Risk Management:**
- Stoploss behind sweep level (tight SL)
- Target: Opposite liquidity zone
- High R:R ratio (typically 1:2 to 1:4)

---

### 3. Breakout Volume Strategy (`breakout_volume.py`)

**Best For:** Consolidation → Explosion phases

**Logic:**
- Detect consolidation (BB squeeze, low ATR)
- Wait for breakout with volume confirmation
- Enter directly or on pullback to breakout level

**Entry Conditions (LONG):**
```python
- Recent consolidation (BB squeeze in last 3-5 candles)
- Breakout above range high
- Volume > 1.5x average (volume spike)
- Strong candle (body > 50% of range)
- RSI > 50, MACD histogram > 0
```

**Entry Conditions (SHORT):**
```python
- Recent consolidation
- Breakout below range low
- Volume spike
- RSI < 50, MACD histogram < 0
```

**Risk Management:**
- Stoploss inside range (failed breakout invalidates trade)
- Target: Measured move (range height from breakout)

---

### 4. Hybrid Strategy (`hybrid_btc_strategy.py`) - RECOMMENDED

**This is the main strategy that adaptively combines all three approaches.**

**Market Regime Detection:**
```python
TRENDING (ADX > 25):     → Use Trend-Pullback
RANGING (ADX < 20):      → Use Liquidity Sweep
SQUEEZE (BB width low):  → Use Breakout
VOLATILE (ATR > 1.5x):   → Use Breakout or Liquidity
```

**Adaptive Logic:**
- Analyzes current market regime automatically
- Applies the most appropriate entry logic
- Manages stoploss and targets based on entry type

---

## Implementation Phases

### Phase 1: Setup & Data (Day 1-2)
```bash
# Install Freqtrade
pip install freqtrade

# Clone this repo
cd /path/to/multi_stg

# Download historical data
freqtrade download-data \
    --exchange binance \
    --pairs BTC/USDT:USDT \
    --timeframes 5m 15m 1h \
    --days 365 \
    --trading-mode futures
```

### Phase 2: Backtesting (Day 3-7)
```bash
# Backtest Hybrid Strategy
freqtrade backtesting \
    --config configs/config_backtest.json \
    --strategy HybridBTCStrategy \
    --timerange 20230101-20231231

# Backtest individual strategies
freqtrade backtesting \
    --config configs/config_backtest.json \
    --strategy TrendPullbackStrategy

freqtrade backtesting \
    --config configs/config_backtest.json \
    --strategy LiquiditySweepStrategy

freqtrade backtesting \
    --config configs/config_backtest.json \
    --strategy BreakoutVolumeStrategy
```

### Phase 3: Optimization (Day 8-14)
```bash
# Hyperparameter optimization
freqtrade hyperopt \
    --config configs/hyperopt_config.json \
    --strategy HybridBTCStrategy \
    --hyperopt-loss SharpeHyperOptLoss \
    --spaces buy sell \
    --epochs 500 \
    --timerange 20230101-20231231
```

### Phase 4: Dry-Run (Day 15-45)
```bash
# Start dry-run (paper trading)
freqtrade trade \
    --config configs/config_futures.json \
    --strategy HybridBTCStrategy
```

**Minimum 30 days dry-run recommended before live trading!**

### Phase 5: Live Trading
```bash
# Update config with real API keys
# Set dry_run: false
# Start with small capital

freqtrade trade \
    --config configs/config_futures.json \
    --strategy HybridBTCStrategy
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Win Rate | 45-55% | Quality over quantity |
| Risk:Reward | 1:1.5 - 1:3 | Varies by strategy |
| Max Drawdown | < 15% | Critical limit |
| Daily Trades | 1-3 | Hard limit |
| Monthly Return | 5-15% | Conservative target |

---

## Risk Management Rules

1. **Position Sizing:** Max 2% risk per trade
2. **Leverage:** 3-7x based on volatility (adaptive)
3. **Daily Limit:** Maximum 3 trades per day
4. **Stoploss:** Always active, never remove
5. **Correlation:** Only BTC/USDT (no multiple positions)

---

## Monitoring & Maintenance

### Daily Checklist
- [ ] Check open positions
- [ ] Review trade journal
- [ ] Verify strategy performance vs backtest
- [ ] Check for any anomalies

### Weekly Review
- [ ] Analyze win/loss ratio
- [ ] Review entry quality
- [ ] Adjust parameters if needed
- [ ] Compare regime detection accuracy

### Monthly Audit
- [ ] Full performance review
- [ ] Re-backtest with recent data
- [ ] Consider re-optimization
- [ ] Risk assessment

---

## Troubleshooting

### Common Issues

1. **Too many signals:** Increase ADX threshold, tighten pullback zone
2. **No signals:** Decrease thresholds, widen zones
3. **High drawdown:** Reduce leverage, tighten stoploss
4. **Whipsaws:** Add confirmation filters, increase volume requirements

### Log Analysis
```bash
# Check strategy logs
freqtrade show-trades --db-url sqlite:///tradesv3.sqlite
```

---

## Disclaimer

This strategy is for educational purposes. Always:
- Backtest thoroughly before live trading
- Start with small capital
- Never risk more than you can afford to lose
- Past performance doesn't guarantee future results
