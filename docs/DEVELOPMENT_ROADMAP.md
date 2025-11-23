# BTC Futures Day Trading Bot - Development Roadmap

## Version: 2.0 | Last Updated: November 2025

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Phase 1: Initial Validation (Weeks 1-2)](#phase-1-initial-validation)
3. [Phase 2: Backtesting & Optimization (Weeks 3-4)](#phase-2-backtesting--optimization)
4. [Phase 3: Dry-Run Testing (Weeks 5-8)](#phase-3-dry-run-testing)
5. [Phase 4: Performance Analysis (Weeks 9-10)](#phase-4-performance-analysis)
6. [Phase 5: FreqAI Integration (Weeks 11-16)](#phase-5-freqai-integration)
7. [Future Goals](#future-goals)
8. [Success Metrics](#success-metrics)
9. [Risk Management Guidelines](#risk-management-guidelines)

---

## Project Overview

This project implements an adaptive hybrid trading strategy for BTC/USDT futures, combining three distinct approaches:

| Strategy | Market Condition | Key Indicators |
|----------|------------------|----------------|
| Trend-Pullback | Trending (ADX > 25) | EMA crossover, RSI |
| Liquidity Sweep | Ranging/Choppy | Swing points, BMS |
| Breakout Volume | Consolidation | BB Squeeze, Volume |

**Current Status**: Ready for Dry-Run Testing

---

## Phase 1: Initial Validation

### Duration: Weeks 1-2

### Objectives

- [ ] Verify strategy loads without errors
- [ ] Confirm all indicators calculate correctly
- [ ] Validate configuration settings
- [ ] Run initial backtest on 30 days of data

### Tasks

#### 1.1 Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 1.2 Data Download
```bash
# Download 365 days of historical data
freqtrade download-data \
    --exchange binance \
    --pairs BTC/USDT:USDT \
    --timeframes 5m 15m 1h 4h \
    --days 365
```

#### 1.3 Initial Backtest
```bash
# Quick validation backtest (30 days)
freqtrade backtesting \
    --config configs/config_backtest.json \
    --strategy HybridBTCStrategy \
    --timerange 20241001-20241101
```

### Success Criteria

| Metric | Target |
|--------|--------|
| Strategy loads | No errors |
| Backtest completes | Without crashes |
| Trade signals generated | > 0 trades |
| No critical warnings | True |

---

## Phase 2: Backtesting & Optimization

### Duration: Weeks 3-4

### Objectives

- [ ] Run comprehensive backtests on 365 days
- [ ] Analyze performance across different market conditions
- [ ] Optimize hyperparameters using Hyperopt
- [ ] Validate risk management parameters

### Tasks

#### 2.1 Full Historical Backtest
```bash
# Full year backtest
freqtrade backtesting \
    --config configs/config_backtest.json \
    --strategy HybridBTCStrategy \
    --timerange 20231101-20241101 \
    --enable-protections
```

#### 2.2 Market Condition Analysis

Run separate backtests for different market phases:

```bash
# Bull Market (example: Q4 2023)
freqtrade backtesting --timerange 20231001-20231231 ...

# Bear Market (example: Q2 2024)
freqtrade backtesting --timerange 20240401-20240630 ...

# Sideways Market (example: Q3 2024)
freqtrade backtesting --timerange 20240701-20240930 ...
```

#### 2.3 Hyperparameter Optimization
```bash
# Run hyperopt with Sharpe Ratio optimization
freqtrade hyperopt \
    --config configs/hyperopt_config.json \
    --strategy HybridBTCStrategy \
    --hyperopt-loss SharpeHyperOptLoss \
    --spaces buy sell \
    --epochs 500 \
    --random-state 42
```

#### 2.4 Key Parameters to Optimize

| Parameter | Range | Default |
|-----------|-------|---------|
| `ema_fast` | 15-25 | 20 |
| `ema_slow` | 40-60 | 50 |
| `adx_trend_threshold` | 20-30 | 25 |
| `swing_lookback` | 3-8 | 5 |
| `bb_period` | 15-25 | 20 |
| `volume_breakout_mult` | 1.3-2.0 | 1.5 |
| `of_min_quality_score` | 40-80 | 60 |

### Success Criteria

| Metric | Target |
|--------|--------|
| Sharpe Ratio | > 1.5 |
| Sortino Ratio | > 2.0 |
| Max Drawdown | < 15% |
| Win Rate | > 45% |
| Profit Factor | > 1.5 |
| Total Trades | 50-150 per month |

---

## Phase 3: Dry-Run Testing

### Duration: Weeks 5-8 (Minimum 30 Days)

### Objectives

- [ ] Validate strategy in real-time market conditions
- [ ] Monitor execution quality and slippage
- [ ] Verify Order Flow indicators accuracy
- [ ] Test system stability over extended periods

### Tasks

#### 3.1 Start Dry-Run
```bash
freqtrade trade \
    --config configs/config_futures.json \
    --strategy HybridBTCStrategy \
    --dry-run
```

#### 3.2 Enable Telegram Notifications

Update `configs/config_futures.json`:
```json
{
    "telegram": {
        "enabled": true,
        "token": "YOUR_BOT_TOKEN",
        "chat_id": "YOUR_CHAT_ID"
    }
}
```

#### 3.3 Daily Monitoring Checklist

- [ ] Check open positions
- [ ] Review entry/exit timing
- [ ] Verify stoploss execution
- [ ] Monitor regime detection accuracy
- [ ] Log any anomalies

#### 3.4 Weekly Review

| Aspect | Questions to Answer |
|--------|---------------------|
| Regime Detection | Is the correct strategy being selected? |
| Order Flow | Are CVD/Imbalance signals accurate? |
| Entry Timing | Are entries near optimal prices? |
| Exit Efficiency | Are profits being captured effectively? |
| Risk Management | Are stoplosses working correctly? |

### Success Criteria

| Metric | Target |
|--------|--------|
| Uptime | > 99% |
| Profitable Days | > 50% |
| Average RRR | > 1:1.5 |
| Max Daily Drawdown | < 5% |
| Execution vs Backtest | < 20% deviation |

---

## Phase 4: Performance Analysis

### Duration: Weeks 9-10

### Objectives

- [ ] Comprehensive performance review
- [ ] Identify strategy weaknesses
- [ ] Document lessons learned
- [ ] Prepare for FreqAI integration

### Tasks

#### 4.1 Generate Performance Reports
```bash
freqtrade backtesting \
    --config configs/config_futures.json \
    --strategy HybridBTCStrategy \
    --export trades \
    --export-filename user_data/backtest_results/analysis.json
```

#### 4.2 Analysis Dimensions

1. **Time Analysis**
   - Best/worst trading hours
   - Day of week performance
   - Monthly patterns

2. **Strategy Analysis**
   - Pullback vs Liquidity vs Breakout performance
   - Regime detection accuracy
   - Order Flow filter effectiveness

3. **Trade Analysis**
   - Average winner vs loser
   - Holding time distribution
   - Entry/exit efficiency

4. **Risk Analysis**
   - Drawdown patterns
   - Correlation with BTC volatility
   - Leverage utilization

#### 4.3 Documentation

Create detailed reports for:
- Trade journal with screenshots
- Weekly performance summaries
- Strategy modification proposals

---

## Phase 5: FreqAI Integration

### Duration: Weeks 11-16

### Overview

[FreqAI](https://docs.freqtrade.io/en/2025.9/freqai/) is a powerful machine learning framework integrated into Freqtrade that enables predictive modeling, automated feature engineering, and reinforcement learning for strategy optimization.

### Why FreqAI?

| Feature | Benefit |
|---------|---------|
| Self-adaptive retraining | Models adapt to changing market conditions |
| Rapid feature engineering | Create 10,000+ features automatically |
| Multi-model support | XGBoost, LightGBM, CatBoost, PyTorch |
| Reinforcement Learning | Advanced adaptive trading agents |

---

### 5.1 FreqAI Fundamentals

#### Supported Model Types

1. **Regressors** - Predict continuous values (e.g., future price change)
2. **Classifiers** - Predict discrete categories (e.g., buy/sell/hold)
3. **Reinforcement Learning** - Learn optimal actions through reward optimization

#### Key Concepts

```
┌─────────────────────────────────────────────────────────────┐
│                    FreqAI Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Historical Data → Feature Engineering → Model Training    │
│         ↓                    ↓                  ↓           │
│   Live Data ─────→ Feature Extraction ──→ Prediction        │
│                                               ↓             │
│                                        Trading Signal       │
│                                                             │
│   Background Thread: Continuous Retraining                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 5.2 Integration Plan

#### Stage 1: Classifier for Entry Confirmation (Weeks 11-12)

**Goal**: Use ML to filter false signals from current strategy

```python
# Example: LightGBM Classifier
class HybridBTCStrategyAI(IStrategy):

    def feature_engineering_expand_all(self, dataframe, period, **kwargs):
        """Generate features for ML model"""
        dataframe["%-rsi"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-adx"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-mfi"] = ta.MFI(dataframe, timeperiod=period)
        # Add Order Flow features
        dataframe["%-cvd"] = self.calculate_cvd(dataframe)
        dataframe["%-imbalance"] = self.calculate_imbalance(dataframe)
        return dataframe

    def set_freqai_targets(self, dataframe, **kwargs):
        """Define prediction target"""
        # Target: Will price be higher in 4 hours?
        dataframe["&-target"] = (
            dataframe["close"].shift(-16) > dataframe["close"]
        ).astype(int)
        return dataframe
```

**Configuration**:
```json
{
    "freqai": {
        "enabled": true,
        "purge_old_models": 2,
        "train_period_days": 30,
        "backtest_period_days": 7,
        "identifier": "hybrid_classifier_v1",
        "feature_parameters": {
            "include_timeframes": ["15m", "1h", "4h"],
            "include_corr_pairlist": ["ETH/USDT:USDT"],
            "indicator_periods_candles": [10, 20, 50]
        },
        "data_split_parameters": {
            "test_size": 0.25,
            "shuffle": false
        },
        "model_training_parameters": {
            "n_estimators": 1000,
            "learning_rate": 0.02,
            "max_depth": 8
        }
    }
}
```

#### Stage 2: Regressor for Exit Optimization (Weeks 13-14)

**Goal**: Predict optimal exit timing

```python
def set_freqai_targets(self, dataframe, **kwargs):
    """Predict maximum favorable excursion"""
    # Calculate MFE (Maximum Favorable Excursion) for next 2 hours
    future_high = dataframe["high"].rolling(8).max().shift(-8)
    dataframe["&-mfe"] = (future_high - dataframe["close"]) / dataframe["close"]
    return dataframe
```

#### Stage 3: Reinforcement Learning (Weeks 15-16)

**Goal**: Fully adaptive entry/exit decisions

[Reinforcement Learning in FreqAI](https://www.freqtrade.io/en/stable/freqai-reinforcement-learning/) involves two critical components:

1. **Agent**: Makes decisions (Long/Short/Neutral)
2. **Environment**: Simulates market and provides rewards

```
┌─────────────────────────────────────────────────────────────┐
│              Reinforcement Learning Flow                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   State (Market Data) ──→ Agent (Neural Network)           │
│          ↑                        ↓                         │
│          │                    Action                        │
│          │                (Buy/Sell/Hold)                   │
│          │                        ↓                         │
│          └──── Reward ←── Environment                       │
│                                                             │
│   Reward = f(profit, drawdown, trade_duration, ...)        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Custom Reward Function**:
```python
def calculate_reward(self, action: int) -> float:
    """
    Custom reward function for RL agent

    Rewards:
    - Profitable trades: +1 to +3 based on profit %
    - Winning with good RRR: +0.5 bonus
    - Quick profits: +0.3 bonus

    Penalties:
    - Losses: -1 to -2 based on loss %
    - Holding too long: -0.1 per period
    - Excessive trading: -0.2
    """
    pnl = self._current_pnl()

    if action == Actions.Neutral:
        return -0.01  # Small penalty for inaction

    if self._position_is_closed():
        if pnl > 0:
            reward = min(pnl * 100, 3.0)  # Cap at 3
            # Bonus for good RRR
            if self._risk_reward_ratio() > 2:
                reward += 0.5
        else:
            reward = max(pnl * 100, -2.0)  # Floor at -2
        return reward

    # Holding penalty (encourages decisive action)
    return -0.01 * self._holding_duration()
```

**RL Configuration**:
```json
{
    "freqai": {
        "enabled": true,
        "rl_config": {
            "train_cycles": 25,
            "max_trade_duration_candles": 100,
            "model_type": "PPO",
            "policy_type": "MlpPolicy",
            "net_arch": [128, 128],
            "randomize_starting_position": true,
            "model_reward_parameters": {
                "rr": 1,
                "profit_aim": 0.02,
                "win_reward_factor": 2
            }
        }
    }
}
```

---

### 5.3 FreqAI Scientific Background

#### Machine Learning in Trading

According to research on [Deep Reinforcement Learning for Trading](https://arxiv.org/abs/2101.07107), DRL frameworks using Proximal Policy Optimization (PPO) have shown significant improvements over traditional algorithmic strategies in high-frequency trading scenarios.

#### Key Advantages of FreqAI

1. **Adaptive Learning**: Models retrain automatically to adapt to changing market conditions
2. **Feature Engineering**: Automated creation of hundreds of technical indicators
3. **Multi-timeframe Analysis**: Incorporates data from multiple timeframes simultaneously
4. **Correlation Features**: Uses correlated assets for enhanced prediction

#### Best Practices (Based on FreqAI Documentation)

| Practice | Rationale |
|----------|-----------|
| Use 30+ days training data | Ensures model sees various market conditions |
| Retrain every 7-14 days | Balances adaptivity with stability |
| Include correlated pairs | ETH, total market cap improve predictions |
| Start with simpler models | LightGBM before Neural Networks |
| Use walk-forward validation | Prevents overfitting |

#### Common Pitfalls to Avoid

1. **Overfitting**: Too complex models memorize noise
2. **Look-ahead bias**: Using future data in features
3. **Data snooping**: Over-optimizing on same dataset
4. **Reward hacking**: RL agents find exploits in reward function

---

### 5.4 FreqAI Implementation Checklist

- [ ] Install FreqAI dependencies: `pip install freqtrade[freqai]`
- [ ] Configure `freqai` section in config
- [ ] Implement `feature_engineering_expand_all()`
- [ ] Define prediction targets
- [ ] Run backtest with FreqAI enabled
- [ ] Monitor TensorBoard for training progress
- [ ] Compare performance with baseline strategy
- [ ] Fine-tune model hyperparameters

---

## Future Goals

### Short-term (3-6 Months)

| Goal | Description | Priority |
|------|-------------|----------|
| Multi-pair expansion | Add ETH/USDT, SOL/USDT | High |
| Advanced Order Flow | Integrate footprint charts | Medium |
| Mobile alerts | Custom Telegram dashboards | Medium |
| Docker deployment | Production-ready containers | High |

### Medium-term (6-12 Months)

| Goal | Description | Priority |
|------|-------------|----------|
| Portfolio management | Cross-asset correlation | High |
| News sentiment | Twitter/news integration | Medium |
| Custom exchange connector | Better latency | Low |
| Web dashboard | Real-time monitoring UI | Medium |

### Long-term (12+ Months)

| Goal | Description | Priority |
|------|-------------|----------|
| Multi-exchange arbitrage | Cross-exchange opportunities | Medium |
| Options strategies | Hedge with BTC options | Low |
| Fund management | Multi-account support | Low |
| Research publication | Share findings with community | Low |

---

## Success Metrics

### Primary KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| Annual Return | > 50% | Compounded monthly |
| Max Drawdown | < 20% | Peak to trough |
| Sharpe Ratio | > 2.0 | Risk-adjusted return |
| Win Rate | > 50% | Winning trades / Total |
| Profit Factor | > 2.0 | Gross profit / Gross loss |

### Secondary KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| Average Trade | > 1% | Mean profit per trade |
| Trade Frequency | 1-3/day | Trades per day |
| Avg Holding Time | 1-4 hours | Time in position |
| Recovery Factor | > 10 | Net profit / Max DD |

---

## Risk Management Guidelines

### Position Sizing

```
Position Size = (Account * Risk%) / (Entry - Stoploss)

Example:
- Account: 1000 USDT
- Risk per trade: 2% = 20 USDT
- Entry: 95,000 USDT/BTC
- Stoploss: 93,100 USDT/BTC (2% below)
- Position Size: 20 / 1900 = 0.0105 BTC
- With 5x leverage: 0.0526 BTC exposure
```

### Risk Limits

| Parameter | Limit |
|-----------|-------|
| Max risk per trade | 2% |
| Max daily drawdown | 5% |
| Max weekly drawdown | 10% |
| Max total drawdown | 20% |
| Max concurrent trades | 3 |
| Max leverage | 10x |

### Emergency Procedures

1. **Circuit Breaker**: Stop trading after 3 consecutive losses
2. **Volatility Filter**: Reduce position size when ATR > 3%
3. **Correlation Check**: Don't add positions in same direction
4. **Manual Override**: Telegram commands for emergency exit

---

## References & Resources

### Official Documentation
- [Freqtrade Documentation](https://docs.freqtrade.io/en/2025.9/)
- [FreqAI Introduction](https://docs.freqtrade.io/en/2025.9/freqai/)
- [FreqAI Reinforcement Learning](https://www.freqtrade.io/en/stable/freqai-reinforcement-learning/)

### Community Resources
- [Awesome Freqtrade](https://github.com/just-nilux/awesome-freqtrade)
- [Freqtrade GitHub Issues](https://github.com/freqtrade/freqtrade/issues)

### Research Papers
- [Deep Reinforcement Learning for Active High Frequency Trading](https://arxiv.org/abs/2101.07107)
- [Multi-Agent Reinforcement Learning for HFT](https://www.researchgate.net/publication/386279469)

### Learning Resources
- [FreqAI Beginner's Guide 2025](https://www.skool.com/the-quantitative-elite/freqai-complete-beginners-guide-2025)
- [Mastering FreqAI DataFrame Patterns](https://medium.com/@hoon33710/mastering-freqai-key-dataframe-patterns-for-predictive-modeling-ccbd58862a00)

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | Nov 2025 | Initial roadmap with FreqAI integration plan |

---

*This document is a living document and will be updated as the project progresses.*
