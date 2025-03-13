# Understanding Lookback Periods in CryptoPilot

This document explains the logic and rationale behind the `lookback_periods` method used in the trading strategy of the CryptoPilot bot.

## What are Lookback Periods?

Lookback periods determine how many historical candles (price data points) the trading bot uses to calculate support and resistance levels for different timeframes. This is a critical factor in determining how the bot analyzes market conditions and identifies potential trading opportunities.

## How Lookback Values Are Derived

The method returns different numbers of candles based on the timeframe, with each designed to cover a specific duration of market history:

| Timeframe | Lookback Periods | Time Covered | Explanation |
|-----------|------------------|--------------|-------------|
| FIVE_MINUTE | 20 candles | 100 minutes | Covers ~1.5 hours of recent price action |
| THIRTY_MINUTE | 12 candles | 360 minutes (6 hours) | Covers roughly a trading day |
| ONE_HOUR | 24 candles | 24 hours (1 day) | Covers a full day of trading |
| SIX_HOUR | 28 candles | 168 hours (7 days) | Covers a full week of trading |
| TWELVE_HOUR | 14 candles | 168 hours (7 days) | Also covers a full week but with fewer candles |
| ONE_DAY | 30 candles | 30 days (1 month) | Covers a full month of trading |

## Rationale for These Values

The lookback periods are carefully chosen to balance several factors:

### 1. Relevance to Timeframe

For shorter timeframes like 5-minute candles, only recent history (last few hours) is relevant for making trading decisions. For longer timeframes like daily candles, a month of history provides better context for identifying meaningful support and resistance levels.

### 2. Statistical Significance

Each timeframe needs enough data points to identify meaningful patterns, but not so many that older (possibly irrelevant) data unduly influences the analysis. The chosen values provide sufficient data points for statistical validity without diluting the analysis with outdated information.

### 3. Consistency Across Timeframes

Notice how 6-hour and 12-hour timeframes both look back 7 days - this provides consistency in the analysis while using different granularity. This approach allows the strategy to maintain consistent market views across different timeframes.

### 4. Computational Efficiency

More candles require more processing power and memory. These values are optimized to provide sufficient data without excessive computational overhead, making the bot more responsive and efficient.

## How This Affects Trading Decisions

When the bot analyzes support and resistance levels:

- **Short timeframes (5min, 30min)**: Focuses on recent market dynamics, suitable for short-term trading and quick market responses
- **Medium timeframes (1h)**: Considers a full day's trading patterns, balancing recent movements with slightly longer-term context
- **Longer timeframes (6h, 12h)**: Considers a full week's patterns, useful for swing trading strategies
- **Longest timeframe (1d)**: Considers a month's worth of market behavior, providing context for longer-term trends

This graduated approach helps the bot identify both short-term trading opportunities and longer-term market trends, depending on which timeframe is being used for analysis.

## Implementation

The implementation in the trading bot is as follows:

```python
def lookback_periods(self) -> int:
    """Number of candles to look back for support/resistance"""
    return {
        "FIVE_MINUTE": 20,    # 100 minutes
        "THIRTY_MINUTE": 12,  # 360 minutes (6 hours)
        "ONE_HOUR": 24,       # 24 hours
        "SIX_HOUR": 28,       # 7 days
        "TWELVE_HOUR": 14,    # 7 days
        "ONE_DAY": 30         # 30 days
    }[self.value]
```

## Customization

These values can be adjusted based on specific trading strategies, market conditions, or asset characteristics. Increasing the lookback period will provide more historical context but may make the bot less responsive to recent price changes. Decreasing the lookback period will make the bot more sensitive to recent price action but may reduce the reliability of the support/resistance calculations. 