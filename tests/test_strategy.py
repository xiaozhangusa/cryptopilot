import pytest
from src.bot_strategy.strategy import SwingStrategy, TradingSignal

def test_strategy_initialization():
    strategy = SwingStrategy()
    assert strategy.rsi_period == 14
    assert strategy.short_ma == 9
    assert strategy.long_ma == 21

def test_rsi_calculation():
    strategy = SwingStrategy()
    prices = [10.0, 11.0, 10.5, 11.5, 12.0] * 3  # Repeat pattern for enough data
    rsi = strategy.calculate_rsi(prices)
    assert isinstance(rsi, float)
    assert 0 <= rsi <= 100 