#!/usr/bin/env python3
"""
Mock backtest script to compare the original and adjusted trading strategies
without requiring real API credentials.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import random

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bot_strategy.strategy import SwingStrategy, TradingSignal
from src.bot_strategy.timeframes import Timeframe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("backtest_mock")

class BacktestResult:
    """Class to store and analyze backtest results"""
    
    def __init__(self, name: str, initial_capital: float = 1000.0):
        self.name = name
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.positions = []
        self.signals = []
        self.equity_curve = [(datetime.now(), initial_capital)]  # (timestamp, capital)
        self.buy_and_hold = 0
        self.in_position = False
        self.position_size = 0
        self.entry_price = 0
        self.entry_time = None
        
    def add_signal(self, timestamp: datetime, signal: TradingSignal):
        """Add a trading signal to the record"""
        self.signals.append({
            'timestamp': timestamp,
            'action': signal.action,
            'price': signal.price,
            'confidence': signal.confidence
        })
        
    def enter_position(self, timestamp: datetime, price: float, size: float):
        """Record entering a position"""
        if self.in_position:
            logger.warning(f"Already in position but trying to enter another at {timestamp}")
            return
            
        self.in_position = True
        self.entry_price = price
        self.entry_time = timestamp
        self.position_size = size
        
        # Record the trade entry
        self.trades.append({
            'entry_time': timestamp,
            'entry_price': price,
            'size': size,
            'exit_time': None,
            'exit_price': None,
            'profit_loss': 0,
            'profit_loss_pct': 0,
            'trade_duration': None
        })
        
        # Update equity curve
        self.equity_curve.append((timestamp, self.current_capital))
        
    def exit_position(self, timestamp: datetime, price: float):
        """Record exiting a position"""
        if not self.in_position:
            logger.warning(f"Not in position but trying to exit at {timestamp}")
            return
            
        # Calculate profit/loss
        pl_dollars = (price - self.entry_price) * self.position_size
        pl_percent = (price / self.entry_price - 1) * 100
        
        # Update capital
        self.current_capital += pl_dollars
        
        # Update the last trade
        self.trades[-1].update({
            'exit_time': timestamp,
            'exit_price': price,
            'profit_loss': pl_dollars,
            'profit_loss_pct': pl_percent,
            'trade_duration': (timestamp - self.entry_time).total_seconds() / 3600  # hours
        })
        
        # Save position details
        self.positions.append({
            'entry_time': self.entry_time,
            'exit_time': timestamp,
            'entry_price': self.entry_price,
            'exit_price': price,
            'size': self.position_size,
            'profit_loss': pl_dollars,
            'profit_loss_pct': pl_percent
        })
        
        # Reset position
        self.in_position = False
        self.position_size = 0
        self.entry_price = 0
        self.entry_time = None
        
        # Update equity curve
        self.equity_curve.append((timestamp, self.current_capital))
        
    def update_equity(self, timestamp: datetime, current_price: float):
        """Update equity curve with current mark-to-market value"""
        # If in a position, calculate unrealized P&L
        capital = self.current_capital
        if self.in_position:
            unrealized_pl = (current_price - self.entry_price) * self.position_size
            capital = self.current_capital + unrealized_pl
            
        self.equity_curve.append((timestamp, capital))
        
    def get_summary(self) -> Dict:
        """Get summary statistics for the backtest"""
        if not self.trades:
            return {
                'name': self.name,
                'total_trades': 0,
                'win_rate': 0,
                'profit_loss': 0,
                'max_drawdown': 0,
                'avg_trade_duration': 0,
                'return_pct': 0
            }
            
        # Calculate performance metrics
        total_trades = len([t for t in self.trades if t['exit_time'] is not None])
        winning_trades = len([t for t in self.trades if t['exit_time'] is not None and t['profit_loss'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        profit_loss = self.current_capital - self.initial_capital
        profit_loss_pct = (profit_loss / self.initial_capital) * 100
        
        # Calculate max drawdown
        equity_values = [e[1] for e in self.equity_curve]
        max_dd = 0
        peak = equity_values[0]
        
        for value in equity_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
            
        # Average trade duration
        completed_trades = [t for t in self.trades if t['exit_time'] is not None]
        avg_duration = np.mean([t['trade_duration'] for t in completed_trades]) if completed_trades else 0
        
        return {
            'name': self.name,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate * 100,  # as percentage
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'max_drawdown': max_dd * 100,  # as percentage
            'avg_trade_duration': avg_duration,  # in hours
            'buy_and_hold_pct': self.buy_and_hold,
            'final_capital': self.current_capital
        }
        
    def plot_equity_curve(self, comparison_results=None, symbol="backtest", timeframe="default"):
        """Plot the equity curve and optionally compare with other results"""
        plt.figure(figsize=(12, 6))
        
        # Convert equity curve to DataFrame for easier plotting
        df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Plot this result
        plt.plot(df.index, df['equity'], label=f"{self.name} (Final: ${self.current_capital:.2f})")
        
        # Plot comparison results if provided
        if comparison_results:
            if not isinstance(comparison_results, list):
                comparison_results = [comparison_results]
                
            for comp in comparison_results:
                comp_df = pd.DataFrame(comp.equity_curve, columns=['timestamp', 'equity'])
                comp_df['timestamp'] = pd.to_datetime(comp_df['timestamp'])
                comp_df.set_index('timestamp', inplace=True)
                plt.plot(comp_df.index, comp_df['equity'], label=f"{comp.name} (Final: ${comp.current_capital:.2f})")
        
        # Plot formatting
        plt.title(f'Strategy Equity Curve Comparison - {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Account Value ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        output_dir = os.path.join('/app', 'backtest_results')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = f'{symbol}_{timeframe.lower()}_equity_{timestamp}.png'
        plt.savefig(os.path.join(output_dir, plot_filename))
        logger.info(f"Equity curve saved to {output_dir}/{plot_filename}")
        
        # Show the plot only if not in a container
        if os.getenv('RUNNING_IN_CONTAINER', 'False').lower() != 'true':
            plt.show()


def generate_mock_data(start_date, end_date, timeframe_minutes=5):
    """
    Generate realistic price data with trends, pullbacks, and volatility
    for backtesting purposes.
    
    Args:
        start_date: The start date for the data
        end_date: The end date for the data
        timeframe_minutes: Candle timeframe in minutes
        
    Returns:
        DataFrame with OHLCV data
    """
    # Calculate number of candles
    total_minutes = int((end_date - start_date).total_seconds() / 60)
    num_candles = total_minutes // timeframe_minutes
    
    # Base price and trend components
    base_price = 100.0  # Starting price
    
    # Generate timestamps
    timestamps = [start_date + timedelta(minutes=i*timeframe_minutes) for i in range(num_candles)]
    
    # Generate price data with trends, reversals and volatility
    prices = []
    current_price = base_price
    
    # Create a trend/cycle pattern
    trend = 0.0
    trend_change_frequency = random.randint(20, 50)  # Change trend direction every 20-50 candles
    
    # Volatility settings
    base_volatility = 0.002  # Base volatility (0.2%)
    volatility_cycle = 100  # Volatility cycles every 100 candles
    
    for i in range(num_candles):
        # Change trend direction periodically
        if i % trend_change_frequency == 0:
            trend = random.uniform(-0.001, 0.001)  # -0.1% to 0.1% per candle
            
        # Add some mean reversion to prevent extreme trends
        mean_reversion = (base_price - current_price) * 0.001
        
        # Cyclical volatility (more volatile during certain periods)
        volatility = base_volatility * (1 + 0.5 * np.sin(i / volatility_cycle * 2 * np.pi))
        
        # Random component (noise)
        noise = random.normalvariate(0, volatility) 
        
        # Calculate price change for this candle
        price_change = current_price * (trend + mean_reversion + noise)
        
        # Occasionally add a shock (news event, etc.)
        if random.random() < 0.01:  # 1% chance of a significant move
            shock = current_price * random.uniform(-0.03, 0.03)  # -3% to 3% shock
            price_change += shock
            
        # Update the current price
        current_price += price_change
        
        # Ensure price doesn't go negative
        current_price = max(current_price, 0.01)
        
        # Calculate OHLC data realistically
        candle_volatility = current_price * volatility * random.uniform(0.5, 2.0)
        open_price = current_price - price_change
        close_price = current_price
        high_price = max(open_price, close_price) + abs(candle_volatility * random.random())
        low_price = min(open_price, close_price) - abs(candle_volatility * random.random())
        
        # Generate realistic volume (higher on big price moves)
        volume = 1000 * (1 + 5 * abs(price_change/current_price)) * random.uniform(0.7, 1.3)
        
        # Add to the list
        prices.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(prices)
    df.set_index('timestamp', inplace=True)
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    return df


class MockStrategyBacktester:
    """Class to backtest a strategy on mock historical data"""
    
    def __init__(self, 
                 symbol: str,
                 timeframe: Timeframe,
                 start_date: datetime,
                 end_date: datetime,
                 initial_capital: float = 1000.0,
                 position_size_pct: float = 0.95,
                 detail_logging: bool = False):
        """
        Initialize the backtester with strategy and data parameters
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USD')
            timeframe: Timeframe for the backtest
            start_date: Start date for the backtest
            end_date: End date for the backtest
            initial_capital: Initial capital for the backtest
            position_size_pct: Percentage of capital to use per trade
            detail_logging: Whether to log detailed information
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.detail_logging = detail_logging
        
        # Create strategies
        self._create_strategies()
        
        # Initialize results
        self.original_result = BacktestResult("Original Strategy", initial_capital)
        self.adjusted_result = BacktestResult("Adjusted Strategy", initial_capital)
        
        # Generate mock data
        self.historical_data = generate_mock_data(
            start_date=start_date,
            end_date=end_date,
            timeframe_minutes=timeframe.minutes
        )
        
    def _create_strategies(self):
        """Create original and adjusted strategy instances"""
        # These will use their respective implementations of _combine_signals
        self.original_strategy = self._create_original_strategy()
        self.adjusted_strategy = self._create_adjusted_strategy()
        
    def _create_original_strategy(self):
        """Create a strategy with the original implementation"""
        # Create a custom strategy class with the original combine_signals
        class OriginalSwingStrategy(SwingStrategy):
            def _combine_signals(self, rsi_signal: Dict, ema_signal: Dict) -> Tuple[str, float]:
                """Original implementation of combine_signals"""
                rsi_action = rsi_signal['signal']
                ema_action = ema_signal['signal']
                
                # Normalize indicator strengths to 0-1 range
                rsi_strength = rsi_signal['strength'] / 100
                ema_strength = ema_signal['strength'] / 100
                
                # Calculate base confidence levels
                rsi_confidence = rsi_strength * 0.6  # RSI contributes 60%
                ema_confidence = ema_strength * 0.4  # EMA contributes 40%
                
                # Default to NEUTRAL
                combined_signal = 'NEUTRAL'
                confidence = 0.0
                
                # Strategy 1: Strong agreement between indicators
                if rsi_action == ema_action and rsi_action != 'NEUTRAL':
                    combined_signal = rsi_action
                    # Both indicators agree, so we have high confidence
                    confidence = (rsi_confidence + ema_confidence) * 1.2  # Bonus for agreement
                    logger.debug(f"Strong agreement: RSI and EMA both suggest {combined_signal}")
                    
                # Strategy 2: RSI extremes with confirming trend
                elif rsi_action == 'BUY' and ema_signal['distance_pct'] > -2:
                    # RSI suggests buy and price is close to or above EMA (confirming uptrend)
                    combined_signal = 'BUY'
                    confidence = rsi_confidence * (1 + 0.3 * ema_confidence)
                    logger.debug(f"RSI oversold in neutral/bullish trend (price to EMA: {ema_signal['distance_pct']:.2f}%)")
                    
                elif rsi_action == 'SELL' and ema_signal['distance_pct'] < 2:
                    # RSI suggests sell and price is close to or below EMA (confirming downtrend)
                    combined_signal = 'SELL'
                    confidence = rsi_confidence * (1 + 0.3 * ema_confidence)
                    logger.debug(f"RSI overbought in neutral/bearish trend (price to EMA: {ema_signal['distance_pct']:.2f}%)")
                    
                # Strategy 3: EMA crossover with confirming RSI
                elif ema_action != 'NEUTRAL' and rsi_signal['value'] > 40 and rsi_signal['value'] < 60:
                    # EMA crossover with RSI in neutral zone (avoiding extremes)
                    combined_signal = ema_action
                    confidence = ema_confidence * (1 + 0.2 * (1 - abs(rsi_signal['value'] - 50) / 10))
                    logger.debug(f"EMA {ema_action} crossover with neutral RSI ({rsi_signal['value']:.2f})")
                
                # Cap confidence at 1.0
                confidence = min(1.0, confidence)
                
                # Require minimum confidence threshold
                if confidence < 0.35:
                    logger.debug(f"Signal {combined_signal} has low confidence ({confidence:.2f}), changing to NEUTRAL")
                    return 'NEUTRAL', confidence
                    
                return combined_signal, confidence
        
        # Create and return the strategy with original behavior
        return OriginalSwingStrategy(timeframe=self.timeframe)

    def _create_adjusted_strategy(self):
        """Create a strategy with the adjusted implementation"""
        class AdjustedSwingStrategy(SwingStrategy):
            def _combine_signals(self, rsi_signal: Dict, ema_signal: Dict) -> Tuple[str, float]:
                """
                Adjusted implementation with:
                1. More balanced indicator weights (50/50)
                2. Stricter trend confirmation requirements
                3. Higher confidence thresholds
                4. Additional momentum conditions
                5. Modified RSI neutral zone
                """
                rsi_action = rsi_signal['signal']
                ema_action = ema_signal['signal']
                
                # Equal weights for both indicators
                rsi_strength = rsi_signal['strength'] / 100
                ema_strength = ema_signal['strength'] / 100
                
                # Base confidence (50/50 split instead of 60/40)
                rsi_confidence = rsi_strength * 0.5
                ema_confidence = ema_strength * 0.5
                
                combined_signal = 'NEUTRAL'
                confidence = 0.0
                
                # Get RSI value and trend
                rsi_value = rsi_signal['value']
                price_to_ema = ema_signal['distance_pct']
                
                # Strategy 1: Strong Trend with Momentum (Stricter Requirements)
                if rsi_action == ema_action and rsi_action != 'NEUTRAL':
                    if rsi_action == 'BUY' and price_to_ema > 0:  # Must be above EMA
                        combined_signal = 'BUY'
                        confidence = (rsi_confidence + ema_confidence) * 1.5  # Increased bonus
                    elif rsi_action == 'SELL' and price_to_ema < 0:  # Must be below EMA
                        combined_signal = 'SELL'
                        confidence = (rsi_confidence + ema_confidence) * 1.5
                
                # Strategy 2: Counter-Trend Reversals (Modified Conditions)
                elif rsi_action == 'BUY' and price_to_ema > -1.5:  # Tighter threshold
                    if rsi_value < 30:  # More extreme RSI requirement
                        combined_signal = 'BUY'
                        confidence = rsi_confidence * (1.4 + abs(price_to_ema) * 0.1)
                elif rsi_action == 'SELL' and price_to_ema < 1.5:
                    if rsi_value > 70:  # More extreme RSI requirement
                        combined_signal = 'SELL'
                        confidence = rsi_confidence * (1.4 + abs(price_to_ema) * 0.1)
                
                # Strategy 3: Trend Following with RSI Confirmation
                elif ema_action != 'NEUTRAL':
                    # Modified RSI neutral zone (45-55 instead of 40-60)
                    if 45 <= rsi_value <= 55:
                        combined_signal = ema_action
                        # Stronger confidence boost for being closer to 50
                        confidence = ema_confidence * (1.3 + (1 - abs(rsi_value - 50) / 5) * 0.2)
                
                # Cap confidence at 1.0
                confidence = min(1.0, confidence)
                
                # Higher minimum confidence threshold (0.45 instead of 0.35)
                if confidence < 0.45:
                    logger.debug(f"Signal {combined_signal} has low confidence ({confidence:.2f}), changing to NEUTRAL")
                    return 'NEUTRAL', confidence
                    
                return combined_signal, confidence
        
        # Create and return the strategy with adjusted behavior
        return AdjustedSwingStrategy(timeframe=self.timeframe)
    
    def _generate_signals(self, strategy: SwingStrategy, candle_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate trading signals for a strategy based on candle data"""
        # Convert DataFrame to format expected by strategy
        prices = candle_data['close'].values.tolist()
        timestamps = [int(ts.timestamp()) for ts in candle_data.index]
        
        if self.detail_logging:
            logger.info(f"Generating signals for {len(prices)} candles...")
        
        # Temporarily disable printing in the strategy to avoid cluttering console
        original_print = print
        try:
            import builtins
            builtins.print = lambda *args, **kwargs: None
            
            # Generate signal
            signal = strategy.generate_signal(
                symbol=self.symbol,
                prices=prices,
                timestamps=timestamps,
                current_price=prices[-1]
            )
            
            if signal:
                return signal
                
        finally:
            # Restore print function
            builtins.print = original_print
            
        return None
        
    def run_backtest(self):
        """Run the backtest for both strategies"""
        if self.historical_data.empty:
            logger.error("No historical data available, cannot run backtest")
            return
            
        logger.info(f"Running backtest from {self.historical_data.index.min()} to {self.historical_data.index.max()}")
        
        # Track buy and hold performance
        start_price = self.historical_data['close'].iloc[0]
        end_price = self.historical_data['close'].iloc[-1]
        buy_and_hold_return = (end_price / start_price - 1) * 100
        
        self.original_result.buy_and_hold = buy_and_hold_return
        self.adjusted_result.buy_and_hold = buy_and_hold_return
        
        # Initialize trading variables
        original_in_position = False
        adjusted_in_position = False
        
        # We'll process the data one candle at a time to simulate real-time trading
        window_size = max(50, self.timeframe.lookback_periods)  # Ensure enough data for indicators
        
        # Process each candle in the date range, with enough prior data for calculations
        for i in range(window_size, len(self.historical_data)):
            # Get the current window of data
            window = self.historical_data.iloc[i-window_size:i+1]
            current_time = window.index[-1]
            current_price = window['close'].iloc[-1]
            
            if self.detail_logging:
                logger.info(f"Processing candle at {current_time}, price: ${current_price:.2f}")
            
            # Generate signals for both strategies
            original_signal = self._generate_signals(self.original_strategy, window)
            adjusted_signal = self._generate_signals(self.adjusted_strategy, window)
            
            # Update equity curves with current mark-to-market value
            self.original_result.update_equity(current_time, current_price)
            self.adjusted_result.update_equity(current_time, current_price)
            
            # Process original strategy signal
            if original_signal:
                self.original_result.add_signal(current_time, original_signal)
                
                if original_signal.action == 'BUY' and not original_in_position:
                    # Enter long position
                    position_size = (self.original_result.current_capital * self.position_size_pct) / current_price
                    self.original_result.enter_position(current_time, current_price, position_size)
                    original_in_position = True
                    logger.info(f"ORIGINAL: BUY at {current_time} @ ${current_price:.2f}, size: {position_size:.6f}")
                    
                elif original_signal.action == 'SELL' and original_in_position:
                    # Exit long position
                    self.original_result.exit_position(current_time, current_price)
                    original_in_position = False
                    logger.info(f"ORIGINAL: SELL at {current_time} @ ${current_price:.2f}")
            
            # Process adjusted strategy signal
            if adjusted_signal:
                self.adjusted_result.add_signal(current_time, adjusted_signal)
                
                if adjusted_signal.action == 'BUY' and not adjusted_in_position:
                    # Enter long position
                    position_size = (self.adjusted_result.current_capital * self.position_size_pct) / current_price
                    self.adjusted_result.enter_position(current_time, current_price, position_size)
                    adjusted_in_position = True
                    logger.info(f"ADJUSTED: BUY at {current_time} @ ${current_price:.2f}, size: {position_size:.6f}")
                    
                elif adjusted_signal.action == 'SELL' and adjusted_in_position:
                    # Exit long position
                    self.adjusted_result.exit_position(current_time, current_price)
                    adjusted_in_position = False
                    logger.info(f"ADJUSTED: SELL at {current_time} @ ${current_price:.2f}")
        
        # Close any open positions at the end of the backtest
        final_price = self.historical_data['close'].iloc[-1]
        final_time = self.historical_data.index[-1]
        
        if original_in_position:
            self.original_result.exit_position(final_time, final_price)
            logger.info(f"ORIGINAL: Closing position at end of backtest @ ${final_price:.2f}")
            
        if adjusted_in_position:
            self.adjusted_result.exit_position(final_time, final_price)
            logger.info(f"ADJUSTED: Closing position at end of backtest @ ${final_price:.2f}")
        
        # Get and display results
        original_summary = self.original_result.get_summary()
        adjusted_summary = self.adjusted_result.get_summary()
        
        logger.info("\n" + "="*50)
        logger.info("BACKTEST RESULTS")
        logger.info("="*50)
        
        logger.info("\nORIGINAL STRATEGY:")
        for key, value in original_summary.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value}")
        
        logger.info("\nADJUSTED STRATEGY:")
        for key, value in adjusted_summary.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value}")
        
        logger.info("\nBUY AND HOLD RETURN: {:.2f}%".format(buy_and_hold_return))
        
        # Plot the results
        self.original_result.plot_equity_curve(self.adjusted_result, self.symbol, self.timeframe.value)
        
        # Create DataFrame with price data for plotting
        price_data = self.historical_data[['close']].copy()
        price_data['Original Signals'] = np.nan
        price_data['Adjusted Signals'] = np.nan
        
        # Add signal markers
        for signal in self.original_result.signals:
            if signal['action'] == 'BUY':
                price_data.loc[signal['timestamp'], 'Original Signals'] = price_data.loc[signal['timestamp'], 'close'] * 0.99  # Below price
            elif signal['action'] == 'SELL':
                price_data.loc[signal['timestamp'], 'Original Signals'] = price_data.loc[signal['timestamp'], 'close'] * 1.01  # Above price
        
        for signal in self.adjusted_result.signals:
            if signal['action'] == 'BUY':
                price_data.loc[signal['timestamp'], 'Adjusted Signals'] = price_data.loc[signal['timestamp'], 'close'] * 0.97  # Further below price
            elif signal['action'] == 'SELL':
                price_data.loc[signal['timestamp'], 'Adjusted Signals'] = price_data.loc[signal['timestamp'], 'close'] * 1.03  # Further above price
        
        # Plot price chart with signals
        plt.figure(figsize=(12, 6))
        plt.plot(price_data.index, price_data['close'], label='Price')
        plt.scatter(price_data.index, price_data['Original Signals'], color='blue', marker='^', label='Original Strategy Signals')
        plt.scatter(price_data.index, price_data['Adjusted Signals'], color='green', marker='o', label='Adjusted Strategy Signals')
        plt.title('Price Chart with Strategy Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        output_dir = os.path.join('/app', 'backtest_results')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = f'{self.symbol}_{self.timeframe.value.lower()}_signals_{timestamp}.png'
        plt.savefig(os.path.join(output_dir, plot_filename))
        
        return {
            'original': original_summary,
            'adjusted': adjusted_summary,
            'buy_and_hold': buy_and_hold_return
        }

def main():
    """
    Run a backtest with mock data
    """
    try:
        # Get parameters from environment variables or use defaults
        symbol = os.getenv('SYMBOL', 'MOCK-USD')
        timeframe_str = os.getenv('TIMEFRAME', 'FIVE_MINUTE')
        backtest_days = int(os.getenv('BACKTEST_DAYS', '30'))
        initial_capital = float(os.getenv('INITIAL_CAPITAL', '1000.0'))
        position_size_pct = float(os.getenv('POSITION_SIZE_PCT', '0.95'))
        detail_logging = os.getenv('DETAIL_LOGGING', 'False').lower() == 'true'
        use_mock_data = os.getenv('USE_MOCK_DATA', 'True').lower() == 'true'
        
        # Convert timeframe string to Timeframe enum
        try:
            timeframe = Timeframe[timeframe_str]
        except KeyError:
            logger.warning(f"Invalid timeframe {timeframe_str}, using FIVE_MINUTE instead")
            timeframe = Timeframe.FIVE_MINUTE
        
        # Set date range for the backtest
        end_date = datetime.now()
        start_date = end_date - timedelta(days=backtest_days)
        
        # Log backtest configuration
        logger.info("="*50)
        logger.info("BACKTEST CONFIGURATION")
        logger.info("="*50)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Timeframe: {timeframe.value} ({timeframe.minutes} minutes)")
        logger.info(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({backtest_days} days)")
        logger.info(f"Initial Capital: ${initial_capital}")
        logger.info(f"Position Size: {position_size_pct*100:.0f}% of capital")
        logger.info(f"Using Mock Data: {use_mock_data}")
        logger.info(f"Detailed Logging: {detail_logging}")
        logger.info("="*50)
        
        # Initialize and run the backtester
        backtester = MockStrategyBacktester(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            detail_logging=detail_logging
        )
        
        # Run the backtest
        results = backtester.run_backtest()
        
        # Save results to JSON
        output_dir = os.path.join('/app', 'backtest_results')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_filename = f'{symbol}_{timeframe.value.lower()}_{timestamp}.json'
        chart_filename = f'{symbol}_{timeframe.value.lower()}_{timestamp}.png'
        
        with open(os.path.join(output_dir, result_filename), 'w') as f:
            # Use the custom DateTimeEncoder for serializing datetime objects
            from src.backtest.json_encoder import DateTimeEncoder
            json.dump(results, f, indent=2, cls=DateTimeEncoder)
            
        logger.info(f"Results saved to {output_dir}/{result_filename}")
        logger.info(f"Chart saved to {output_dir}/{chart_filename}")
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 