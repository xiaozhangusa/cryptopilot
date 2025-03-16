#!/usr/bin/env python3
"""
Backtest script to compare the original and adjusted trading strategies.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import copy
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bot_strategy.strategy import SwingStrategy, TradingSignal
from src.bot_strategy.timeframes import Timeframe
from src.coinbase_api.client import CoinbaseAdvancedClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("backtest")

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
        
    def plot_equity_curve(self, comparison_results=None):
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
        plt.title('Strategy Equity Curve Comparison')
        plt.xlabel('Date')
        plt.ylabel('Account Value ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'backtest_results')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(output_dir, f'equity_curve_{timestamp}.png'))
        logger.info(f"Equity curve saved to {output_dir}/equity_curve_{timestamp}.png")
        
        # Show the plot
        plt.show()

class StrategyBacktester:
    """Class to backtest a strategy on historical data"""
    
    def __init__(self, 
                 client: CoinbaseAdvancedClient,
                 symbol: str,
                 timeframe: Timeframe,
                 start_date: datetime,
                 end_date: datetime,
                 initial_capital: float = 1000.0,
                 position_size_pct: float = 0.95,  # Use 95% of capital per trade
                 detail_logging: bool = False):
        """
        Initialize the backtester with strategy and data parameters
        
        Args:
            client: CoinbaseAdvancedClient instance for data fetching
            symbol: Trading pair symbol (e.g., 'BTC-USD')
            timeframe: Timeframe for the backtest
            start_date: Start date for the backtest
            end_date: End date for the backtest
            initial_capital: Initial capital for the backtest
            position_size_pct: Percentage of capital to use per trade
            detail_logging: Whether to log detailed information
        """
        self.client = client
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
        
        # Fetch data
        self.historical_data = self._fetch_historical_data()
        
    def _create_strategies(self):
        """Create original and adjusted strategy instances"""
        # These will use their respective implementations of _combine_signals
        self.original_strategy = self._create_original_strategy()
        self.adjusted_strategy = SwingStrategy(timeframe=self.timeframe)
        
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
        
    def _fetch_historical_data(self) -> pd.DataFrame:
        """Fetch historical data for the specified symbol and timeframe"""
        logger.info(f"Fetching historical data for {self.symbol} on {self.timeframe.value} timeframe")
        
        try:
            # Calculate the number of days to look back
            # This method will simulate fetching historical data by converting from API candles format
            # In a real backtest, you might want to fetch data from a source that provides historical data
            # like a CSV file or database
            
            # Convert timeframe minutes to seconds
            candle_seconds = self.timeframe.minutes * 60
            
            # Get candles from the API
            candles = self.client.get_product_candles(
                product_id=self.symbol,
                granularity=self.timeframe.value
            )
            
            if not candles:
                logger.error(f"No candles received for {self.symbol}")
                return pd.DataFrame()
            
            # Convert candles to DataFrame
            data = []
            for candle in candles:
                timestamp = datetime.fromtimestamp(candle.start)
                data.append({
                    'timestamp': timestamp,
                    'open': float(candle.open),
                    'high': float(candle.high),
                    'low': float(candle.low),
                    'close': float(candle.close),
                    'volume': float(candle.volume)
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)  # Ensure chronological order
            
            logger.info(f"Fetched {len(df)} historical candles from {df.index.min()} to {df.index.max()}")
            
            # Calculate additional metrics like returns
            df['returns'] = df['close'].pct_change()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
    
    def _generate_signals(self, strategy: SwingStrategy, candle_data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals for a strategy based on candle data"""
        signals = []
        
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
        
        # Get start and end indices based on our date range
        date_filtered_data = self.historical_data[
            (self.historical_data.index >= self.start_date) & 
            (self.historical_data.index <= self.end_date)
        ]
        
        if len(date_filtered_data) == 0:
            logger.error(f"No data available in the specified date range ({self.start_date} to {self.end_date})")
            return
            
        start_idx = self.historical_data.index.get_loc(date_filtered_data.index[0])
        end_idx = self.historical_data.index.get_loc(date_filtered_data.index[-1])
        
        # Process each candle in the date range
        for i in range(start_idx, end_idx + 1):
            # Skip if we don't have enough prior data
            if i < window_size:
                continue
                
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
        self.original_result.plot_equity_curve(self.adjusted_result)
        
        return {
            'original': original_summary,
            'adjusted': adjusted_summary,
            'buy_and_hold': buy_and_hold_return
        }

def main():
    """
    Run a backtest on historical data
    """
    # Load API credentials from a secrets file
    secrets_file = os.getenv('SECRETS_FILE', 'secrets.json')
    
    try:
        with open(secrets_file) as f:
            secrets = json.load(f)
            
        # Initialize the API client
        client = CoinbaseAdvancedClient(
            api_key=secrets['api_key'],
            api_secret=secrets['api_secret']
        )
        
        # Parameters for the backtest
        symbol = 'SOL-USD'
        timeframe = Timeframe.FIVE_MINUTE
        
        # Set date range for the backtest (use the last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Initialize and run the backtester
        backtester = StrategyBacktester(
            client=client,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=1000.0,
            position_size_pct=0.95,
            detail_logging=False
        )
        
        # Run the backtest
        results = backtester.run_backtest()
        
        # Save results to JSON
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'backtest_results')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(output_dir, f'backtest_results_{timestamp}.json'), 'w') as f:
            # Use the custom DateTimeEncoder for serializing datetime objects
            from backtest.json_encoder import DateTimeEncoder
            json.dump(results, f, indent=2, cls=DateTimeEncoder)
            
        logger.info(f"Results saved to {output_dir}/backtest_results_{timestamp}.json")
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 