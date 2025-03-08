from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import logging
from .timeframes import Timeframe
from datetime import datetime, timezone
import pytz  # Add this import

logger = logging.getLogger(__name__)

# Add timezone constants at the top with other constants
est_tz = pytz.timezone('America/New_York')
utc_tz = pytz.UTC

@dataclass
class TradingSignal:
    symbol: str
    action: str  # 'BUY' or 'SELL'
    price: float
    timeframe: Timeframe

class SwingStrategy:
    def __init__(self, 
                 timeframe: Timeframe = Timeframe.FIVE_MIN,
                 rsi_period: int = 14):
        """Initialize strategy with timeframe-specific parameters
        
        Args:
            timeframe: Trading timeframe (default: 5 minutes)
            rsi_period: RSI period (default: 14)
        """
        self.timeframe = timeframe
        self.rsi_period = rsi_period
        
        # Adjust thresholds based on timeframe
        self.oversold, self.overbought = self._get_rsi_thresholds()
        logger.info(f"Initialized {timeframe.value} strategy: "
                   f"RSI({rsi_period}) thresholds: {self.oversold}/{self.overbought}")
    
    def _ensure_chronological_order(self, prices: List[float]) -> List[float]:
        """
        Ensure price data is in chronological order (oldest first)
        
        Args:
            prices: List of price values
            
        Returns:
            List of prices in chronological order (oldest first)
        """
        # Make a copy to avoid modifying the original data
        chronological_prices = prices.copy()
        
        # Check if prices appear to be in reverse order (newest first)
        if len(prices) > 1 and prices[0] > prices[-1]:
            logger.debug("Prices appear to be in reverse order, reordering to oldest-first")
            chronological_prices = list(reversed(chronological_prices))
            
        return chronological_prices
    
    def _get_rsi_thresholds(self) -> Tuple[float, float]:
        """Get RSI thresholds based on timeframe"""
        return {
            Timeframe.FIVE_MIN: (25, 75),    # More extreme for short timeframes
            Timeframe.ONE_HOUR: (30, 70),    # Standard thresholds
            Timeframe.SIX_HOUR: (35, 65),    # Less extreme
            Timeframe.TWELVE_HOUR: (35, 65), # Less extreme
            Timeframe.ONE_DAY: (40, 60),     # Most conservative
        }[self.timeframe]
    
    def calculate_rsi(self, prices: List[float]) -> float:
        """Calculate RSI using standard formula"""
        if len(prices) < self.rsi_period + 1:
            logger.warning(f"Not enough data for {self.timeframe.value} RSI calculation")
            return 50
        
        # Make sure prices are in chronological order (oldest first)
        chronological_prices = self._ensure_chronological_order(prices)
        
        # Calculate price changes (next - current)
        deltas = np.diff(chronological_prices)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gain and loss
        avg_gain = np.mean(gains[-self.rsi_period:])  # Include zeros
        avg_loss = np.mean(losses[-self.rsi_period:]) # Include zeros
        
        if avg_loss == 0:
            return 100
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        logger.debug(f"{self.timeframe.value} RSI: {rsi:.2f}")
        return rsi

    def generate_signal(self, symbol: str, prices: List[float], timestamps: List[int]) -> Optional[TradingSignal]:
        """Generate trading signal with timeframe-specific logic"""
        if len(prices) < self.timeframe.lookback_periods:
            logger.warning(f"Insufficient data for {self.timeframe.value} analysis")
            return None
            
        # Print candle analysis before generating signal
        self.print_candle_analysis(prices, timestamps)
        
        # Log price data direction for debugging
        is_reversed = len(prices) > 1 and prices[0] > prices[-1]
        price_order = "newest first (reversed)" if is_reversed else "oldest first (chronological)"
        logger.info(f"Prices appear to be in {price_order} order. First price: {prices[0]}, Last price: {prices[-1]}")
        
        # Calculate RSI
        rsi = self.calculate_rsi(prices)
        
        # Get the current price (most recent)
        current_price = prices[0] if is_reversed else prices[-1]
        
        # Log details about the RSI calculation
        logger.info(f"{self.timeframe.value} RSI calculation result: {rsi:.2f}, using {self.rsi_period} periods")
        logger.info(f"Current price: ${current_price:.2f}")
        
        if rsi < self.oversold:
            logger.info(f"{self.timeframe.value} RSI({rsi:.2f}) below {self.oversold} - Generating BUY signal")
            return TradingSignal(
                symbol=symbol,
                action='BUY',
                price=current_price,
                timeframe=self.timeframe
            )
        elif rsi > self.overbought:
            logger.info(f"{self.timeframe.value} RSI({rsi:.2f}) above {self.overbought} - Generating SELL signal")
            return TradingSignal(
                symbol=symbol,
                action='SELL',
                price=current_price,
                timeframe=self.timeframe
            )
        else:
            logger.info(f"{self.timeframe.value} RSI({rsi:.2f}) between thresholds {self.oversold}-{self.overbought} - No signal")
        
        return None

    def print_candle_analysis(self, prices: List[float], timestamps: List[int]) -> None:
        """Print detailed candle analysis and RSI calculation"""
        if not prices or not timestamps or len(prices) != len(timestamps):
            logger.error("Invalid price or timestamp data")
            return
        
        print("\n📈 Candle Analysis:")
        print(f"Timeframe: {self.timeframe.value}")
        
        # Define consistent column widths to ensure alignment
        time_width = 12
        price_width = 14
        change_width = 10
        gain_width = 10
        loss_width = 10
        
        # Header with proper spacing
        print("\n" + "│ " + "Time".ljust(time_width) + "Price".ljust(price_width) + 
              "Change".ljust(change_width) + "Gain".ljust(gain_width) + "Loss".ljust(loss_width))
        print("│ " + "-" * (time_width + price_width + change_width + gain_width + loss_width))
        
        # Determine if prices are in reverse order
        is_reversed = len(prices) > 1 and prices[0] > prices[-1]
        
        # Create copies for display (newest first)
        display_prices = prices.copy()
        display_timestamps = timestamps.copy()
        
        # Make sure display data is newest-first for better readability
        if not is_reversed:
            display_prices = list(reversed(display_prices))
            display_timestamps = list(reversed(display_timestamps))
            
        # Get chronological prices (oldest first) for calculations
        chronological_prices = self._ensure_chronological_order(prices)
        
        # Calculate deltas in chronological order
        deltas = np.diff(chronological_prices)
        
        # Prepare changes array for display (newest first)
        if is_reversed:
            # If original data was newest-first, flip the deltas to match
            changes = list(np.flip(deltas)) + [0]
        else:
            # If original data was oldest-first, keep reversed order from display
            changes = list(deltas) + [0]
            changes.reverse()
        
        # Separate gains and losses for display
        gains = [max(0, change) for change in changes]
        losses = [max(0, -change) for change in changes]
        
        # Print most recent periods first with proper column alignment
        periods_to_show = self.rsi_period + 1
        for i in range(min(periods_to_show, len(display_prices))):
            try:
                dt = datetime.fromtimestamp(display_timestamps[i], tz=utc_tz).astimezone(est_tz)
                time_str = dt.strftime("%H:%M")
                price_str = f"${display_prices[i]:,.2f}"
                
                # Format change with proper sign
                if changes[i] != 0:
                    change_str = f"{changes[i]:+.2f}"
                else:
                    change_str = "-"
                
                # Format gains and losses
                gain_str = f"{gains[i]:.2f}" if gains[i] > 0 else "-"
                loss_str = f"{losses[i]:.2f}" if losses[i] > 0 else "-"
                
                # Print with consistent column widths
                print("│ " + time_str.ljust(time_width) + 
                      price_str.ljust(price_width) + 
                      change_str.ljust(change_width) + 
                      gain_str.ljust(gain_width) + 
                      loss_str.ljust(loss_width))
            except IndexError as e:
                logger.error(f"Index error at position {i}: {str(e)}")
                break
        
        # Calculate RSI using the same method as calculate_rsi
        # Use chronological prices that are already properly ordered
        
        # Calculate deltas in chronological order
        deltas = np.diff(chronological_prices)
        
        # Separate gains and losses
        calc_gains = np.where(deltas > 0, deltas, 0)
        calc_losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gain and loss
        avg_gain = np.mean(calc_gains[-self.rsi_period:])  # Include zeros
        avg_loss = np.mean(calc_losses[-self.rsi_period:]) # Include zeros
        rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
        rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 100
        
        print("\n📊 RSI Calculation:")
        print(f"Average Gain: ${avg_gain:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Relative Strength (RS) = Avg Gain / Avg Loss = {rs:.2f}")
        print(f"RSI = 100 - (100 / (1 + RS)) = {rsi:.2f}")
        
        # Verify the RSI matches with the calculate_rsi method
        verify_rsi = self.calculate_rsi(prices)
        logger.info(f"RSI Verification - Printed: {rsi:.2f}, calculate_rsi method: {verify_rsi:.2f}")

    def analyze_rsi_swings(self, prices: List[float], timestamps: List[int]) -> dict:
        """Analyze RSI swings from oversold to overbought conditions
        
        Args:
            prices: List of historical prices
            timestamps: List of corresponding timestamps
            
        Returns:
            Dictionary containing swing statistics:
            - avg_swing: Average price movement %
            - min_swing: Smallest successful swing %
            - max_swing: Largest swing %
            - avg_duration: Average swing duration
            - success_rate: % of swings that reached overbought
            - recent_swings: Last 3 swings for reference
        """
        # Calculate RSI for each period
        rsi_values = []
        for i in range(self.rsi_period, len(prices)):
            window = prices[i-self.rsi_period:i+1]
            rsi = self.calculate_rsi(window)
            rsi_values.append(rsi)
        
        # Find completed oversold to overbought swings
        swings = []
        in_swing = False
        swing_start_price = 0
        swing_start_time = 0
        
        for i in range(len(rsi_values)-1):
            if not in_swing and rsi_values[i] < self.oversold:
                in_swing = True
                swing_start_price = prices[i+self.rsi_period]
                swing_start_time = timestamps[i+self.rsi_period]
            elif in_swing and rsi_values[i] > self.overbought:
                swing_end_price = prices[i+self.rsi_period]
                swing_end_time = timestamps[i+self.rsi_period]
                swing_pct = (swing_end_price - swing_start_price) / swing_start_price
                swings.append({
                    'start_price': swing_start_price,
                    'end_price': swing_end_price,
                    'pct_change': swing_pct,
                    'duration': swing_end_time - swing_start_time
                })
                in_swing = False
        
        if not swings:
            return None
        
        # Calculate swing statistics
        pct_changes = [s['pct_change'] for s in swings]
        durations = [s['duration'] for s in swings]
        
        return {
            'avg_swing': np.mean(pct_changes),
            'min_swing': min(pct_changes),
            'max_swing': max(pct_changes),
            'avg_duration': np.mean(durations),
            'success_rate': len([p for p in pct_changes if p > 0]) / len(pct_changes),
            'recent_swings': swings[-3:]  # Last 3 swings for reference
        } 