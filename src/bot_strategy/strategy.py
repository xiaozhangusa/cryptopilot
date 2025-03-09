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
    
    def _ensure_chronological_order(self, prices: List[float], timestamps: Optional[List[int]] = None) -> List[float]:
        """
        Ensure price data is in chronological order (oldest first)
        
        Args:
            prices: List of price values
            timestamps: Optional list of corresponding timestamps
            
        Returns:
            List of prices in chronological order (oldest first)
        """
        # If timestamps are provided, use them to determine order
        if timestamps and len(timestamps) == len(prices):
            # Check if timestamps are in descending order (newest first)
            if len(timestamps) > 1 and timestamps[0] > timestamps[-1]:
                logger.debug("Timestamps are in reverse order (newest first), reordering to oldest-first")
                return list(reversed(prices.copy()))
            return prices.copy()  # Already in oldest-first order
        
        # Fallback to the simple heuristic if no timestamps
        # Make a copy to avoid modifying the original data
        chronological_prices = prices.copy()
        
        # This is a simple heuristic and may not always be correct
        # It assumes if first price > last price, data is in reverse (newest first)
        if len(prices) > 1 and prices[0] > prices[-1]:
            logger.debug("No timestamps available. Prices appear to be in reverse order, reordering to oldest-first")
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
    
    def calculate_rsi(self, prices: List[float], timestamps: Optional[List[int]] = None) -> float:
        """
        Calculate RSI using standard formula
        
        Args:
            prices: List of price values
            timestamps: Optional list of corresponding timestamps for accurate ordering
            
        Returns:
            RSI value (0-100)
        """
        if len(prices) < self.rsi_period + 1:
            logger.warning(f"Not enough data for {self.timeframe.value} RSI calculation")
            return 50
        
        # Make sure prices are in chronological order (oldest first)
        chronological_prices = self._ensure_chronological_order(prices, timestamps)
        
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
        
        # Calculate RSI using timestamps for proper ordering
        rsi = self.calculate_rsi(prices, timestamps)
        
        # Get the current price (most recent)
        # Find the index of the most recent timestamp
        if timestamps and len(timestamps) > 0:
            most_recent_idx = timestamps.index(max(timestamps))
            current_price = prices[most_recent_idx]
        else:
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
        
        # Note: Timestamps are displayed in Eastern Time (ET) for market hours reference,
        # regardless of the server's timezone. This is standard practice for US financial markets.
        print("\n📈 Candle Analysis:")
        print(f"Timeframe: {self.timeframe.value}")
        
        # Add current time marker for reference
        current_time_utc = datetime.now(utc_tz)
        current_time_est = current_time_utc.astimezone(est_tz)
        print(f"Current time: {current_time_est.strftime('%H:%M:%S')} EST - Data may lag behind")
        
        # Get timezone abbreviation for display
        tz_abbr = current_time_est.strftime('%Z')
        
        # Define consistent column widths to ensure alignment
        time_width = 12
        price_width = 14
        change_width = 10
        gain_width = 10
        loss_width = 10
        
        # Header with proper spacing and timezone info
        print("\n" + "│ " + f"Time ({tz_abbr})".ljust(time_width) + "Price".ljust(price_width) + 
              "Change".ljust(change_width) + "Gain".ljust(gain_width) + "Loss".ljust(loss_width))
        print("│ " + "-" * (time_width + price_width + change_width + gain_width + loss_width))
        
        # CRITICAL: First, let's make sure our data is correctly sorted by timestamp
        # Create combined data of prices and timestamps
        combined_data = list(zip(prices, timestamps))
        
        # Sort by timestamp in descending order (newest first)
        combined_data.sort(key=lambda x: x[1], reverse=True)
        
        # Unpack the sorted data
        sorted_prices, sorted_timestamps = zip(*combined_data)
        
        # Now we have our data in the correct order - newest first
        display_prices = list(sorted_prices)
        display_timestamps = list(sorted_timestamps)
        
        # Debug the timestamp range
        newest_timestamp = display_timestamps[0]
        oldest_timestamp = display_timestamps[-1]
        newest_time = datetime.fromtimestamp(newest_timestamp, tz=utc_tz).astimezone(est_tz)
        oldest_time = datetime.fromtimestamp(oldest_timestamp, tz=utc_tz).astimezone(est_tz)
        
        print(f"DEBUG: Full data range {oldest_time.strftime('%H:%M')} to {newest_time.strftime('%H:%M')} {tz_abbr}")
        print(f"DEBUG: Current time: {current_time_est.strftime('%H:%M')}, newest candle time: {newest_time.strftime('%H:%M')}")
        time_diff_minutes = (current_time_utc.timestamp() - newest_timestamp) // 60
        print(f"DEBUG: Most recent candle is {time_diff_minutes} minutes old")
        
        # Calculate deltas for price changes (for newest-first data)
        deltas = []
        for i in range(1, len(display_prices)):
            deltas.append(display_prices[i-1] - display_prices[i])
        deltas.append(0)  # Add a zero at the end for the oldest candle
        
        # Separate gains and losses
        gains = [max(0, change) for change in deltas]
        losses = [max(0, -change) for change in deltas]
        
        # Display the most recent candles (limiting to 15 for readability)
        periods_to_show = min(15, len(display_prices))
        
        # Calculate the time range we're showing
        first_shown_timestamp = display_timestamps[0]
        last_shown_timestamp = display_timestamps[periods_to_show-1] if periods_to_show > 1 else first_shown_timestamp
        first_shown_time = datetime.fromtimestamp(first_shown_timestamp, tz=utc_tz).astimezone(est_tz)
        last_shown_time = datetime.fromtimestamp(last_shown_timestamp, tz=utc_tz).astimezone(est_tz)
        
        # Display time range from oldest to newest that we're showing
        print(f"Showing {periods_to_show} most recent candles from {last_shown_time.strftime('%H:%M')} to {first_shown_time.strftime('%H:%M')} {tz_abbr}")
        
        # Print the candles
        for i in range(periods_to_show):
            try:
                dt = datetime.fromtimestamp(display_timestamps[i], tz=utc_tz).astimezone(est_tz)
                # Include date in format if looking at data spanning multiple days
                time_str = dt.strftime("%m-%d %H:%M") if self.timeframe.minutes >= 60 else dt.strftime("%H:%M")
                price_str = f"${display_prices[i]:,.2f}"
                
                # Format change with proper sign
                if i < len(deltas):
                    change = deltas[i]
                    if change != 0:
                        change_str = f"{change:+.2f}"
                    else:
                        change_str = "-"
                else:
                    change_str = "-"
                
                # Format gains and losses
                gain_str = f"{gains[i]:.2f}" if i < len(gains) and gains[i] > 0 else "-"
                loss_str = f"{losses[i]:.2f}" if i < len(losses) and losses[i] > 0 else "-"
                
                # Print with consistent column widths
                print("│ " + time_str.ljust(time_width) + 
                      price_str.ljust(price_width) + 
                      change_str.ljust(change_width) + 
                      gain_str.ljust(gain_width) + 
                      loss_str.ljust(loss_width))
            except IndexError as e:
                logger.error(f"Index error at position {i}: {str(e)}")
                break
                
        # Add back the RSI calculation
        # Create chronological prices (oldest first) for RSI calculation
        chronological_prices = list(reversed(display_prices.copy()))
        chronological_timestamps = list(reversed(display_timestamps.copy()))
        
        # Calculate deltas in chronological order for RSI
        rsi_deltas = np.diff(chronological_prices)
        
        # Separate gains and losses for RSI calculation
        calc_gains = np.where(rsi_deltas > 0, rsi_deltas, 0)
        calc_losses = np.where(rsi_deltas < 0, -rsi_deltas, 0)
        
        # Get the actual gain/loss values used in the RSI calculation
        rsi_gains_used = calc_gains[-self.rsi_period:].tolist()
        rsi_losses_used = calc_losses[-self.rsi_period:].tolist()
        
        # Filter out zeros to get only the actual gains and losses
        non_zero_gains = [g for g in rsi_gains_used if g > 0]
        non_zero_losses = [l for l in rsi_losses_used if l > 0]
        
        # Format the gain/loss values for display - only show non-zero values
        # If the string becomes too long, truncate it
        if len(non_zero_gains) > 5:
            gain_values_str = " + ".join([f"${g:.2f}" for g in non_zero_gains[:5]]) + f" + ... ({len(non_zero_gains)-5} more)"
        else:
            gain_values_str = " + ".join([f"${g:.2f}" for g in non_zero_gains]) if non_zero_gains else "$0"
        
        if len(non_zero_losses) > 5:
            loss_values_str = " + ".join([f"${l:.2f}" for l in non_zero_losses[:5]]) + f" + ... ({len(non_zero_losses)-5} more)"
        else:
            loss_values_str = " + ".join([f"${l:.2f}" for l in non_zero_losses]) if non_zero_losses else "$0"
        
        # Calculate average gain and loss
        avg_gain = np.mean(calc_gains[-self.rsi_period:])  # Include zeros
        avg_loss = np.mean(calc_losses[-self.rsi_period:]) # Include zeros
        rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
        rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 100
        
        # Count non-zero gains and losses for better understanding
        non_zero_gain_count = len(non_zero_gains)
        non_zero_loss_count = len(non_zero_losses)
        
        # Find the time range of candles used for RSI calculation
        rsi_start_idx = max(0, len(chronological_timestamps) - self.rsi_period)
        rsi_end_idx = len(chronological_timestamps) - 1
        
        # Convert RSI candle timestamps to datetime for display
        if rsi_start_idx < len(chronological_timestamps) and rsi_end_idx < len(chronological_timestamps):
            rsi_start_time = datetime.fromtimestamp(chronological_timestamps[rsi_start_idx], tz=utc_tz).astimezone(est_tz)
            rsi_end_time = datetime.fromtimestamp(chronological_timestamps[rsi_end_idx], tz=utc_tz).astimezone(est_tz)
            
            print("\n📊 RSI Calculation:")
            print(f"Using {self.rsi_period} periods from {rsi_start_time.strftime('%H:%M')} to {rsi_end_time.strftime('%H:%M')} {tz_abbr}")
            
            # Display the gain/loss values used in calculation
            print(f"Gains in period ({non_zero_gain_count}/{self.rsi_period} candles): {gain_values_str}")
            print(f"Losses in period ({non_zero_loss_count}/{self.rsi_period} candles): {loss_values_str}")
            
            # Show the full calculation with concrete values
            # Use the same gain_values_str and loss_values_str from above
            print(f"Average Gain ({gain_values_str}) / {self.rsi_period}: ${avg_gain:.2f}")
            print(f"Average Loss ({loss_values_str}) / {self.rsi_period}: ${avg_loss:.2f}")
            print(f"Relative Strength (RS) = Avg Gain / Avg Loss = {rs:.2f}")
            print(f"RSI = 100 - (100 / (1 + RS)) = {rsi:.2f}")
            
            # Verify the RSI matches with the calculate_rsi method
            verify_rsi = self.calculate_rsi(prices, timestamps)
            logger.info(f"RSI Verification - Printed: {rsi:.2f}, calculate_rsi method: {verify_rsi:.2f}")
        else:
            print("\n❌ Not enough data for RSI calculation")

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
            window_prices = prices[i-self.rsi_period:i+1]
            window_timestamps = timestamps[i-self.rsi_period:i+1] if timestamps else None
            rsi = self.calculate_rsi(window_prices, window_timestamps)
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