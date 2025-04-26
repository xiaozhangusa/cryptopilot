"""
Moving Average Convergence Divergence (MACD) indicator implementation.
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import logging
from .indicator_base import IndicatorBase
from .ema import EMA

logger = logging.getLogger(__name__)

class MACD(IndicatorBase):
    """
    Moving Average Convergence Divergence (MACD) indicator.
    
    This indicator calculates the difference between two exponential moving averages
    (typically 12 and 26 periods) and a signal line (typically 9-period EMA of the MACD).
    It's used to identify changes in the strength, direction, momentum, and duration of a trend.
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialize MACD indicator.
        
        Args:
            fast_period: The period for the fast EMA (default: 12)
            slow_period: The period for the slow EMA (default: 26)
            signal_period: The period for the signal line EMA (default: 9)
        """
        super().__init__("MACD", slow_period)  # Use slow period as the main period
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        # Initialize EMAs
        self.fast_ema = EMA(period=fast_period)
        self.slow_ema = EMA(period=slow_period)
        
        logger.info(f"Initialized MACD({fast_period},{slow_period},{signal_period}) indicator")
    
    def calculate(self, prices: List[float], timestamps: Optional[List[int]] = None) -> Tuple[float, float, float]:
        """
        Calculate MACD, signal line, and histogram values.
        
        Args:
            prices: List of price values
            timestamps: Optional list of corresponding timestamps
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        if len(prices) < self.slow_period + self.signal_period:
            logger.warning(f"Not enough data for MACD calculation (need {self.slow_period + self.signal_period}, got {len(prices)})")
            return 0, 0, 0
        
        # Make sure prices are in chronological order (oldest first)
        chronological_prices = self._ensure_chronological_order(prices, timestamps)
        
        # Calculate fast and slow EMAs
        fast_ema_values = self.fast_ema.calculate_multiple(chronological_prices, timestamps)
        slow_ema_values = self.slow_ema.calculate_multiple(chronological_prices, timestamps)
        
        # Calculate MACD line (fast EMA - slow EMA)
        macd_line = []
        for i in range(len(fast_ema_values)):
            if np.isnan(fast_ema_values[i]) or np.isnan(slow_ema_values[i]):
                macd_line.append(np.nan)
            else:
                macd_line.append(fast_ema_values[i] - slow_ema_values[i])
        
        # Calculate signal line (EMA of MACD line)
        # Filter out NaN values
        valid_macd = [x for x in macd_line if not np.isnan(x)]
        
        if len(valid_macd) < self.signal_period:
            logger.warning(f"Not enough valid MACD values for signal line calculation")
            return macd_line[-1] if macd_line and not np.isnan(macd_line[-1]) else 0, 0, 0
        
        # Calculate signal line as EMA of MACD line
        signal_line = []
        
        # First calculate an initial SMA of the MACD for the signal line
        initial_sma_idx = self.slow_period - 1  # Start index for initial calculation
        
        # If we have enough data points, use them for the SMA calculation
        if initial_sma_idx >= 0 and initial_sma_idx + self.signal_period <= len(macd_line):
            # Extract valid values for SMA calculation
            sma_values = [x for x in macd_line[initial_sma_idx:(initial_sma_idx + self.signal_period)] if not np.isnan(x)]
            
            # Pad with NaN values up to initial_sma_idx
            signal_line = [np.nan] * initial_sma_idx
            
            # Add the initial SMA value if we have valid values
            if sma_values:
                signal_line.append(np.mean(sma_values))
            else:
                signal_line.append(np.nan)
                
            # Now calculate the EMA for remaining values
            multiplier = 2.0 / (self.signal_period + 1)
            
            for i in range(initial_sma_idx + 1, len(macd_line)):
                prev_signal = signal_line[-1]
                # Skip calculation if previous signal or current MACD is NaN
                if np.isnan(prev_signal) or np.isnan(macd_line[i]):
                    signal_line.append(np.nan)
                else:
                    # EMA formula: (Current - Previous) * Multiplier + Previous
                    new_signal = (macd_line[i] - prev_signal) * multiplier + prev_signal
                    signal_line.append(new_signal)
        else:
            # Fallback: If we don't have enough data, initialize with MACD value
            signal_line = [np.nan] * len(macd_line)
            
            # Find the first valid MACD value to start the signal line
            for i in range(len(macd_line)):
                if not np.isnan(macd_line[i]):
                    signal_line[i] = macd_line[i]
                    
                    # Calculate forward from this point
                    multiplier = 2.0 / (self.signal_period + 1)
                    for j in range(i+1, len(macd_line)):
                        if np.isnan(macd_line[j]):
                            signal_line[j] = np.nan
                        else:
                            signal_line[j] = (macd_line[j] - signal_line[j-1]) * multiplier + signal_line[j-1]
                    break
        
        # Calculate histogram (MACD line - signal line)
        histogram = []
        for i in range(len(macd_line)):
            if np.isnan(macd_line[i]) or np.isnan(signal_line[i]):
                histogram.append(np.nan)
            else:
                histogram.append(macd_line[i] - signal_line[i])
        
        # Return the most recent values
        current_macd = macd_line[-1] if not np.isnan(macd_line[-1]) else 0
        current_signal = signal_line[-1] if not np.isnan(signal_line[-1]) else 0
        current_histogram = histogram[-1] if not np.isnan(histogram[-1]) else 0
        
        self._last_calculated_value = current_macd
        logger.debug(f"MACD: {current_macd:.4f}, Signal: {current_signal:.4f}, Histogram: {current_histogram:.4f}")
        
        return current_macd, current_signal, current_histogram
    
    def calculate_series(self, prices: List[float], timestamps: Optional[List[int]] = None) -> Dict[str, List[float]]:
        """
        Calculate full MACD, signal line, and histogram series.
        
        Args:
            prices: List of price values
            timestamps: Optional list of corresponding timestamps
            
        Returns:
            Dictionary with keys 'macd', 'signal', and 'histogram', each containing a list of values
        """
        if len(prices) < self.slow_period + self.signal_period:
            return {'macd': [], 'signal': [], 'histogram': []}
        
        # Make sure prices are in chronological order (oldest first)
        chronological_prices = self._ensure_chronological_order(prices, timestamps)
        
        # Calculate fast and slow EMAs
        fast_ema_values = self.fast_ema.calculate_multiple(chronological_prices, timestamps)
        slow_ema_values = self.slow_ema.calculate_multiple(chronological_prices, timestamps)
        
        # Calculate MACD line (fast EMA - slow EMA)
        macd_line = []
        for i in range(len(fast_ema_values)):
            if np.isnan(fast_ema_values[i]) or np.isnan(slow_ema_values[i]):
                macd_line.append(np.nan)
            else:
                macd_line.append(fast_ema_values[i] - slow_ema_values[i])
        
        # Calculate signal line (EMA of MACD line)
        # First calculate an initial SMA of the MACD for the signal line
        initial_sma_idx = self.slow_period - 1  # Start index for initial calculation
        signal_line = []
        
        # If we have enough data points, use them for the SMA calculation
        if initial_sma_idx >= 0 and initial_sma_idx + self.signal_period <= len(macd_line):
            # Extract valid values for SMA calculation
            sma_values = [x for x in macd_line[initial_sma_idx:(initial_sma_idx + self.signal_period)] if not np.isnan(x)]
            
            # Pad with NaN values up to initial_sma_idx
            signal_line = [np.nan] * initial_sma_idx
            
            # Add the initial SMA value if we have valid values
            if sma_values:
                signal_line.append(np.mean(sma_values))
            else:
                signal_line.append(np.nan)
                
            # Now calculate the EMA for remaining values
            multiplier = 2.0 / (self.signal_period + 1)
            
            for i in range(initial_sma_idx + 1, len(macd_line)):
                prev_signal = signal_line[-1]
                # Skip calculation if previous signal or current MACD is NaN
                if np.isnan(prev_signal) or np.isnan(macd_line[i]):
                    signal_line.append(np.nan)
                else:
                    # EMA formula: (Current - Previous) * Multiplier + Previous
                    new_signal = (macd_line[i] - prev_signal) * multiplier + prev_signal
                    signal_line.append(new_signal)
        else:
            # Fallback: If we don't have enough data, initialize with MACD value
            signal_line = [np.nan] * len(macd_line)
            
            # Find the first valid MACD value to start the signal line
            for i in range(len(macd_line)):
                if not np.isnan(macd_line[i]):
                    signal_line[i] = macd_line[i]
                    
                    # Calculate forward from this point
                    multiplier = 2.0 / (self.signal_period + 1)
                    for j in range(i+1, len(macd_line)):
                        if np.isnan(macd_line[j]):
                            signal_line[j] = np.nan
                        else:
                            signal_line[j] = (macd_line[j] - signal_line[j-1]) * multiplier + signal_line[j-1]
                    break
        
        # Calculate histogram (MACD line - signal line)
        histogram = []
        for i in range(len(macd_line)):
            if np.isnan(macd_line[i]) or np.isnan(signal_line[i]):
                histogram.append(np.nan)
            else:
                histogram.append(macd_line[i] - signal_line[i])
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def get_signal(self, prices: List[float], timestamps: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Get trading signal based on MACD crossovers and histogram changes.
        
        Args:
            prices: List of price values
            timestamps: Optional list of corresponding timestamps
            
        Returns:
            Dictionary with signal information:
            - 'signal': 'BUY', 'SELL', or 'NEUTRAL'
            - 'value': The MACD value
            - 'strength': Signal strength (0-100)
            - 'signal_line': The signal line value
            - 'histogram': The histogram value
            - 'histogram_change': The change in histogram value
        """
        if len(prices) < self.slow_period + self.signal_period + 5:  # Need extra points for crossover detection
            return {
                'signal': 'NEUTRAL',
                'value': 0,
                'strength': 0,
                'signal_line': 0,
                'histogram': 0,
                'histogram_change': 0
            }
        
        # Calculate full series to detect crossovers
        series = self.calculate_series(prices, timestamps)
        macd_series = series['macd']
        signal_series = series['signal']
        histogram_series = series['histogram']
        
        # Get the last few valid values
        valid_idx = -1
        while valid_idx >= -len(macd_series) and (
                np.isnan(macd_series[valid_idx]) or 
                np.isnan(signal_series[valid_idx]) or 
                np.isnan(histogram_series[valid_idx])):
            valid_idx -= 1
        
        if valid_idx < -len(macd_series):
            logger.warning("No valid MACD values found")
            return {
                'signal': 'NEUTRAL',
                'value': 0,
                'strength': 0,
                'signal_line': 0,
                'histogram': 0,
                'histogram_change': 0
            }
        
        # Current and previous values
        current_macd = macd_series[valid_idx]
        current_signal = signal_series[valid_idx]
        current_histogram = histogram_series[valid_idx]
        
        # Get previous values (if available)
        prev_idx = valid_idx - 1
        if prev_idx >= -len(macd_series) and not np.isnan(macd_series[prev_idx]):
            prev_macd = macd_series[prev_idx]
            prev_signal = signal_series[prev_idx]
            prev_histogram = histogram_series[prev_idx]
        else:
            prev_macd = current_macd
            prev_signal = current_signal
            prev_histogram = current_histogram
        
        # Calculate histogram change
        histogram_change = current_histogram - prev_histogram
        
        # Check for crossovers
        macd_above_signal = current_macd > current_signal
        macd_was_above_signal = prev_macd > prev_signal
        
        # Determine signal
        signal = 'NEUTRAL'
        strength = 0
        
        if macd_above_signal and not macd_was_above_signal:
            # Bullish crossover (MACD crosses above signal line)
            signal = 'BUY'
            # Base strength on histogram value and MACD value
            strength = min(100, max(0, 50 + (current_histogram / abs(current_macd) * 50 if current_macd != 0 else 25)))
        elif not macd_above_signal and macd_was_above_signal:
            # Bearish crossover (MACD crosses below signal line)
            signal = 'SELL'
            # Base strength on histogram value and MACD value
            strength = min(100, max(0, 50 + (abs(current_histogram) / abs(current_macd) * 50 if current_macd != 0 else 25)))
        elif macd_above_signal:
            # MACD remaining above signal (continuation of uptrend)
            if histogram_change > 0:
                # Increasing histogram in uptrend is bullish continuation
                signal = 'BUY'
                strength = min(100, max(0, 30 + (histogram_change / abs(current_macd) * 40 if current_macd != 0 else 15)))
            elif current_histogram > 0:
                # Positive histogram in uptrend is slightly bullish
                signal = 'NEUTRAL'  # or weak BUY
                strength = min(100, max(0, 20 + (current_histogram / abs(current_macd) * 30 if current_macd != 0 else 10)))
        else:
            # MACD remaining below signal (continuation of downtrend)
            if histogram_change < 0:
                # Decreasing histogram in downtrend is bearish continuation
                signal = 'SELL'
                strength = min(100, max(0, 30 + (abs(histogram_change) / abs(current_macd) * 40 if current_macd != 0 else 15)))
            elif current_histogram < 0:
                # Negative histogram in downtrend is slightly bearish
                signal = 'NEUTRAL'  # or weak SELL
                strength = min(100, max(0, 20 + (abs(current_histogram) / abs(current_macd) * 30 if current_macd != 0 else 10)))
        
        return {
            'signal': signal,
            'value': current_macd,
            'strength': strength,
            'signal_line': current_signal,
            'histogram': current_histogram,
            'histogram_change': histogram_change
        } 