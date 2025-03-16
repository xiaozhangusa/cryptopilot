"""
Exponential Moving Average (EMA) indicator implementation.
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import logging
from .indicator_base import IndicatorBase
from datetime import datetime

logger = logging.getLogger(__name__)

class EMA(IndicatorBase):
    """
    Exponential Moving Average (EMA) indicator.
    
    This indicator calculates an exponentially weighted moving average of prices,
    giving more weight to recent prices. It can be used to identify trends and
    potential reversals.
    """
    
    def __init__(self, period: int = 20, smoothing: float = 2.0):
        """
        Initialize EMA indicator.
        
        Args:
            period: The lookback period for EMA calculation (default: 20)
            smoothing: Smoothing factor for EMA (default: 2.0)
        """
        super().__init__("EMA", period)
        self.smoothing = smoothing
        self._last_ema = None
        
        logger.info(f"Initialized EMA({period}) indicator")
    
    def calculate(self, prices: List[float], timestamps: Optional[List[int]] = None) -> float:
        """
        Calculate EMA using standard formula.
        
        Args:
            prices: List of price values
            timestamps: Optional list of corresponding timestamps
            
        Returns:
            EMA value
        """
        if len(prices) < self.period:
            logger.warning(f"Not enough data for EMA calculation (need {self.period}, got {len(prices)})")
            return prices[-1] if prices else 0  # Default to last price or 0
        
        # Make sure prices are in chronological order (oldest first)
        chronological_prices = self._ensure_chronological_order(prices, timestamps)
        chronological_timestamps = timestamps
        if timestamps and len(timestamps) == len(prices) and timestamps[0] > timestamps[-1]:
            chronological_timestamps = list(reversed(timestamps.copy()))
        
        # Calculate multiplier
        multiplier = self.smoothing / (self.period + 1)
        
        # Initialize EMA with SMA if _last_ema is None
        if self._last_ema is None:
            sma = np.mean(chronological_prices[:self.period])
            ema = sma
        else:
            ema = self._last_ema
            
        # Get the prices used for EMA calculation
        ema_prices = chronological_prices[self.period:]
        
        # Store the initial EMA value
        current_ema = ema
        
        # Calculate EMA for each price after the initial SMA period
        for i, price in enumerate(ema_prices):
            current_ema = (price - current_ema) * multiplier + current_ema
        
        # Generate concise EMA calculation display
        if self.show_detailed_calculations and chronological_timestamps:
            # For display purposes, we want to use the most recent data
            # We'll use the most recent self.period prices for SMA and the next period prices for EMA
            
            # Get total length of data
            total_length = len(chronological_prices)
            
            # Determine indices for recent data display
            if total_length >= 2 * self.period:
                # We have enough data for both SMA and EMA display
                # Use the last 2*period elements, with the first half for SMA and second half for EMA steps
                last_index = total_length - 1
                first_index = max(0, last_index - (2 * self.period) + 1)
                
                # SMA period is the first half of this window
                sma_start_idx = first_index
                sma_end_idx = first_index + self.period - 1
                
                # EMA calculation period is the second half
                ema_start_idx = sma_end_idx + 1
                ema_end_idx = last_index
            else:
                # Not enough data for full display, use what we have
                sma_start_idx = 0
                sma_end_idx = min(self.period - 1, total_length - 1)
                ema_start_idx = sma_end_idx + 1
                ema_end_idx = total_length - 1
            
            # Make sure indices are valid
            ema_start_idx = min(ema_start_idx, total_length - 1)
            
            # Get timezone for display
            timezone = datetime.now().astimezone().tzname()
            
            # Format and display if we have valid timestamps
            if chronological_timestamps and len(chronological_timestamps) == total_length:
                # Format timestamp range for display
                if sma_start_idx < len(chronological_timestamps) and sma_end_idx < len(chronological_timestamps):
                    sma_start_time = datetime.fromtimestamp(chronological_timestamps[sma_start_idx]).strftime('%H:%M')
                    sma_end_time = datetime.fromtimestamp(chronological_timestamps[sma_end_idx]).strftime('%H:%M')
                    
                    # Get the SMA prices for display
                    display_sma_prices = chronological_prices[sma_start_idx:sma_end_idx+1]
                    display_sma = np.mean(display_sma_prices)
                    
                    # Define EMA indices and times for display first
                    # Ensure ema_start_time and ema_end_time are defined before the visualization
                    if ema_start_idx <= ema_end_idx and ema_start_idx < len(chronological_timestamps) and ema_end_idx < len(chronological_timestamps):
                        ema_start_time = datetime.fromtimestamp(chronological_timestamps[ema_start_idx]).strftime('%H:%M')
                        ema_end_time = datetime.fromtimestamp(chronological_timestamps[ema_end_idx]).strftime('%H:%M')
                        display_ema_prices = chronological_prices[ema_start_idx:ema_end_idx+1]
                    else:
                        # Fallback if EMA indices are invalid
                        ema_start_time = "N/A"
                        ema_end_time = "N/A"
                        display_ema_prices = []
                    
                    # Start displaying calculation
                    print("\n📈 EMA Calculation:")
                    print(f"Period: {self.period}, Smoothing Factor: {multiplier:.4f}")
                    
                    # Add the timeline illustration to explain the two time periods with better alignment
                    print("\nEMA uses two distinct time periods:")
                    
                    # Calculate widths for better alignment
                    sma_range = f"{sma_start_time}...{sma_end_time}"
                    ema_range = f"{ema_start_time}...{ema_end_time}"
                    sma_width = len(sma_range) + 2  # +2 for brackets
                    ema_width = len(ema_range) + 2  # +2 for brackets
                    
                    # Create the timeline with better alignment
                    print(f"[{sma_range}] [{ema_range}]")
                    
                    # Create underlines with exact width to match the ranges above
                    sma_line = "─" * (sma_width - 2)  # -2 to account for the └ and ┘
                    ema_line = "─" * (ema_width - 2)  # -2 to account for the └ and ┘
                    print(f"└{sma_line}┘ └{ema_line}┘")
                    
                    # Center the explanatory text under each section
                    sma_text = f"First {self.period} periods"
                    ema_text = f"Next {len(display_ema_prices)} periods"
                    
                    # Pad text to center it under each section
                    sma_pad = max(0, (sma_width - len(sma_text)) // 2)
                    ema_pad = max(0, (ema_width - len(ema_text)) // 2)
                    
                    print(f"{' ' * sma_pad}{sma_text}{' ' * sma_pad} {' ' * ema_pad}{ema_text}{' ' * ema_pad}")
                    print("")
                    
                    print(f"SMA Calculation: Using {len(display_sma_prices)} periods from {sma_start_time} to {sma_end_time} {timezone}")
                    
                    # Show SMA calculation with sample prices
                    if len(display_sma_prices) > 8:
                        # Show first 3 and last 3 prices if too many
                        sma_prices_display = [f"${p:.2f}" for p in display_sma_prices[:3]]
                        sma_prices_display.append("...")
                        sma_prices_display.extend([f"${p:.2f}" for p in display_sma_prices[-3:]])
                    else:
                        # Show all prices if not too many
                        sma_prices_display = [f"${p:.2f}" for p in display_sma_prices]
                    
                    sma_display = " + ".join(sma_prices_display)
                    print(f"SMA = ({sma_display}) / {len(display_sma_prices)} = ${display_sma:.2f}")
                    print(f"Initial EMA = SMA = ${display_sma:.2f}")
                    
                    # Show EMA calculation steps if we have data
                    if len(display_ema_prices) > 0:
                        print(f"EMA Calculation: Using {len(display_ema_prices)} periods from {ema_start_time} to {ema_end_time} {timezone}")
                        
                        # Show first few EMA calculation steps
                        steps_to_show = min(3, len(display_ema_prices))
                        
                        # Start with SMA for first step
                        ema_value = display_sma
                        for i in range(steps_to_show):
                            if i < len(display_ema_prices):
                                price = display_ema_prices[i]
                                new_ema = (price - ema_value) * multiplier + ema_value
                                print(f"Step {i+1}: EMA = (${price:.2f} - ${ema_value:.2f}) × {multiplier:.4f} + ${ema_value:.2f} = ${new_ema:.2f}")
                                ema_value = new_ema
                        
                        # If there are more than 6 steps, show an ellipsis
                        if len(display_ema_prices) > 6:
                            print(f"... ({len(display_ema_prices) - 6} intermediate steps) ...")
                        
                        # Show the last 3 steps if more than 3 steps
                        if len(display_ema_prices) > 3:
                            # Calculate up to last 3 steps
                            ema_value = display_sma
                            for i in range(len(display_ema_prices) - 3):
                                price = display_ema_prices[i]
                                ema_value = (price - ema_value) * multiplier + ema_value
                            
                            # Show last 3 steps
                            for i in range(max(0, len(display_ema_prices) - 3), len(display_ema_prices)):
                                price = display_ema_prices[i]
                                new_ema = (price - ema_value) * multiplier + ema_value
                                print(f"Step {i+1}: EMA = (${price:.2f} - ${ema_value:.2f}) × {multiplier:.4f} + ${ema_value:.2f} = ${new_ema:.2f}")
                                ema_value = new_ema
                    
                    # Show the final calculated EMA value
                    print(f"Final EMA({self.period}) = ${current_ema:.2f}")
                    
                    # Verification log
                    logger.info(f"EMA Verification - Calculated: {current_ema:.2f}")
                else:
                    # Fallback if indices are out of range
                    print("\n📈 EMA Calculation:")
                    print(f"Period: {self.period}, Smoothing Factor: {multiplier:.4f}")
                    print(f"SMA = (price data from {self.period} periods) / {self.period} = ${sma:.2f}")
                    print(f"EMA({self.period}) = ${current_ema:.2f}")
            else:
                # Fallback if no timestamps
                print("\n📈 EMA Calculation:")
                print(f"Period: {self.period}, Smoothing Factor: {multiplier:.4f}")
                print(f"EMA({self.period}) = ${current_ema:.2f}")
        
        self._last_calculated_value = current_ema
        self._last_ema = current_ema
        return current_ema
    
    def calculate_multiple(self, prices: List[float], timestamps: Optional[List[int]] = None) -> List[float]:
        """
        Calculate EMA for each point in the price series.
        
        Args:
            prices: List of price values
            timestamps: Optional list of corresponding timestamps
            
        Returns:
            List of EMA values for each price point
        """
        if len(prices) < self.period:
            logger.warning(f"Not enough data for EMA calculation (need {self.period}, got {len(prices)})")
            return prices.copy()  # Return copy of prices as fallback
        
        # Make sure prices are in chronological order (oldest first)
        chronological_prices = self._ensure_chronological_order(prices, timestamps)
        
        # Calculate multiplier
        multiplier = self.smoothing / (self.period + 1)
        
        # Initialize with SMA
        ema_values = [np.nan] * (self.period - 1)
        ema = np.mean(chronological_prices[:self.period])
        ema_values.append(ema)
        
        # Calculate EMA for each remaining price
        for i in range(self.period, len(chronological_prices)):
            ema = (chronological_prices[i] - ema) * multiplier + ema
            ema_values.append(ema)
        
        self._last_calculated_value = ema
        self._last_ema = ema
        return ema_values
    
    def get_signal(self, prices: List[float], timestamps: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Get trading signal based on price crossing EMA.
        
        Args:
            prices: List of price values
            timestamps: Optional list of corresponding timestamps
            
        Returns:
            Dictionary with signal information:
            - 'signal': 'BUY', 'SELL', or 'NEUTRAL'
            - 'value': The EMA value
            - 'strength': Signal strength (0-100)
            - 'price': Current price
            - 'distance_pct': Distance from price to EMA as percentage
        """
        if len(prices) < self.period + 2:  # Need at least period + 2 points to detect a crossing
            return {
                'signal': 'NEUTRAL',
                'value': prices[-1] if prices else 0,
                'strength': 0,
                'price': prices[-1] if prices else 0,
                'distance_pct': 0
            }
        
        # Make sure prices are in chronological order (oldest first)
        chronological_prices = self._ensure_chronological_order(prices, timestamps)
        
        # Calculate EMA series
        ema_series = self.calculate_multiple(chronological_prices)
        
        # Current and previous values
        current_price = chronological_prices[-1]
        previous_price = chronological_prices[-2]
        current_ema = ema_series[-1]
        previous_ema = ema_series[-2]
        
        # Check for crossovers
        price_above_ema = current_price > current_ema
        price_was_above_ema = previous_price > previous_ema
        
        # Calculate distance as percentage
        distance_pct = (current_price - current_ema) / current_ema * 100
        
        # Determine signal
        if price_above_ema and not price_was_above_ema:
            # Bullish crossover (price crosses above EMA)
            signal = 'BUY'
            strength = min(100, max(0, 50 + abs(distance_pct) * 5))  # Base strength on distance
        elif not price_above_ema and price_was_above_ema:
            # Bearish crossover (price crosses below EMA)
            signal = 'SELL'
            strength = min(100, max(0, 50 + abs(distance_pct) * 5))  # Base strength on distance
        elif price_above_ema:
            # Price staying above EMA (uptrend)
            signal = 'NEUTRAL'  # or could be weak 'BUY'
            strength = min(100, max(0, 30 + abs(distance_pct) * 2))  # Lower strength for continuation
        else:
            # Price staying below EMA (downtrend)
            signal = 'NEUTRAL'  # or could be weak 'SELL'
            strength = min(100, max(0, 30 + abs(distance_pct) * 2))  # Lower strength for continuation
        
        return {
            'signal': signal,
            'value': current_ema,
            'strength': strength,
            'price': current_price,
            'distance_pct': distance_pct
        } 