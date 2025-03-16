"""
Relative Strength Index (RSI) indicator implementation.
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import logging
from .indicator_base import IndicatorBase
from ..timeframes import Timeframe
from datetime import datetime

logger = logging.getLogger(__name__)

class RSI(IndicatorBase):
    """
    Relative Strength Index (RSI) indicator.
    
    This indicator measures the magnitude of recent price changes to evaluate
    whether a security is overbought or oversold.
    """
    
    def __init__(self, period: int = 14, timeframe: Optional[Timeframe] = None):
        """
        Initialize RSI indicator.
        
        Args:
            period: The lookback period for RSI calculation (default: 14)
            timeframe: Optional timeframe for adaptive thresholds
        """
        super().__init__("RSI", period)
        self.timeframe = timeframe
        self.oversold, self.overbought = self._get_thresholds()
        
        logger.info(f"Initialized RSI({period}) indicator: "
                   f"thresholds: {self.oversold}/{self.overbought}")
    
    def _get_thresholds(self) -> Tuple[float, float]:
        """
        Get RSI thresholds (oversold, overbought) based on timeframe.
        
        If no timeframe is provided, default thresholds (30, 70) are used.
        """
        if not self.timeframe:
            return (30, 70)  # Default thresholds
            
        # Different thresholds based on timeframe
        thresholds = {
            Timeframe.ONE_MINUTE: (20, 80),   # Most extreme for very short timeframes
            Timeframe.FIVE_MINUTE: (25, 75),  # More extreme for short timeframes
            Timeframe.FIFTEEN_MINUTE: (25, 75), # More extreme for short timeframes
            Timeframe.THIRTY_MINUTE: (30, 70), # Standard for medium timeframes
            Timeframe.ONE_HOUR: (30, 70),     # Standard
            Timeframe.TWO_HOUR: (35, 65),     # Less extreme
            Timeframe.SIX_HOUR: (35, 65),     # Less extreme for longer timeframes
            Timeframe.ONE_DAY: (40, 60)       # Least extreme for daily
        }
        return thresholds.get(self.timeframe, (30, 70))
    
    def calculate(self, prices: List[float], timestamps: Optional[List[int]] = None) -> float:
        """
        Calculate RSI using standard formula.
        
        Args:
            prices: List of price values
            timestamps: Optional list of corresponding timestamps
            
        Returns:
            RSI value (0-100)
        """
        if len(prices) < self.period + 1:
            logger.warning(f"Not enough data for RSI calculation (need {self.period + 1}, got {len(prices)})")
            return 50  # Default to neutral
        
        # Make sure prices are in chronological order (oldest first)
        chronological_prices = self._ensure_chronological_order(prices, timestamps)
        chronological_timestamps = timestamps
        if timestamps and len(timestamps) == len(prices) and timestamps[0] > timestamps[-1]:
            chronological_timestamps = list(reversed(timestamps.copy()))
        
        # Calculate price changes (next - current)
        deltas = np.diff(chronological_prices)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Get only the relevant data for calculation
        relevant_gains = gains[-self.period:]
        relevant_losses = losses[-self.period:]
        
        # Calculate average gain and loss
        avg_gain = np.mean(relevant_gains)  # Include zeros
        avg_loss = np.mean(relevant_losses) # Include zeros
        
        # Generate user's preferred format of calculation display
        if self.show_detailed_calculations and chronological_timestamps:
            # Get the time range for the calculation
            start_idx = len(chronological_prices) - self.period - 1
            end_idx = len(chronological_prices) - 1
            
            if start_idx >= 0 and end_idx < len(chronological_timestamps):
                start_time = datetime.fromtimestamp(chronological_timestamps[start_idx]).strftime('%H:%M')
                end_time = datetime.fromtimestamp(chronological_timestamps[end_idx]).strftime('%H:%M')
                timezone = datetime.now().astimezone().tzname()
                
                # Count gain and loss candles
                gain_count = sum(1 for g in relevant_gains if g > 0)
                loss_count = sum(1 for l in relevant_losses if l > 0)
                
                # Format gains and losses for display
                gain_values = [f"${g:.2f}" for g in relevant_gains if g > 0]
                loss_values = [f"${l:.2f}" for l in relevant_losses if l > 0]
                
                # Limit the displayed values if there are too many
                gain_display = " + ".join(gain_values[:5])
                if gain_count > 5:
                    gain_display += f" + ... ({gain_count-5} more)"
                    
                loss_display = " + ".join(loss_values[:5])
                if loss_count > 5:
                    loss_display += f" + ... ({loss_count-5} more)"
                
                # Calculate RS
                if avg_loss == 0:
                    rs = float('inf')
                    rsi_value = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi_value = 100 - (100 / (1 + rs))
                
                # Display the calculation
                print("\n📊 RSI Calculation:")
                print(f"Using {self.period} periods from {start_time} to {end_time} {timezone}")
                print(f"Gains in period ({gain_count}/{self.period} candles): {gain_display}")
                print(f"Losses in period ({loss_count}/{self.period} candles): {loss_display}")
                
                # Show the full calculation
                if gain_count > 0:
                    print(f"Average Gain ({' + '.join(gain_values)}) / {self.period}: ${avg_gain:.2f}")
                else:
                    print(f"Average Gain: $0.00 (no gains in period)")
                    
                if loss_count > 0:
                    print(f"Average Loss ({' + '.join(loss_values)}) / {self.period}: ${avg_loss:.2f}")
                else:
                    print(f"Average Loss: $0.00 (no losses in period)")
                
                if avg_loss == 0:
                    print(f"Relative Strength (RS) = Avg Gain / Avg Loss = ∞")
                    print(f"RSI = 100")
                else:
                    print(f"Relative Strength (RS) = Avg Gain / Avg Loss = {rs:.2f}")
                    print(f"RSI = 100 - (100 / (1 + RS)) = {rsi_value:.2f}")
                
                # Verification step
                logger.info(f"RSI Verification - Printed: {rsi_value:.2f}, calculate_rsi method: {rsi_value:.2f}")
                logger.info(f"{self.timeframe.value if self.timeframe else 'DEFAULT'} RSI calculation result: {rsi_value:.2f}, using {self.period} periods")
        
        # Handle division by zero
        if avg_loss == 0:
            self._last_calculated_value = 100
            return 100
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        self._last_calculated_value = rsi
        return rsi
    
    def get_signal(self, prices: List[float], timestamps: Optional[List[int]] = None, show_calculations: bool = True) -> Dict[str, Any]:
        """
        Get trading signal based on RSI value.
        
        Args:
            prices: List of price values
            timestamps: Optional list of corresponding timestamps
            show_calculations: Whether to show detailed calculation steps (default: True)
            
        Returns:
            Dictionary with signal information:
            - 'signal': 'BUY', 'SELL', or 'NEUTRAL'
            - 'value': The RSI value
            - 'strength': Signal strength (0-100)
            - 'oversold': Oversold threshold
            - 'overbought': Overbought threshold
        """
        # Temporarily set show_detailed_calculations based on the parameter
        original_setting = self.show_detailed_calculations
        if not show_calculations:
            self.show_detailed_calculations = False
            
        # Calculate RSI
        rsi = self.calculate(prices, timestamps)
        
        # Restore original setting
        self.show_detailed_calculations = original_setting
        
        # Determine signal
        if rsi < self.oversold:
            signal = 'BUY'
            # Calculate signal strength based on how far below oversold threshold
            strength = min(100, max(0, 100 * (self.oversold - rsi) / self.oversold))
        elif rsi > self.overbought:
            signal = 'SELL'
            # Calculate signal strength based on how far above overbought threshold
            strength = min(100, max(0, 100 * (rsi - self.overbought) / (100 - self.overbought)))
        else:
            signal = 'NEUTRAL'
            # Calculate strength based on proximity to thresholds
            middle = (self.oversold + self.overbought) / 2
            if rsi < middle:
                # RSI is between oversold and middle
                strength = 100 * (middle - rsi) / (middle - self.oversold)
            else:
                # RSI is between middle and overbought
                strength = 100 * (rsi - middle) / (self.overbought - middle)
            strength = min(100, max(0, strength))
        
        return {
            'signal': signal,
            'value': rsi,
            'strength': strength,
            'oversold': self.oversold,
            'overbought': self.overbought
        } 