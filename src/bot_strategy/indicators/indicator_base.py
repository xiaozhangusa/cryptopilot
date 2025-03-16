"""
Base class for technical indicators.
"""

from typing import List, Optional, Union, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

class IndicatorBase:
    """Base class for all technical indicators."""
    
    # Class-level toggle for detailed calculations (global setting)
    show_detailed_calculations = False
    
    def __init__(self, name: str, period: int):
        """
        Initialize indicator with name and period.
        
        Args:
            name: The name of the indicator
            period: The lookback period used for calculation
        """
        self.name = name
        self.period = period
        self._last_calculated_value = None
    
    @classmethod
    def set_verbose_mode(cls, verbose: bool = True):
        """
        Enable or disable detailed calculation logs for all indicators.
        
        Args:
            verbose: True to show detailed calculations, False to hide
        """
        cls.show_detailed_calculations = verbose
        logger.info(f"Detailed calculation logs for indicators: {'ENABLED' if verbose else 'DISABLED'}")
    
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
                logger.debug(f"{self.name}: Timestamps are in reverse order (newest first), reordering to oldest-first")
                return list(reversed(prices.copy()))
            return prices.copy()  # Already in oldest-first order
        
        # Fallback to the simple heuristic if no timestamps
        # Make a copy to avoid modifying the original data
        chronological_prices = prices.copy()
        
        # This is a simple heuristic and may not always be correct
        # It assumes if first price > last price, data is in reverse (newest first)
        if len(prices) > 1 and prices[0] > prices[-1]:
            logger.debug(f"{self.name}: No timestamps available. Prices appear to be in reverse order, reordering to oldest-first")
            chronological_prices = list(reversed(chronological_prices))
            
        return chronological_prices
    
    def calculate(self, prices: List[float], timestamps: Optional[List[int]] = None) -> float:
        """
        Calculate the indicator value.
        
        Args:
            prices: List of price values
            timestamps: Optional list of corresponding timestamps
            
        Returns:
            Indicator value
        """
        raise NotImplementedError("Subclasses must implement calculate()")
    
    def get_signal(self, prices: List[float], timestamps: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Get trading signal from indicator.
        
        Args:
            prices: List of price values
            timestamps: Optional list of corresponding timestamps
            
        Returns:
            Dictionary with signal information:
            - 'signal': 'BUY', 'SELL', or 'NEUTRAL'
            - 'value': The indicator value
            - 'strength': Signal strength (0-100)
            - Additional indicator-specific fields
        """
        raise NotImplementedError("Subclasses must implement get_signal()") 