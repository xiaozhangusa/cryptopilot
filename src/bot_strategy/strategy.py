from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import logging
from .timeframes import Timeframe

logger = logging.getLogger(__name__)

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
        """Calculate RSI with timeframe-specific parameters"""
        if len(prices) < self.rsi_period + 1:
            logger.warning(f"Not enough data for {self.timeframe.value} RSI calculation")
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        logger.debug(f"{self.timeframe.value} RSI: {rsi:.2f}")
        return rsi

    def generate_signal(self, symbol: str, prices: List[float]) -> Optional[TradingSignal]:
        """Generate trading signal with timeframe-specific logic"""
        if len(prices) < self.timeframe.lookback_periods:
            logger.warning(f"Insufficient data for {self.timeframe.value} analysis")
            return None
            
        rsi = self.calculate_rsi(prices)
        current_price = prices[-1]
        
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
        
        return None 