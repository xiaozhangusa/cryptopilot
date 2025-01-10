from dataclasses import dataclass
import numpy as np
from typing import List, Optional

@dataclass
class TradingSignal:
    symbol: str
    action: str  # 'BUY' or 'SELL'
    price: float

class SwingStrategy:
    def __init__(self, rsi_period: int = 14, 
                 oversold: float = 30, 
                 overbought: float = 70):
        """Initialize Swing Trading Strategy with RSI
        
        Args:
            rsi_period: Period for RSI calculation (default: 14)
            oversold: RSI level to consider oversold (default: 30)
            overbought: RSI level to consider overbought (default: 70)
        """
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def calculate_rsi(self, prices: List[float]) -> float:
        """Calculate RSI for given prices"""
        if len(prices) < self.rsi_period + 1:
            return 50  # Return neutral RSI if not enough data
            
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100
        
        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def generate_signal(self, symbol: str, prices: List[float]) -> Optional[TradingSignal]:
        """Generate trading signal based on RSI
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USD')
            prices: List of closing prices
            
        Returns:
            TradingSignal if conditions are met, None otherwise
        """
        if len(prices) < self.rsi_period + 1:
            return None
            
        rsi = self.calculate_rsi(prices)
        current_price = prices[-1]
        
        if rsi < self.oversold:
            return TradingSignal(
                symbol=symbol,
                action='BUY',
                price=current_price
            )
        elif rsi > self.overbought:
            return TradingSignal(
                symbol=symbol,
                action='SELL',
                price=current_price
            )
            
        return None 