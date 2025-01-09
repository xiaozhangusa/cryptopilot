from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

@dataclass
class TradingSignal:
    symbol: str
    action: str  # 'BUY' or 'SELL'
    price: float
    timestamp: str
    confidence: float
    indicators: Dict[str, float]

class SwingStrategy:
    def __init__(self):
        # ... initialization code ...
        pass

    def calculate_rsi(self, prices: List[float]) -> float:
        df = pd.Series(prices)
        delta = df.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))

    def calculate_ma_crossover(self, prices: List[float]) -> bool:
        df = pd.Series(prices)
        short_ma = df.rolling(window=self.short_ma).mean()
        long_ma = df.rolling(window=self.long_ma).mean()
        
        # Check if short MA crossed above long MA
        return (short_ma.iloc[-2] <= long_ma.iloc[-2] and 
                short_ma.iloc[-1] > long_ma.iloc[-1])

    def generate_signal(self, 
                       symbol: str,
                       prices: List[float],
                       current_position: Optional[str] = None) -> Optional[TradingSignal]:
        if len(prices) < max(self.rsi_period, self.long_ma):
            return None

        rsi = self.calculate_rsi(prices)
        ma_crossover = self.calculate_ma_crossover(prices)
        current_price = prices[-1]

        # Generate buy signal
        if rsi < 30 and ma_crossover and current_position != 'LONG':
            return TradingSignal(
                symbol=symbol,
                action='BUY',
                price=current_price,
                timestamp=pd.Timestamp.now().isoformat(),
                confidence=0.8,
                indicators={'rsi': rsi}
            )

        # Generate sell signal
        if rsi > 70 and not ma_crossover and current_position == 'LONG':
            return TradingSignal(
                symbol=symbol,
                action='SELL',
                price=current_price,
                timestamp=pd.Timestamp.now().isoformat(),
                confidence=0.8,
                indicators={'rsi': rsi}
            )

        return None 