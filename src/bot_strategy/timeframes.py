from enum import Enum

class Timeframe(Enum):
    # Updated to match Coinbase Advanced API granularity values
    ONE_MINUTE = "ONE_MINUTE"
    FIVE_MINUTE = "FIVE_MINUTE"
    FIFTEEN_MINUTE = "FIFTEEN_MINUTE"
    THIRTY_MINUTE = "THIRTY_MINUTE"
    ONE_HOUR = "ONE_HOUR"
    TWO_HOUR = "TWO_HOUR"
    SIX_HOUR = "SIX_HOUR"
    ONE_DAY = "ONE_DAY"

    @property
    def minutes(self) -> int:
        return {
            "ONE_MINUTE": 1,
            "FIVE_MINUTE": 5,
            "FIFTEEN_MINUTE": 15,
            "THIRTY_MINUTE": 30,
            "ONE_HOUR": 60,
            "TWO_HOUR": 120,
            "SIX_HOUR": 360,
            "ONE_DAY": 1440
        }[self.value]
    
    @property
    def lookback_periods(self) -> int:
        """Number of candles to look back for support/resistance"""
        return {
            "ONE_MINUTE": 60,    # 60 minutes
            "FIVE_MINUTE": 20,   # 100 minutes
            "FIFTEEN_MINUTE": 16, # 240 minutes (4 hours)
            "THIRTY_MINUTE": 12, # 360 minutes (6 hours)
            "ONE_HOUR": 24,      # 24 hours
            "TWO_HOUR": 12,      # 24 hours
            "SIX_HOUR": 28,      # 7 days
            "ONE_DAY": 30        # 30 days
        }[self.value] 
        
