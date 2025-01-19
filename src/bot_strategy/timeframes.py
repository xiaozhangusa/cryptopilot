from enum import Enum

class Timeframe(Enum):
    FIVE_MIN = "FIVE_MINUTE"
    ONE_HOUR = "ONE_HOUR"
    SIX_HOUR = "SIX_HOUR"
    TWELVE_HOUR = "TWELVE_HOUR"
    ONE_DAY = "ONE_DAY"

    @property
    def minutes(self) -> int:
        return {
            "FIVE_MINUTE": 5,
            "ONE_HOUR": 60,
            "SIX_HOUR": 360,
            "TWELVE_HOUR": 720,
            "ONE_DAY": 1440
        }[self.value]
    
    @property
    def lookback_periods(self) -> int:
        """Number of candles to look back for support/resistance"""
        return {
            "FIVE_MINUTE": 20,    # 100 minutes
            "ONE_HOUR": 24,       # 24 hours
            "SIX_HOUR": 28,       # 7 days
            "TWELVE_HOUR": 14,    # 7 days
            "ONE_DAY": 30         # 30 days
        }[self.value] 