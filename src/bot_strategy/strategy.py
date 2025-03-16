from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import logging
from .timeframes import Timeframe
from datetime import datetime, timezone
import pytz  # Add this import
from .indicators import RSI, EMA

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
    confidence: float = 0.0  # 0.0 to 1.0
    indicators: Dict[str, Any] = None

class SwingStrategy:
    def __init__(self, 
                 timeframe: Timeframe = Timeframe.FIVE_MINUTE,
                 rsi_period: int = 14,
                 ema_period: int = 20):
        """Initialize strategy with timeframe-specific parameters
        
        Args:
            timeframe: Trading timeframe (default: 5 minutes)
            rsi_period: RSI period (default: 14)
            ema_period: EMA period (default: 20)
        """
        self.timeframe = timeframe
        
        # Initialize indicators
        self.rsi = RSI(period=rsi_period, timeframe=timeframe)
        self.ema = EMA(period=ema_period)
        
        logger.info(f"Initialized {timeframe.value} strategy with: "
                   f"RSI({rsi_period}), EMA({ema_period})")
    
    def set_verbose_calculations(self, verbose: bool = True):
        """
        Enable or disable detailed calculation logs for indicators.
        
        Args:
            verbose: True to show step-by-step calculations, False to hide
        """
        from .indicators.indicator_base import IndicatorBase
        IndicatorBase.set_verbose_mode(verbose)
        logger.info(f"Detailed indicator calculations: {'ENABLED' if verbose else 'DISABLED'}")
        return self
    
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
    
    def generate_signal(self, symbol: str, prices: List[float], timestamps: List[int], current_price: Optional[float] = None) -> Optional[TradingSignal]:
        """Generate trading signal by combining momentum and trend indicators
        
        Args:
            symbol: Trading pair symbol
            prices: List of historical prices
            timestamps: List of corresponding timestamps
            current_price: Optional real-time price from exchange
        """
        if len(prices) < self.timeframe.lookback_periods:
            logger.warning(f"Insufficient data for {self.timeframe.value} analysis")
            return None
            
        # Print candle analysis before generating signal
        self.print_candle_analysis(prices, timestamps, symbol, current_price)
        
        # Log price data direction for debugging
        is_reversed = len(prices) > 1 and prices[0] > prices[-1]
        price_order = "newest first (reversed)" if is_reversed else "oldest first (chronological)"
        logger.info(f"Prices appear to be in {price_order} order. First price: {prices[0]}, Last price: {prices[-1]}")
        
        # Get signals from each indicator
        # Set show_calculations to False for the second call to prevent duplicating the RSI calculation display
        rsi_signal = self.rsi.get_signal(prices, timestamps, show_calculations=False)
        ema_signal = self.ema.get_signal(prices, timestamps)
        
        # Get the current price (most recent)
        # If real-time price is provided, use it
        if current_price is not None:
            logger.info(f"Using real-time market price: ${current_price:.2f}")
        else:
            # Extract from candle data
            if timestamps and len(timestamps) > 0:
                most_recent_idx = timestamps.index(max(timestamps))
                current_price = prices[most_recent_idx]
            else:
                current_price = prices[0] if is_reversed else prices[-1]
            logger.info(f"Using candle price: ${current_price:.2f}")
        
        # Log indicator values
        logger.info(f"{self.timeframe.value} RSI: {rsi_signal['value']:.2f}, "
                   f"EMA({self.ema.period}): {ema_signal['value']:.2f}, "
                   f"Current price: ${current_price:.2f}")
        
        # Combine indicators to generate a trading signal
        signal, confidence = self._combine_signals(rsi_signal, ema_signal)
        
        if signal != 'NEUTRAL':
            logger.info(f"{self.timeframe.value} Combined signal: {signal} with {confidence:.2f} confidence")
            return TradingSignal(
                symbol=symbol,
                action=signal,
                price=current_price,
                timeframe=self.timeframe,
                confidence=confidence,
                indicators={'rsi': rsi_signal, 'ema': ema_signal}
            )
        else:
            logger.info(f"{self.timeframe.value} No actionable signal (confidence: {confidence:.2f})")
        
        return None
    
    def _combine_signals(self, rsi_signal: Dict[str, Any], ema_signal: Dict[str, Any]) -> Tuple[str, float]:
        """
        Combine RSI and EMA signals to generate a trading decision.
        
        Args:
            rsi_signal: Signal dictionary from RSI indicator
            ema_signal: Signal dictionary from EMA indicator
            
        Returns:
            Tuple of (signal, confidence)
            - signal: 'BUY', 'SELL', or 'NEUTRAL'
            - confidence: A value between 0.0 and 1.0 indicating confidence level
        """
        rsi_action = rsi_signal['signal']
        ema_action = ema_signal['signal']
        rsi_value = rsi_signal['value']
        
        # Normalize indicator strengths to 0-1 range
        rsi_strength = rsi_signal['strength'] / 100
        ema_strength = ema_signal['strength'] / 100
        
        # Calculate base confidence levels
        # Give RSI 70% weight (was 60%) and EMA 30% weight (was 40%)
        rsi_confidence = rsi_strength * 0.7  # RSI contributes 70%
        ema_confidence = ema_strength * 0.3  # EMA contributes 30%
        
        # Calculate EMA slope from distance percentage
        ema_distance = ema_signal['distance_pct']
        
        # Default to NEUTRAL
        combined_signal = 'NEUTRAL'
        confidence = 0.0
        
        # Strategy 1: Strong agreement between indicators
        if rsi_action == ema_action and rsi_action != 'NEUTRAL':
            combined_signal = rsi_action
            # Both indicators agree, so we have high confidence
            confidence = (rsi_confidence + ema_confidence) * 1.2  # Bonus for agreement
            logger.info(f"Strong agreement: RSI and EMA both suggest {combined_signal}")
            
        # Strategy 2: RSI extremes with confirming trend
        # More sensitive to early trend reversal - allow up to -2.5% (was -2%)
        elif rsi_action == 'BUY' and ema_signal['distance_pct'] > -2.5:
            # RSI suggests buy and price is close to or approaching EMA (early trend reversal)
            combined_signal = 'BUY'
            
            # Special case: Deeply oversold RSI (below 20) - higher confidence even with price below EMA
            if rsi_value < 20:
                # Deep oversold is a stronger signal, increase confidence
                confidence = rsi_confidence * (1 + 0.5 * (1 + ema_confidence))
                logger.info(f"Deep RSI oversold ({rsi_value:.2f}) indicates potential reversal (price to EMA: {ema_signal['distance_pct']:.2f}%)")
            else:
                # Regular oversold case - adjust formula to give more weight to RSI
                confidence = rsi_confidence * (1 + 0.4 * ema_confidence)
                logger.info(f"RSI oversold in neutral/bullish trend (price to EMA: {ema_signal['distance_pct']:.2f}%)")
            
            # Add a boost for prices getting closer to EMA (potential early reversal)
            if ema_signal['distance_pct'] > -1:
                confidence *= 1.15  # 15% confidence boost when price is very close to EMA
                logger.info(f"Price approaching EMA - potential trend reversal signal")
            
        elif rsi_action == 'SELL' and ema_signal['distance_pct'] < 2:
            # RSI suggests sell and price is close to or below EMA (confirming downtrend)
            combined_signal = 'SELL'
            confidence = rsi_confidence * (1 + 0.3 * ema_confidence)
            logger.info(f"RSI overbought in neutral/bearish trend (price to EMA: {ema_signal['distance_pct']:.2f}%)")
            
        # Strategy 3: EMA crossover with confirming RSI
        elif ema_action != 'NEUTRAL' and rsi_signal['value'] > 40 and rsi_signal['value'] < 60:
            # EMA crossover with RSI in neutral zone (avoiding extremes)
            combined_signal = ema_action
            confidence = ema_confidence * (1 + 0.2 * (1 - abs(rsi_signal['value'] - 50) / 10))
            logger.info(f"EMA {ema_action} crossover with neutral RSI ({rsi_signal['value']:.2f})")
        
        # Cap confidence at 1.0
        confidence = min(1.0, confidence)
        
        # Require minimum confidence threshold
        # Lower threshold for BUY signals to 0.30 (was 0.35)
        min_confidence = 0.30 if combined_signal == 'BUY' else 0.35
        if confidence < min_confidence:
            logger.info(f"Signal {combined_signal} has low confidence ({confidence:.2f}), changing to NEUTRAL")
            return 'NEUTRAL', confidence
            
        return combined_signal, confidence

    def print_candle_analysis(self, prices: List[float], timestamps: List[int], symbol: Optional[str] = None, current_price: Optional[float] = None) -> None:
        """Print detailed candle analysis and RSI calculation
        
        Args:
            prices: List of price values
            timestamps: List of corresponding timestamps
            symbol: Optional trading pair symbol (for display purposes)
            current_price: Optional real-time price from external source
        """
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
        
        # Add indicator analysis section
        print("\n📊 Indicator Analysis:")
        
        # Calculate both indicators
        rsi_value = self.rsi.calculate(prices, timestamps)
        ema_value = self.ema.calculate(prices, timestamps)
        
        # Create chronological prices (oldest first) for additional analysis
        chronological_prices = list(reversed(display_prices.copy()))
        chronological_timestamps = list(reversed(display_timestamps.copy()))
        
        # Get prices and EMAs for the last few periods to show trend
        ema_series = self.ema.calculate_multiple(chronological_prices, chronological_timestamps)
        
        # Display only the last 5 EMA values and corresponding prices
        print("\nPrice vs EMA Trend:")
        for i in range(min(5, len(ema_series))):
            if i >= len(chronological_prices) - 5:
                idx = len(chronological_prices) - (len(chronological_prices) - i)
                if idx < len(chronological_prices) and not np.isnan(ema_series[idx]):
                    ts_idx = idx if idx < len(chronological_timestamps) else -1
                    dt = datetime.fromtimestamp(chronological_timestamps[ts_idx], tz=utc_tz).astimezone(est_tz)
                    time_str = dt.strftime("%H:%M")
                    print(f"{time_str}: Price ${chronological_prices[idx]:.2f} | EMA ${ema_series[idx]:.2f} | "
                          f"Diff: {(chronological_prices[idx] - ema_series[idx]):.2f} "
                          f"({(chronological_prices[idx] / ema_series[idx] - 1) * 100:.2f}%)")
        
        # If current_price is provided, use it (from real-time API)
        # Otherwise, extract from candle data
        if current_price is not None:
            print(f"\n🔄 Real-time market price: ${current_price:.2f}")
        else:
            # Extract current price from candle data
            if timestamps and len(timestamps) > 0:
                most_recent_idx = timestamps.index(max(timestamps))
                current_price = prices[most_recent_idx]
            else:
                is_reversed = len(prices) > 1 and prices[0] > prices[-1]
                current_price = prices[0] if is_reversed else prices[-1]
            print(f"\n📊 Using candle data price: ${current_price:.2f}")
        
        # Show RSI details
        print(f"\nRSI({self.rsi.period}): {rsi_value:.2f}")
        print(f"RSI Thresholds: Oversold < {self.rsi.oversold} | Overbought > {self.rsi.overbought}")
        
        # Show EMA details
        print(f"\nEMA({self.ema.period}): {ema_value:.2f}")
        print(f"Current Price to EMA: {(current_price / ema_value - 1) * 100:.2f}%")
        
        # Show cross-indicator analysis
        if rsi_value < self.rsi.oversold and current_price > ema_value:
            print("\n🔔 SIGNAL: RSI oversold while price above EMA - Potential bullish setup")
        elif rsi_value > self.rsi.overbought and current_price < ema_value:
            print("\n🔔 SIGNAL: RSI overbought while price below EMA - Potential bearish setup")
        elif rsi_value < self.rsi.oversold and current_price < ema_value:
            print("\n⚠️ MIXED: RSI oversold but price below EMA - Conflicting signals")
        elif rsi_value > self.rsi.overbought and current_price > ema_value:
            print("\n⚠️ MIXED: RSI overbought but price above EMA - Conflicting signals")
    
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
        for i in range(self.rsi.period, len(prices)):
            window_prices = prices[i-self.rsi.period:i+1]
            window_timestamps = timestamps[i-self.rsi.period:i+1] if timestamps else None
            rsi = self.rsi.calculate(window_prices, window_timestamps)
            rsi_values.append(rsi)
        
        # Find completed oversold to overbought swings
        swings = []
        in_swing = False
        swing_start_price = 0
        swing_start_time = 0
        
        for i in range(len(rsi_values)-1):
            if not in_swing and rsi_values[i] < self.rsi.oversold:
                in_swing = True
                swing_start_price = prices[i+self.rsi.period]
                swing_start_time = timestamps[i+self.rsi.period]
            elif in_swing and rsi_values[i] > self.rsi.overbought:
                swing_end_price = prices[i+self.rsi.period]
                swing_end_time = timestamps[i+self.rsi.period]
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