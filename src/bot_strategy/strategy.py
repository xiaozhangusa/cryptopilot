from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import logging
from .timeframes import Timeframe
from datetime import datetime, timezone
import pytz  # Add this import
from .indicators import RSI, EMA, MACD

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
                 ema_period: int = 20,
                 macd_fast_period: int = 12,
                 macd_slow_period: int = 26,
                 macd_signal_period: int = 9):
        """Initialize strategy with timeframe-specific parameters
        
        Args:
            timeframe: Trading timeframe (default: 5 minutes)
            rsi_period: RSI period (default: 14)
            ema_period: EMA period (default: 20)
            macd_fast_period: MACD fast period (default: 12)
            macd_slow_period: MACD slow period (default: 26)
            macd_signal_period: MACD signal period (default: 9)
        """
        self.timeframe = timeframe
        
        # Initialize indicators
        self.rsi = RSI(period=rsi_period, timeframe=timeframe)
        self.ema = EMA(period=ema_period)
        self.macd = MACD(fast_period=macd_fast_period, slow_period=macd_slow_period, signal_period=macd_signal_period)
        
        # Store RSI thresholds that can be adjusted for more frequent signals
        self.rsi_oversold = 30  # Default 30, more aggressive: 40
        self.rsi_overbought = 70  # Default 70, more aggressive: 60
        
        logger.info(f"Initialized {timeframe.value} strategy with: "
                   f"RSI({rsi_period}), EMA({ema_period}), "
                   f"MACD({macd_fast_period},{macd_slow_period},{macd_signal_period})")
    
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
        macd_signal = self.macd.get_signal(prices, timestamps)
        
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
                   f"MACD: {macd_signal['value']:.4f}, Signal: {macd_signal['signal_line']:.4f}, "
                   f"Current price: ${current_price:.2f}")
        
        # Combine indicators to generate a trading signal
        signal, confidence = self._combine_signals(rsi_signal, ema_signal, macd_signal)
        
        if signal != 'NEUTRAL':
            logger.info(f"{self.timeframe.value} Combined signal: {signal} with {confidence:.2f} confidence")
            return TradingSignal(
                symbol=symbol,
                action=signal,
                price=current_price,
                timeframe=self.timeframe,
                confidence=confidence,
                indicators={'rsi': rsi_signal, 'ema': ema_signal, 'macd': macd_signal}
            )
        else:
            logger.info(f"{self.timeframe.value} No actionable signal (confidence: {confidence:.2f})")
        
        return None
    
    def _combine_signals(self, rsi_signal: Dict[str, Any], ema_signal: Dict[str, Any], 
                         macd_signal: Dict[str, Any]) -> Tuple[str, float]:
        """
        Combine RSI, EMA and MACD signals to generate a trading decision.
        More aggressive strategy to generate more frequent trading signals.
        
        Args:
            rsi_signal: Signal dictionary from RSI indicator
            ema_signal: Signal dictionary from EMA indicator
            macd_signal: Signal dictionary from MACD indicator
            
        Returns:
            Tuple of (signal, confidence)
            - signal: 'BUY', 'SELL', or 'NEUTRAL'
            - confidence: A value between 0.0 and 1.0 indicating confidence level
        """
        rsi_action = rsi_signal['signal']
        ema_action = ema_signal['signal']
        macd_action = macd_signal['signal']
        rsi_value = rsi_signal['value']
        
        # Normalize indicator strengths to 0-1 range
        rsi_strength = rsi_signal['strength'] / 100
        ema_strength = ema_signal['strength'] / 100
        macd_strength = macd_signal['strength'] / 100
        
        # Calculate base confidence levels with new weights
        # RSI: 50%, EMA: 20%, MACD: 30%
        rsi_confidence = rsi_strength * 0.5
        ema_confidence = ema_strength * 0.2
        macd_confidence = macd_strength * 0.3
        
        # Calculate EMA slope from distance percentage
        ema_distance = ema_signal['distance_pct']
        
        # Default to NEUTRAL
        combined_signal = 'NEUTRAL'
        confidence = 0.0
        
        # Extract MACD values for additional analysis
        macd_value = macd_signal['value']
        signal_line = macd_signal['signal_line']
        histogram = macd_signal['histogram']
        histogram_change = macd_signal['histogram_change']
        
        # Strategy 1: Strong agreement between indicators (all three)
        if rsi_action == ema_action == macd_action and rsi_action != 'NEUTRAL':
            combined_signal = rsi_action
            # All indicators agree, so we have high confidence
            confidence = (rsi_confidence + ema_confidence + macd_confidence) * 1.2  # Bonus for agreement
            logger.info(f"Strong agreement: RSI, EMA, and MACD all suggest {combined_signal}")
        
        # Strategy 2: MACD crossover (highest priority)
        elif macd_action != 'NEUTRAL':
            combined_signal = macd_action
            # MACD crossovers are strong signals
            confidence = macd_confidence * 1.5
            
            # Add confirmation bonus if RSI or EMA agree
            if rsi_action == macd_action:
                confidence += rsi_confidence * 0.3
                logger.info(f"MACD {macd_action} crossover confirmed by RSI")
            
            if ema_action == macd_action:
                confidence += ema_confidence * 0.2
                logger.info(f"MACD {macd_action} crossover confirmed by EMA trend")
            
            logger.info(f"MACD {macd_action} signal with histogram: {histogram:.4f}, change: {histogram_change:.4f}")
        
        # Strategy 3: RSI extreme values (medium priority)
        elif rsi_action != 'NEUTRAL':
            combined_signal = rsi_action
            
            # Special case: Deep oversold (below 25) - high confidence even with minimal confirmation
            if rsi_action == 'BUY' and rsi_value < 25:
                confidence = rsi_confidence * 1.5
                logger.info(f"Deep RSI oversold ({rsi_value:.2f}) indicates strong reversal potential")
            
            # Special case: Deep overbought (above 75) - high confidence even with minimal confirmation
            elif rsi_action == 'SELL' and rsi_value > 75:
                confidence = rsi_confidence * 1.5
                logger.info(f"Deep RSI overbought ({rsi_value:.2f}) indicates strong reversal potential")
            
            # Regular RSI signal
            else:
                confidence = rsi_confidence
                
                # Add confirmation bonus if MACD histogram agrees with direction
                if (rsi_action == 'BUY' and histogram_change > 0) or (rsi_action == 'SELL' and histogram_change < 0):
                    confidence += 0.1
                    logger.info(f"RSI {rsi_action} confirmed by MACD histogram direction")
            
            # Add EMA confirmation bonus
            if (rsi_action == 'BUY' and ema_distance > -3) or (rsi_action == 'SELL' and ema_distance < 3):
                confidence += 0.1
                logger.info(f"RSI {rsi_action} has favorable EMA position: {ema_distance:.2f}%")
        
        # Strategy 4: EMA significant crosses (lowest priority)
        elif ema_action != 'NEUTRAL' and abs(ema_distance) < 0.5:
            combined_signal = ema_action
            confidence = ema_confidence
            
            # Add confirmation from MACD direction
            if (ema_action == 'BUY' and histogram > 0) or (ema_action == 'SELL' and histogram < 0):
                confidence += 0.15
                logger.info(f"EMA {ema_action} confirmed by MACD histogram sign")
            
            logger.info(f"EMA {ema_action} signal with price-to-EMA distance: {ema_distance:.2f}%")
        
        # Strategy 5: MACD histogram turning points (new strategy for more signals)
        elif abs(histogram_change) > abs(histogram) * 0.2:  # 20% change in histogram
            if histogram_change > 0 and histogram < 0:
                # Negative histogram starting to turn up - potential buy
                combined_signal = 'BUY'
                confidence = 0.3 + min(0.3, abs(histogram_change) * 5)
                logger.info(f"MACD histogram turning up from negative: potential reversal BUY signal")
            elif histogram_change < 0 and histogram > 0:
                # Positive histogram starting to turn down - potential sell
                combined_signal = 'SELL'
                confidence = 0.3 + min(0.3, abs(histogram_change) * 5)
                logger.info(f"MACD histogram turning down from positive: potential reversal SELL signal")
        
        # Cap confidence at 1.0
        confidence = min(1.0, confidence)
        
        # Require minimum confidence threshold - LOWERED for more frequent signals
        # Lower threshold for both BUY and SELL signals to 0.25 (was 0.3/0.35)
        min_confidence = 0.25
        if confidence < min_confidence:
            logger.info(f"Signal {combined_signal} has low confidence ({confidence:.2f}), changing to NEUTRAL")
            return 'NEUTRAL', confidence
            
        return combined_signal, confidence

    def print_candle_analysis(self, prices: List[float], timestamps: List[int], symbol: Optional[str] = None, current_price: Optional[float] = None) -> None:
        """Print detailed candle analysis and indicator calculations
        
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
        
        # Calculate indicators
        rsi_value = self.rsi.calculate(prices, timestamps)
        ema_value = self.ema.calculate(prices, timestamps)
        macd_value, signal_line, histogram = self.macd.calculate(prices, timestamps)
        
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
        print(f"\n📈 RSI Calculation:")
        print(f"Using {self.rsi.period} periods from {oldest_time.strftime('%H:%M')} to {newest_time.strftime('%H:%M')} {tz_abbr}")
        
        # Display actual RSI calculation similar to the screenshot
        up_moves = [g for g in gains if g > 0][:self.rsi.period]
        down_moves = [l for l in losses if l > 0][:self.rsi.period]
        
        # Format the gains and losses display
        gains_str = " + ".join([f"${g:.2f}" for g in up_moves[:5]])
        if len(up_moves) > 5:
            gains_str += f" + ... ({len(up_moves) - 5} more)"
            
        losses_str = " + ".join([f"${l:.2f}" for l in down_moves[:5]])
        if len(down_moves) > 5:
            losses_str += f" + ... ({len(down_moves) - 5} more)"
            
        # Calculate average gain and loss for display
        avg_gain = sum(up_moves) / self.rsi.period if up_moves else 0
        avg_loss = sum(down_moves) / self.rsi.period if down_moves else 0
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        
        print(f"Gains in period ({len(up_moves)}/{self.rsi.period} candles): {gains_str}")
        print(f"Losses in period ({len(down_moves)}/{self.rsi.period} candles): {losses_str}")
        print(f"Average Gain: ${avg_gain:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Relative Strength (RS) = Avg Gain / Avg Loss = {rs:.2f}")
        print(f"RSI = 100 - (100 / (1 + RS)) = {rsi_value:.2f}")
        print(f"RSI Thresholds: Oversold < {self.rsi_oversold} | Overbought > {self.rsi_overbought}")
        
        # Show EMA details
        print(f"\n📉 EMA Calculation:")
        print(f"Period: {self.ema.period}, Smoothing Factor: {2.0 / (self.ema.period + 1):.4f}")
        print(f"Final EMA({self.ema.period}) = ${ema_value:.2f}")
        print(f"Current Price to EMA: {(current_price / ema_value - 1) * 100:.2f}%")
        
        # Show MACD details
        print(f"\n📊 MACD Calculation:")
        print(f"Fast EMA({self.macd.fast_period}), Slow EMA({self.macd.slow_period}), Signal({self.macd.signal_period})")
        
        # Calculate MACD series for visualization
        macd_series = self.macd.calculate_series(chronological_prices, chronological_timestamps)
        macd_line = macd_series['macd']
        signal_line_series = macd_series['signal']
        histogram_series = macd_series['histogram']
        
        # Display recent MACD values (last 6 periods)
        recent_periods = 6
        start_idx = max(0, len(macd_line) - recent_periods)
        
        print("\nRecent MACD Values:")
        print("Date/Time".ljust(12) + "MACD".ljust(12) + "Signal".ljust(12) + "Histogram".ljust(12) + "Trend")
        print("-" * 60)
        
        for i in range(start_idx, len(macd_line)):
            if i < len(chronological_timestamps):
                dt = datetime.fromtimestamp(chronological_timestamps[i], tz=utc_tz).astimezone(est_tz)
                time_str = dt.strftime("%H:%M")
                
                # Get values, handling NaN
                macd_val = macd_line[i] if not np.isnan(macd_line[i]) else 0
                signal_val = signal_line_series[i] if not np.isnan(signal_line_series[i]) else 0
                hist_val = histogram_series[i] if not np.isnan(histogram_series[i]) else 0
                
                # Determine trend indicators
                trend = ""
                if i > 0 and i < len(histogram_series):
                    prev_hist = histogram_series[i-1] if not np.isnan(histogram_series[i-1]) else 0
                    hist_change = hist_val - prev_hist
                    
                    if macd_val > signal_val and hist_val > 0:
                        trend = "🟢 Bullish"
                    elif macd_val < signal_val and hist_val < 0:
                        trend = "🔴 Bearish"
                    elif hist_val > 0 and hist_change > 0:
                        trend = "↗️ Gaining"
                    elif hist_val < 0 and hist_change < 0:
                        trend = "↘️ Weakening"
                    elif hist_val > 0 and hist_change < 0:
                        trend = "↗️➡️ Slowing"
                    elif hist_val < 0 and hist_change > 0:
                        trend = "↘️➡️ Improving"
                
                print(f"{time_str}".ljust(12) + 
                      f"{macd_val:.4f}".ljust(12) + 
                      f"{signal_val:.4f}".ljust(12) + 
                      f"{hist_val:.4f}".ljust(12) + 
                      f"{trend}")
        
        # Print a simple MACD histogram visualization
        print("\nMACD Histogram:")
        print("+" + "-" * 30 + "0" + "-" * 30 + "-")
        
        # Determine scale for histogram
        max_hist = max([abs(h) for h in histogram_series if not np.isnan(h)]) if histogram_series else 1
        scale_factor = 30 / max_hist if max_hist > 0 else 1
        
        for i in range(start_idx, len(histogram_series)):
            if i < len(chronological_timestamps):
                dt = datetime.fromtimestamp(chronological_timestamps[i], tz=utc_tz).astimezone(est_tz)
                time_str = dt.strftime("%H:%M")
                
                # Get histogram value, handling NaN
                hist_val = histogram_series[i] if not np.isnan(histogram_series[i]) else 0
                
                # Calculate bar length and character based on sign
                bar_len = abs(int(hist_val * scale_factor))
                bar_char = "█" if hist_val > 0 else "▓"
                
                # Create a minimal dot for zero values
                if hist_val == 0:
                    bar = " " * 30 + "•"
                
                # Create the bar with proper alignment
                if hist_val > 0:
                    bar = " " * 30 + bar_char * bar_len
                else:
                    bar = " " * (30 - bar_len) + bar_char * bar_len + " " * 30
                
                # Print with time and value
                print(f"{time_str} ({hist_val:.4f}): {bar}")
        
        # Current MACD values summary
        print(f"\nCurrent MACD: {macd_value:.4f}")
        print(f"Signal Line: {signal_line:.4f}")
        print(f"Histogram: {histogram:.4f}")
        
        if macd_value > signal_line:
            # Handle division by zero
            if signal_line != 0:
                # Special case for opposite signs
                if signal_line < 0 and macd_value > 0:
                    print(f"MACD above Signal by {macd_value - signal_line:.4f} (MACD positive, Signal negative)")
                else:
                    pct_diff = ((macd_value / signal_line) - 1) * 100
                    # Cap extreme values
                    if abs(pct_diff) > 1000:
                        print(f"MACD above Signal by {macd_value - signal_line:.4f} (>1000% difference)")
                    else:
                        print(f"MACD above Signal by {macd_value - signal_line:.4f} ({pct_diff:.2f}%)")
            else:
                print(f"MACD above Signal by {macd_value - signal_line:.4f}")
        else:
            # Handle division by zero
            if macd_value != 0:
                # Special case for opposite signs
                if macd_value < 0 and signal_line > 0:
                    print(f"MACD below Signal by {signal_line - macd_value:.4f} (MACD negative, Signal positive)")
                else:
                    pct_diff = ((signal_line / macd_value) - 1) * 100
                    # Cap extreme values
                    if abs(pct_diff) > 1000:
                        print(f"MACD below Signal by {signal_line - macd_value:.4f} (>1000% difference)")
                    else:
                        print(f"MACD below Signal by {signal_line - macd_value:.4f} ({pct_diff:.2f}%)")
            else:
                print(f"MACD below Signal by {signal_line - macd_value:.4f}")
        
        # Show cross-indicator analysis
        if rsi_value < self.rsi_oversold and current_price > ema_value:
            print("\n🔔 SIGNAL: RSI oversold while price above EMA - Potential bullish setup")
        elif rsi_value > self.rsi_overbought and current_price < ema_value:
            print("\n🔔 SIGNAL: RSI overbought while price below EMA - Potential bearish setup")
        elif rsi_value < self.rsi_oversold and histogram > 0:
            print("\n🔔 SIGNAL: RSI oversold with positive MACD histogram - Bullish divergence")
        elif rsi_value > self.rsi_overbought and histogram < 0:
            print("\n🔔 SIGNAL: RSI overbought with negative MACD histogram - Bearish divergence")
        elif macd_value > signal_line and macd_value - signal_line > abs(macd_value) * 0.05:
            print("\n🔔 SIGNAL: Strong MACD crossover above signal line - Bullish momentum")
        elif macd_value < signal_line and signal_line - macd_value > abs(macd_value) * 0.05:
            print("\n🔔 SIGNAL: Strong MACD crossover below signal line - Bearish momentum")
        
        # Add Signal Generation Logic explanation with ASCII art
        print("\n🧠 SIGNAL GENERATION LOGIC:")
        print("┌───────────────────────────────────────────────────────────────────────┐")
        print("│ PRIORITY 1: ALL INDICATORS AGREE                                      │")
        print("│   BUY when: RSI, EMA, MACD all bullish ➡️ High confidence (x1.2)      │")
        print("│   SELL when: RSI, EMA, MACD all bearish ➡️ High confidence (x1.2)     │")
        print("│                                                                       │")
        print("│ PRIORITY 2: MACD CROSSOVER                                           │")
        print("│   BUY when: MACD crosses above Signal ➡️ Base confidence (x1.5)       │")
        print("│   SELL when: MACD crosses below Signal ➡️ Base confidence (x1.5)      │")
        print("│   +RSI/EMA confirmation: Additional confidence bonus                  │")
        print("│                                                                       │")
        print("│ PRIORITY 3: EXTREME RSI                                              │")
        print("│   BUY when: RSI < 25 (deeply oversold) ➡️ Medium confidence (x1.5)    │")
        print("│   SELL when: RSI > 75 (deeply overbought) ➡️ Medium confidence (x1.5) │")
        print("│   Regular RSI signals (30/70): Lower confidence                       │")
        print("│                                                                       │")
        print("│ PRIORITY 4: EMA CROSSES                                              │")
        print("│   BUY when: Price crosses above EMA ➡️ Lower confidence               │")
        print("│   SELL when: Price crosses below EMA ➡️ Lower confidence              │")
        print("│                                                                       │")
        print("│ PRIORITY 5: MACD HISTOGRAM CHANGES                                   │")
        print("│   BUY when: Histogram turns up while negative ➡️ Lower confidence     │")
        print("│   SELL when: Histogram turns down while positive ➡️ Lower confidence  │")
        print("│                                                                       │")
        print("│ MINIMUM CONFIDENCE THRESHOLD: 0.25 (increased frequency)             │")
        print("└───────────────────────────────────────────────────────────────────────┘")
    
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
            if not in_swing and rsi_values[i] < self.rsi_oversold:
                in_swing = True
                swing_start_price = prices[i+self.rsi.period]
                swing_start_time = timestamps[i+self.rsi.period]
            elif in_swing and rsi_values[i] > self.rsi_overbought:
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