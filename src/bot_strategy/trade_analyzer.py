from dataclasses import dataclass
from typing import List
import logging
from datetime import datetime
import time
from .timeframes import Timeframe

logger = logging.getLogger(__name__)

def print_price_chart(prices: List[float], width: int = 50):
    """Print a simple ASCII chart of price movement"""
    if len(prices) < 2:
        return
    
    min_price = min(prices)
    max_price = max(prices)
    price_range = max_price - min_price
    
    print("\n📊 Price Chart:")
    print(f"${max_price:,.2f} ┐")
    
    for i in range(5):  # 5 price levels
        price = max_price - (price_range * i / 4)
        normalized = [(p - min_price) / price_range * (width - 1) for p in prices]
        line = [" " for _ in range(width)]
        for j, n in enumerate(normalized):
            if abs(n - (4 - i) * (width - 1) / 4) < 1:
                line[int(n)] = "•"
        print(f"${price:,.2f} {''.join(line)}")
    
    print(f"${min_price:,.2f} ┘")

@dataclass
class TradeAnalysis:
    investment: float
    entry_price: float
    timeframe: Timeframe
    trading_fee_rate: float = 0.006  # 0.6% total fees (entry + exit)
    
    def __init__(self, 
                 investment: float,
                 entry_price: float,
                 timeframe: Timeframe = Timeframe.FIVE_MIN):
        self.investment = investment
        self.entry_price = entry_price
        self.timeframe = timeframe
        self.trading_fee_rate = 0.006  # 0.6% total fees
        
        # Adjust stop loss based on timeframe
        self.stop_loss_pct = {
            Timeframe.FIVE_MIN: 0.01,      # 1% for 5min
            Timeframe.ONE_HOUR: 0.02,      # 2% for 1h
            Timeframe.SIX_HOUR: 0.03,      # 3% for 6h
            Timeframe.TWELVE_HOUR: 0.04,   # 4% for 12h
            Timeframe.ONE_DAY: 0.05,       # 5% for 1d
        }[timeframe]
    
    def _calculate_metrics(self, prices: List[float], action: str) -> dict:
        """Calculate all trading metrics"""
        qty = self.investment / self.entry_price
        trading_fees = self.investment * self.trading_fee_rate
        
        # Calculate support/resistance from recent prices
        support = min(prices[-20:])    # Using last 20 candles
        resistance = max(prices[-20:]) # Using last 20 candles
        
        # RSI-based profit targets
        if action == 'BUY':
            stop_loss = support * (1 - self.stop_loss_pct)
            # Target the overbought zone (resistance area)
            take_profit = resistance  # Sell when price reaches recent high/resistance
            
            max_loss = (stop_loss - self.entry_price) * qty
            potential_profit = (take_profit - self.entry_price) * qty
        else:  # SELL
            stop_loss = resistance * (1 + self.stop_loss_pct)
            # Target the oversold zone (support area)
            take_profit = support  # Buy back when price reaches recent low/support
            
            max_loss = (self.entry_price - stop_loss) * qty
            potential_profit = (self.entry_price - take_profit) * qty
        
        net_profit = potential_profit - trading_fees
        profit_cost_ratio = net_profit / (self.investment + trading_fees)
        risk_reward_ratio = abs(potential_profit / max_loss) if max_loss != 0 else float('inf')
        
        return {
            'qty': qty,
            'fees': trading_fees,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'max_loss': max_loss,
            'potential_profit': potential_profit,
            'net_profit': net_profit,
            'profit_cost_ratio': profit_cost_ratio,
            'risk_reward_ratio': risk_reward_ratio,
            'support': support,
            'resistance': resistance
        }
    
    def analyze(self, prices: List[float], action: str, swing_stats: dict = None) -> dict:
        """Analyze trade potential and risks using historical swing data"""
        analysis = self._calculate_metrics(prices, action)
        
        if swing_stats:
            # Refine take profit based on historical swings
            typical_move = swing_stats['avg_swing']
            conservative_move = swing_stats['min_swing']
            aggressive_move = min(swing_stats['max_swing'], 0.1)  # Cap at 10%
            
            if action == 'BUY':
                analysis['take_profit'] = self.entry_price * (1 + typical_move)
                analysis['potential_profit'] = (analysis['take_profit'] - self.entry_price) * analysis['qty']
            else:
                analysis['take_profit'] = self.entry_price * (1 - typical_move)
                analysis['potential_profit'] = (self.entry_price - analysis['take_profit']) * analysis['qty']
                
            # Add historical context
            analysis['historical_success_rate'] = swing_stats['success_rate']
            analysis['avg_swing_duration'] = swing_stats['avg_duration']
            
            # Adjust viability check
            analysis['is_viable'] = (
                analysis['risk_reward_ratio'] >= 2.0 and
                analysis['profit_cost_ratio'] > 0.05 and
                analysis['net_profit'] > 0 and
                swing_stats['success_rate'] > 0.6  # At least 60% success rate
            )
        
        # Add risk rating
        if analysis['risk_reward_ratio'] >= 3.0 and analysis['profit_cost_ratio'] > 0.10:
            analysis['risk_rating'] = "🌟 Excellent"
        elif analysis['risk_reward_ratio'] >= 2.0 and analysis['profit_cost_ratio'] > 0.05:
            analysis['risk_rating'] = "✅ Good"
        elif analysis['risk_reward_ratio'] >= 1.5 and analysis['profit_cost_ratio'] > 0.02:
            analysis['risk_rating'] = "⚠️ Marginal"
        else:
            analysis['risk_rating'] = "❌ Poor"
        
        return analysis
    
    def print_analysis(self, analysis: dict, symbol: str, action: str, prices: List[float] = None):
        """Print formatted trade analysis with order details and profit calculations"""
        print("\n🎯 Signal generated:")
        print(f"   Symbol: {symbol}")
        print(f"   Action: {action}")
        print(f"   Price: ${self.entry_price:,.2f}")

        print("\n💰 Order Details:")
        print(f"   Investment: ${self.investment:.2f}")
        print(f"   Quantity: {analysis['qty']:.8f} {symbol.split('-')[0]}")
        print(f"   Quote Amount: ${self.investment:.2f} {symbol.split('-')[1]}")
        print(f"   Trading Fees: ${analysis['fees']:.2f}")

        print("\n📊 Risk Analysis:")
        print(f"   Entry Price: ${self.entry_price:.2f}")
        print(f"   Stop Loss: ${analysis['stop_loss']:.2f}")
        print(f"   Take Profit: ${analysis['take_profit']:.2f}")
        print(f"   Max Loss: ${abs(analysis['max_loss']):.2f}")
        print(f"   Potential Profit: ${analysis['potential_profit']:.2f}")
        if action == 'BUY':
            print(f"      Profit = Position × (Target - Entry) where Position = Investment/Entry")
            print(f"      = ${self.investment:.2f}/{self.entry_price:.2f} × (${analysis['take_profit']:.2f} - ${self.entry_price:.2f})")
        else:
            print(f"      Profit = Position × (Entry - Target) where Position = Investment/Entry")
            print(f"      = ${self.investment:.2f}/{self.entry_price:.2f} × (${self.entry_price:.2f} - ${analysis['take_profit']:.2f})")
        print(f"   Net Profit: ${analysis['net_profit']:.2f}  (after ${analysis['fees']:.2f} fees)")
        print(f"   Risk/Reward Ratio: 1:{analysis['risk_reward_ratio']:.2f}")
        print(f"   Profit/Cost Ratio: {analysis['profit_cost_ratio']:.2%}")
        
        if 'historical_success_rate' in analysis:
            print(f"   Historical Success Rate: {analysis['historical_success_rate']:.0%}")
            print(f"   Average Swing Duration: {int(analysis['avg_swing_duration']/60)} minutes")
        
        print(f"   Risk Rating: {analysis['risk_rating']}")

        # Print efficiency rating
        if analysis['profit_cost_ratio'] < 0.01:
            print("   Efficiency: ❌ Trade may not be worth the fees")
        elif analysis['profit_cost_ratio'] < 0.02:
            print("   Efficiency: ⚠️ Minimal profit over costs")
        elif analysis['profit_cost_ratio'] < 0.05:
            print("   Efficiency: ✅ Decent profit potential")
        else:
            print("   Efficiency: 🌟 Excellent profit potential")

        print("\n📋 Trading Criteria Met:")
        if not analysis['is_viable']:
            print("   ❌ Trade rejected due to:")
            if analysis['risk_reward_ratio'] < 2.0:
                print("   - Poor risk/reward ratio (needs 1:2 minimum)")
            if analysis['profit_cost_ratio'] <= 0.05:
                print("   - Insufficient profit over costs")
            if analysis['net_profit'] <= 0:
                print("   - Negative profit after fees")
            if 'historical_success_rate' in analysis and analysis['historical_success_rate'] <= 0.6:
                print("   - Historical success rate below threshold")
            return

        if action == 'BUY':
            print(f"   ✓ Price near support level (${analysis['support']:.2f})")
            print(f"   ✓ RSI indicates oversold condition")
            print(f"   ✓ Potential upside: {(analysis['take_profit']/self.entry_price - 1):.1%}")
            print(f"   ✓ Risk/Reward ratio acceptable: 1:{analysis['risk_reward_ratio']:.2f}")
            if 'historical_success_rate' in analysis:
                print(f"   ✓ Historical success rate: {analysis['historical_success_rate']:.0%}")
        else:
            print(f"   ✓ Price near resistance level (${analysis['resistance']:.2f})")
            print(f"   ✓ RSI indicates overbought condition")
            print(f"   ✓ Potential downside: {(1 - analysis['take_profit']/self.entry_price):.1%}")
            if 'historical_success_rate' in analysis:
                print(f"   ✓ Historical success rate: {analysis['historical_success_rate']:.0%}")

        # Print succinct profit calculation
        print("\n💵 Profit Summary:")
        print(f"   Entry: ${self.entry_price:.2f}")
        print(f"   Target: ${analysis['take_profit']:.2f}")
        print(f"   Stop: ${analysis['stop_loss']:.2f}")
        print(f"   Risk: ${abs(analysis['max_loss']):.2f}")
        print(f"   Reward: ${analysis['potential_profit']:.2f}")
        print(f"   Net After Fees: ${analysis['net_profit']:.2f}")
        print(f"   Expected Return: {analysis['profit_cost_ratio']:.1%}") 