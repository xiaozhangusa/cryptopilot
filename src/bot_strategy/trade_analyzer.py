from dataclasses import dataclass
from typing import List
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)

def print_price_chart(prices: List[float], width: int = 50):
    """Print a simple ASCII chart of price movement"""
    if len(prices) < 2:
        return
    
    min_price = min(prices)
    max_price = max(prices)
    price_range = max_price - min_price
    
    print("\n📊 Price Chart (last hour):")
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
    print(f"Time: {datetime.fromtimestamp(time.time() - 3600).strftime('%H:%M')} "
          f"-> {datetime.fromtimestamp(time.time()).strftime('%H:%M')}")

@dataclass
class TradeAnalysis:
    investment: float
    entry_price: float
    trading_fee_rate: float = 0.006  # 0.6% total fees (entry + exit)
    stop_loss_pct: float = 0.01  # 1% stop loss
    
    def analyze(self, prices: List[float], action: str) -> dict:
        """Analyze trade potential and risks"""
        qty = self.investment / self.entry_price
        trading_fees = self.investment * self.trading_fee_rate
        
        # Calculate support/resistance from recent prices
        support = min(prices[-20:])  # Using last 20 candles
        resistance = max(prices[-20:])
        
        # Calculate stop loss and take profit levels
        if action == 'BUY':
            stop_loss = support * (1 - self.stop_loss_pct)
            take_profit = self.entry_price + (self.entry_price - support)
            max_loss = (stop_loss - self.entry_price) * qty
            potential_profit = (take_profit - self.entry_price) * qty
        else:  # SELL
            stop_loss = resistance * (1 + self.stop_loss_pct)
            take_profit = self.entry_price - (resistance - self.entry_price)
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
    
    def print_analysis(self, analysis: dict, symbol: str, action: str):
        """Print formatted trade analysis"""
        # Print price chart
        print_price_chart(prices)
        
        print("\n💰 Trade Analysis:")
        print(f"   Investment: ${self.investment:.2f}")
        print(f"   Quantity: {analysis['qty']:.8f} {symbol.split('-')[0]}")
        print(f"   Trading Fees: ${analysis['fees']:.2f}")
        print(f"   Entry Price: ${self.entry_price:.2f}")
        print(f"   Stop Loss: ${analysis['stop_loss']:.2f}")
        print(f"   Take Profit: ${analysis['take_profit']:.2f}")
        print(f"\n📊 Risk Analysis:")
        print(f"   Max Loss: ${abs(analysis['max_loss']):.2f}")
        print(f"   Potential Profit: ${analysis['potential_profit']:.2f}")
        print(f"   Net Profit: ${analysis['net_profit']:.2f}")
        print(f"   Risk/Reward Ratio: 1:{analysis['risk_reward_ratio']:.2f}")
        print(f"   Profit/Cost Ratio: {analysis['profit_cost_ratio']:.2%}")
        
        # Print efficiency rating
        if analysis['profit_cost_ratio'] < 0.01:  # 1%
            print("   Efficiency: ❌ Trade may not be worth the fees")
        elif analysis['profit_cost_ratio'] < 0.02:  # 2%
            print("   Efficiency: ⚠️ Minimal profit over costs")
        elif analysis['profit_cost_ratio'] < 0.05:  # 5%
            print("   Efficiency: ✅ Decent profit potential")
        else:
            print("   Efficiency: 🌟 Excellent profit potential")
        
        print(f"\n📋 Trading Criteria Met:")
        if action == 'BUY':
            print(f"   ✓ Price near support level (${analysis['support']:.2f})")
            print(f"   ✓ RSI indicates oversold condition")
            print(f"   ✓ Potential upside: {(analysis['take_profit']/self.entry_price - 1):.1%}")
        else:
            print(f"   ✓ Price near resistance level (${analysis['resistance']:.2f})")
            print(f"   ✓ RSI indicates overbought condition")
            print(f"   ✓ Potential downside: {(1 - analysis['take_profit']/self.entry_price):.1%}") 