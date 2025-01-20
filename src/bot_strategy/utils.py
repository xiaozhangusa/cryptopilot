from typing import List

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