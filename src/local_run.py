import os
import json
import logging
import time
from bot_strategy.strategy import SwingStrategy
from coinbase_api.client import CoinbaseAdvancedClient, OrderRequest
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_local_secrets():
    """Load secrets from local file for development"""
    secrets_file = os.getenv('SECRETS_FILE', 'secrets.json')
    try:
        with open(secrets_file) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Secrets file {secrets_file} not found. Using dummy credentials for simulation.")
        return {
            "api_key": "dummy_key",
            "api_secret": "dummy_secret",
            "passphrase": "dummy_passphrase"
        }

def main():
    try:
        trading_mode = os.getenv('TRADING_MODE', 'simulation')
        logger.info(f"Starting trading bot in {trading_mode} mode")
        
        secrets = load_local_secrets()
        coinbase_client = CoinbaseAdvancedClient(
            api_key=secrets['api_key'],
            api_secret=secrets['api_secret']
        )
        
        # Initialize strategy
        strategy = SwingStrategy()
        
        # Trading loop
        while True:
            try:
                # Get market data
                symbol = 'BTC-USD'
                print("\n" + "="*50)
                print(f"📊 Fetching market data for {symbol}...")
                
                # Get candles for the last hour
                end = str(int(time.time()))
                start = str(int(time.time() - 3600))
                response = coinbase_client.rest_client.get_candles(
                    product_id=symbol,
                    start=start,
                    end=end,
                    granularity="FIVE_MINUTE",
                    limit=300
                )
                prices = [float(candle.close) for candle in response.candles]
                print(f"📈 Latest price: ${prices[-1]:,.2f}")
                print(f"📉 Price range: ${min(prices):,.2f} - ${max(prices):,.2f}")
                
                # Generate trading signal
                print("\n🤖 Analyzing market conditions...")
                signal = strategy.generate_signal(symbol, prices)
                
                if signal:
                    print(f"\n🎯 Signal generated:")
                    print(f"   Symbol: {signal.symbol}")
                    print(f"   Action: {signal.action}")
                    print(f"   Price: ${signal.price:,.2f}")
                    
                    if trading_mode == 'simulation':
                        print(f"\n🔸 SIMULATION MODE:")
                        print(f"   Would {signal.action} {symbol} at ${signal.price:,.2f}")
                    else:
                        print(f"\n🔶 LIVE MODE: Executing trade...")
                        order = OrderRequest(
                            product_id=signal.symbol,
                            side=signal.action.lower(),
                            order_type='MARKET',
                            quote_size='10'
                        )
                        response = coinbase_client.create_market_order(order)
                        logger.info(f"Order placed: {response}")
                else:
                    print("\n😴 No trading signals generated")
                    
                # Wait for next iteration
                print(f"\n⏳ Waiting {300/60:.1f} minutes for next analysis...")
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                print(f"\n❌ Error: {str(e)}")
                time.sleep(60)  # Wait before retrying
                
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 