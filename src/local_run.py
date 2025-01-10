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
                # Get candles for the last hour
                end = time.strftime('%Y-%m-%dT%H:%M:%SZ')  # Current time in ISO 8601
                start = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() - 3600))  # 1 hour ago
                response = coinbase_client.rest_client.get_candles(
                    product_id=symbol,
                    start=start,
                    end=end,
                    granularity="FIVE_MINUTE",  # Using FIVE_MINUTE as it's more stable
                    limit=300  # Maximum number of candles
                )
                prices = [float(candle.close) for candle in response.candles]
                
                # Generate trading signal
                signal = strategy.generate_signal(symbol, prices)
                
                if signal:
                    logger.info(f"Generated signal: {signal}")
                    
                    if trading_mode == 'simulation':
                        logger.info(f"Simulation mode: Would execute {signal.action} "
                                  f"for {symbol} at {signal.price}")
                    else:
                        order = OrderRequest(
                            product_id=signal.symbol,
                            side=signal.action.lower(),
                            order_type='MARKET',
                            quote_size='10'  # Trade with $10 for testing
                        )
                        response = coinbase_client.create_market_order(order)
                        logger.info(f"Order placed: {response}")
                
                # Wait for next iteration
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(60)  # Wait before retrying
                
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 