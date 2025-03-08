import os
import json
import logging
import time
from bot_strategy.strategy import SwingStrategy
from bot_strategy.trade_analyzer import TradeAnalysis
from coinbase_api.client import CoinbaseAdvancedClient, OrderRequest
import sys
from bot_strategy.timeframes import Timeframe
from utils.chart import print_price_chart
from trading.order_manager import OrderManager

# Configure logging to output to stdout with proper formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout  # Ensure logs go to stdout
)
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
        timeframe = Timeframe(os.getenv('TIMEFRAME', 'FIVE_MINUTE'))
        logger.info(f"Starting trading bot in {trading_mode} mode")
        logger.info(f"Using {timeframe.value} timeframe")
        
        secrets = load_local_secrets()
        logger.info(f"Secrets loaded successfully")
        coinbase_client = CoinbaseAdvancedClient(
            api_key=secrets['api_key'],
            api_secret=secrets['api_secret']
        )
        
        # Initialize strategy
        strategy = SwingStrategy(timeframe=timeframe)
        logger.info("Strategy initialized successfully")
        
        # Initialize order manager
        order_manager = OrderManager(coinbase_client)

        # Trading loop
        while True:
            try:
                # Get market data
                symbol = 'BTC-USD'
                print("\n" + "="*50, flush=True)
                print(f"📊 Fetching market data for {symbol}...", flush=True)
                
                # Let client.py handle all the timestamp and API details
                candles = coinbase_client.get_product_candles(
                    product_id=symbol,
                    granularity=timeframe.value
                )

                # Process the candles
                if not candles:
                    logger.error("No candles received")
                    time.sleep(60)  # Wait before retrying
                    continue

                # Ensure we have enough candles for analysis
                min_candles_needed = max(timeframe.lookback_periods, 14)  # Use larger of lookback or RSI periods
                if len(candles) < min_candles_needed:
                    logger.error(f"Not enough candles received. Got {len(candles)}, need {min_candles_needed}")
                    time.sleep(60)  # Wait before retrying
                    continue

                # Process the candles
                try:
                    # Extract both prices and timestamps
                    prices = [float(candle.close) for candle in candles]
                    timestamps = [int(candle.start) for candle in candles]
                    
                    # Print price information
                    print(f"📈 Latest price: ${prices[-1]:,.2f}", flush=True)
                    print(f"\n💹 Price Range:")
                    print(f"High: ${max(prices):.2f}")
                    print(f"Low:  ${min(prices):.2f}")
                    print(f"Current: ${prices[-1]:.2f}")

                    # Print price chart
                    print_price_chart(prices)

                    # Generate trading signal
                    print("\n🤖 Analyzing market conditions...")
                    signal = strategy.generate_signal(symbol, prices, timestamps)
                    
                    # if signal:
                    if True:
                        # Analyze trade potential
                        # analyzer = TradeAnalysis(
                        #     investment=10.0,  # $10 test trade
                        #     entry_price=signal.price,
                        #     timeframe=timeframe  # Pass the timeframe
                        # )
                        # analysis = analyzer.analyze(prices, signal.action)
                        # analyzer.print_analysis(analysis, signal.symbol, signal.action, prices)
                        
                        if trading_mode == 'simulation':
                            print(f"\n🔸 SIMULATION MODE:")
                            print(f"\n🔶 Creating limit order for USDT-USDC...")
                            order = OrderRequest(
                                product_id='USDT-USDC',
                                side='BUY',
                                order_type='LIMIT',
                                base_size='2',
                                limit_price='0.9800',
                                time_in_force='GTC'
                            )
                            response = order_manager.place_order(order)
                            if response['success']:
                                logger.info(f"✅ Order placed successfully: {response}")
                            else:
                                logger.warning(f"❌ Order not placed: {response['error']}")
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
                    logger.error(f"Error processing candles: {str(e)}")
                    time.sleep(60)  # Wait before retrying
                    continue

            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                print(f"\n❌ Error: {str(e)}")
                time.sleep(60)  # Wait before retrying
                
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 