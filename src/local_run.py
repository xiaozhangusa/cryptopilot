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
from decimal import Decimal

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
        
        # Display all accounts and their metadata
        display_account_details(coinbase_client)
        
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
                            # print(f"\n🔸 SIMULATION MODE:")
                            # print(f"\n🔶 Creating limit order for USDT-USDC...")
                            # order = OrderRequest(
                            #     product_id='USDT-USDC',
                            #     side='BUY',
                            #     order_type='LIMIT',
                            #     base_size='2',
                            #     limit_price='0.9800',
                            #     time_in_force='GTC'
                            # )
                            # response = order_manager.place_order(order)
                            # if response['success']:
                            #     logger.info(f"✅ Order placed successfully: {response}")
                            # else:
                            #     logger.warning(f"❌ Order not placed: {response['error']}")
                            print(f"\n🔸 SIMULATION MODE:")
                            print(f"\n🔶 Creating selling limit order for BTC-USDT...")
                            order = OrderRequest(
                                product_id='BTC-USDT',
                                side='SELL',
                                order_type='LIMIT',
                                base_size='0.001',
                                limit_price='110000',
                                time_in_force='GTC'
                            )
                            response = order_manager.place_order(order)
                            if response['success']:
                                logger.info(f"✅ Limit sell order placed successfully: {response}")
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

def display_account_details(client):
    """Display detailed information about all accounts"""
    try:
        print("\n" + "="*80)
        print("📊 ACCOUNT DETAILS".center(80))
        print("="*80)
        
        # Get accounts through the main API method
        accounts = client.get_accounts()
        if not accounts:
            print("No accounts found!")
            return
            
        # Try to get alternate view of accounts through portfolio endpoint
        portfolio_accounts = client.get_portfolio_accounts()
        
        # Keep track of duplicate currencies
        currency_counts = {}
        for account in accounts:
            currency = account.currency
            currency_counts[currency] = currency_counts.get(currency, 0) + 1
        
        # Print header
        print("\n{:<6} {:<6} {:<12} {:<15} {:<15} {:<20} {:<15}".format(
            "Index", "Curr.", "Available", "Hold", "Total", "UUID", "Notes"))
        print("-"*100)
        
        # Group accounts by currency for better organization
        accounts_by_currency = {}
        for account in accounts:
            currency = account.currency
            if currency not in accounts_by_currency:
                accounts_by_currency[currency] = []
            accounts_by_currency[currency].append(account)
        
        # Create a lookup for portfolio balances
        portfolio_balances = {}
        for currency, balance in portfolio_accounts:
            portfolio_balances[currency] = balance
            
        # Track discrepancies between API accounts and portfolio
        discrepancies = []
        
        # Sort currencies and print accounts
        index = 1
        for currency in sorted(accounts_by_currency.keys()):
            # Sort accounts by available balance (highest first)
            accounts_by_currency[currency].sort(key=lambda acc: acc.available_balance, reverse=True)
            
            # Check for discrepancy with portfolio
            portfolio_balance = portfolio_balances.get(currency, Decimal('0'))
            api_total_balance = sum((acc.available_balance + acc.hold) for acc in accounts_by_currency[currency])
            
            # If portfolio shows higher balance than API accounts combined
            if portfolio_balance > api_total_balance and portfolio_balance > Decimal('0.001'):
                discrepancies.append((currency, api_total_balance, portfolio_balance))
            
            for i, account in enumerate(accounts_by_currency[currency]):
                total_balance = account.available_balance + account.hold
                
                # Highlight if this is a duplicate currency
                prefix = "* " if currency_counts[currency] > 1 else "  "
                
                # Add suffix to show which one has highest balance
                suffix = ""
                if i == 0 and currency_counts[currency] > 1:
                    suffix += "(highest)"
                
                # Add a note if portfolio shows different balance
                if i == 0 and currency in portfolio_balances and portfolio_balances[currency] > Decimal('0.001'):
                    port_balance = portfolio_balances[currency]
                    if abs(port_balance - total_balance) > Decimal('0.00001'):
                        suffix += " ⚠️Portfolio: " + str(port_balance)
                
                print("{:<6} {:<6} {:<12.8f} {:<15.8f} {:<15.8f} {:<20} {:<15}".format(
                    prefix + str(index),
                    account.currency,
                    account.available_balance,
                    account.hold,
                    total_balance,
                    account.uuid,
                    suffix
                ))
                index += 1
        
        # Print currencies from portfolio that don't appear in the API accounts
        for currency, balance in portfolio_accounts:
            if currency not in accounts_by_currency and balance > Decimal('0.001'):
                print("{:<6} {:<6} {:<12.8f} {:<15.8f} {:<15.8f} {:<20} {:<15}".format(
                    "⚠️" + str(index),
                    currency,
                    balance,
                    Decimal('0'),
                    balance,
                    "UNKNOWN (PORTFOLIO ONLY)",
                    "Not in API"
                ))
                index += 1
                discrepancies.append((currency, Decimal('0'), balance))
        
        # Display summary of duplicate accounts
        print("\n" + "-"*100)
        duplicates = {curr: count for curr, count in currency_counts.items() if count > 1}
        if duplicates:
            print("* Currencies with multiple accounts: " + ", ".join(
                [f"{curr} ({count})" for curr, count in duplicates.items()]
            ))
        
        # Display API access issues if discrepancies found
        if discrepancies:
            print("\n⚠️  API ACCESS DISCREPANCIES DETECTED:")
            print("   The following accounts have different balances in your portfolio vs API:")
            for currency, api_balance, port_balance in discrepancies:
                print(f"   - {currency}: API shows {api_balance}, Portfolio shows {port_balance}")
            print("   This usually indicates limited API permissions.")
        
        print("="*100)
        
        # Suggest solutions if discrepancies were found
        if discrepancies:
            print("\n📌 RECOMMENDED SOLUTIONS:")
            print("1. Create a new API key with expanded permissions:")
            print("   - Go to Coinbase Advanced → Settings → API → Add New Key")
            print("   - Make sure to select 'View' permission for all portfolios")
            print("   - Include 'Trade' permission if you want the bot to execute trades")
            print("2. If creating a new key doesn't work, you may need to use OAuth2 authentication")
            print("   instead of API keys to access all accounts.")
            print("="*100)
    except Exception as e:
        logger.error(f"Error displaying account details: {str(e)}")

if __name__ == "__main__":
    main() 