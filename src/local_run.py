import os
import json
import logging
import time
import argparse
from bot_strategy.strategy import SwingStrategy
from bot_strategy.trade_analyzer import TradeAnalysis
from coinbase_api.client import CoinbaseAdvancedClient, OrderRequest
import sys
from bot_strategy.timeframes import Timeframe
from utils.chart import print_price_chart
from trading.order_manager import OrderManager, OrderCooldownError
from decimal import Decimal
from datetime import datetime, timedelta

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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading Bot')
    parser.add_argument('--verbose-calc', action='store_true', 
                        help='Show detailed step-by-step indicator calculations')
    parser.add_argument('--symbol', type=str, default='SOL-USD',
                        help='Trading pair symbol (e.g., BTC-USD, ETH-USD)')
    parser.add_argument('--timeframe', type=str, default=os.getenv('TIMEFRAME', 'THIRTY_MINUTE'),
                        choices=[t.name for t in Timeframe], 
                        help='Trading timeframe to use')
    return parser.parse_args()

def main():
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        trading_mode = os.getenv('TRADING_MODE', 'simulation')
        # Use timeframe from command line args or environment variable
        timeframe = Timeframe[args.timeframe]
        symbol = args.symbol
        
        logger.info(f"Starting trading bot in {trading_mode} mode")
        logger.info(f"Using {timeframe.value} timeframe ({timeframe.minutes} minutes)")
        logger.info(f"Trading pair: {symbol}")
        
        # Log the order placement cooldown periods
        secrets = load_local_secrets()
        logger.info(f"Secrets loaded successfully")
        client = CoinbaseAdvancedClient(
            api_key=secrets['api_key'],
            api_secret=secrets['api_secret']
        )
        
        # Display all accounts and their metadata
        display_account_details(client)
        
        # Initialize strategy
        strategy = SwingStrategy(timeframe=timeframe)
        
        # Enable detailed calculations if requested
        # if args.verbose_calc:
        if True:
            strategy.set_verbose_calculations(True)
            logger.info("Detailed calculation mode: ENABLED - Will show step-by-step indicator calculations")
        
        logger.info("Strategy initialized successfully")
        
        # Initialize order manager
        order_manager = OrderManager(client)
        
        # Initialize order stack with historical data
        logger.info(f"Initializing order stack for {symbol}...")
        order_manager.order_stack.refresh_from_api(symbol, client)
        stack_size = order_manager.order_stack.get_stack_size(symbol)
        logger.info(f"Order stack initialized with {stack_size} unfilled buy orders")
        
        if stack_size > 0:
            latest_buy = order_manager.order_stack.peek_latest_buy(symbol)
            if latest_buy:
                logger.info(f"Latest buy order in stack (most recent - LIFO): {latest_buy.size} @ ${latest_buy.price:.2f} from {latest_buy.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Log cooldown periods for reference
        buy_cooldown = order_manager.get_order_cooldown_period(timeframe, 'BUY')
        sell_cooldown = order_manager.get_order_cooldown_period(timeframe, 'SELL')
        logger.info(f"Order cooldown periods: BUY: {buy_cooldown//60} minutes, SELL: {sell_cooldown//60} minutes")

        # Loop forever
        while True:
            try:
                # Get current time and minute
                current_time = datetime.now()
                current_minute = current_time.minute
                current_second = current_time.second
                
                # Calculate exact candle boundary information
                minutes_in_timeframe = timeframe.minutes
                
                # Calculate which candle we're in
                current_candle_minute = (current_minute // minutes_in_timeframe) * minutes_in_timeframe
                seconds_into_current_candle = (current_minute - current_candle_minute) * 60 + current_second
                
                # Calculate when the next candle should start
                current_candle_start = current_time.replace(
                    minute=current_candle_minute, 
                    second=0, 
                    microsecond=0
                )
                next_candle_start = current_candle_start + timedelta(minutes=minutes_in_timeframe)
                seconds_to_next_candle = (next_candle_start - current_time).total_seconds()
                
                # Debug message with precise timing info
                print(f"\n🕒 TIMING DEBUG: {current_time.strftime('%H:%M:%S')} | ", end="")
                print(f"Current {timeframe.value} candle: {current_candle_start.strftime('%H:%M:%S')} to {next_candle_start.strftime('%H:%M:%S')}")
                print(f"  Seconds into current candle: {seconds_into_current_candle:.1f}s | Seconds until next candle: {seconds_to_next_candle:.1f}s")
                
                # Determine if we're at a candle boundary (either just after or just before)
                # We check more frequently around candle boundaries to catch new data quickly
                boundary_window = min(30, minutes_in_timeframe * 60 * 0.1)  # 10% of timeframe or 30 sec, whichever is smaller
                is_boundary_time = seconds_into_current_candle < boundary_window or seconds_to_next_candle < boundary_window
                
                if is_boundary_time:
                    print(f"🔄 Checking at {timeframe.value} candle boundary time")
                
                print("\n" + "="*50, flush=True)
                print(f"📊 Fetching market data for {symbol}...", flush=True)
                
                # Add timestamp to API call to prevent caching
                candles = client.get_product_candles(
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
                    
                    # Get real-time price from API (separate from historical candles)
                    current_price = None
                    try:
                        current_price = client.get_product_price(symbol)
                        print(f"📊 Real-time price: ${current_price:.2f}")
                    except Exception as e:
                        logger.warning(f"Could not get real-time price: {str(e)}")
                        # Will fall back to candle data in the strategy
                    
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
                    # Pass the current price to strategy instead of the client
                    signal = strategy.generate_signal(symbol, prices, timestamps, current_price)
                    
                    # Process the signal if it exists
                    if signal:
                        print(f"\n🔔 Got trading signal: {signal.action} {signal.symbol} at ${signal.price:.2f}")
                        
                        # Display additional signal information if available
                        if hasattr(signal, 'confidence') and signal.confidence > 0:
                            print(f"📊 Signal confidence: {signal.confidence:.2f} (scale 0-1)")
                        
                        # Display indicator details if available
                        if hasattr(signal, 'indicators') and signal.indicators:
                            if 'rsi' in signal.indicators:
                                rsi_data = signal.indicators['rsi']
                                print(f"  RSI: {rsi_data['value']:.2f} (thresholds: {rsi_data['oversold']}/{rsi_data['overbought']})")
                            if 'ema' in signal.indicators:
                                ema_data = signal.indicators['ema']
                                print(f"  EMA: ${ema_data['value']:.2f} (price distance: {ema_data['distance_pct']:.2f}%)")
                        
                        try:
                            if trading_mode == 'simulation':
                                print(f"\n🔸 SIMULATION MODE:")
                                
                                # Use symbol from signal if available, otherwise default to SOL-USD
                                trading_pair = signal.symbol if hasattr(signal, 'symbol') else symbol
                                trading_action = signal.action if hasattr(signal, 'action') else 'BUY'
                                # trading_action = 'SELL'
                                
                                if trading_action == 'BUY':
                                    # Analyze trade potential
                                    balance_fraction = 0.05  # Use 5% of available balance
                                    # investment = balance_fraction * 10
                                    investment = 10
                                    price_percentage = 0.999 # 95% of current price for buy limit order
                                    entry_price = signal.price * price_percentage
                                    analyzer = TradeAnalysis(
                                        investment=investment,
                                        entry_price=entry_price,
                                        timeframe=timeframe
                                    )
                                    analysis = analyzer.analyze(prices, signal.action)
                                    analyzer.print_analysis(analysis, signal.symbol, signal.action, prices)    
                                
                                    print(f"\n🔶 Creating smart limit buy order for {trading_pair}...")
                                    
                                    # Create a smart limit order with custom parameters
                                    # Default parameters: 95% of market price, 10% of available balance
                                    order = order_manager.create_smart_limit_order(
                                        product_id=trading_pair,
                                        side='BUY',
                                        price_percentage=price_percentage,  
                                        balance_fraction=balance_fraction   
                                    )

                                    try:
                                        # Debug the order before placing
                                        logger.debug(f"ABOUT TO PLACE ORDER: {order.__dict__ if hasattr(order, '__dict__') else order}")
                                        
                                        # Place the order with cooldown check
                                        response = order_manager.place_order(order, timeframe=timeframe)
                                        
                                        # Log the response for debugging
                                        print(f"Response from place_order: {response}")
                                        if response and response['success']:
                                            order_id = response['success_response']['order_id']
                                            print(f"Limit buy order created: {order_id}")
                                        elif hasattr(response, 'order_id'):
                                            print(f"Limit buy order created: {response.order_id}")
                                        else:
                                            logger.warning(f"❌ Order not placed: {response}")
                                    except OrderCooldownError as e:
                                        print(f"\n⏳ Order placement throttled: {str(e)}")
                                        print(f"This prevents overtrading and follows best practices for the {timeframe.value} timeframe")
                                    except Exception as e:
                                        logger.error(f"Error placing buy order: {str(e)}")
                                    
                                elif trading_action == 'SELL':
                                    print(f"\n🔶 Creating limit sell order for {trading_pair} based on trading signal...")
                                    
                                    try:
                                        # Create a sell order using the enhanced smart limit order method
                                        # which now implements LIFO stack-based matching
                                        order = order_manager.create_smart_limit_order(
                                            product_id=trading_pair,
                                            side='SELL',
                                            price_percentage=1.001,  # 100.1% of current price for sell limit order
                                            balance_fraction=1.0    # Use full size of matched buy order
                                        )
                                        
                                        if order:  # Only proceed if an order was created
                                            # Place the order with cooldown check
                                            response = order_manager.place_order(order, timeframe=timeframe)
                                            
                                            # Log the response for debugging
                                            print(f"SELL Response from place_order: {response}")
                                            # Check the response based on its type
                                            if response and response.get('success', False):
                                                order_id = response['success_response']['order_id']
                                                print(f"Limit sell order created: {order_id}")
                                            elif hasattr(response, 'order_id'):
                                                print(f"Limit sell order created: {response.order_id}")
                                            else:
                                                logger.warning(f"❌ Order not placed: {response}")
                                        else:
                                            # No order was created - check why
                                            # Get stack and balance info for better diagnostics
                                            stack_size = order_manager.order_stack.get_stack_size(trading_pair)
                                            base_asset = trading_pair.split('-')[0]
                                            asset_account = order_manager.balance_manager.get_balance(base_asset)
                                            available_balance = float(asset_account.available_balance) if asset_account else 0
                                            
                                            # Provide specific feedback based on the situation
                                            if stack_size == 0 and available_balance == 0:
                                                print(f"⚠️ Sell signal ignored: No {base_asset} to sell (empty stack and zero balance)")
                                            elif stack_size == 0 and available_balance > 0:
                                                print(f"⚠️ Sell signal ignored: You have {available_balance} {base_asset} but the sell price may not be favorable")
                                            elif stack_size > 0:
                                                # We have orders in the stack but didn't sell - must be price not profitable
                                                latest_buy = order_manager.order_stack.peek_latest_buy(trading_pair)
                                                if latest_buy:
                                                    current_price = order_manager.client.get_product_price(trading_pair)
                                                    profit_pct = (current_price / latest_buy.price - 1) * 100
                                                    print(f"⚠️ Sell signal ignored: Current price not profitable")
                                                    print(f"  Buy price: ${latest_buy.price:.2f}")
                                                    print(f"  Current price: ${current_price:.2f}")
                                                    print(f"  Profit/loss: {profit_pct:.2f}%")
                                                    print(f"  Need at least 0.5% profit to cover fees")
                                            
                                            # Display the stack status for reference
                                            print(f"ℹ️ System Status:")
                                            print(f"  Buy orders in stack: {stack_size}")
                                            print(f"  {base_asset} balance: {available_balance}")
                                            if stack_size > 0:
                                                latest_buy = order_manager.order_stack.peek_latest_buy(trading_pair)
                                                if latest_buy:
                                                    print(f"  Most recent buy: {latest_buy.size} @ ${latest_buy.price:.2f} on {latest_buy.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                                    except OrderCooldownError as e:
                                        print(f"\n⏳ Order placement throttled: {str(e)}")
                                        print(f"This prevents overtrading and follows best practices for the {timeframe.value} timeframe")
                                    except Exception as e:
                                        logger.error(f"Error placing sell order: {str(e)}")
                                
                            else:
                                print(f"\n🔶 LIVE MODE: Executing trade...")
                                try:
                                    order = OrderRequest(
                                        product_id=signal.symbol,
                                        side=signal.action.lower(),
                                        order_type='MARKET',
                                        quote_size='10'
                                    )
                                    response = client.create_market_order(order, timeframe=timeframe)
                                    logger.info(f"Order placed: {response}")
                                except OrderCooldownError as e:
                                    print(f"\n⏳ Order placement throttled: {str(e)}")
                                    print(f"This prevents overtrading and follows best practices for the {timeframe.value} timeframe")
                                except Exception as e:
                                    logger.error(f"Error placing market order: {str(e)}")
                        except Exception as e:
                            logger.error(f"Error processing signal: {str(e)}")
                            
                    else:
                        print("\n😴 No trading signals generated")
                    
                    # Wait for next iteration - use timing based on the specific timeframe
                    last_run_time = datetime.now()
                    
                    # Calculate next check time based on timeframe and current time
                    # Get timeframe in minutes and seconds
                    minutes_in_timeframe = timeframe.minutes
                    seconds_in_timeframe = minutes_in_timeframe * 60
                    
                    # Calculate time to next boundary for any timeframe
                    current_minute = last_run_time.minute
                    current_second = last_run_time.second
                    
                    # For any timeframe, calculate minutes to the next boundary
                    minutes_to_boundary = minutes_in_timeframe - (current_minute % minutes_in_timeframe)
                    if minutes_to_boundary == minutes_in_timeframe:
                        minutes_to_boundary = 0
                    
                    seconds_to_boundary = minutes_to_boundary * 60 - current_second
                    
                    if seconds_to_boundary <= 0:
                        seconds_to_boundary += seconds_in_timeframe
                    
                    # Define check frequency based on timeframe size
                    # For shorter timeframes, check more often
                    if minutes_in_timeframe <= 15:  # 15 min or less
                        # standard_check_interval = min(60, seconds_in_timeframe // 5)  # At least 5 checks per timeframe
                        standard_check_interval = seconds_in_timeframe // 2
                        boundary_threshold = 60  # Check more frequently within 60 seconds of boundary
                    elif minutes_in_timeframe <= 60:  # Hour or less
                        # standard_check_interval = min(300, seconds_in_timeframe // 10)  # At least 10 checks per timeframe
                        standard_check_interval = seconds_in_timeframe // 2
                        boundary_threshold = 120  # Check more frequently within 2 minutes of boundary
                    else:  # Longer timeframes
                        standard_check_interval = min(900, seconds_in_timeframe // 20)  # At least 20 checks per timeframe
                        boundary_threshold = 300  # Check more frequently within 5 minutes of boundary
                    
                    # If we're far from a boundary, wait standard interval
                    # If we're close to a boundary, wait until just after the boundary
                    if seconds_to_boundary > boundary_threshold:
                        wait_time = standard_check_interval
                    else:
                        wait_time = seconds_to_boundary + 5  # Wait until 5 seconds after boundary
                    
                    boundary_time = last_run_time + timedelta(seconds=seconds_to_boundary)
                    print(f"\n⏳ Next {timeframe.value} candle boundary at {boundary_time.strftime('%H:%M:%S')}")
                    
                    next_check = last_run_time + timedelta(seconds=wait_time)
                    print(f"⏳ Checking again at {next_check.strftime('%H:%M:%S')} ({wait_time/60:.1f} minutes)")
                    time.sleep(wait_time)
                    
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