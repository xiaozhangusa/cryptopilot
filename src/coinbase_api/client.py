from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import logging
import json
from decimal import Decimal
import asyncio
import math
from datetime import datetime, timedelta
import pytz
import time
import traceback

from coinbase.rest import RESTClient
from coinbase.websocket import WSClient
# from coinbase.rest.models.enums import OrderSide, OrderType, TimeInForce
# from coinbase.rest.models import CreateOrderResponse
from bot_strategy.timeframes import Timeframe  # Add this import at the top
from src.coinbase_api.models import Account  # Add this import at the top

logger = logging.getLogger(__name__)

est_tz = pytz.timezone('America/New_York')
utc_tz = pytz.UTC

@dataclass
class OrderRequest:
    """Request parameters for creating an order"""
    product_id: str
    side: str  # 'BUY' or 'SELL'
    order_type: str  # 'MARKET' or 'LIMIT'
    quote_size: Optional[str] = None  # For market orders: amount in quote currency
    base_size: Optional[str] = None   # For limit orders: amount in base currency
    limit_price: Optional[str] = None # For limit orders: price per unit
    client_order_id: Optional[str] = None
    time_in_force: Optional[str] = 'GTC'  # 'GTC', 'GTD', 'IOC', or 'FOK'
    cancel_after: Optional[str] = None  # 'min', 'hour', 'day' for GTD orders

class CoinbaseAdvancedClient:
    """Wrapper for Coinbase Advanced Trading API using official SDK"""
    
    def __init__(self, api_key: str, api_secret: str, verbose: bool = False):
        """
        Initialize the client with CDP API credentials
        
        Args:
            api_key: CDP API key (format: "organizations/{org_id}/apiKeys/{key_id}")
            api_secret: CDP API secret (PEM format private key)
            verbose: Enable debug logging
        """
        self.rest_client = RESTClient(
            api_key=api_key,
            api_secret=api_secret
        )
        self.verbose = verbose  # Store verbose setting at client level
        self.ws_client: Optional[WSClient] = None
        self.order_filled = False
        self.limit_order_id = None

    def initialize_websocket(self, 
                            product_ids: List[str],
                            channels: List[str] = ["heartbeats", "ticker", "user"],
                            verbose: bool = False):
        """Initialize WebSocket connection for price monitoring"""
        
        def ws_handler(message: str):
            try:
                message_data = json.loads(message)
                logger.debug(f"WebSocket message received: {message_data}")
                
                if 'channel' in message_data:
                    if message_data['channel'] == 'ticker':
                        events = message_data.get('events', [])
                        for event in events:
                            tickers = event.get('tickers', [])
                            for ticker in tickers:
                                product_id = ticker.get('product_id')
                                price = ticker.get('price')
                                logger.info(f"Price update for {product_id}: {price}")
                    
                    elif message_data['channel'] == 'user':
                        orders = message_data['events'][0].get('orders', [])
                        for order in orders:
                            order_id = order.get('order_id')
                            if order_id == self.limit_order_id and order.get('status') == 'FILLED':
                                self.order_filled = True
                                logger.info(f"Order {order_id} filled!")
                                
            except Exception as e:
                logger.error(f"WebSocket handler error: {str(e)}")

        try:
            # Configure WebSocket client
            self.ws_client = WSClient(
                api_key=self.rest_client.api_key,
                api_secret=self.rest_client.api_secret,
                on_message=ws_handler,
                verbose=verbose,
                retry=True  # Enable automatic reconnection
            )
            
            # Open connection and subscribe to channels
            self.ws_client.open()
            self.ws_client.subscribe(product_ids, channels)
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket: {str(e)}")
            raise

    def wait_for_order_fill(self, timeout: Optional[int] = None):
        """Wait for order to be filled
        
        Args:
            timeout: Optional timeout in seconds. If None, waits indefinitely.
        """
        if timeout:
            self.ws_client.sleep_with_exception_check(timeout)
        else:
            while not self.order_filled:
                self.ws_client.sleep_with_exception_check(1)

    def get_accounts(self) -> List[Account]:
        """
        Get all accounts from Coinbase
        Returns list of Account objects with proper balance information
        """
        try:
            response = self.rest_client.get_accounts()
            accounts = []
            currency_counts = {}  # To track duplicate currencies
            
            # Check if verbose mode is enabled (defaults to False if not found)
            verbose_mode = getattr(self, 'verbose', False)
            
            for account in response.accounts:
                # Handle the nested dictionary structure from the API
                account_dict = account.to_dict()
                
                # Log the full account structure for debugging only in verbose mode
                if verbose_mode:
                    logger.debug(f"Account structure: {account_dict}")
                
                # Safely extract values with better error handling
                try:
                    # Get the currency (check if it's a duplicate)
                    currency = account_dict.get('currency', '')
                    if currency:
                        currency_counts[currency] = currency_counts.get(currency, 0) + 1
                    
                    # Get the available balance value, defaulting to '0' if missing
                    available_balance_value = account_dict.get('available_balance', {}).get('value', '0')
                    hold_value = account_dict.get('hold', {}).get('value', '0')
                    
                    # Account ID or any unique identifier
                    uuid = account_dict.get('uuid', '')
                    
                    # Create Account object with proper values
                    accounts.append(Account(
                        uuid=uuid,
                        name=account_dict.get('name', ''),
                        currency=currency,
                        available_balance=Decimal(str(available_balance_value)),
                        hold=Decimal(str(hold_value))
                    ))
                except (KeyError, TypeError, ValueError) as e:
                    logger.error(f"Error parsing account data: {e}, account: {account_dict}")
                    # Still add the account but with safe default values
                    accounts.append(Account(
                        uuid=account_dict.get('uuid', ''),
                        name=account_dict.get('name', ''),
                        currency=account_dict.get('currency', ''),
                        available_balance=Decimal('0'),
                        hold=Decimal('0')
                    ))
            
            # Log duplicate currencies if any (only in verbose mode)
            duplicates = {curr: count for curr, count in currency_counts.items() if count > 1}
            if duplicates and verbose_mode:
                logger.info(f"Found duplicate currency accounts: {duplicates}")
                
                # Group accounts by currency for clearer logging
                for currency, count in duplicates.items():
                    matching = [acc for acc in accounts if acc.currency == currency]
                    logger.info(f"Details for {currency} accounts ({count}):")
                    for i, acc in enumerate(matching):
                        logger.info(f"  - {currency} account {i+1}: Balance = {acc.available_balance}, UUID = {acc.uuid}")
            
            # Debug log all accounts and their balances (only in verbose mode)
            if verbose_mode:
                for account in accounts:
                    logger.info(f"Account {account.currency}: available={account.available_balance}, hold={account.hold}")
                
            logger.info(f"Retrieved {len(accounts)} accounts" + ("" if verbose_mode else " (silent mode)"))
            return accounts
        except Exception as e:
            logger.error(f"Error getting accounts: {str(e)}")
            raise

    def get_portfolio_accounts(self) -> List[tuple]:
        """
        Get account balances through Coinbase Advanced API portfolios
        Returns list of (currency, balance) tuples from portfolio endpoint
        
        This implementation focuses on the methods that are confirmed to be
        available in the REST client based on the logs.
        """
        try:
            # Portfolio data to return
            portfolio_data = []
            
            # Try portfolio breakdown method first - this was found in the available methods list
            try:
                logger.info("Trying portfolio breakdown method")
                
                # 1. First get the portfolios
                portfolios_response = self.rest_client.get_portfolios()
                
                if hasattr(portfolios_response, 'portfolios') and portfolios_response.portfolios:
                    portfolios = portfolios_response.portfolios
                    logger.info(f"Found {len(portfolios)} portfolios")
                    
                    # 2. Go through each portfolio
                    for portfolio in portfolios:
                        portfolio_id = None
                        if hasattr(portfolio, 'uuid'):
                            portfolio_id = portfolio.uuid
                            logger.info(f"Found portfolio ID using uuid: {portfolio_id}")
                        elif hasattr(portfolio, 'id'):
                            portfolio_id = portfolio.id
                            logger.info(f"Found portfolio ID using id: {portfolio_id}")
                        
                        if portfolio_id:
                            # 3. Get portfolio breakdown - this method is available according to the logs
                            try:
                                logger.info(f"Getting portfolio breakdown for portfolio: {portfolio_id}")
                                breakdown = self.rest_client.get_portfolio_breakdown(portfolio_uuid=portfolio_id)
                                
                                # Log what we got
                                logger.debug(f"Portfolio breakdown type: {type(breakdown)}")
                                if hasattr(breakdown, 'to_dict'):
                                    try:
                                        logger.debug(f"Breakdown dict: {breakdown.to_dict()}")
                                    except:
                                        pass
                                
                                # Try to extract assets/currencies from different possible responses
                                
                                # Option 1: Check for assets attribute
                                if hasattr(breakdown, 'assets') and breakdown.assets:
                                    logger.info(f"Found assets in portfolio breakdown")
                                    for asset in breakdown.assets:
                                        try:
                                            currency = None
                                            balance = None
                                            
                                            # Try different attribute names for currency
                                            for curr_attr in ['currency', 'currency_id', 'symbol', 'base_currency_id']:
                                                if hasattr(asset, curr_attr):
                                                    currency = getattr(asset, curr_attr)
                                                    if currency:
                                                        break
                                            
                                            # Try different attribute names for balance
                                            for bal_attr in ['balance', 'total_balance', 'value', 'amount']:
                                                if hasattr(asset, bal_attr):
                                                    bal_val = getattr(asset, bal_attr)
                                                    if isinstance(bal_val, (int, float, str, Decimal)):
                                                        balance = Decimal(str(bal_val))
                                                        break
                                                    elif hasattr(bal_val, 'value'):
                                                        balance = Decimal(str(bal_val.value))
                                                        break
                                                    elif isinstance(bal_val, dict) and 'value' in bal_val:
                                                        balance = Decimal(str(bal_val['value']))
                                                        break
                                            
                                            if currency and balance and balance > 0:
                                                portfolio_data.append((currency, balance))
                                                logger.info(f"Found in breakdown: {currency}: {balance}")
                                        except Exception as e:
                                            logger.warning(f"Error processing asset in breakdown: {str(e)}")
                                
                                # Option 2: Check for currencies or allocations
                                for attr in ['currencies', 'allocations', 'positions']:
                                    if hasattr(breakdown, attr):
                                        items = getattr(breakdown, attr)
                                        if items:
                                            logger.info(f"Found {len(items)} items in breakdown.{attr}")
                                            for item in items:
                                                try:
                                                    # Similar extraction as above but for different structure
                                                    # ... extraction code would be similar to the above ...
                                                    pass
                                                except Exception as e:
                                                    logger.warning(f"Error processing item in {attr}: {str(e)}")
                            except Exception as e:
                                logger.warning(f"Error getting portfolio breakdown: {str(e)}")
            except Exception as e:
                logger.warning(f"Error in portfolio breakdown approach: {str(e)}")
            
            # If we found data from portfolio breakdown, return it
            if portfolio_data:
                return portfolio_data
                
            # Fallback to regular account data which we know works
            logger.info("Using regular account data as fallback for portfolios")
            try:
                accounts = self.get_accounts()
                for account in accounts:
                    if account.available_balance > 0:
                        portfolio_data.append((account.currency, account.available_balance))
                        logger.info(f"Using regular account data: {account.currency}: {account.available_balance}")
            except Exception as e:
                logger.warning(f"Error using regular account data: {str(e)}")
                
            return portfolio_data
        except Exception as e:
            logger.error(f"Error in get_portfolio_accounts: {str(e)}")
            return []

    def create_market_order(self, order: OrderRequest) -> Dict:
        """Create a market order"""
        try:
            if order.side.lower() == 'buy':
                response = self.rest_client.market_order_buy(
                    client_order_id=order.client_order_id,
                    product_id=order.product_id,
                    quote_size=order.quote_size
                )
            else:
                response = self.rest_client.market_order_sell(
                    client_order_id=order.client_order_id,
                    product_id=order.product_id,
                    quote_size=order.quote_size
                )
            
            if response['success']:
                order_id = response['success_response']['order_id']
                logger.info(f"Market order created: {order_id}")
                
                # Get fill information
                fills = self.rest_client.get_fills(order_id=order_id)
                logger.info(f"Order fills: {fills.to_dict()}")
            else:
                logger.error(f"Failed to create order: {response['error_response']}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to create market order: {str(e)}")
            raise

    def create_limit_order(self, order: OrderRequest) -> Dict:
        """Create a limit order with specified parameters.
        
        Args:
            order: OrderRequest object containing:
                - product_id: Trading pair (e.g., 'BTC-USD')
                - side: Order side ('BUY' or 'SELL')
                - base_size: Amount in base currency
                - limit_price: Limit price per unit
                - client_order_id: Optional client-specified ID
                - time_in_force: Optional time in force ('GTC', 'GTD', 'IOC', 'FOK')
                    GTC: Good till cancelled (default)
                    GTD: Good till date
                    IOC: Immediate or cancel
                    FOK: Fill or kill
                - cancel_after: Optional cancel time for GTD ('min', 'hour', 'day')
                
        Returns:
            Dict containing response with fields:
                - success: bool indicating if order was created
                - success_response: Order details if successful
                - error_response: Error details if failed
                
        Raises:
            Exception: If order creation fails
        """
        try:
            # Reset order tracking
            self.order_filled = False
            self.limit_order_id = None
            
            # Determine which SDK method to call based on side and time_in_force
            time_in_force = getattr(order, 'time_in_force', 'GTC').upper()
            side = order.side.lower()
            
            order_params = {
                'client_order_id': order.client_order_id,
                'product_id': order.product_id,
                'base_size': order.base_size,
                'limit_price': order.limit_price
            }
            
            # Add cancel_after if GTD
            if time_in_force == 'GTD' and hasattr(order, 'cancel_after'):
                order_params['cancel_after'] = order.cancel_after
            
            # Select appropriate SDK method
            if side == 'buy':
                if time_in_force == 'GTD':
                    response = self.rest_client.limit_order_gtd_buy(**order_params)
                elif time_in_force == 'IOC':
                    response = self.rest_client.limit_order_ioc_buy(**order_params)
                elif time_in_force == 'FOK':
                    response = self.rest_client.limit_order_fok_buy(**order_params)
                else:  # GTC is default
                    response = self.rest_client.limit_order_gtc_buy(**order_params)
            else:  # sell
                if time_in_force == 'GTD':
                    response = self.rest_client.limit_order_gtd_sell(**order_params)
                elif time_in_force == 'IOC':
                    response = self.rest_client.limit_order_ioc_sell(**order_params)
                elif time_in_force == 'FOK':
                    response = self.rest_client.limit_order_fok_sell(**order_params)
                else:  # GTC is default
                    response = self.rest_client.limit_order_gtc_sell(**order_params)
            
            if response['success']:
                order_id = response['success_response']['order_id']
                self.limit_order_id = order_id  # Store for WebSocket tracking
                logger.info(f"Limit order created: {order_id} ({time_in_force})")
            else:
                logger.error(f"Failed to create order: {response['error_response']}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to create limit order: {str(e)}")
            raise

    def close(self):
        """Close WebSocket connection if active"""
        if self.ws_client:
            try:
                self.ws_client.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {str(e)}")

    def get_product_price(self, product_id: str) -> float:
        """Get current price for a product"""
        try:
            product = self.rest_client.get_product(product_id)
            return float(product["price"])
        except Exception as e:
            logger.error(f"Failed to get product price: {str(e)}")
            raise

    def get_granularity_seconds(self, granularity: str) -> int:
        """Convert granularity string to seconds"""
        return {
            'ONE_MINUTE': 60,
            'FIVE_MINUTE': 300,        # 60 * 5
            'FIFTEEN_MINUTE': 900,     # 60 * 15
            'THIRTY_MINUTE': 1800,     # 60 * 30
            'ONE_HOUR': 3600,         # 60 * 60
            'TWO_HOUR': 7200,         # 60 * 60 * 2
            'SIX_HOUR': 21600,        # 60 * 60 * 6
            'ONE_DAY': 86400          # 60 * 60 * 24
        }[granularity]

    def get_product_candles(self, product_id: str, granularity: str = 'FIVE_MINUTE') -> List[dict]:
        """Get historical candles for a product"""
        try:
            # Get timeframe properties
            timeframe = Timeframe(granularity)
            minutes_in_timeframe = timeframe.minutes
            
            # Debug header with timestamp info
            print("\n" + "="*80)
            print(f"CANDLE RETRIEVAL DEBUG - {granularity}")
            print("="*80)
            
            # Current time in various formats for debugging
            current_time = datetime.now(utc_tz)
            current_time_utc = int(current_time.timestamp())
            current_time_est = current_time.astimezone(est_tz)
            
            print(f"Current exact time: {current_time_est.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} {current_time_est.strftime('%Z')}")
            print(f"Current UTC timestamp: {current_time_utc}")
            
            # For FIVE_MINUTE candles, calculate the start of the current candle
            # This is critical for understanding if we should have the latest candle yet
            minutes_since_hour = current_time_est.minute
            current_candle_minute = (minutes_since_hour // minutes_in_timeframe) * minutes_in_timeframe
            seconds_into_current_candle = (minutes_since_hour - current_candle_minute) * 60 + current_time_est.second
            
            current_candle_start = current_time_est.replace(
                minute=current_candle_minute, 
                second=0, 
                microsecond=0
            )
            
            # Calculate when the next candle should start
            next_candle_start = current_candle_start + timedelta(minutes=minutes_in_timeframe)
            
            print(f"Current {granularity} candle period: {current_candle_start.strftime('%H:%M:%S')} to {next_candle_start.strftime('%H:%M:%S')} {current_time_est.strftime('%Z')}")
            print(f"Seconds into current candle period: {seconds_into_current_candle} of {minutes_in_timeframe*60}")
            
            # Use a far future end time to ensure we get all candles including the most recent
            # This is important for avoiding API timestamp quirks
            future_time = current_time_utc + 3600  # Look 1 hour ahead (far future for candle purposes)
            end = str(future_time)
            
            # Calculate the start time based on how many candles we need
            # Using a large number to ensure we get enough historical data
            candle_count = max(timeframe.lookback_periods + 20, 50)  # Get extra candles for good measure
            seconds_per_candle = minutes_in_timeframe * 60
            lookback_seconds = seconds_per_candle * candle_count
            start_time = current_time_utc - lookback_seconds
            start = str(start_time)
            
            # Detailed timing info for the request
            print(f"\nRequest parameters:")
            print(f"  Start time: {datetime.fromtimestamp(int(start), utc_tz).astimezone(est_tz).strftime('%Y-%m-%d %H:%M:%S')} {current_time_est.strftime('%Z')}")
            print(f"  End time: {datetime.fromtimestamp(int(end), utc_tz).astimezone(est_tz).strftime('%Y-%m-%d %H:%M:%S')} {current_time_est.strftime('%Z')} (future)")
            print(f"  Candle count: {candle_count}")
            print(f"  Granularity: {granularity}")
            
            # For the REST client we need to stick to the documented parameters
            response = self.rest_client.get_candles(
                product_id=product_id,
                start=start,  # Required - start time for historical data
                end=end,      # Future time
                granularity=granularity,
                limit=candle_count
            )
            
            # Sort candles newest first for easier analysis
            candles = sorted(response.candles, key=lambda x: x.start, reverse=True)
            
            if not candles:
                print("❌ No candles received.")
                return []
            
            # Detailed analysis of the received candles
            print(f"\nReceived {len(candles)} candles.")
            
            # Check the newest and oldest candles for time range
            newest_candle = candles[0]
            oldest_candle = candles[-1]
            
            newest_time = datetime.fromtimestamp(int(newest_candle.start), utc_tz)
            oldest_time = datetime.fromtimestamp(int(oldest_candle.start), utc_tz)
            
            newest_time_est = newest_time.astimezone(est_tz)
            oldest_time_est = oldest_time.astimezone(est_tz)
            
            print(f"Candle time range: {oldest_time_est.strftime('%Y-%m-%d %H:%M:%S')} to {newest_time_est.strftime('%Y-%m-%d %H:%M:%S')} {current_time_est.strftime('%Z')}")
            
            # Calculate lag
            lag_time = current_time - newest_time
            lag_seconds = lag_time.total_seconds()
            lag_minutes = lag_seconds / 60
            
            print(f"Time since newest candle: {lag_minutes:.2f} minutes ({lag_seconds:.2f} seconds)")
            
            # Check if we should have the current candle yet
            if seconds_into_current_candle < 10:
                print(f"⚠️ Current candle period just started {seconds_into_current_candle:.2f} seconds ago - normal not to have it yet")
            elif lag_minutes > minutes_in_timeframe:
                print(f"⚠️ WARNING: Data is significantly delayed by {lag_minutes:.2f} minutes (more than one candle period)")
            else:
                print(f"ℹ️ Latest candle is {lag_minutes:.2f} minutes old - normal for {granularity}")
            
            # Show the 5 most recent candles for detailed debugging
            print("\nMost recent candles:")
            for i, candle in enumerate(candles[:5]):
                candle_time = datetime.fromtimestamp(int(candle.start), utc_tz).astimezone(est_tz)
                candle_end = candle_time + timedelta(minutes=minutes_in_timeframe)
                print(f"Candle {i}: {candle_time.strftime('%H:%M:%S')} - {candle_end.strftime('%H:%M:%S')} | ${float(candle.close):.2f} | Vol: {float(candle.volume):.4f}")
            
            print("="*80 + "\n")
            return candles
            
        except Exception as e:
            logger.error(f"Failed to get {granularity} candles: {str(e)}")
            print(f"❌ Error getting candles: {str(e)}")
            print(f"Stack trace: ", traceback.format_exc())
            return []

    def cancel_orders(self, order_ids: List[str]) -> Dict:
        """Cancel one or more orders by ID"""
        try:
            response = self.rest_client.cancel_orders(order_ids=order_ids)
            if response['success']:
                logger.info(f"Successfully cancelled orders: {order_ids}")
            else:
                logger.error(f"Failed to cancel orders: {response['error_response']}")
            return response
        except Exception as e:
            logger.error(f"Failed to cancel orders: {str(e)}")
            raise 