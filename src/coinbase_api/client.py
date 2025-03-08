from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import logging
import json
from decimal import Decimal
import asyncio
import math
from datetime import datetime
import pytz
import time

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
            api_secret=api_secret,
            verbose=verbose
        )
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
            
            for account in response.accounts:
                # Handle the nested dictionary structure from the API
                account_dict = account.to_dict()
                
                # Log the full account structure for debugging
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
            
            # Log duplicate currencies if any
            duplicates = {curr: count for curr, count in currency_counts.items() if count > 1}
            if duplicates:
                logger.info(f"Found duplicate currency accounts: {duplicates}")
                
                # Group accounts by currency for clearer logging
                for currency, count in duplicates.items():
                    matching = [acc for acc in accounts if acc.currency == currency]
                    logger.info(f"Details for {currency} accounts ({count}):")
                    for i, acc in enumerate(matching):
                        logger.info(f"  - {currency} account {i+1}: Balance = {acc.available_balance}, UUID = {acc.uuid}")
            
            # Debug log all accounts and their balances
            for account in accounts:
                logger.info(f"Account {account.currency}: available={account.available_balance}, hold={account.hold}")
                
            logger.info(f"Retrieved {len(accounts)} accounts")
            return accounts
        except Exception as e:
            logger.error(f"Error getting accounts: {str(e)}")
            raise

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
            # 1. Get timeframe properties
            timeframe = Timeframe(granularity)
            lookback_minutes = timeframe.minutes * (timeframe.lookback_periods + 14)
            
            # 2. Calculate timestamps in UTC with debug info
            current_time_utc = int(datetime.now(utc_tz).timestamp())
            current_time_est = datetime.now(est_tz)
            
            end = str(current_time_utc)
            start = str(current_time_utc - lookback_minutes * 60)
            
            # Debug timestamps
            print("\n" + "="*50)
            print(f"DETAILED TIMING CHECK:")
            print(f"Current time EST: {current_time_est}")
            print(f"Current time UTC: {datetime.now(utc_tz)}")
            print(f"Request window:")
            print(f"  Start: {datetime.fromtimestamp(int(start), utc_tz).astimezone(est_tz)} EST")
            print(f"  End: {datetime.fromtimestamp(int(end), utc_tz).astimezone(est_tz)} EST")
            
            # 3. Make API request using string timestamps as required by API
            response = self.rest_client.get_candles(
                product_id=product_id,
                start=start,      # String timestamp as required
                end=end,         # String timestamp as required
                granularity=timeframe.value,
                limit=300
            )
            
            # 4. Sort and check candles
            candles = sorted(response.candles, key=lambda x: x.start, reverse=True)
            # print(f"Candles: {candles}")
            
            if candles:
                print("\nReceived candles:")
                for i, candle in enumerate(candles[:3]):
                    # Convert string timestamp to integer before creating datetime
                    candle_time = datetime.fromtimestamp(int(candle.start), utc_tz).astimezone(est_tz)
                    print(f"Candle {i}: {candle_time} EST")
                newest_candle = candles[0]  # First candle should be newest
                oldest_candle = candles[-1]  # Last candle should be oldest
                print("\nCANDLE TIME RANGE:")
                print(f"Newest candle: {datetime.fromtimestamp(int(newest_candle.start), utc_tz).astimezone(est_tz)} EST")
                print(f"Oldest candle: {datetime.fromtimestamp(int(oldest_candle.start), utc_tz).astimezone(est_tz)} EST")
            print("="*50 + "\n")
            
            return candles
            
        except Exception as e:
            logger.error(f"Failed to get candles: {str(e)}")
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