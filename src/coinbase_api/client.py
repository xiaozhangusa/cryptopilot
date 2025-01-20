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

    def get_accounts(self) -> Dict:
        """Fetch all available trading accounts"""
        try:
            accounts = self.rest_client.get_accounts()
            logger.info(f"Retrieved {len(accounts.accounts)} accounts")
            return accounts.to_dict()
        except Exception as e:
            logger.error(f"Failed to fetch accounts: {str(e)}")
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
        """Create a limit order"""
        try:
            if order.side.lower() == 'buy':
                response = self.rest_client.limit_order_gtc_buy(
                    client_order_id=order.client_order_id,
                    product_id=order.product_id,
                    base_size=order.base_size,
                    limit_price=order.limit_price
                )
            else:
                response = self.rest_client.limit_order_gtc_sell(
                    client_order_id=order.client_order_id,
                    product_id=order.product_id,
                    base_size=order.base_size,
                    limit_price=order.limit_price
                )
            
            if response['success']:
                order_id = response['success_response']['order_id']
                logger.info(f"Limit order created: {order_id}")
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
        """Get historical price candles for a cryptocurrency product.

        This method retrieves OHLCV (Open, High, Low, Close, Volume) candle data for a specified
        trading pair over a time period. The time period is calculated based on the number of
        periods needed for analysis (default 20 periods for RSI calculation) and the specified
        granularity.

        Args:
            product_id (str): The trading pair identifier (e.g., 'BTC-USD', 'ETH-USD')
            granularity (str, optional): The time interval for each candle. Defaults to 'FIVE_MINUTE'.
                Valid values:
                - 'ONE_MINUTE': 1-minute candles
                - 'FIVE_MINUTE': 5-minute candles
                - 'FIFTEEN_MINUTE': 15-minute candles
                - 'THIRTY_MINUTE': 30-minute candles
                - 'ONE_HOUR': 1-hour candles
                - 'TWO_HOUR': 2-hour candles
                - 'SIX_HOUR': 6-hour candles
                - 'ONE_DAY': 24-hour candles

        Returns:
            List[dict]: A list of candle data sorted by time (newest first). Each candle contains:
                - start: Unix timestamp for the start of the candle
                - low: Lowest price during the period
                - high: Highest price during the period
                - open: Opening price for the period
                - close: Closing price for the period
                - volume: Trading volume during the period

        Raises:
            Exception: If there's an error fetching the candle data from Coinbase API.
            The exception is caught and logged, returning an empty list.

        Example:
            >>> client = CoinbaseAdvancedClient(api_key, api_secret)
            >>> candles = client.get_product_candles('BTC-USD', 'ONE_HOUR')
            >>> if candles:
            >>>     print(f"Most recent BTC price: ${candles[0].close}")
        """
        try:
            # 1. Get current time in Unix timestamp (seconds since epoch)
            end = int(time.time())
            
            # 2. Define how many candles we need for analysis
            periods_needed = 20  # For RSI calculation
            
            # 3. Convert timeframe to seconds
            granularity_secs = self.get_granularity_seconds(granularity)
            
            # 4. Calculate start time by going back (periods * interval) seconds
            start = end - (periods_needed * granularity_secs)
            
            # 5. Log the time range we're requesting (in EST)
            start_time = datetime.fromtimestamp(start, utc_tz).astimezone(est_tz)
            end_time = datetime.fromtimestamp(end, utc_tz).astimezone(est_tz)
            logger.info(f"Fetching candles from {start_time} to {end_time} EST")
            
            # 6. Make API request to Coinbase
            response = self.rest_client.get_candles(
                product_id=product_id,
                start=str(start),
                end=str(end),
                granularity=granularity
            )
            
            # 7. Sort candles newest first
            candles = sorted(response.candles, key=lambda x: x.start, reverse=True)
            
            # 8. Log what we actually received
            if candles:
                newest = datetime.fromtimestamp(candles[0].start, utc_tz).astimezone(est_tz)
                oldest = datetime.fromtimestamp(candles[-1].start, utc_tz).astimezone(est_tz)
                logger.info(f"Got candles from {oldest} to {newest} EST")
            
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