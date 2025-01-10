from typing import Dict, Optional, Callable
import asyncio
import logging
from decimal import Decimal

from coinbase_advanced_py import RESTClient, WSClient
from coinbase_advanced_py.websocket import Channel, ChannelType
from coinbase_advanced_py.rest import OrderSide, OrderType, TimeInForce
from coinbase_advanced_py.rest.models import CreateOrderResponse

logger = logging.getLogger(__name__)

class CoinbaseAdvancedSDKClient:
    """
    A wrapper class for Coinbase Advanced Trading SDK that provides both REST and WebSocket
    functionality with proper error handling and logging.
    """
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.rest_client = RESTClient(api_key=api_key, api_secret=api_secret)
        self.ws_client: Optional[WSClient] = None
        
    async def initialize_websocket(self, 
                                 product_id: str,
                                 price_threshold: Decimal,
                                 on_price_update: Optional[Callable[[Decimal], None]] = None):
        """
        Initialize and start WebSocket connection with price monitoring
        
        Args:
            product_id: Trading pair ID (e.g., 'BTC-USD')
            price_threshold: Price threshold for monitoring
            on_price_update: Callback function for price updates
        """
        try:
            # Define WebSocket message handler
            async def ws_handler(message: Dict):
                try:
                    logger.debug(f"WebSocket message received: {message}")
                    
                    if message.get('type') == 'ticker':
                        price = Decimal(message.get('price', '0'))
                        logger.info(f"Price update for {product_id}: {price}")
                        
                        if on_price_update:
                            on_price_update(price)
                            
                        if price > price_threshold:
                            logger.info(f"Price threshold {price_threshold} reached. Closing connection.")
                            await self.ws_client.close()
                            
                except Exception as e:
                    logger.error(f"Error in WebSocket handler: {str(e)}")
            
            # Configure WebSocket channels
            channel_data = [
                Channel(
                    name=ChannelType.TICKER,
                    product_ids=[product_id]
                )
            ]
            
            # Initialize WebSocket client
            self.ws_client = WSClient(
                api_key=self.api_key,
                api_secret=self.api_secret,
                channels=channel_data,
                message_handler=ws_handler
            )
            
            # Start WebSocket connection
            await self.ws_client.start()
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket: {str(e)}")
            raise
            
    def get_accounts(self) -> Dict:
        """Fetch all available trading accounts"""
        try:
            accounts = self.rest_client.get_accounts()
            logger.info(f"Retrieved {len(accounts)} accounts")
            return accounts
        except Exception as e:
            logger.error(f"Failed to fetch accounts: {str(e)}")
            raise
            
    def place_limit_order(self, 
                         product_id: str,
                         side: str,
                         size: Decimal,
                         price: Decimal) -> CreateOrderResponse:
        """
        Place a limit order
        
        Args:
            product_id: Trading pair ID (e.g., 'BTC-USD')
            side: 'buy' or 'sell'
            size: Order size in base currency
            price: Limit price
            
        Returns:
            CreateOrderResponse object containing order details
        """
        try:
            # Convert side string to OrderSide enum
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            # Place the order
            response = self.rest_client.create_order(
                product_id=product_id,
                side=order_side,
                order_type=OrderType.LIMIT,
                base_size=str(size),
                limit_price=str(price),
                time_in_force=TimeInForce.GOOD_TILL_CANCELLED
            )
            
            logger.info(
                f"Placed {side} limit order for {size} {product_id} "
                f"at price {price}: Order ID {response.order_id}"
            )
            return response
            
        except Exception as e:
            logger.error(f"Failed to place limit order: {str(e)}")
            raise
            
    async def close(self):
        """Close WebSocket connection if active"""
        if self.ws_client:
            try:
                await self.ws_client.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {str(e)}") 