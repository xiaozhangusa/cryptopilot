import asyncio
import logging
from decimal import Decimal
import os
import json
from src.coinbase_api.advanced_client import CoinbaseAdvancedSDKClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PRODUCT_ID = 'BTC-USD'
PRICE_THRESHOLD = Decimal('45000.00')  # Adjust based on current market price
ORDER_SIZE = Decimal('0.001')  # Small size for testing

async def main():
    # Load credentials
    try:
        with open('secrets.json') as f:
            secrets = json.load(f)
            api_key = secrets['api_key']
            api_secret = secrets['api_secret']
    except Exception as e:
        logger.error(f"Failed to load credentials: {str(e)}")
        return

    # Initialize client
    client = CoinbaseAdvancedSDKClient(api_key, api_secret)
    
    try:
        # Fetch and display accounts
        accounts = client.get_accounts()
        logger.info("Available accounts:")
        for account in accounts:
            logger.info(f"Account {account.account_id}: {account.available_balance}")
        
        # Place a limit order
        current_price = Decimal('44000.00')  # In practice, fetch this from the API
        limit_price = current_price * Decimal('0.95')  # 5% below current price
        
        order_response = client.place_limit_order(
            product_id=PRODUCT_ID,
            side='buy',
            size=ORDER_SIZE,
            price=limit_price
        )
        logger.info(f"Placed limit order: {order_response.order_id}")
        
        # Define price update callback
        def on_price_update(price: Decimal):
            logger.info(f"Price update callback: {price}")
        
        # Start WebSocket connection
        await client.initialize_websocket(
            product_id=PRODUCT_ID,
            price_threshold=PRICE_THRESHOLD,
            on_price_update=on_price_update
        )
        
        # Keep the script running until WebSocket closes
        while client.ws_client and client.ws_client.is_connected():
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main()) 