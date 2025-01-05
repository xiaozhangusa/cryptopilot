import os
import json
import boto3
import logging
from typing import Dict
from ..bot_strategy.strategy import SwingStrategy
from ..coinbase_api.client import CoinbaseAdvancedClient, OrderRequest

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_secrets() -> Dict:
    """Retrieve secrets from AWS Secrets Manager"""
    trading_mode = os.environ.get('TRADING_MODE', 'simulation')
    secret_name = f'trading-bot/{trading_mode}/coinbase-credentials'
    
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

def lambda_handler(event, context):
    try:
        # Get configuration
        trading_mode = os.environ.get('TRADING_MODE', 'simulation')
        secrets = get_secrets()
        
        # Initialize clients
        coinbase_client = CoinbaseAdvancedClient(
            api_key=secrets['api_key'],
            api_secret=secrets['api_secret'],
            passphrase=secrets['passphrase'],
            mode=trading_mode
        )
        
        strategy = SwingStrategy()
        
        # Get market data
        symbol = 'BTC-USD'
        candles = coinbase_client.get_product_candles(symbol)
        prices = [float(candle[4]) for candle in candles]  # Close prices
        
        # Generate trading signal
        signal = strategy.generate_signal(symbol, prices)
        
        if signal:
            logger.info(f"Generated signal: {signal}")
            
            if trading_mode == 'simulation':
                # Log simulated trade
                logger.info(f"Simulation mode: Would execute {signal.action} "
                          f"for {symbol} at {signal.price}")
            else:
                # Execute real trade
                order = OrderRequest(
                    symbol=signal.symbol,
                    side=signal.action.lower(),
                    size=0.01  # Minimum order size, adjust based on your risk management
                )
                response = coinbase_client.place_order(order)
                logger.info(f"Order placed: {response}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Trading bot executed successfully in {trading_mode} mode',
                'signal': signal.__dict__ if signal else None
            })
        }
        
    except Exception as e:
        logger.error(f"Error executing trading bot: {str(e)}")
        raise 