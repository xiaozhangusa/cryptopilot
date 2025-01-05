from typing import Dict, List, Optional
import hmac
import hashlib
import time
import base64
import requests
from dataclasses import dataclass

@dataclass
class OrderRequest:
    symbol: str
    side: str
    size: float
    price: Optional[float] = None
    type: str = 'market'

class CoinbaseAdvancedClient:
    def __init__(self, api_key: str, api_secret: str, passphrase: str, mode: str = 'simulation'):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.mode = mode
        
        # Use sandbox URLs for simulation mode
        self.base_url = ('https://api-public.sandbox.exchange.coinbase.com' 
                        if mode == 'simulation' 
                        else 'https://api.exchange.coinbase.com')

    def _generate_signature(self, timestamp: str, method: str, 
                          request_path: str, body: str = '') -> str:
        message = f'{timestamp}{method}{request_path}{body}'
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message.encode('ascii'),
            hashlib.sha256
        )
        return base64.b64encode(signature.digest()).decode('utf-8')

    def _request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        timestamp = str(int(time.time()))
        url = f'{self.base_url}{endpoint}'
        
        headers = {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': self._generate_signature(timestamp, method, endpoint),
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }

        response = requests.request(method, url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def get_product_candles(self, symbol: str, 
                           granularity: int = 3600) -> List[Dict]:
        """Get historical candles for a product"""
        endpoint = f'/products/{symbol}/candles'
        return self._request('GET', endpoint)

    def place_order(self, order: OrderRequest) -> Dict:
        """Place an order"""
        endpoint = '/orders'
        data = {
            'product_id': order.symbol,
            'side': order.side,
            'size': str(order.size),
            'type': order.type
        }
        
        if order.price:
            data['price'] = str(order.price)

        return self._request('POST', endpoint, data)

    def get_account(self) -> Dict:
        """Get account information"""
        endpoint = '/accounts'
        return self._request('GET', endpoint) 