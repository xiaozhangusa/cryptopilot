from typing import Dict, List, Optional
import hmac
import hashlib
import time
import base64
import requests
from dataclasses import dataclass
import json

@dataclass
class OrderRequest:
    product_id: str
    side: str  # 'BUY' or 'SELL'
    order_type: str  # 'MARKET' or 'LIMIT'
    price: Optional[float] = None
    size: Optional[float] = None

class CoinbaseAdvancedClient:
    BASE_URL = "https://api.coinbase.com/api/v3/brokerage"
    
    def __init__(self, api_key: str, api_secret: str, passphrase: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase

    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = '') -> str:
        message = f"{timestamp}{method}{path}{body}"
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(signature.digest()).decode('utf-8')

    def _request(self, method: str, path: str, params: Dict = None, data: Dict = None) -> Dict:
        timestamp = str(int(time.time()))
        url = f"{self.BASE_URL}{path}"
        
        headers = {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }

        body = json.dumps(data) if data else ''
        headers['CB-ACCESS-SIGN'] = self._generate_signature(timestamp, method, path, body)

        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=data
        )
        response.raise_for_status()
        return response.json()

    def get_accounts(self) -> Dict:
        """List all accounts"""
        return self._request('GET', '/accounts')

    def get_account(self, account_id: str) -> Dict:
        """Get specific account details"""
        return self._request('GET', f'/accounts/{account_id}')

    def get_best_bid_ask(self, product_ids: List[str]) -> Dict:
        """Get best bid/ask for specified products"""
        params = {'product_ids': ','.join(product_ids)}
        return self._request('GET', '/best_bid_ask', params=params)

    def get_product_book(self, product_id: str, limit: int = 10) -> Dict:
        """Get order book for a product"""
        params = {
            'product_id': product_id,
            'limit': limit
        }
        return self._request('GET', '/product_book', params=params)

    def get_product_candles(self, 
                          product_id: str, 
                          start: str = None, 
                          end: str = None,
                          granularity: str = '1D') -> Dict:
        """Get historical candles for a product"""
        params = {
            'product_id': product_id,
            'granularity': granularity
        }
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        return self._request('GET', f'/products/{product_id}/candles', params=params)

    def create_order(self, order: OrderRequest) -> Dict:
        """Create a new order"""
        data = {
            'product_id': order.product_id,
            'side': order.side,
            'order_configuration': {
                order.order_type.lower(): {
                    'quote_size': str(order.size) if order.size else None,
                    'base_size': str(order.size) if order.size else None,
                    'limit_price': str(order.price) if order.price else None
                }
            }
        }
        return self._request('POST', '/orders', data=data)

    def get_fills(self, order_id: str = None, product_id: str = None) -> Dict:
        """Get order fills"""
        params = {}
        if order_id:
            params['order_id'] = order_id
        if product_id:
            params['product_id'] = product_id
        return self._request('GET', '/orders/fills', params=params) 