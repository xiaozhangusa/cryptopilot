from decimal import Decimal
from typing import Optional
import logging
from coinbase_api.client import CoinbaseAdvancedClient, OrderRequest, Account  # Import Account from our client

logger = logging.getLogger(__name__)

class InsufficientBalanceError(Exception):
    pass

class OrderPlacementError(Exception):
    pass

class BalanceManager:
    def __init__(self, client: CoinbaseAdvancedClient):
        self.client = client

    def get_balance(self, asset: str) -> Optional[Account]:
        """Get balance for a specific asset"""
        try:
            accounts = self.client.get_accounts()
            for account in accounts:
                if account.currency == asset:
                    return account
            logger.warning(f"No account found for {asset}")
            return None
        except Exception as e:
            logger.error(f"Error getting balance for {asset}: {str(e)}")
            raise

class OrderValidator:
    def __init__(self, balance_manager: BalanceManager):
        self.balance_manager = balance_manager

    def validate_order(self, order: OrderRequest) -> None:
        """Validate order, raise InsufficientBalanceError if invalid"""
        base_asset, quote_asset = order.product_id.split('-')
        size = Decimal(order.base_size or order.quote_size)
        
        if order.side.upper() == 'BUY':
            check_asset = quote_asset
            required_amount = size * Decimal(order.limit_price) if order.limit_price else size
        else:
            check_asset = base_asset
            required_amount = size

        account = self.balance_manager.get_balance(check_asset)
        if not account or account.available_balance < required_amount:
            raise InsufficientBalanceError(
                f"Insufficient {check_asset} balance. "
                f"Required: {required_amount}, "
                f"Available: {account.available_balance if account else 0}"
            )

class OrderManager:
    def __init__(self, client: CoinbaseAdvancedClient):
        self.client = client
        self.balance_manager = BalanceManager(client)
        self.validator = OrderValidator(self.balance_manager)

    def place_order(self, order: OrderRequest):
        """Place an order with validation"""
        try:
            # Validate order
            self.validator.validate_order(order)
            
            # Place order
            if order.order_type == 'LIMIT':
                return self.client.create_limit_order(order)
            return self.client.create_market_order(order)
            
        except InsufficientBalanceError as e:
            logger.warning(str(e))
            raise
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            raise OrderPlacementError(str(e)) 