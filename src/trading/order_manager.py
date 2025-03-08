from decimal import Decimal
from typing import Optional
import logging
from coinbase_api.client import CoinbaseAdvancedClient, OrderRequest, Account  # Import Account from our client
import time
from os import environ

logger = logging.getLogger(__name__)

class InsufficientBalanceError(Exception):
    pass

class OrderPlacementError(Exception):
    pass

class BalanceManager:
    def __init__(self, client: CoinbaseAdvancedClient):
        self.client = client
        self._accounts_cache = None
        self._last_update = 0
        self._cache_ttl = 60  # Cache accounts for 60 seconds

    def get_balance(self, asset: str) -> Optional[Account]:
        """Get balance for a specific asset with caching"""
        try:
            # Refresh accounts cache if needed or if it's the first time
            current_time = time.time()
            if self._accounts_cache is None or (current_time - self._last_update) > self._cache_ttl:
                logger.info(f"Refreshing accounts cache")
                self._accounts_cache = self.client.get_accounts()
                self._last_update = current_time
            
            # Find ALL accounts matching the currency and choose the one with highest balance
            matching_accounts = []
            for account in self._accounts_cache:
                if account.currency.upper() == asset.upper():
                    matching_accounts.append(account)
                    
            if matching_accounts:
                # Sort by available balance (highest first)
                matching_accounts.sort(key=lambda acc: acc.available_balance, reverse=True)
                best_account = matching_accounts[0]
                
                # Log all matching accounts for debugging
                if len(matching_accounts) > 1:
                    logger.info(f"Found {len(matching_accounts)} accounts for {asset}:")
                    for i, acc in enumerate(matching_accounts):
                        logger.info(f"  - Account {i+1}: Balance = {acc.available_balance}, UUID = {acc.uuid}")
                    logger.info(f"Selected account with highest balance: {best_account.available_balance}")
                else:
                    logger.info(f"Found {asset} account with balance: {best_account.available_balance}")
                    
                return best_account
                    
            # Try refreshing cache and searching again if not found (in case of new accounts)
            if current_time - self._last_update > 1:  # Only if cache is at least 1 second old
                logger.info(f"Account {asset} not found in cache, refreshing")
                self._accounts_cache = self.client.get_accounts()
                self._last_update = current_time
                
                # Repeat the search on refreshed cache
                matching_accounts = []
                for account in self._accounts_cache:
                    if account.currency.upper() == asset.upper():
                        matching_accounts.append(account)
                
                if matching_accounts:
                    # Sort by available balance (highest first)
                    matching_accounts.sort(key=lambda acc: acc.available_balance, reverse=True)
                    best_account = matching_accounts[0]
                    
                    # Log all matching accounts for debugging
                    if len(matching_accounts) > 1:
                        logger.info(f"Found {len(matching_accounts)} accounts for {asset} after refresh:")
                        for i, acc in enumerate(matching_accounts):
                            logger.info(f"  - Account {i+1}: Balance = {acc.available_balance}, UUID = {acc.uuid}")
                        logger.info(f"Selected account with highest balance: {best_account.available_balance}")
                    else:
                        logger.info(f"Found {asset} account with balance: {best_account.available_balance}")
                        
                    return best_account
            
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
        
        # Log order details for debugging
        logger.info(f"Validating order: {order.product_id}, side={order.side}, size={size}")
        
        if order.side.upper() == 'BUY':
            check_asset = quote_asset
            required_amount = size * Decimal(order.limit_price) if order.limit_price else size
        else:
            check_asset = base_asset
            required_amount = size

        # Get all accounts first to log them for debugging
        all_accounts = self.balance_manager.client.get_accounts()
        for acc in all_accounts:
            logger.info(f"Balance check - Account {acc.currency}: available={acc.available_balance}")
            
        # Now get the specific account we need
        account = self.balance_manager.get_balance(check_asset)
        
        logger.info(f"Checking {check_asset} balance: required={required_amount}, " +
                   f"available={account.available_balance if account else 'None'}")
        
        if not account:
            raise InsufficientBalanceError(
                f"No {check_asset} account found"
            )
            
        if account.available_balance < required_amount:
            raise InsufficientBalanceError(
                f"Insufficient {check_asset} balance. "
                f"Required: {required_amount}, "
                f"Available: {account.available_balance}"
            )
        
        logger.info(f"Order validation passed: sufficient {check_asset} balance")

class OrderManager:
    def __init__(self, client: CoinbaseAdvancedClient):
        self.client = client
        self.balance_manager = BalanceManager(client)
        self.validator = OrderValidator(self.balance_manager)

    def place_order(self, order: OrderRequest):
        """Place an order with validation"""
        try:
            # Check if we're running in simulation mode
            # This assumes there's some way to determine if we're in simulation mode
            # You might need to adjust this based on how your app determines simulation mode
            trading_mode = environ.get('TRADING_MODE', 'simulation')
            
            # Only validate balance for live trading, not for simulation
            if trading_mode.lower() == 'live':
                logger.info(f"Live trading mode - validating order balance")
                # Validate order
                self.validator.validate_order(order)
            else:
                logger.info(f"Simulation mode - skipping balance validation")
            
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