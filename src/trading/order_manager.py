from decimal import Decimal
from typing import Optional
import logging
from coinbase_api.client import CoinbaseAdvancedClient, OrderRequest, Account  # Import Account from our client
import time
from os import environ
import uuid
import os

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
                logger.info(f"Refreshing accounts cache (silent mode)")
                # Get the client's verbosity setting safely
                client_verbose = getattr(self.client, 'verbose', False)
                # Fetch accounts without excessive logging
                self._accounts_cache = self.client.get_accounts()
                self._last_update = current_time
            
            # Find ALL accounts matching the currency and choose the one with highest balance
            matching_accounts = []
            for account in self._accounts_cache:
                if account.currency.upper() == asset.upper():
                    matching_accounts.append(account)
                    
            if matching_accounts:
                # Sort by available balance (highest first)
                matching_accounts.sort(key=lambda acc: float(acc.available_balance), reverse=True)
                best_account = matching_accounts[0]
                
                # Log limited info about matching accounts (only if there are multiple)
                if len(matching_accounts) > 1:
                    logger.debug(f"Found {len(matching_accounts)} accounts for {asset}, using highest balance: {best_account.available_balance}")
                
                return best_account
                    
            # Try refreshing cache and searching again if not found (in case of new accounts)
            if current_time - self._last_update > 1:  # Only if cache is at least 1 second old
                logger.info(f"Account {asset} not found in cache, refreshing")
                # Fetch accounts without excessive logging
                self._accounts_cache = self.client.get_accounts()
                self._last_update = current_time
                
                # Repeat the search on refreshed cache with minimal logging
                matching_accounts = []
                for account in self._accounts_cache:
                    if account.currency.upper() == asset.upper():
                        matching_accounts.append(account)
                
                if matching_accounts:
                    # Sort by available balance (highest first)
                    matching_accounts.sort(key=lambda acc: float(acc.available_balance), reverse=True)
                    best_account = matching_accounts[0]
                    
                    # Use debug level to minimize console output
                    logger.debug(f"Found {asset} account after refresh with balance: {best_account.available_balance}")
                    
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
        """Validate if there is sufficient balance for the order.
        
        Args:
            order: The order to validate
            
        Raises:
            InsufficientBalanceError: If there isn't enough balance to place the order
        """
        # First check: Ensure order sizes are not zero
        base_size = float(order.base_size or 0)
        quote_size = float(order.quote_size or 0)
        
        # Validate non-zero sizes based on order type
        if order.order_type == 'LIMIT' and base_size <= 0:
            raise ValueError(f"Invalid order: base_size must be greater than 0 for LIMIT orders, got {base_size}")
        
        if order.order_type == 'MARKET':
            if order.side == 'BUY' and quote_size <= 0:
                raise ValueError(f"Invalid order: quote_size must be greater than 0 for MARKET BUY orders, got {quote_size}")
            if order.side == 'SELL' and base_size <= 0:
                raise ValueError(f"Invalid order: base_size must be greater than 0 for MARKET SELL orders, got {base_size}")
        
        # Extract base and quote assets from product_id
        base_asset, quote_asset = order.product_id.split('-')
        
        # Only log the relevant accounts for this trading pair
        relevant_assets = [base_asset, quote_asset]
        
        # Get account info only for relevant assets
        base_account = self.balance_manager.get_balance(base_asset)
        quote_account = self.balance_manager.get_balance(quote_asset)
        
        # Log only the relevant account balances
        if base_account:
            logger.info(f"Balance check - Account {base_asset}: available={base_account.available_balance}")
        else:
            logger.info(f"Balance check - Account {base_asset}: not found")
            
        if quote_account:
            logger.info(f"Balance check - Account {quote_asset}: available={quote_account.available_balance}")
        else:
            logger.info(f"Balance check - Account {quote_asset}: not found")
        
        # Continue with validation logic
        if order.side == 'BUY':
            # For buy orders, check quote currency (e.g., USD) balance
            if not quote_account:
                raise InsufficientBalanceError(f"No {quote_asset} account found")
            
            available_balance = float(quote_account.available_balance)
            
            if order.order_type == 'MARKET':
                # For market orders, check quote_size against available balance
                if not order.quote_size:
                    raise ValueError("quote_size must be specified for MARKET BUY orders")
                
                required_balance = float(order.quote_size)
                logger.info(f"Checking {quote_asset} balance: required={required_balance}, available={available_balance}")
                
                if required_balance > available_balance:
                    raise InsufficientBalanceError(
                        f"Insufficient {quote_asset} balance: {available_balance} available, {required_balance} required"
                    )
            else:  # LIMIT
                # For limit buy orders, multiply base_size by limit_price
                if not order.base_size or not order.limit_price:
                    raise ValueError("base_size and limit_price must be specified for LIMIT orders")
                
                required_balance = float(order.base_size) * float(order.limit_price)
                logger.info(f"Checking {quote_asset} balance: required={required_balance}, available={available_balance}")
                
                if required_balance > available_balance:
                    raise InsufficientBalanceError(
                        f"Insufficient {quote_asset} balance: {available_balance} available, {required_balance} required"
                    )
        else:  # SELL
            # For sell orders, check base currency (e.g., BTC) balance
            if not base_account:
                raise InsufficientBalanceError(f"No {base_asset} account found")
            
            available_balance = float(base_account.available_balance)
            
            if not order.base_size:
                raise ValueError("base_size must be specified for SELL orders")
            
            required_balance = float(order.base_size)
            logger.info(f"Checking {base_asset} balance: required={required_balance}, available={available_balance}")
            
            if required_balance > available_balance:
                raise InsufficientBalanceError(
                    f"Insufficient {base_asset} balance: {available_balance} available, {required_balance} required"
                )

class OrderManager:
    def __init__(self, client: CoinbaseAdvancedClient):
        self.client = client
        self.balance_manager = BalanceManager(client)
        self.validator = OrderValidator(self.balance_manager)

    def create_smart_limit_order(self, 
                             product_id: str, 
                             side: str, 
                             price_percentage: float = None, 
                             balance_fraction: float = 0.1,
                             time_in_force: str = 'GTC') -> OrderRequest:
        """Create a limit order with a smart price and size strategy.
        
        Args:
            product_id: The trading pair (e.g., 'BTC-USD')
            side: 'BUY' or 'SELL'
            price_percentage: For buy orders, percentage of current price (e.g., 0.95 = 95% of current price)
                             For sell orders, percentage of current price (e.g., 1.05 = 105% of current price)
            balance_fraction: Fraction of available balance to use (e.g., 0.1 = 10% of available balance)
            time_in_force: Order time-in-force (GTC, GTD, IOC, FOK)
            
        Returns:
            OrderRequest object for the limit order
        """
        # Get base and quote assets from the product_id
        base_asset, quote_asset = product_id.split('-')
        
        # Get current market price
        current_price = self.client.get_product_price(product_id)
        print(f"Current market price for {product_id}: ${current_price}")
        
        # Set default price percentage if not provided
        if price_percentage is None:
            price_percentage = 0.95 if side == 'BUY' else 1.05
        
        # Calculate limit price as a percentage of current price
        limit_price = current_price * price_percentage
        
        # Different logic for buy vs sell orders
        if side == 'BUY':
            # For buy orders, use a percentage of available quote balance
            asset_account = self.balance_manager.get_balance(quote_asset)
            available_balance = float(asset_account.available_balance) if asset_account else 0
            
            # Calculate order parameters
            # Amount of quote currency to spend (e.g., USD)
            quote_amount = available_balance * balance_fraction
            # Calculate base amount (e.g., BTC) to buy with the quote amount
            base_amount = quote_amount / limit_price if limit_price > 0 else 0
            
            # Order strategy description
            strategy_description = f"Using {balance_fraction*100:.0f}% of available {quote_asset} balance ({available_balance:.2f})"
            
        else:  # SELL
            # For sell orders, try to get the last buy order
            last_buy_order = None
            last_buy_size = None
            try:
                last_buy_order = self.client.get_last_filled_buy_order(product_id)
                if last_buy_order and hasattr(last_buy_order, 'filled_size'):
                    last_buy_size = float(last_buy_order.filled_size)
                    print(f"Last buy order size: {last_buy_size} {base_asset}")
            except Exception as e:
                logger.warning(f"Could not get last filled buy order: {str(e)}")
            
            # Get current balance of base asset
            asset_account = self.balance_manager.get_balance(base_asset)
            available_balance = float(asset_account.available_balance) if asset_account else 0
            
            # Determine base amount to sell based on the requested strategy:
            # min(last buy order size, available balance)
            if last_buy_size is not None:
                base_amount = min(available_balance, last_buy_size) * balance_fraction
                strategy_description = (
                    f"Using min(last buy order size, available balance) * balance_fraction\n"
                    f"= min({last_buy_size:.8f}, {available_balance:.8f}) * {balance_fraction:.2f}\n"
                    f"= {base_amount:.8f} {base_asset}"
                )
            else:
                # Just use available balance
                base_amount = available_balance * balance_fraction
                strategy_description = (
                    f"Using {balance_fraction*100:.0f}% of available {base_asset} balance ({available_balance:.8f})\n"
                    f"= {base_amount:.8f} {base_asset}"
                )
            
            # Check if we have enough to sell
            if base_amount <= 0:
                logger.warning(f"Cannot create SELL order: insufficient {base_asset} balance ({available_balance:.8f})")
                return None
        
        # Round to 8 decimal places for crypto amounts
        base_amount = round(base_amount, 8)
        
        # Format limit price to 2 decimal places for USD
        formatted_limit_price = "{:.2f}".format(limit_price)
        
        # Print order strategy details
        print(f"\n💡 {side} Order Strategy for {product_id}:")
        print(f"  ● Market price: ${current_price:.2f}")
        print(f"  ● Limit price: ${limit_price:.2f} ({price_percentage*100:.0f}% of market price)")
        print(f"  ● Available balance: {available_balance:.8f} {base_asset if side == 'SELL' else quote_asset}")
        print(f"  ● {strategy_description}")
        
        # Create order request
        order = OrderRequest(
            product_id=product_id,
            side=side,
            order_type='LIMIT',
            base_size=str(base_amount),
            limit_price=formatted_limit_price,
            time_in_force=time_in_force
        )
        
        return order
        
    def place_order(self, order: OrderRequest):
        """Place an order using the client API

        Args:
            order: Order request parameters

        Returns:
            The response from the API
        """
        # Always validate balance regardless of mode
        trading_mode = os.environ.get('TRADING_MODE', 'simulation')
        logger.info(f"Validating balance for {order.product_id} order in {trading_mode} mode")
        
        # Log order details for debugging
        logger.info(f"Order details: {order.product_id}, side={order.side}, size={order.base_size}")
        
        try:
            # Show projected balances after order fill
            base_asset, quote_asset = order.product_id.split('-')
            base_size_float = float(order.base_size) if order.base_size else 0
            limit_price_float = float(order.limit_price) if order.limit_price else 0
            quote_size_float = base_size_float * limit_price_float if order.order_type == 'LIMIT' else float(order.quote_size or 0)
            
            # Get current balances for both assets
            base_account = self.balance_manager.get_balance(base_asset)
            quote_account = self.balance_manager.get_balance(quote_asset)
            
            # Current balances
            base_balance = float(base_account.available_balance) if base_account else 0
            quote_balance = float(quote_account.available_balance) if quote_account else 0
            
            # Determine order cost/profit based on order type
            price = 0
            base_amount = 0
            quote_amount = 0
            
            # Get current market price for calculations if needed
            try:
                current_price = self.client.get_product_price(order.product_id)
            except:
                current_price = float(order.limit_price) if order.limit_price else 0
            
            # Calculate amounts involved in the trade
            if order.order_type == 'MARKET':
                # For market orders
                if order.side == 'BUY':
                    # For market buy, we know the quote amount (e.g. $100 of BTC)
                    quote_amount = float(order.quote_size) if order.quote_size else 0
                    # Estimate base amount (approximate, as actual execution price may vary)
                    base_amount = quote_amount / current_price if current_price > 0 else 0
                else:  # SELL
                    # For market sell, we know the base amount (e.g. 0.01 BTC)
                    base_amount = float(order.base_size) if order.base_size else 0
                    # Estimate quote amount
                    quote_amount = base_amount * current_price
                price = current_price
            else:  # LIMIT
                # For limit orders, price is specified
                price = float(order.limit_price) if order.limit_price else 0
                base_amount = float(order.base_size) if order.base_size else 0
                quote_amount = base_amount * price
            
            # Calculate projected balances after order fill
            projected_base_balance = 0
            projected_quote_balance = 0
            
            if order.side == 'BUY':
                projected_base_balance = base_balance + base_amount
                projected_quote_balance = quote_balance - quote_amount
            else:  # SELL
                projected_base_balance = base_balance - base_amount
                projected_quote_balance = quote_balance + quote_amount
            
            # Display balance information
            print("\n📊 ORDER BALANCE ANALYSIS")
            print(f"Order: {order.side} {order.order_type} {order.product_id}")
            
            if order.order_type == 'LIMIT':
                print(f"Amount: {base_amount} {base_asset} @ ${price:.2f}")
            else:  # MARKET
                if order.side == 'BUY':
                    print(f"Amount: ${quote_amount:.2f} worth of {base_asset} (est. {base_amount:.8f} {base_asset})")
                else:  # SELL
                    print(f"Amount: {base_amount} {base_asset} (est. ${quote_amount:.2f})")
            
            print("\nCurrent Balances:")
            print(f"  {base_asset}: {base_balance:.8f}")
            print(f"  {quote_asset}: ${quote_balance:.2f}")
            
            print("\nOrder Impact:")
            if order.side == 'BUY':
                print(f"  Cost: ${quote_amount:.2f}")
                print(f"  Gain: +{base_amount:.8f} {base_asset}")
            else:  # SELL
                print(f"  Cost: {base_amount:.8f} {base_asset}")
                print(f"  Gain: +${quote_amount:.2f}")
            
            print("\nProjected Balances After Fill:")
            print(f"  {base_asset}: {projected_base_balance:.8f}")
            print(f"  {quote_asset}: ${projected_quote_balance:.2f}")
            print("="*50)
            
            # Perform validation
            try:
                self.validator.validate_order(order)
                logger.info("Balance validation passed ✅")
            except InsufficientBalanceError as e:
                # In simulation mode, we'll log the warning but still allow the order
                if trading_mode.lower() == 'simulation':
                    logger.warning(f"Balance validation failed, but continuing in simulation mode: {str(e)}")
                else:
                    # In live mode, raise the error to prevent the order
                    logger.error(f"Balance validation failed in live mode: {str(e)}")
                    raise
            
            # Place order
            if order.order_type == 'LIMIT':
                response = self.client.create_limit_order(order)
                # Debug log the response structure
                logger.debug(f"RESPONSE TYPE: {type(response)}")
                logger.debug(f"RESPONSE STRUCTURE: {response}")
                if isinstance(response, dict):
                    logger.debug(f"RESPONSE KEYS: {list(response.keys())}")
                    if 'success_response' in response:
                        logger.debug(f"SUCCESS_RESPONSE STRUCTURE: {response['success_response']}")
                    if 'error_response' in response:
                        logger.debug(f"ERROR_RESPONSE STRUCTURE: {response['error_response']}")
                
                # Modify the response for simulation mode to make it easier to work with
                if trading_mode.lower() == 'simulation' and isinstance(response, dict) and response.get('success'):
                    # Create a simpler response object with just the order_id
                    from types import SimpleNamespace
                    simple_response = SimpleNamespace()
                    
                    # Extract order_id from success_response if available
                    if 'success_response' in response and 'order_id' in response['success_response']:
                        order_id = response['success_response']['order_id']
                        simple_response.order_id = order_id
                        logger.debug(f"SIMULATION MODE: Creating simplified response with order_id: {order_id}")
                    else:
                        simulated_id = "simulated-order-" + str(uuid.uuid4())
                        simple_response.order_id = simulated_id
                        logger.debug(f"SIMULATION MODE: Creating simplified response with generated order_id: {simulated_id}")
                    
                    # Debug output to verify the simple_response object
                    logger.debug(f"SIMPLIFIED RESPONSE TYPE: {type(simple_response)}")
                    logger.debug(f"SIMPLIFIED RESPONSE ATTRIBUTES: {dir(simple_response)}")
                    logger.debug(f"HAS order_id ATTRIBUTE: {hasattr(simple_response, 'order_id')}")
                    if hasattr(simple_response, 'order_id'):
                        logger.debug(f"order_id VALUE: {simple_response.order_id}")
                    
                    logger.info(f"Transformed API response to simple object for simulation mode")
                    return simple_response
                
                # Add debug for non-simulation or unsuccessful response
                logger.debug(f"Returning original response: simulation_mode={trading_mode.lower()=='simulation'}, success={response.get('success', False) if isinstance(response, dict) else 'N/A'}")
                return response
            return self.client.create_market_order(order)
            
        except InsufficientBalanceError as e:
            logger.warning(str(e))
            raise
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            raise OrderPlacementError(str(e)) 