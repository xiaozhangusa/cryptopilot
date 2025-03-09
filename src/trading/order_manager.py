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

    def create_smart_limit_order(self, 
                               product_id: str, 
                               side: str, 
                               price_percentage: float = 0.95, 
                               balance_fraction: float = 0.1,
                               time_in_force: str = 'GTC') -> OrderRequest:
        """
        Create a smart limit order using a percentage of market price and fraction of available balance
        
        Args:
            product_id: Trading pair (e.g., 'BTC-USD', 'SOL-USDT')
            side: 'BUY' or 'SELL'
            price_percentage: For buy orders, percentage of current price (e.g., 0.95 for 95%)
                              For sell orders, percentage would be > 1 (e.g., 1.05 for 105%)
            balance_fraction: Fraction of available balance to use (e.g., 0.1 for 10%)
            time_in_force: Order time in force (default: 'GTC')
            
        Returns:
            OrderRequest object ready to be submitted
        """
        # Extract base and quote assets from product ID
        if '-' in product_id:
            base_asset, quote_asset = product_id.split('-')
        else:
            # Handle products without dash (e.g., BTCUSD)
            base_asset = product_id.replace('USD', '').replace('USDT', '')
            quote_asset = 'USD' if 'USD' in product_id else 'USDT'
        
        # Get current market price
        try:
            current_price = self.client.get_product_price(product_id)
            print(f"Current market price for {product_id}: ${current_price:.2f}")
        except Exception as e:
            logger.error(f"Error getting market price: {str(e)}")
            raise ValueError(f"Could not get current price for {product_id}: {str(e)}")
        
        # Get available balance
        if side == 'BUY':
            # For buy orders, we need quote asset balance (e.g., USD)
            asset_account = self.balance_manager.get_balance(quote_asset)
            available_balance = float(asset_account.available_balance) if asset_account else 0
            
            # Calculate order parameters
            limit_price = current_price * price_percentage
            # Amount of quote currency to spend (e.g., USD)
            quote_amount = available_balance * balance_fraction
            # Calculate base amount (e.g., BTC) to buy with the quote amount
            base_amount = quote_amount / limit_price if limit_price > 0 else 0
            
        else:  # SELL
            # For sell orders, we need base asset balance (e.g., BTC)
            asset_account = self.balance_manager.get_balance(base_asset)
            available_balance = float(asset_account.available_balance) if asset_account else 0
            
            # Calculate order parameters
            limit_price = current_price * price_percentage  # For sell, this should be > 1
            # Amount of base currency to sell (e.g., BTC)
            base_amount = available_balance * balance_fraction
            # Calculate quote amount that will be received
            quote_amount = base_amount * limit_price
        
        # Round base amount to 8 decimal places and convert to string
        base_size = '{:.8f}'.format(base_amount).rstrip('0').rstrip('.') if base_amount > 0 else '0'
        # Round limit price appropriately
        limit_price_str = '{:.2f}'.format(limit_price)
        
        # Print the strategy details
        print("\n📊 SMART LIMIT ORDER STRATEGY")
        print(f"Product: {product_id}")
        print(f"Side: {side}")
        print(f"Market Price: ${current_price:.2f}")
        print(f"Price Adjustment: {price_percentage*100:.0f}% of market price")
        print(f"Limit Price: ${limit_price:.2f}")
        print(f"Available {quote_asset if side == 'BUY' else base_asset} Balance: {available_balance:.8f}")
        print(f"Balance Fraction: {balance_fraction*100:.0f}%")
        print(f"Amount to {'Spend' if side == 'BUY' else 'Sell'}: " + 
              (f"${quote_amount:.2f}" if side == 'BUY' else f"{base_amount:.8f} {base_asset}"))
        print(f"Base Size: {base_size} {base_asset}")
        
        # Create the order request
        order = OrderRequest(
            product_id=product_id,
            side=side,
            order_type='LIMIT',
            base_size=base_size,
            limit_price=limit_price_str,
            time_in_force=time_in_force
        )
        
        return order
        
    def place_order(self, order: OrderRequest):
        """Place an order with validation"""
        try:
            # Check if we're running in simulation mode
            # This assumes there's some way to determine if we're in simulation mode
            # You might need to adjust this based on how your app determines simulation mode
            trading_mode = environ.get('TRADING_MODE', 'simulation')
            
            # Parse product ID to get base and quote assets
            if '-' in order.product_id:
                base_asset, quote_asset = order.product_id.split('-')
            else:
                # For product IDs without a dash, assume quote is USD or USDT
                base_asset = order.product_id.replace('USD', '').replace('USDT', '')
                quote_asset = 'USD' if 'USD' in order.product_id else 'USDT'
            
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
            
            # Always validate balance regardless of mode
            logger.info(f"Validating order balance ({trading_mode} mode)")
            
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
                return self.client.create_limit_order(order)
            return self.client.create_market_order(order)
            
        except InsufficientBalanceError as e:
            logger.warning(str(e))
            raise
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            raise OrderPlacementError(str(e)) 