from decimal import Decimal
from typing import Optional, Dict, Tuple, List
import logging
from coinbase_api.client import CoinbaseAdvancedClient, OrderRequest, Account  # Import Account from our client
import time
from os import environ
import uuid
import os
from datetime import datetime
from bot_strategy.timeframes import Timeframe

logger = logging.getLogger(__name__)

class InsufficientBalanceError(Exception):
    pass

class OrderPlacementError(Exception):
    pass

class OrderCooldownError(Exception):
    """Raised when an order is attempted during its cooldown period"""
    pass

class PriceNotProfitableError(Exception):
    """Raised when selling at a price that is below the average purchase price"""
    pass

class NoMatchingBuyOrdersError(Exception):
    """Raised when no matching buy orders can be found for FIFO selling"""
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

class FillData:
    """Class to store information about a filled order"""
    def __init__(self, order_id: str, product_id: str, side: str, size: float, price: float, 
                 created_at: datetime, fees: float = 0.0):
        self.order_id = order_id
        self.product_id = product_id
        self.side = side
        self.size = size
        self.price = price
        self.created_at = created_at
        self.fees = fees
        self.cost_basis = price * size + fees  # Total cost including fees
        
    def __repr__(self):
        return f"Fill({self.side} {self.size} @ {self.price}, created={self.created_at.strftime('%Y-%m-%d %H:%M:%S')})"

class OrderStack:
    """Stack-based data structure to track buy orders for LIFO matching with sell orders"""
    def __init__(self):
        # Dict mapping product_id to a list of FillData ordered with newest at the top (LIFO stack)
        self.buy_orders: Dict[str, List[FillData]] = {}
        # Track when the stack was last refreshed
        self.last_refresh: Dict[str, datetime] = {}
        # TTL for refreshing from API (5 minutes)
        self.refresh_ttl = 300
        
    def refresh_from_api(self, product_id: str, client: CoinbaseAdvancedClient) -> bool:
        """Refresh the buy order stack from the API"""
        try:
            current_time = datetime.now()
            
            # Skip refresh if we did it recently
            if (product_id in self.last_refresh and 
                (current_time - self.last_refresh[product_id]).total_seconds() < self.refresh_ttl):
                return False
                
            logger.info(f"Refreshing order stack for {product_id} from API")
            
            # Get filled orders from the API
            filled_orders = client.get_filled_orders(product_id=product_id, limit=50)
            
            # Convert to FillData objects
            fill_data_list = []
            for order in filled_orders:
                side = getattr(order, 'side', '').upper()
                if side != 'BUY':
                    continue  # Skip non-buy orders
                    
                size = float(getattr(order, 'filled_size', 0))
                price = float(getattr(order, 'price', 0))
                created_at = getattr(order, 'created_at', current_time)
                if isinstance(created_at, str):
                    try:
                        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    except:
                        created_at = current_time
                order_id = getattr(order, 'order_id', '')
                fees = float(getattr(order, 'fees', 0))
                
                fill_data = FillData(
                    order_id=order_id,
                    product_id=product_id,
                    side=side,
                    size=size,
                    price=price,
                    created_at=created_at,
                    fees=fees
                )
                fill_data_list.append(fill_data)
            
            # Find the most recent sell order as cutoff
            cutoff_time = None
            for order in filled_orders:
                if getattr(order, 'side', '').upper() == 'SELL':
                    cutoff_time = getattr(order, 'created_at', None)
                    if isinstance(cutoff_time, str):
                        try:
                            cutoff_time = datetime.fromisoformat(cutoff_time.replace('Z', '+00:00'))
                        except:
                            cutoff_time = None
                    break
            
            # Filter buy orders to only include those after the most recent sell
            active_buy_orders = []
            for order in fill_data_list:
                if cutoff_time and order.created_at <= cutoff_time:
                    continue
                active_buy_orders.append(order)
            
            # Sort newest first (LIFO - Last In, First Out)
            # This ensures the most recent buy order is at the top of the stack
            active_buy_orders.sort(key=lambda x: x.created_at, reverse=True)
            
            # Update the stack
            self.buy_orders[product_id] = active_buy_orders
            self.last_refresh[product_id] = current_time
            
            logger.info(f"Refreshed {product_id} order stack with {len(active_buy_orders)} active buy orders")
            return True
        
        except Exception as e:
            logger.error(f"Error refreshing order stack: {str(e)}")
            return False
    
    def push_buy_order(self, fill_data: FillData) -> None:
        """Add a new buy order to the top of the stack (LIFO - most recent first)"""
        product_id = fill_data.product_id
        if product_id not in self.buy_orders:
            self.buy_orders[product_id] = []
        
        # Insert at the beginning (top of stack - LIFO)
        self.buy_orders[product_id].insert(0, fill_data)
        logger.info(f"Added buy order to stack: {fill_data}")
    
    def peek_latest_buy(self, product_id: str) -> Optional[FillData]:
        """Look at the most recent buy order (top of stack) without removing it"""
        if product_id not in self.buy_orders or not self.buy_orders[product_id]:
            return None
        
        # Return the first item (top of stack - most recent)
        return self.buy_orders[product_id][0]
    
    def pop_latest_buy(self, product_id: str) -> Optional[FillData]:
        """Remove and return the most recent buy order (top of stack)"""
        if product_id not in self.buy_orders or not self.buy_orders[product_id]:
            return None
        
        # Pop the first item (top of stack - most recent)
        return self.buy_orders[product_id].pop(0)
    
    def is_sell_profitable(self, product_id: str, sell_price: float) -> bool:
        """Check if selling at the given price would be profitable compared to the latest buy"""
        latest_buy = self.peek_latest_buy(product_id)
        if not latest_buy:
            return False
        
        # Require some minimum profit (e.g., 0.5% to cover fees)
        min_profit_pct = 0.005
        min_profitable_price = latest_buy.price * (1 + min_profit_pct)
        
        return sell_price > min_profitable_price
    
    def get_stack_size(self, product_id: str) -> int:
        """Get the number of buy orders in the stack for a product"""
        if product_id not in self.buy_orders:
            return 0
        return len(self.buy_orders[product_id])

class OrderManager:
    def __init__(self, client: CoinbaseAdvancedClient):
        self.client = client
        self.balance_manager = BalanceManager(client)
        self.validator = OrderValidator(self.balance_manager)
        
        # Order cooldown tracking
        self._last_order_times: Dict[Tuple[str, str], datetime] = {}  # (product_id, side) -> last_order_time
        
        # Cache for filled orders to reduce API calls
        self._filled_orders_cache: Dict[str, List[FillData]] = {}  # product_id -> list of FillData
        self._filled_orders_last_update: Dict[str, datetime] = {}  # product_id -> last update time
        self._filled_orders_cache_ttl = 300  # Cache filled orders for 5 minutes (300 seconds)
        
        # Track the last buy order index that was sold (for FIFO selling)
        self._last_matched_buy_index: Dict[str, int] = {}  # product_id -> index

        # Initialize order stack
        self.order_stack = OrderStack()

    def get_order_cooldown_period(self, timeframe: Timeframe, order_side: str) -> int:
        """
        Calculate the cooldown period between order placements based on timeframe.
        Industry practice typically suggests waiting at least 1-3 candles between trades.
        
        Args:
            timeframe: The trading timeframe
            order_side: 'BUY' or 'SELL' - we allow shorter cooldowns for exits (SELL)
            
        Returns:
            Cooldown period in seconds
        """
        # Minutes in each timeframe
        minutes = timeframe.minutes
        
        # Default to 2-3 candle periods for BUY orders (entries)
        if order_side == 'BUY':
            # For smaller timeframes, wait longer in terms of candle count
            # For larger timeframes, wait fewer candles (but still longer in absolute time)
            if minutes <= 5:  # 1min or 5min
                return minutes * 60 * 3  # Wait 3 candles
            elif minutes <= 30:  # 15min or 30min
                return minutes * 60 * 2  # Wait 2 candles
            else:  # 1h and above
                return minutes * 60 * 1.5  # Wait 1.5 candles
        else:  # SELL orders (exits) - allow shorter cooldowns
            # For exits, we generally want to be more responsive, so shorter cooldowns
            if minutes <= 5:  # 1min or 5min
                return minutes * 60 * 1.5  # Wait 1.5 candles
            elif minutes <= 30:  # 15min or 30min
                return minutes * 60 * 1  # Wait 1 candle
            else:  # 1h and above
                return minutes * 60 * 0.75  # Wait 0.75 candles
                
    def can_place_order(self, product_id: str, side: str, timeframe: Timeframe, bypass_cooldown: bool = False) -> bool:
        """
        Determine if enough time has elapsed since the last order placement to allow a new order.
        
        Args:
            product_id: The trading pair (e.g. 'BTC-USD')
            side: 'BUY' or 'SELL'
            timeframe: Current trading timeframe
            bypass_cooldown: If True, skip the cooldown check (for manual orders or other special cases)
            
        Returns:
            True if order can be placed, False if still in cooldown
        """
        if bypass_cooldown:
            return True
            
        order_key = (product_id, side)
        last_order_time = self._last_order_times.get(order_key)
        
        # If no previous order for this product and side, allow placing
        if not last_order_time:
            return True
            
        # Calculate cooldown period based on timeframe and order side
        cooldown_period = self.get_order_cooldown_period(timeframe, side)
        
        # Check if enough time has elapsed
        current_time = datetime.now()
        time_elapsed = (current_time - last_order_time).total_seconds()
        can_place = time_elapsed >= cooldown_period
        
        # If still in cooldown, log details about remaining wait time
        if not can_place:
            remaining_time = cooldown_period - time_elapsed
            remaining_minutes = int(remaining_time // 60)
            remaining_seconds = int(remaining_time % 60)
            logger.info(f"⏳ Order cooldown active for {product_id} {side} orders - {remaining_minutes}m {remaining_seconds}s remaining")
            logger.info(f"Last {side} order placed at {last_order_time.strftime('%H:%M:%S')}, " 
                      f"cooldown period: {cooldown_period//60} minutes")
        
        return can_place
        
    def update_last_order_time(self, product_id: str, side: str):
        """
        Update the last order time for the specified product and side.
        
        Args:
            product_id: The trading pair (e.g. 'BTC-USD')
            side: 'BUY' or 'SELL'
        """
        order_key = (product_id, side)
        current_time = datetime.now()
        self._last_order_times[order_key] = current_time
        logger.info(f"✅ Updated last {product_id} {side} order time to {current_time.strftime('%H:%M:%S')}")

    def get_filled_orders(self, product_id: str, limit: int = 50) -> List[FillData]:
        """
        Get a list of filled orders for a specific product, with caching
        
        Args:
            product_id: Trading pair (e.g., 'BTC-USD')
            limit: Maximum number of orders to retrieve
            
        Returns:
            List of FillData objects, with most recent first
        """
        current_time = datetime.now()
        
        # Check if we have cached data that's still fresh
        if (product_id in self._filled_orders_cache and 
            product_id in self._filled_orders_last_update and
            (current_time - self._filled_orders_last_update[product_id]).total_seconds() < self._filled_orders_cache_ttl):
            return self._filled_orders_cache[product_id]
        
        # Otherwise, fetch from API
        logger.info(f"Fetching filled orders for {product_id}...")
        
        try:
            # This assumes the client has a method to get filled orders
            # You may need to adjust based on the actual client API
            filled_orders = self.client.get_filled_orders(product_id=product_id, limit=limit)
            
            # Convert to FillData objects
            fill_data_list = []
            for order in filled_orders:
                # Adjust these based on the actual structure returned by the API
                side = getattr(order, 'side', '').upper()
                size = float(getattr(order, 'filled_size', 0))
                price = float(getattr(order, 'price', 0))
                created_at = getattr(order, 'created_at', current_time)
                if isinstance(created_at, str):
                    # Parse date string if needed
                    try:
                        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    except:
                        created_at = current_time
                order_id = getattr(order, 'order_id', '')
                fees = float(getattr(order, 'fees', 0))
                
                fill_data = FillData(
                    order_id=order_id,
                    product_id=product_id,
                    side=side,
                    size=size,
                    price=price,
                    created_at=created_at,
                    fees=fees
                )
                fill_data_list.append(fill_data)
            
            # Sort by creation time (newest first)
            fill_data_list.sort(key=lambda x: x.created_at, reverse=True)
            
            # Cache the results
            self._filled_orders_cache[product_id] = fill_data_list
            self._filled_orders_last_update[product_id] = current_time
            
            return fill_data_list
            
        except Exception as e:
            logger.error(f"Error fetching filled orders for {product_id}: {str(e)}")
            
            # Return empty list or cached data if available
            if product_id in self._filled_orders_cache:
                logger.warning(f"Using cached filled orders due to error")
                return self._filled_orders_cache[product_id]
            return []

    def get_average_purchase_price(self, product_id: str) -> float:
        """
        Calculate the average purchase price for a given product using filled buy orders
        
        Args:
            product_id: Trading pair (e.g., 'BTC-USD')
            
        Returns:
            Average purchase price or 0 if no buy orders found
        """
        filled_orders = self.get_filled_orders(product_id)
        
        # Find the most recent sell order first (as a cut-off point)
        cutoff_time = None
        for order in filled_orders:
            if order.side == 'SELL':
                cutoff_time = order.created_at
                break
        
        # Process buy orders (possibly limited by cutoff)
        total_buy_size = 0.0
        total_cost = 0.0
        
        for order in filled_orders:
            # Skip sell orders
            if order.side != 'BUY':
                continue
                
            # Stop if we hit the cutoff time
            if cutoff_time and order.created_at <= cutoff_time:
                break
                
            # Add to totals
            total_buy_size += order.size
            total_cost += order.cost_basis
        
        # Calculate average price
        if total_buy_size > 0:
            return total_cost / total_buy_size
        return 0.0
    
    def get_next_fifo_buy_order(self, product_id: str, current_price: float) -> Optional[Tuple[float, float, datetime]]:
        """
        Get the next buy order to sell, using FIFO approach (oldest first)
        
        Args:
            product_id: Trading pair (e.g., 'BTC-USD')
            current_price: Current market price
            
        Returns:
            Tuple of (size, price, buy_date) or None if no suitable order found
            
        Raises:
            PriceNotProfitableError: If current price is below average purchase price
        """
        # Get average purchase price for profit check
        avg_price = self.get_average_purchase_price(product_id)
        
        if avg_price <= 0:
            logger.warning(f"No buy orders found for {product_id}, can't determine average price")
            return None
            
        if current_price < avg_price:
            raise PriceNotProfitableError(
                f"Current price (${current_price:.2f}) is below average purchase price (${avg_price:.2f})"
            )
        
        # Get filled orders
        filled_orders = self.get_filled_orders(product_id)
        
        # Find the most recent sell order as cutoff
        cutoff_time = None
        for order in filled_orders:
            if order.side == 'SELL':
                cutoff_time = order.created_at
                break
        
        # Extract all buy orders since the cutoff
        buy_orders = []
        for order in filled_orders:
            # Only process buy orders
            if order.side != 'BUY':
                continue
                
            # Stop if we hit the cutoff time
            if cutoff_time and order.created_at <= cutoff_time:
                break
                
            # Add to our list of potential orders
            buy_orders.append(order)
        
        # Sort buy orders by date (oldest first for FIFO) - note our filled_orders are newest first
        buy_orders.reverse()
        
        # Get the index of the last matched buy order for this product
        last_matched_index = self._last_matched_buy_index.get(product_id, -1)
        
        # Find the next buy order to match
        if last_matched_index + 1 < len(buy_orders):
            # We have a next buy order to match
            next_order = buy_orders[last_matched_index + 1]
            
            # Update the last matched index
            self._last_matched_buy_index[product_id] = last_matched_index + 1
            
            # Return the details needed for creating a matching sell order
            return (next_order.size, next_order.price, next_order.created_at)
        
        # Reset the index if we've reached the end of all available buy orders
        # This allows future sell signals to start over with the oldest unfilled buy
        if len(buy_orders) > 0:
            logger.info(f"All {len(buy_orders)} buy orders have been matched for {product_id}, resetting FIFO index")
            self._last_matched_buy_index[product_id] = -1
            
        # No more buy orders to match
        return None
        
    def create_smart_limit_order(self, 
                             product_id: str, 
                             side: str, 
                             price_percentage: float = None, 
                             balance_fraction: float = 0.1,
                             time_in_force: str = 'GTC') -> Optional[OrderRequest]:
        """Create a limit order with a smart price and size strategy.
        
        Args:
            product_id: The trading pair (e.g., 'BTC-USD')
            side: 'BUY' or 'SELL'
            price_percentage: For buy orders, percentage of current price (e.g., 0.95 = 95% of current price)
                             For sell orders, percentage of current price (e.g., 1.05 = 105% of current price)
            balance_fraction: Fraction of available balance to use (e.g., 0.1 = 10% of available balance)
            time_in_force: Order time-in-force (GTC, GTD, IOC, FOK)
            
        Returns:
            OrderRequest object for the limit order or None if no order can be created
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
            # Refresh the order stack from the API if needed
            self.order_stack.refresh_from_api(product_id, self.client)
            
            # Use stack-based LIFO approach for SELL orders
            # This matches the most recent buy order (Last In, First Out)
            latest_buy = self.order_stack.peek_latest_buy(product_id)
            
            if latest_buy:
                # Check if selling at current price would be profitable
                if not self.order_stack.is_sell_profitable(product_id, limit_price):
                    logger.warning(
                        f"Not creating sell order: Current price (${limit_price:.2f}) is not profitable "
                        f"compared to latest buy price (${latest_buy.price:.2f})"
                    )
                    print(f"\n⚠️ Price not profitable for selling {product_id}:")
                    print(f"  Current price: ${current_price:.2f}")
                    print(f"  Limit price: ${limit_price:.2f}")
                    print(f"  Latest buy price: ${latest_buy.price:.2f}")
                    return None
                
                # Use the exact size from the buy order
                base_amount = latest_buy.size
                
                # Apply price percentage to the current price
                adjusted_price = current_price * price_percentage
                
                # Format price to 2 decimal places for USD
                formatted_price = "{:.2f}".format(adjusted_price)
                
                # Calculate profit percentage
                profit_pct = (adjusted_price / latest_buy.price - 1) * 100
                
                print(f"\n📊 Creating LIFO-matched sell order for {product_id}:")
                print(f"  ● Matching the most recent buy order from {latest_buy.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  ● Order size: {base_amount:.8f} {base_asset}")
                print(f"  ● Buy price: ${latest_buy.price:.2f}")
                print(f"  ● Sell price: ${adjusted_price:.2f}")
                print(f"  ● Profit margin: {profit_pct:.2f}%")
                
                # Create order request
                order = OrderRequest(
                    product_id=product_id,
                    side='SELL',
                    order_type='LIMIT',
                    base_size=str(base_amount),
                    limit_price=formatted_price,
                    time_in_force=time_in_force
                )
                return order
            else:
                # FALLBACK: No buy orders in stack, but check if there's an actual balance we can use
                asset_account = self.balance_manager.get_balance(base_asset)
                available_balance = float(asset_account.available_balance) if asset_account else 0
                
                if available_balance > 0:
                    # We have a balance even though our stack is empty, so we can sell some of it
                    base_amount = available_balance * balance_fraction
                    
                    if base_amount <= 0:
                        logger.warning(f"Cannot create SELL order: insufficient {base_asset} balance ({available_balance:.8f})")
                        return None
                    
                    # Apply price percentage to the current price
                    adjusted_price = current_price * price_percentage
                    formatted_price = "{:.2f}".format(adjusted_price)
                    
                    print(f"\n📊 Creating SELL order from account balance (stack empty):")
                    print(f"  ● Available balance: {available_balance:.8f} {base_asset}")
                    print(f"  ● Order size: {base_amount:.8f} {base_asset} ({balance_fraction*100:.0f}% of balance)")
                    print(f"  ● Market price: ${current_price:.2f}")
                    print(f"  ● Sell price: ${adjusted_price:.2f}")
                    print(f"  ⚠️ Note: Selling from account balance since no buy orders in stack")
                    
                    # Create order request
                    order = OrderRequest(
                        product_id=product_id,
                        side='SELL',
                        order_type='LIMIT',
                        base_size=str(base_amount),
                        limit_price=formatted_price,
                        time_in_force=time_in_force
                    )
                    return order
                else:
                    # No buy orders in stack AND no balance - truly can't sell anything
                    logger.warning(f"No buy orders in stack for {product_id} and no balance available - skipping sell signal")
                    print(f"\n⚠️ Cannot create sell order for {product_id}:")
                    print(f"  ● No matching buy orders in stack")
                    print(f"  ● No {base_asset} balance available in account")
                    print(f"  ● Skipping this sell signal and waiting for the next opportunity")
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
        
    def place_order(self, order: OrderRequest, timeframe: Optional[Timeframe] = None, bypass_cooldown: bool = False):
        """Place an order using the client API, with cooldown checks and update order stack for successful orders
        
        Args:
            order: Order request parameters
            timeframe: Current trading timeframe (needed for cooldown checks)
            bypass_cooldown: If True, skip the cooldown check

        Returns:
            The response from the API
            
        Raises:
            OrderCooldownError: If the order is attempted during its cooldown period
            InsufficientBalanceError: If there isn't enough balance
            OrderPlacementError: For other errors during order placement
        """
        # Check for cooldown if timeframe is provided
        if timeframe and not bypass_cooldown:
            if not self.can_place_order(order.product_id, order.side, timeframe):
                raise OrderCooldownError(f"Order cooldown active for {order.product_id} {order.side}")
        
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
            
            # Validate the order
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
                
                # Check if the order was successful
                order_successful = False
                if isinstance(response, dict) and response.get('success'):
                    order_successful = True
                elif hasattr(response, 'order_id'):
                    order_successful = True
                
                # Update the last order time if the order was successful
                if order_successful:
                    self.update_last_order_time(order.product_id, order.side)
                    
                    # Handle stack updates for successful orders
                    if order.side == 'BUY':
                        # Add the buy order to the top of the stack (LIFO)
                        current_time = datetime.now()
                        order_id = None
                        if isinstance(response, dict) and 'success_response' in response:
                            order_id = response['success_response'].get('order_id', '')
                        elif hasattr(response, 'order_id'):
                            order_id = response.order_id
                            
                        # Create fill data for the successful order
                        fill_data = FillData(
                            order_id=order_id or str(uuid.uuid4()),  # Use UUID if we don't have an order ID
                            product_id=order.product_id,
                            side='BUY',
                            size=float(order.base_size),
                            price=float(order.limit_price),
                            created_at=current_time,
                            fees=0.0  # We don't know fees yet for a just-placed order
                        )
                        
                        # Add to the stack (LIFO - most recent at the top)
                        self.order_stack.push_buy_order(fill_data)
                        logger.info(f"Added new buy order to stack: {order.product_id} {order.base_size} @ {order.limit_price}")
                        
                    elif order.side == 'SELL':
                        # Pop the most recent buy order from the stack since it's now being sold (LIFO)
                        popped_buy = self.order_stack.pop_latest_buy(order.product_id)
                        if popped_buy:
                            logger.info(f"Popped most recent buy order from stack after successful sell: {popped_buy}")
                            
                            # Calculate profit
                            buy_price = popped_buy.price
                            sell_price = float(order.limit_price)
                            profit_pct = (sell_price / buy_price - 1) * 100
                            profit_amount = (sell_price - buy_price) * float(order.base_size)
                            
                            print(f"\n💰 PROFIT TRACKING (LIFO Stack-based):")
                            print(f"  Buy: {popped_buy.size} @ ${buy_price:.2f} on {popped_buy.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                            print(f"  Sell: {order.base_size} @ ${sell_price:.2f}")
                            print(f"  Profit: ${profit_amount:.2f} ({profit_pct:.2f}%)")
                            
                            # Calculate remaining buy orders for tracking
                            remaining = self.order_stack.get_stack_size(order.product_id)
                            if remaining > 0:
                                print(f"  Remaining buy orders in stack: {remaining}")
                
                # Modify the response for simulation mode to make it easier to work with
                if trading_mode.lower() == 'simulation' and isinstance(response, dict) and response.get('success'):
                    # Create a simpler response object with just the order_id
                    from types import SimpleNamespace
                    simple_response = SimpleNamespace()
                    
                    # Extract order_id from success_response if available
                    if 'success_response' in response and 'order_id' in response['success_response']:
                        order_id = response['success_response']['order_id']
                        simple_response.order_id = order_id
                    else:
                        simulated_id = "simulated-order-" + str(uuid.uuid4())
                        simple_response.order_id = simulated_id
                    
                    return simple_response
                
                return response
                
            # For market orders
            response = self.client.create_market_order(order)
            
            # Update the last order time if we got here (means the order was successful)
            self.update_last_order_time(order.product_id, order.side)
            
            # Handle stack updates for market orders too
            if hasattr(response, 'success') and response.success:
                # ... similar stack update logic as for limit orders ...
                pass
            
            return response
            
        except InsufficientBalanceError as e:
            logger.warning(str(e))
            raise
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            raise OrderPlacementError(str(e)) 