from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

@dataclass
class Account:
    uuid: str
    name: str
    currency: str
    available_balance: Decimal
    hold: Decimal
