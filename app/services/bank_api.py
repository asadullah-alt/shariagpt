import pybreaker
from datetime import datetime, timedelta

# Circuit breaker: Trip after 3 failures, reset after 30 seconds
bank_api_breaker = pybreaker.CircuitBreaker(fail_max=3, reset_timeout=30)

@bank_api_breaker
def get_account_balance(account_number: str) -> dict:
    """Mock bank API for balance check."""
    # In a real scenario, this would use httpx to call a core banking service
    return {
        "account_number": account_number,
        "balance": 15000.50,
        "currency": "AED",
        "last_updated": datetime.now().isoformat()
    }

@bank_api_breaker
def get_recent_transactions(account_number: str) -> dict:
    """Mock bank API for recent transactions."""
    return {
        "account_number": account_number,
        "transactions": [
            {"date": (datetime.now() - timedelta(days=1)).isoformat(), "amount": -150.0, "description": "Grocery Store"},
            {"date": (datetime.now() - timedelta(days=3)).isoformat(), "amount": 5000.0, "description": "Salary Deposit"},
            {"date": (datetime.now() - timedelta(days=5)).isoformat(), "amount": -200.0, "description": "Utility Bill"}
        ]
    }

@bank_api_breaker
def get_account_status(account_number: str) -> dict:
    """Mock bank API for account status."""
    return {
        "account_number": account_number,
        "status": "ACTIVE",
        "type": "Retail Savings (Mudarabah)",
        "profit_rate": "3.5%"
    }
