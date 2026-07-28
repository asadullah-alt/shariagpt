"""
Secrets Manager
────────────────
Simulates a secure vault integration (e.g., HashiCorp Vault) and provides
utilities for zeroing memory of sensitive strings to prevent RAM dump leaks.
"""
import sys
import ctypes
from app.config import get_settings

class SecretsVault:
    @staticmethod
    def get_jwt_secret() -> str:
        """Fetch the JWT secret securely (simulated)."""
        # In a real enterprise app, this would make an mTLS call to HashiCorp Vault.
        return get_settings().jwt_secret


def zero_memory(secret: str):
    """
    Overwrites the string buffer in memory with zeros.
    WARNING: In CPython, strings are immutable and sometimes interned.
    This is a demonstration of the memory-zeroing pattern for high-security environments.
    """
    if not isinstance(secret, str) or not secret:
        return
    
    # Get the memory address of the string object
    address = id(secret)
    
    # CPython 3 string struct has headers. The actual character data usually starts at:
    offset = sys.getsizeof("") - 1
    size = len(secret)
    
    try:
        # Overwrite memory with 0s
        ctypes.memset(address + offset, 0, size)
    except Exception as e:
        print(f"[SecretsManager] Failed to zero memory: {e}")
