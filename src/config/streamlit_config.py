"""
Streamlit-compatible configuration helper that works with both 
local environment variables and Streamlit secrets.
"""

import os
from typing import Any, Union

def get_config_value(key: str, default: Any = None) -> Union[str, int, float, bool]:
    """
    Get configuration value from Streamlit secrets or environment variables.
    
    Args:
        key: Configuration key name
        default: Default value if key not found
        
    Returns:
        Configuration value
    """
    try:
        # Try to import streamlit and use secrets
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except ImportError:
        # Streamlit not available, continue to environment variables
        pass
    except Exception:
        # Streamlit available but secrets not accessible, continue to environment variables
        pass
    
    # Fallback to environment variables
    return os.getenv(key, default)

def get_int_config(key: str, default: int = 0) -> int:
    """Get integer configuration value."""
    value = get_config_value(key, default)
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def get_float_config(key: str, default: float = 0.0) -> float:
    """Get float configuration value."""
    value = get_config_value(key, default)
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def get_bool_config(key: str, default: bool = False) -> bool:
    """Get boolean configuration value."""
    value = get_config_value(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return default