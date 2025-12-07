"""
Logging Configuration
=====================
Centralized logging setup for the application.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    name: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string (optional)
        name: Logger name (optional, uses root if None)
    
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Pre-configured loggers for different components
def get_etl_logger() -> logging.Logger:
    """Get logger for ETL operations."""
    return get_logger("hipaa.etl")


def get_api_logger() -> logging.Logger:
    """Get logger for API operations."""
    return get_logger("hipaa.api")


def get_search_logger() -> logging.Logger:
    """Get logger for search operations."""
    return get_logger("hipaa.search")


def get_db_logger() -> logging.Logger:
    """Get logger for database operations."""
    return get_logger("hipaa.db")
