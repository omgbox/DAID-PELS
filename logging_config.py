"""
BookBot Logging Configuration
Structured logging for file and console output.
"""

import logging
import sys
from pathlib import Path

from .config import LOG_PATH, LOGGING


def setup_logging(name: str = 'bookbot') -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        name: Logger name

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOGGING['level']))

    # Clear existing handlers
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(LOGGING['format'])

    # Console handler
    if LOGGING['to_console']:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if LOGGING['to_file']:
        log_path = Path(LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.

    Args:
        name: Logger name (usually module name)

    Returns:
        Named logger
    """
    return logging.getLogger(f'bookbot.{name}')
