"""
Logging configuration module
"""

import logging
import os
from datetime import datetime


def setup_logging(log_to_file: bool = True, log_level: str = "INFO", log_to_console: bool = False) -> str:
    """
    Setup logging configuration
    
    Args:
        log_to_file: Whether to write to log file
        log_level: Logging level
        log_to_console: Whether to output to console, default False (only output to file)
    """
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = []
    log_filename = None
    
    if log_to_file:
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Generate log filename
        log_filename = f"logs/verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # File handler
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Console handler (optional)
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    return log_filename
