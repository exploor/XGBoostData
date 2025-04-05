# core/logger.py
import logging
import sys
from mrd.core.config import config  # Assuming config is a module that provides settings

def setup_logger(name: str) -> logging.Logger:
    """Set up and return a configured logger with error handling."""
    logger = logging.getLogger(name)
    
    # Only configure the logger if it doesn’t already have handlers
    if not logger.handlers:
        try:
            # Get log level from config, default to INFO if not set
            level = config.get('logging', 'level', 'INFO')
            numeric_level = getattr(logging, level.upper(), logging.INFO)
            logger.setLevel(numeric_level)

            # Add console handler to output logs to stdout
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # Add file handler if specified in config
            handlers = config.get('logging', 'handlers', '').split(',')
            if 'file' in handlers:
                log_file = config.get('logging', 'log_file')
                if log_file:
                    try:
                        file_handler = logging.FileHandler(log_file)
                        file_handler.setLevel(numeric_level)
                        file_handler.setFormatter(formatter)
                        logger.addHandler(file_handler)
                    except Exception as e:
                        print(f"Warning: Could not create file handler for '{log_file}': {e}")
                else:
                    print("Warning: 'file' handler specified but no log_file in config.")
        except Exception as e:
            print(f"Error setting up logger '{name}': {e}")
            raise  # Re-raise the exception so it’s visible in the calling code
    
    return logger