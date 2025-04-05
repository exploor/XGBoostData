import logging
from core.logger import setup_logger

logger = setup_logger(__name__)

class AppError(Exception):
    """Custom exception for application errors."""
    pass

def handle_error(error: Exception, context: str = "") -> None:
    """Handle and log application errors."""
    error_message = f"Error in {context}: {str(error)}"
    logger.error(error_message, exc_info=True)
    raise AppError(error_message)