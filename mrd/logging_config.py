import logging
import sys

def setup_logging():
    """Configure logging to stdout and file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log', mode='a')
        ]
    )