import pandas as pd
import logging

def validate_data(df: pd.DataFrame) -> bool:
    """Validate data integrity."""
    required_columns = ['directional_momentum', 'volatility_signature', 'volume_intensity', 
                       'body_proportion', 'range_position', 'future_price_change_4h']
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"Missing column: {col}")
            return False
    if df.isnull().any().any():
        logging.error("Data contains NaNs")
        return False
    # Check for all-zero columns
    if (df[required_columns] == 0).all().any():
        logging.warning("Data contains all-zero columns")
    return True