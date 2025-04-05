"""
Module for calculating the volatility signature feature for 4-hour BTC data.

The volatility signature is calculated as (high-low)/ATR, scaled by percentile
rank over 100 periods.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_true_range(df):
    """
    Calculate the True Range (TR) for each row.

    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', 'close' columns

    Returns:
        pd.Series: Series containing the True Range values
    """
    try:
        # Make a copy to avoid SettingWithCopyWarning
        df_copy = df.copy()
        
        # Calculate previous close
        df_copy['prev_close'] = df_copy['close'].shift(1)
        
        # Calculate the three TR components
        high_low = df_copy['high'] - df_copy['low']
        high_prev_close = abs(df_copy['high'] - df_copy['prev_close'])
        low_prev_close = abs(df_copy['low'] - df_copy['prev_close'])
        
        # Take the maximum of the three components
        df_copy['tr'] = np.maximum(high_low, np.maximum(high_prev_close, low_prev_close))
        
        # Handle first row where prev_close is NaN
        if len(df_copy) > 0:
            df_copy.loc[df_copy.index[0], 'tr'] = df_copy.loc[df_copy.index[0], 'high'] - df_copy.loc[df_copy.index[0], 'low']
        
        return df_copy['tr']
    
    except Exception as e:
        logger.error(f"Error calculating True Range: {str(e)}")
        raise

def calculate_atr(tr_series, period=14):
    """
    Calculate the Average True Range (ATR) over the specified period.

    Args:
        tr_series (pd.Series): Series containing the True Range values
        period (int, optional): The period over which to calculate ATR. Defaults to 14.

    Returns:
        pd.Series: Series containing the ATR values
    """
    try:
        if len(tr_series) < period:
            logger.warning(f"Insufficient data for ATR calculation: {len(tr_series)} points, {period} required")
        
        return tr_series.rolling(window=period, min_periods=1).mean()
    
    except Exception as e:
        logger.error(f"Error calculating ATR: {str(e)}")
        raise

def calculate(df):
    """
    Calculate the volatility signature feature.

    Volatility signature is (high-low)/ATR, scaled by percentile rank over 100 periods.

    Args:
        df (pd.DataFrame): DataFrame with at least 'high', 'low', 'close' columns

    Returns:
        pd.Series: Series containing the volatility signature values
    """
    try:
        # Validate input
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.Series(index=df.index)
            
        # Calculate True Range and ATR
        tr = calculate_true_range(df)
        atr = calculate_atr(tr, period=14)
        
        # Calculate price range and raw volatility signature
        price_range = df['high'] - df['low']
        vol_sig_raw = price_range / atr
        
        # Handle potential divide-by-zero
        vol_sig_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Log intermediate results for debugging
        logger.info(f"vol_sig_raw head: {vol_sig_raw.head().to_list()}")
        logger.info(f"vol_sig_raw NaNs: {vol_sig_raw.isna().sum()}")
        
        # Percentile rank scaling
        min_periods = min(100, len(df))
        vol_signature = vol_sig_raw.rolling(window=100, min_periods=min_periods).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True
        )
        
        # Log final results for debugging
        logger.info(f"vol_signature head: {vol_signature.head().to_list()}")
        logger.info(f"vol_signature NaNs: {vol_signature.isna().sum()}")
        
        logger.info(f"Successfully calculated volatility signature for {len(df)} rows")
        return vol_signature
    
    except Exception as e:
        logger.error(f"Error calculating volatility signature: {str(e)}")
        raise