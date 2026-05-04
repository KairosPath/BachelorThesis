"""
Data Processor Module
=====================
Cleans, transforms, and prepares data for analysis
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Processes and transforms financial data for econometric analysis
    
    Features:
    - Missing value handling
    - Log returns calculation
    - Feature engineering
    - Data synchronization across different frequencies
    """
    
    def __init__(self):
        self.config = config.data
    
    def clean_data(
        self,
        df: pd.DataFrame,
        fill_method: Optional[str] = None,
        max_missing: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Clean data by handling missing values
        
        Args:
            df: Input DataFrame
            fill_method: Method for filling missing values
            max_missing: Maximum allowed missing ratio per column
            
        Returns:
            Cleaned DataFrame
        """
        fill_method = fill_method or self.config.fill_method
        max_missing = max_missing or self.config.max_missing_ratio
        
        logger.info(f"Cleaning data with {fill_method} method")
        
        missing_before = df.isnull().sum()
        if missing_before.sum() > 0:
            logger.info(f"Missing values before cleaning:\n{missing_before[missing_before > 0]}")
        
        missing_ratio = df.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > max_missing].index.tolist()
        if cols_to_drop:
            logger.warning(f"Dropping columns with >{max_missing*100}% missing: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        if fill_method == "ffill":
            df = df.ffill().bfill()
        elif fill_method == "bfill":
            df = df.bfill().ffill()
        elif fill_method == "interpolate":
            df = df.interpolate(method="time")
            df = df.ffill().bfill()
        
        missing_after = df.isnull().sum().sum()
        if missing_after > 0:
            logger.warning(f"Remaining missing values: {missing_after}")
            df = df.dropna()
        
        logger.info(f"Data cleaned: {len(df)} records remaining")
        
        return df
    
    def calculate_returns(
        self,
        prices: pd.DataFrame,
        method: str = "log"
    ) -> pd.DataFrame:
        """
        Calculate returns from price data
        
        Args:
            prices: DataFrame with price data
            method: 'log' for log returns, 'simple' for simple returns
            
        Returns:
            DataFrame with returns
        """
        if method == "log":
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        
        returns = returns.dropna()
        
        logger.info(f"Calculated {method} returns: {len(returns)} records")
        
        return returns
    
    def calculate_volatility(
        self,
        returns: pd.DataFrame,
        window: int = 21
    ) -> pd.DataFrame:
        """
        Calculate rolling volatility (annualized standard deviation)
        
        Args:
            returns: DataFrame with returns
            window: Rolling window size
            
        Returns:
            DataFrame with volatility estimates
        """
        annualization = np.sqrt(252)
        
        volatility = returns.rolling(window=window).std() * annualization
        volatility.columns = [f"{col}_vol" for col in volatility.columns]
        
        return volatility.dropna()
    
    def create_lagged_features(
        self,
        df: pd.DataFrame,
        lags: List[int] = [1, 2, 3, 5, 10, 21],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create lagged features for time series analysis
        
        Args:
            df: Input DataFrame
            lags: List of lag periods
            columns: Columns to create lags for (None = all)
            
        Returns:
            DataFrame with original and lagged features
        """
        result = df.copy()
        columns = columns or df.columns.tolist()
        
        for col in columns:
            for lag in lags:
                result[f"{col}_lag{lag}"] = df[col].shift(lag)
        
        logger.info(f"Created lagged features for {len(lags)} lags")
        
        return result.dropna()
    
    def create_momentum_features(
        self,
        prices: pd.DataFrame,
        windows: List[int] = [5, 10, 21, 63]
    ) -> pd.DataFrame:
        """
        Create momentum and trend features
        
        Args:
            prices: DataFrame with price data
            windows: List of lookback windows
            
        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=prices.index)
        
        for col in prices.columns:
            for w in windows:
                features[f"{col}_mom_{w}"] = prices[col].pct_change(w)
                
                features[f"{col}_ma_ratio_{w}"] = prices[col] / prices[col].rolling(w).mean()
                
                features[f"{col}_roc_{w}"] = (prices[col] - prices[col].shift(w)) / prices[col].shift(w)
        
        logger.info(f"Created momentum features for {len(windows)} windows")
        
        return features.dropna()
    
    def prepare_dataset(
        self,
        target_data: pd.DataFrame,
        features: pd.DataFrame,
        target_column: str = "Close"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare final dataset for modeling
        
        Args:
            target_data: Target asset price data (EEM)
            features: Feature data
            target_column: Column to use as target
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Preparing final dataset")
        
        if target_column in target_data.columns:
            target_prices = target_data[target_column]
        else:
            target_prices = target_data.iloc[:, 0]
        
        if isinstance(target_prices, pd.DataFrame):
            target_prices = target_prices.iloc[:, 0]
        
        target_df = pd.DataFrame({'price': target_prices})
        target_returns = self.calculate_returns(target_df, method="log")
        target_returns.columns = ["target"]
        
        features_clean = self.clean_data(features)
        
        feature_returns = self.calculate_returns(features_clean, method="log")
        
        volatility = self.calculate_volatility(feature_returns, window=21)
        
        momentum = self.create_momentum_features(features_clean, windows=[5, 10, 21])
        
        all_features = pd.concat([feature_returns, volatility, momentum], axis=1)
        
        target_lagged = self.create_lagged_features(
            target_returns, 
            lags=[1, 2, 3, 5],
            columns=["target"]
        )
        target_lagged = target_lagged.drop(columns=["target"])
        
        combined = pd.concat([all_features, target_lagged], axis=1)
        
        common_idx = combined.index.intersection(target_returns.index)
        X = combined.loc[common_idx]
        y = target_returns.loc[common_idx, "target"]
        
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        logger.info(f"Final dataset: {len(X)} samples, {X.shape[1]} features")
        
        return X, y
    
    def save_processed_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        name: str = "processed_data"
    ) -> None:
        """
        Save processed data to disk
        
        Args:
            X: Features DataFrame
            y: Target Series
            name: File name prefix
        """
        X.to_csv(PROCESSED_DATA_DIR / f"{name}_features.csv")
        y.to_csv(PROCESSED_DATA_DIR / f"{name}_target.csv")
        logger.info(f"Saved processed data to {PROCESSED_DATA_DIR}")
    
    def load_processed_data(
        self,
        name: str = "processed_data"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load processed data from disk
        
        Args:
            name: File name prefix
            
        Returns:
            Tuple of (features, target)
        """
        X = pd.read_csv(PROCESSED_DATA_DIR / f"{name}_features.csv", index_col=0, parse_dates=True)
        y = pd.read_csv(PROCESSED_DATA_DIR / f"{name}_target.csv", index_col=0, parse_dates=True)
        y = y.squeeze()
        
        logger.info(f"Loaded processed data: {len(X)} samples")
        
        return X, y

