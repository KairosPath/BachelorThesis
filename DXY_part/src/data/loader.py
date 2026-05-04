"""
Data Loader Module
==================
Fetches financial data from Yahoo Finance and FRED
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Optional, Dict
from pathlib import Path
import logging

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Loads financial data from multiple sources
    
    Sources:
    - Yahoo Finance: DXY, VIX, Treasury yields, commodities, FX rates
    - FRED: Fed Funds Rate, Treasury spreads, Trade-weighted dollar index
    """
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize DataLoader
        
        Args:
            fred_api_key: API key for FRED data (optional)
        """
        self.config = config.data
        self.fred_api_key = fred_api_key
        self.fred = None
        
        if fred_api_key and FRED_AVAILABLE:
            self.fred = Fred(api_key=fred_api_key)
    
    def fetch_yahoo_data(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with adjusted close prices
        """
        start = start_date or self.config.start_date
        end = end_date or self.config.end_date
        
        logger.info(f"Fetching Yahoo data for {len(tickers)} tickers from {start} to {end}")
        
        try:
            df = yf.download(tickers, start=start, end=end, progress=False)
            
            if df.empty:
                logger.warning("No data received from Yahoo Finance")
                return pd.DataFrame()
            
            if isinstance(df.columns, pd.MultiIndex):
                if 'Adj Close' in df.columns.get_level_values(0):
                    result = df['Adj Close']
                elif 'Close' in df.columns.get_level_values(0):
                    result = df['Close']
                else:
                    result = df.iloc[:, df.columns.get_level_values(0) == df.columns.get_level_values(0)[0]]
                    result.columns = result.columns.droplevel(0)
            else:
                if 'Adj Close' in df.columns:
                    result = df[['Adj Close']]
                    result.columns = tickers[:1]
                elif 'Close' in df.columns:
                    result = df[['Close']]
                    result.columns = tickers[:1]
                else:
                    result = df
            
            result.index = pd.to_datetime(result.index)
            
            for ticker in result.columns:
                valid_count = result[ticker].notna().sum()
                logger.info(f"✓ {ticker}: {valid_count} records")
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
    
    def fetch_fred_data(
        self,
        series: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch data from FRED (Federal Reserve Economic Data)
        
        Args:
            series: List of FRED series IDs
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with FRED data
        """
        if not self.fred:
            logger.warning("FRED API not configured. Skipping FRED data.")
            return pd.DataFrame()
        
        start = start_date or self.config.start_date
        end = end_date or self.config.end_date
        
        logger.info(f"Fetching FRED data for {len(series)} series")
        
        data_frames = {}
        
        for series_id in series:
            try:
                data = self.fred.get_series(series_id, start, end)
                if data is not None and len(data) > 0:
                    data_frames[series_id] = data
                    logger.info(f"✓ {series_id}: {len(data)} records")
                else:
                    logger.warning(f"✗ {series_id}: No data available")
            except Exception as e:
                logger.error(f"✗ {series_id}: Error - {str(e)}")
        
        if data_frames:
            result = pd.DataFrame(data_frames)
            result.index = pd.to_datetime(result.index)
            return result
        else:
            return pd.DataFrame()
    
    def fetch_dxy(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch DXY (Dollar Index) data
        
        Returns:
            DataFrame with DXY OHLCV data
        """
        start = start_date or self.config.start_date
        end = end_date or self.config.end_date
        
        logger.info(f"Fetching DXY data from {start} to {end}")
        
        df = yf.download(self.config.target_ticker, start=start, end=end, progress=False)
        
        if df.empty:
            logger.warning("DXY data from Yahoo is empty. Trying alternative ticker...")
            df = yf.download("DX=F", start=start, end=end, progress=False)
        
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            logger.info(f"✓ DXY: {len(df)} records fetched")
        else:
            logger.error("✗ Failed to fetch DXY data")
        
        return df
    
