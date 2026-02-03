"""
data_loader.py
Data acquisition module for Risk Parity Portfolio project
Fetches market data (Yahoo Finance) and macroeconomic data (FRED)
"""

import pandas as pd
import yfinance as yf
from fredapi import Fred
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Handles data acquisition from Yahoo Finance and FRED
    """
    
    def __init__(self, fred_api_key: str = "9fb3507ccba20e766e4972a45c57c18c"):
        """
        Initialize DataLoader with FRED API key
        
        Parameters:
        -----------
        fred_api_key : str
            FRED API key for macroeconomic data
        """
        self.fred = Fred(api_key=fred_api_key)
        
        # Asset universe
        self.tickers = {
            'SPY': 'US Equities',
            'TLT': 'Bonds',
            'GLD': 'Gold',
            'DBC': 'Commodities',
            'VNQ': 'REITs'
        }
        
        # FRED series codes
        self.fred_series = {
            'CPI': 'CPIAUCSL',
            'FedFunds': 'FEDFUNDS',
            '10Y': 'DGS10',
            'Recession': 'USREC'
        }
    
    def fetch_market_data(self, start_date: str = "2007-01-01", 
                         end_date: str = None) -> pd.DataFrame:
        """
        Fetch daily adjusted close prices for all assets
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format (None for today)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with adjusted close prices for all tickers
        """
        print("📊 Fetching market data from Yahoo Finance...")
        
        prices = pd.DataFrame()
        
        for ticker in self.tickers.keys():
            try:
                data = yf.download(ticker, start=start_date, end=end_date, 
                                 progress=False)['Adj Close']
                prices[ticker] = data
                print(f"   ✓ {ticker} ({self.tickers[ticker]}): {len(data)} records")
            except Exception as e:
                print(f"   ✗ Error fetching {ticker}: {e}")
        
        # Remove any NaN rows at the beginning
        prices = prices.dropna()
        
        print(f"\n✅ Market data loaded: {prices.shape[0]} days, {prices.shape[1]} assets")
        print(f"   Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
        
        return prices
    
    def fetch_macro_data(self) -> pd.DataFrame:
        """
        Fetch macroeconomic indicators from FRED
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with macro indicators
        """
        print("\n📈 Fetching macroeconomic data from FRED...")
        
        macro_data = {}
        
        for name, code in self.fred_series.items():
            try:
                series = self.fred.get_series(code)
                macro_data[name] = series
                print(f"   ✓ {name} ({code}): {len(series)} records")
            except Exception as e:
                print(f"   ✗ Error fetching {name}: {e}")
        
        # Combine into single DataFrame
        macro = pd.concat([
            series.rename(name) for name, series in macro_data.items()
        ], axis=1)
        
        # Forward fill missing values
        macro = macro.ffill()
        
        print(f"\n✅ Macro data loaded: {macro.shape[0]} periods, {macro.shape[1]} indicators")
        
        return macro
    
    def save_data(self, prices: pd.DataFrame, filepath: str = "../data/raw_prices.csv"):
        """
        Save price data to CSV
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Price data to save
        filepath : str
            Path to save CSV file
        """
        prices.to_csv(filepath)
        print(f"\n💾 Data saved to {filepath}")
    
    def load_data(self, filepath: str = "../data/raw_prices.csv") -> pd.DataFrame:
        """
        Load price data from CSV
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded price data
        """
        prices = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"📂 Data loaded from {filepath}: {prices.shape}")
        return prices
