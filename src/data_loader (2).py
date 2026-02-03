"""
data_loader.py
FRED-based data acquisition module for Risk Parity Portfolio project

Provides:
- Long-horizon financial asset proxies
- Clean macroeconomic series
- Professional-grade research data pipeline
"""

import pandas as pd
import numpy as np
from fredapi import Fred
from typing import Dict
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class DataLoader:
    """
    Handles financial + macroeconomic data acquisition from FRED
    """

    def __init__(self, fred_api_key: str):
        """
        Initialize DataLoader with FRED API key
        """
        self.fred = Fred(api_key=fred_api_key)

        # Core Risk Parity Asset Universe (FRED Proxies)
        self.asset_series = {
            "Stocks": "SP500",                 # S&P 500 Index
            "Bonds": "DGS10",                  # 10Y Treasury Yield
            "Gold": "GOLDAMGBD228NLBM",        # Gold Price (London Fix)
            "Commodities": "PALLFNFINDEXQ",    # Broad Commodity Index
        }

        # Macro indicators (optional ML features)
        self.macro_series = {
            "CPI": "CPIAUCSL",
            "FedFunds": "FEDFUNDS",
            "YieldCurve": "T10Y2Y",
            "Recession": "USREC",
        }

    # ============================================================
    # Financial Market Data
    # ============================================================

    def fetch_market_data(
        self,
        start_date: str = "2007-01-01",
        cache_path: str = "../data/raw_prices.csv"
    ) -> pd.DataFrame:
        """
        Fetch financial asset data from FRED

        Returns clean daily asset price proxies suitable for
        risk parity and portfolio modeling.
        """
        print("📊 Fetching market data from FRED...")

        df = pd.DataFrame()

        for asset, code in self.asset_series.items():
            try:
                s = self.fred.get_series(code)
                s = s[s.index >= start_date]
                df[asset] = s
                print(f"   ✓ {asset} ({code}): {len(s)} records")
            except Exception as e:
                print(f"   ✗ Failed {asset}: {e}")

        df = df.ffill().dropna()

        print(f"\n✅ Market data loaded: {df.shape}")
        print(f"   Date range: {df.index[0].date()} → {df.index[-1].date()}")

        self.save_data(df, cache_path)

        return df

    # ============================================================
    # Macro Data
    # ============================================================

    def fetch_macro_data(
        self,
        start_date: str = "2007-01-01"
    ) -> pd.DataFrame:
        """
        Fetch macroeconomic indicators from FRED
        """
        print("\n📈 Fetching macroeconomic data from FRED...")

        df = pd.DataFrame()

        for name, code in self.macro_series.items():
            try:
                s = self.fred.get_series(code)
                s = s[s.index >= start_date]
                df[name] = s
                print(f"   ✓ {name} ({code}): {len(s)} records")
            except Exception as e:
                print(f"   ✗ Failed {name}: {e}")

        df = df.ffill().dropna()

        print(f"\n✅ Macro data loaded: {df.shape}")
        return df

    # ============================================================
    # Transformations
    # ============================================================

    def compute_returns(
        self,
        prices: pd.DataFrame,
        bond_duration: float = 8.5
    ) -> pd.DataFrame:
        """
        Convert price series → returns

        Special handling for bond yields:
            ΔP ≈ -Duration * ΔYield

        This is institutional-grade bond modeling.
        """
        print("\n🔄 Computing asset returns...")

        returns = prices.pct_change()

        # Proper bond price approximation
        if "Bonds" in prices.columns:
            dy = prices["Bonds"].diff() / 100
            returns["Bonds"] = -bond_duration * dy

        returns = returns.dropna()

        print(f"✅ Returns computed: {returns.shape}")
        return returns

    # ============================================================
    # Caching
    # ============================================================
    
    def save_data(self, df: pd.DataFrame, filepath: str):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        print(f"\n💾 Data saved → {filepath}")

    def load_data(self, filepath: str):
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"\n📂 Loaded cached data → {df.shape}")
        return df
