"""
risk_models.py
Risk modeling and volatility estimation module
Calculates returns, volatility, correlations, and risk contributions
"""

import numpy as np
import pandas as pd
from typing import Tuple


class RiskModels:
    """
    Risk modeling utilities for portfolio construction
    """
    
    @staticmethod
    def calculate_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
        """
        Calculate returns from price data
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Price data
        method : str
            'log' for log returns, 'simple' for simple returns
            
        Returns:
        --------
        pd.DataFrame
            Returns
        """
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        
        return returns.dropna()
    
    @staticmethod
    def calculate_rolling_volatility(returns: pd.DataFrame, 
                                     window: int = 60) -> pd.DataFrame:
        """
        Calculate rolling annualized volatility
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Daily returns
        window : int
            Rolling window size in days
            
        Returns:
        --------
        pd.DataFrame
            Rolling annualized volatility
        """
        # Annualization factor for daily data
        annualization_factor = np.sqrt(252)
        
        rolling_vol = returns.rolling(window=window).std() * annualization_factor
        
        return rolling_vol.dropna()
    
    @staticmethod
    def calculate_covariance_matrix(returns: pd.DataFrame, 
                                   window: int = None) -> np.ndarray:
        """
        Calculate covariance matrix (annualized)
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Daily returns
        window : int
            If provided, uses last 'window' days; otherwise uses all data
            
        Returns:
        --------
        np.ndarray
            Annualized covariance matrix
        """
        if window:
            returns_subset = returns.tail(window)
        else:
            returns_subset = returns
        
        # Annualize (252 trading days)
        cov_matrix = returns_subset.cov() * 252
        
        return cov_matrix.values
    
    @staticmethod
    def calculate_correlation_matrix(returns: pd.DataFrame, 
                                    window: int = None) -> pd.DataFrame:
        """
        Calculate correlation matrix
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Daily returns
        window : int
            If provided, uses last 'window' days
            
        Returns:
        --------
        pd.DataFrame
            Correlation matrix
        """
        if window:
            returns_subset = returns.tail(window)
        else:
            returns_subset = returns
        
        return returns_subset.corr()
    
    @staticmethod
    def calculate_portfolio_volatility(weights: np.ndarray, 
                                      cov_matrix: np.ndarray) -> float:
        """
        Calculate portfolio volatility
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights (must sum to 1)
        cov_matrix : np.ndarray
            Covariance matrix
            
        Returns:
        --------
        float
            Portfolio volatility (annualized)
        """
        portfolio_variance = weights.T @ cov_matrix @ weights
        return np.sqrt(portfolio_variance)
    
    @staticmethod
    def calculate_risk_contributions(weights: np.ndarray, 
                                    cov_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate risk contribution of each asset
        
        Risk Contribution: RC_i = w_i * (Σw)_i / σ_p
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        cov_matrix : np.ndarray
            Covariance matrix
            
        Returns:
        --------
        np.ndarray
            Risk contributions (sum to 1)
        """
        portfolio_vol = RiskModels.calculate_portfolio_volatility(weights, cov_matrix)
        
        # Marginal contribution to risk: (Σw)_i
        marginal_contrib = cov_matrix @ weights
        
        # Risk contribution: w_i * (Σw)_i / σ_p
        risk_contrib = weights * marginal_contrib / portfolio_vol
        
        # Normalize to percentages
        risk_contrib_pct = risk_contrib / risk_contrib.sum()
        
        return risk_contrib_pct
    
    @staticmethod
    def calculate_maximum_drawdown(prices_or_returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown and dates
        
        Parameters:
        -----------
        prices_or_returns : pd.Series
            Price series or cumulative returns
            
        Returns:
        --------
        tuple
            (max_drawdown, peak_date, trough_date)
        """
        # Convert to cumulative returns if returns are provided
        if prices_or_returns.min() < 0:  # Likely returns
            cumulative = (1 + prices_or_returns).cumprod()
        else:
            cumulative = prices_or_returns
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        
        # Find dates
        trough_date = drawdown.idxmin()
        peak_date = cumulative[:trough_date].idxmax()
        
        return max_dd, peak_date, trough_date
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate annualized Sharpe ratio
        
        Parameters:
        -----------
        returns : pd.Series
            Daily returns
        risk_free_rate : float
            Annual risk-free rate
            
        Returns:
        --------
        float
            Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / 252
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
        return sharpe
