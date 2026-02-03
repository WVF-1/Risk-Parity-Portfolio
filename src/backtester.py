"""
backtester.py
Portfolio backtesting module
Simulates portfolio performance over time with rebalancing
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .risk_models import RiskModels
from .optimizer import PortfolioOptimizer


class PortfolioBacktester:
    """
    Backtest portfolio strategies over historical data
    """
    
    def __init__(self, prices: pd.DataFrame, returns: pd.DataFrame):
        """
        Initialize backtester
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Price data
        returns : pd.DataFrame
            Returns data
        """
        self.prices = prices
        self.returns = returns
        self.asset_names = list(prices.columns)
    
    def backtest_portfolio(self, weights: np.ndarray, 
                          rebalance_freq: str = 'M') -> pd.Series:
        """
        Backtest portfolio with periodic rebalancing
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        rebalance_freq : str
            Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
        --------
        pd.Series
            Cumulative portfolio value over time
        """
        # Calculate portfolio returns
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        # Apply rebalancing if needed (simplified: assumes constant weights)
        # In practice, would reweight periodically
        
        # Cumulative returns
        cumulative_value = (1 + portfolio_returns).cumprod()
        cumulative_value.name = 'Portfolio Value'
        
        return cumulative_value
    
    def backtest_multiple_strategies(self, 
                                    strategies: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Backtest multiple portfolio strategies
        
        Parameters:
        -----------
        strategies : dict
            Dictionary of {strategy_name: weights}
            
        Returns:
        --------
        pd.DataFrame
            Cumulative values for all strategies
        """
        results = {}
        
        for name, weights in strategies.items():
            cumulative = self.backtest_portfolio(weights)
            results[name] = cumulative
        
        return pd.DataFrame(results)
    
    def calculate_crisis_performance(self, 
                                    weights: np.ndarray,
                                    crisis_periods: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
        """
        Evaluate portfolio performance during crisis periods
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        crisis_periods : dict
            Dictionary of {crisis_name: (start_date, end_date)}
            
        Returns:
        --------
        pd.DataFrame
            Performance metrics during each crisis
        """
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        crisis_stats = {}
        
        for crisis_name, (start, end) in crisis_periods.items():
            # Filter returns for crisis period
            crisis_returns = portfolio_returns.loc[start:end]
            
            if len(crisis_returns) == 0:
                continue
            
            # Calculate metrics
            cumulative = (1 + crisis_returns).cumprod()
            total_return = cumulative.iloc[-1] - 1
            
            volatility = crisis_returns.std() * np.sqrt(252)
            
            # Maximum drawdown during crisis
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            
            crisis_stats[crisis_name] = {
                'Total Return': total_return,
                'Volatility': volatility,
                'Max Drawdown': max_dd,
                'Days': len(crisis_returns)
            }
        
        return pd.DataFrame(crisis_stats).T
    
    def rolling_sharpe_ratio(self, weights: np.ndarray, 
                           window: int = 252,
                           risk_free_rate: float = 0.02) -> pd.Series:
        """
        Calculate rolling Sharpe ratio
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        window : int
            Rolling window size
        risk_free_rate : float
            Annual risk-free rate
            
        Returns:
        --------
        pd.Series
            Rolling Sharpe ratio
        """
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        # Excess returns
        excess_returns = portfolio_returns - risk_free_rate / 252
        
        # Rolling mean and std
        rolling_mean = excess_returns.rolling(window).mean()
        rolling_std = portfolio_returns.rolling(window).std()
        
        # Sharpe ratio
        sharpe = np.sqrt(252) * rolling_mean / rolling_std
        
        return sharpe.dropna()
    
    def calculate_drawdown_series(self, weights: np.ndarray) -> pd.Series:
        """
        Calculate drawdown series over time
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
            
        Returns:
        --------
        pd.Series
            Drawdown series
        """
        portfolio_returns = (self.returns * weights).sum(axis=1)
        cumulative = (1 + portfolio_returns).cumprod()
        
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown
    
    def get_summary_statistics(self, weights: np.ndarray) -> Dict:
        """
        Calculate comprehensive summary statistics
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
            
        Returns:
        --------
        dict
            Summary statistics
        """
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        # Annualized metrics
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol
        
        # Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Worst day
        worst_day = portfolio_returns.min()
        best_day = portfolio_returns.max()
        
        # Win rate
        win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)
        
        return {
            'Annualized Return': f"{annual_return:.2%}",
            'Annualized Volatility': f"{annual_vol:.2%}",
            'Sharpe Ratio': f"{sharpe:.3f}",
            'Maximum Drawdown': f"{max_dd:.2%}",
            'Best Day': f"{best_day:.2%}",
            'Worst Day': f"{worst_day:.2%}",
            'Win Rate': f"{win_rate:.2%}",
            'Total Days': len(portfolio_returns)
        }
