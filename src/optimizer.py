"""
optimizer.py
Portfolio optimization module
Implements risk parity optimization and benchmark portfolios
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple
import pandas as pd


class PortfolioOptimizer:
    """
    Portfolio optimization with focus on risk parity
    """
    
    def __init__(self, cov_matrix: np.ndarray, asset_names: list = None):
        """
        Initialize optimizer with covariance matrix
        
        Parameters:
        -----------
        cov_matrix : np.ndarray
            Covariance matrix of assets
        asset_names : list
            Names of assets (for display)
        """
        self.cov_matrix = cov_matrix
        self.n_assets = cov_matrix.shape[0]
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(self.n_assets)]
    
    def risk_parity_objective(self, weights: np.ndarray) -> float:
        """
        Objective function for risk parity optimization
        Minimizes squared deviation of risk contributions from equality
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
            
        Returns:
        --------
        float
            Objective value (lower is better)
        """
        # Calculate portfolio volatility
        portfolio_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        
        # Marginal contribution to risk
        marginal_contrib = self.cov_matrix @ weights
        
        # Risk contribution for each asset
        risk_contrib = weights * marginal_contrib / portfolio_vol
        
        # Target: equal risk contribution
        target_risk = portfolio_vol / self.n_assets
        
        # Minimize sum of squared deviations from target
        objective = np.sum((risk_contrib - target_risk) ** 2)
        
        return objective
    
    def optimize_risk_parity(self, initial_weights: np.ndarray = None) -> Dict:
        """
        Optimize portfolio for risk parity
        
        Parameters:
        -----------
        initial_weights : np.ndarray
            Starting weights (defaults to equal weight)
            
        Returns:
        --------
        dict
            Optimization results including weights and risk contributions
        """
        # Initial guess: equal weight
        if initial_weights is None:
            initial_weights = np.ones(self.n_assets) / self.n_assets
        
        # Constraints: weights sum to 1
        constraints = {
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1
        }
        
        # Bounds: weights between 0 and 1 (no shorting)
        bounds = tuple((0.0, 1.0) for _ in range(self.n_assets))
        
        # Optimize
        result = minimize(
            fun=self.risk_parity_objective,
            x0=initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        if not result.success:
            print(f"⚠️  Optimization warning: {result.message}")
        
        weights = result.x
        
        # Calculate risk contributions
        portfolio_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        marginal_contrib = self.cov_matrix @ weights
        risk_contrib = weights * marginal_contrib / portfolio_vol
        risk_contrib_pct = risk_contrib / risk_contrib.sum()
        
        return {
            'weights': weights,
            'risk_contributions': risk_contrib_pct,
            'portfolio_volatility': portfolio_vol,
            'success': result.success,
            'objective_value': result.fun
        }
    
    @staticmethod
    def equal_weight_portfolio(n_assets: int) -> np.ndarray:
        """
        Create equal weight portfolio
        
        Parameters:
        -----------
        n_assets : int
            Number of assets
            
        Returns:
        --------
        np.ndarray
            Equal weights
        """
        return np.ones(n_assets) / n_assets
    
    @staticmethod
    def sixty_forty_portfolio(asset_names: list) -> np.ndarray:
        """
        Create 60/40 portfolio (60% stocks, 40% bonds)
        
        Parameters:
        -----------
        asset_names : list
            List of asset names (must include 'SPY' and 'TLT')
            
        Returns:
        --------
        np.ndarray
            60/40 weights
        """
        weights = np.zeros(len(asset_names))
        
        # Map common ticker patterns
        for i, name in enumerate(asset_names):
            if name == 'SPY':
                weights[i] = 0.60
            elif name == 'TLT':
                weights[i] = 0.40
        
        # Normalize in case SPY/TLT not found
        if weights.sum() == 0:
            print("⚠️  SPY/TLT not found, using equal weight instead")
            weights = np.ones(len(asset_names)) / len(asset_names)
        
        return weights
    
    def create_allocation_table(self, weights_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Create allocation comparison table
        
        Parameters:
        -----------
        weights_dict : dict
            Dictionary of {portfolio_name: weights_array}
            
        Returns:
        --------
        pd.DataFrame
            Allocation table
        """
        allocation_df = pd.DataFrame(weights_dict, index=self.asset_names)
        allocation_df = (allocation_df * 100).round(2)  # Convert to percentages
        
        return allocation_df
    
    def calculate_portfolio_metrics(self, weights: np.ndarray, 
                                   returns: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive portfolio metrics
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        returns : pd.DataFrame
            Historical returns data
            
        Returns:
        --------
        dict
            Portfolio metrics
        """
        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Annualized return
        annual_return = portfolio_returns.mean() * 252
        
        # Annualized volatility
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = annual_return / annual_vol
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown
        }
