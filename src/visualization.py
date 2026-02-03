"""
visualization.py
Visualization module with STONKS meme aesthetic
All plots follow dark background with neon colors
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# STONKS Meme Aesthetic
plt.style.use("dark_background")
STONKS_GREEN = '#00ff9c'
STONKS_RED = '#ff005c'
STONKS_BLUE = '#00c3ff'
STONKS_PURP = '#9d4edd'
STONKS_YELLOW = '#ffbe0b'
STONKS_ORANGE = '#fb5607'

# Color palette for multiple series
STONKS_PALETTE = [STONKS_GREEN, STONKS_BLUE, STONKS_PURP, STONKS_YELLOW, STONKS_ORANGE]


class PortfolioVisualizer:
    """
    Visualization utilities for portfolio analysis
    """
    
    @staticmethod
    def plot_price_history(prices: pd.DataFrame, figsize: Tuple = (14, 7)):
        """
        Plot normalized price history for all assets
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Price data
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Normalize to starting value of 100
        normalized = (prices / prices.iloc[0]) * 100
        
        for i, col in enumerate(normalized.columns):
            ax.plot(normalized.index, normalized[col], 
                   label=col, linewidth=2, 
                   color=STONKS_PALETTE[i % len(STONKS_PALETTE)])
        
        ax.set_title('Asset Price Performance (Normalized to 100)', 
                    fontsize=16, fontweight='bold', color=STONKS_GREEN)
        ax.set_xlabel('Date', fontsize=12, color='white')
        ax.set_ylabel('Normalized Price', fontsize=12, color='white')
        ax.legend(loc='upper left', framealpha=0.8)
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_rolling_volatility(volatility: pd.DataFrame, figsize: Tuple = (14, 7)):
        """
        Plot rolling volatility for all assets
        
        Parameters:
        -----------
        volatility : pd.DataFrame
            Rolling volatility data
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, col in enumerate(volatility.columns):
            ax.plot(volatility.index, volatility[col], 
                   label=col, linewidth=2,
                   color=STONKS_PALETTE[i % len(STONKS_PALETTE)])
        
        ax.set_title('Rolling 60-Day Volatility (Annualized)', 
                    fontsize=16, fontweight='bold', color=STONKS_GREEN)
        ax.set_xlabel('Date', fontsize=12, color='white')
        ax.set_ylabel('Volatility', fontsize=12, color='white')
        ax.legend(loc='upper left', framealpha=0.8)
        ax.grid(True, alpha=0.2)
        
        # Add recession shading if available
        ax.axhline(y=0.20, color=STONKS_RED, linestyle='--', 
                  alpha=0.5, label='20% Vol Level')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_correlation_heatmap(corr_matrix: pd.DataFrame, figsize: Tuple = (10, 8)):
        """
        Plot correlation heatmap
        
        Parameters:
        -----------
        corr_matrix : pd.DataFrame
            Correlation matrix
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Custom colormap for correlation
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
                   cmap=cmap, center=0, square=True,
                   linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title('Asset Correlation Matrix', 
                    fontsize=16, fontweight='bold', 
                    color=STONKS_GREEN, pad=20)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_risk_contributions(risk_contrib: np.ndarray, 
                               asset_names: List[str],
                               title: str = 'Risk Contributions',
                               figsize: Tuple = (12, 6)):
        """
        Plot risk contribution bar chart
        
        Parameters:
        -----------
        risk_contrib : np.ndarray
            Risk contributions (as percentages)
        asset_names : list
            Asset names
        title : str
            Plot title
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(asset_names))
        colors = [STONKS_PALETTE[i % len(STONKS_PALETTE)] for i in range(len(asset_names))]
        
        bars = ax.bar(x, risk_contrib * 100, color=colors, 
                     edgecolor='white', linewidth=1.5, alpha=0.9)
        
        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', 
                   fontsize=11, fontweight='bold', color='white')
        
        ax.set_xlabel('Asset', fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel('Risk Contribution (%)', fontsize=12, fontweight='bold', color='white')
        ax.set_title(title, fontsize=16, fontweight='bold', color=STONKS_GREEN)
        ax.set_xticks(x)
        ax.set_xticklabels(asset_names, fontsize=11)
        ax.grid(True, alpha=0.2, axis='y')
        
        # Add target line (equal risk)
        target = 100 / len(asset_names)
        ax.axhline(y=target, color=STONKS_RED, linestyle='--', 
                  linewidth=2, alpha=0.7, label=f'Target ({target:.1f}%)')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_allocation_comparison(allocation_df: pd.DataFrame, 
                                   figsize: Tuple = (14, 6)):
        """
        Plot allocation comparison across portfolios
        
        Parameters:
        -----------
        allocation_df : pd.DataFrame
            Allocation table (assets x portfolios)
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(allocation_df.index))
        width = 0.25
        n_portfolios = len(allocation_df.columns)
        
        for i, col in enumerate(allocation_df.columns):
            offset = width * (i - n_portfolios/2 + 0.5)
            ax.bar(x + offset, allocation_df[col], width, 
                  label=col, color=STONKS_PALETTE[i % len(STONKS_PALETTE)],
                  edgecolor='white', linewidth=1, alpha=0.9)
        
        ax.set_xlabel('Asset', fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel('Allocation (%)', fontsize=12, fontweight='bold', color='white')
        ax.set_title('Portfolio Allocation Comparison', 
                    fontsize=16, fontweight='bold', color=STONKS_GREEN)
        ax.set_xticks(x)
        ax.set_xticklabels(allocation_df.index, fontsize=11)
        ax.legend(loc='upper right', framealpha=0.8)
        ax.grid(True, alpha=0.2, axis='y')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_portfolio_performance(performance_df: pd.DataFrame, 
                                   figsize: Tuple = (14, 8)):
        """
        Plot cumulative performance of multiple portfolios
        
        Parameters:
        -----------
        performance_df : pd.DataFrame
            Cumulative returns for each portfolio
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, col in enumerate(performance_df.columns):
            ax.plot(performance_df.index, performance_df[col], 
                   label=col, linewidth=2.5,
                   color=STONKS_PALETTE[i % len(STONKS_PALETTE)])
        
        ax.set_title('Portfolio Performance Comparison', 
                    fontsize=16, fontweight='bold', color=STONKS_GREEN)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel('Cumulative Value ($1 Initial)', 
                     fontsize=12, fontweight='bold', color='white')
        ax.legend(loc='upper left', framealpha=0.8, fontsize=11)
        ax.grid(True, alpha=0.2)
        
        # Add horizontal line at 1.0
        ax.axhline(y=1.0, color='white', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_drawdown_comparison(drawdown_dict: Dict[str, pd.Series],
                                 figsize: Tuple = (14, 7)):
        """
        Plot drawdown comparison for multiple portfolios
        
        Parameters:
        -----------
        drawdown_dict : dict
            Dictionary of {portfolio_name: drawdown_series}
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, (name, drawdown) in enumerate(drawdown_dict.items()):
            ax.fill_between(drawdown.index, 0, drawdown * 100,
                           label=name, alpha=0.6,
                           color=STONKS_PALETTE[i % len(STONKS_PALETTE)])
        
        ax.set_title('Drawdown Comparison', 
                    fontsize=16, fontweight='bold', color=STONKS_GREEN)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold', color='white')
        ax.legend(loc='lower left', framealpha=0.8)
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_crisis_performance(crisis_df: pd.DataFrame,
                               figsize: Tuple = (14, 8)):
        """
        Plot crisis performance metrics
        
        Parameters:
        -----------
        crisis_df : pd.DataFrame
            Crisis performance data
        figsize : tuple
            Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        metrics = ['Total Return', 'Volatility', 'Max Drawdown', 'Days']
        colors = [STONKS_GREEN, STONKS_BLUE, STONKS_RED, STONKS_PURP]
        
        for i, metric in enumerate(metrics):
            if metric in crisis_df.columns:
                ax = axes[i]
                data = crisis_df[metric] * 100 if metric != 'Days' else crisis_df[metric]
                
                bars = ax.bar(range(len(crisis_df)), data, 
                             color=colors[i], alpha=0.9,
                             edgecolor='white', linewidth=1.5)
                
                ax.set_title(metric, fontsize=13, fontweight='bold', 
                           color=colors[i])
                ax.set_xticks(range(len(crisis_df)))
                ax.set_xticklabels(crisis_df.index, rotation=45, ha='right')
                ax.grid(True, alpha=0.2, axis='y')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}',
                           ha='center', va='bottom', fontsize=9, color='white')
        
        plt.suptitle('Crisis Performance Analysis', 
                    fontsize=16, fontweight='bold', 
                    color=STONKS_GREEN, y=1.00)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_efficient_frontier_overlay(portfolios_dict: Dict[str, Dict],
                                       figsize: Tuple = (12, 8)):
        """
        Plot portfolios on risk-return space
        
        Parameters:
        -----------
        portfolios_dict : dict
            Dictionary of {name: {'return': float, 'volatility': float}}
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, (name, metrics) in enumerate(portfolios_dict.items()):
            ax.scatter(metrics['volatility'] * 100, 
                      metrics['return'] * 100,
                      s=300, alpha=0.8,
                      color=STONKS_PALETTE[i % len(STONKS_PALETTE)],
                      edgecolor='white', linewidth=2,
                      label=name, zorder=5)
            
            # Add labels
            ax.annotate(name, 
                       xy=(metrics['volatility'] * 100, metrics['return'] * 100),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor=STONKS_PALETTE[i % len(STONKS_PALETTE)], 
                                alpha=0.7))
        
        ax.set_xlabel('Volatility (%)', fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel('Return (%)', fontsize=12, fontweight='bold', color='white')
        ax.set_title('Risk-Return Profile', 
                    fontsize=16, fontweight='bold', color=STONKS_GREEN)
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        return fig
