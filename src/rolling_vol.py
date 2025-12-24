"""
Rolling Volatility Module

Computes simple moving window volatility using standard deviation of returns.

Key Features:
- Rolling window volatility calculation
- Multiple window sizes
- Annualization (daily to annual)
- Visualization tools
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List
import warnings


class RollingVolatility:
    """
    Calculate rolling window volatility from returns data.
    
    Attributes:
        returns (pd.DataFrame): Returns data
        volatility (pd.DataFrame): Computed rolling volatility
        window (int): Rolling window size
        annualized (bool): Whether volatility is annualized
    """
    
    def __init__(self, returns: Optional[pd.DataFrame] = None):
        """
        Initialize RollingVolatility calculator.
        
        Parameters:
        -----------
        returns : pd.DataFrame, optional
            Returns data with DatetimeIndex
        """
        self.returns = returns
        self.volatility = None
        self.window = None
        self.annualized = False
    
    def compute_volatility(
        self,
        returns: Optional[pd.DataFrame] = None,
        window: int = 20,
        min_periods: Optional[int] = None,
        center: bool = False
    ) -> pd.DataFrame:
        """
        Compute rolling window volatility.
        
        Formula: σ_t = std(r_{t-window+1}, ..., r_t)
        
        Parameters:
        -----------
        returns : pd.DataFrame, optional
            Returns data. If None, uses self.returns
        window : int, default=20
            Rolling window size (e.g., 20 days ≈ 1 month)
        min_periods : int, optional
            Minimum number of observations required. If None, uses window
        center : bool, default=False
            Whether to center the rolling window
            
        Returns:
        --------
        pd.DataFrame
            Rolling volatility (daily, not annualized)
        """
        if returns is None:
            if self.returns is None:
                raise ValueError("No returns data provided")
            returns = self.returns
        else:
            self.returns = returns
        
        if min_periods is None:
            min_periods = window
        
        # Compute rolling standard deviation
        volatility = returns.rolling(
            window=window,
            min_periods=min_periods,
            center=center
        ).std()
        
        self.volatility = volatility
        self.window = window
        self.annualized = False
        
        return volatility
    
    def annualize(
        self,
        volatility: Optional[pd.DataFrame] = None,
        periods_per_year: int = 252
    ) -> pd.DataFrame:
        """
        Annualize volatility.
        
        Formula: σ_annual = σ_daily * sqrt(periods_per_year)
        
        Parameters:
        -----------
        volatility : pd.DataFrame, optional
            Daily volatility. If None, uses self.volatility
        periods_per_year : int, default=252
            Number of periods per year (252 for daily trading days)
            
        Returns:
        --------
        pd.DataFrame
            Annualized volatility
        """
        if volatility is None:
            if self.volatility is None:
                raise ValueError("No volatility data to annualize")
            volatility = self.volatility
        
        annualized_vol = volatility * np.sqrt(periods_per_year)
        
        if volatility is self.volatility:
            self.volatility = annualized_vol
            self.annualized = True
        
        return annualized_vol
    
    def compute_multiple_windows(
        self,
        returns: Optional[pd.DataFrame] = None,
        windows: List[int] = [20, 60, 252],
        annualize: bool = False
    ) -> dict:
        """
        Compute volatility for multiple window sizes.
        
        Parameters:
        -----------
        returns : pd.DataFrame, optional
            Returns data
        windows : List[int], default=[20, 60, 252]
            List of window sizes to compute
        annualize : bool, default=False
            Whether to annualize the volatility
            
        Returns:
        --------
        dict
            Dictionary with window sizes as keys and volatility DataFrames as values
        """
        if returns is None:
            if self.returns is None:
                raise ValueError("No returns data provided")
            returns = self.returns
        
        results = {}
        
        for window in windows:
            vol = self.compute_volatility(returns, window=window)
            
            if annualize:
                vol = self.annualize(vol)
            
            results[window] = vol
        
        return results
    
    def plot_volatility(
        self,
        volatility: Optional[pd.DataFrame] = None,
        figsize: tuple = (14, 6),
        show_returns: bool = True
    ) -> None:
        """
        Plot rolling volatility with optional returns overlay.
        
        Parameters:
        -----------
        volatility : pd.DataFrame, optional
            Volatility data. If None, uses self.volatility
        figsize : tuple, default=(14, 6)
            Figure size
        show_returns : bool, default=True
            Whether to show returns in subplot
        """
        if volatility is None:
            if self.volatility is None:
                raise ValueError("No volatility data to plot")
            volatility = self.volatility
        
        n_cols = volatility.shape[1]
        
        if show_returns:
            if self.returns is None:
                raise ValueError("No returns data available for plotting")
            fig, axes = plt.subplots(n_cols, 2, figsize=figsize)
            if n_cols == 1:
                axes = axes.reshape(1, -1)
        else:
            fig, axes = plt.subplots(n_cols, 1, figsize=figsize)
            if n_cols == 1:
                axes = axes.reshape(1, -1)
        
        for idx, col in enumerate(volatility.columns):
            # Plot volatility
            ax_idx = 0 if show_returns else 0
            if show_returns:
                ax = axes[idx, 0]
            else:
                ax = axes[idx, 0] if n_cols > 1 else axes[0]
            
            ax.plot(volatility.index, volatility[col], linewidth=1.5, color='darkred')
            ax.set_title(f'{col} - Rolling Volatility ({self.window}-day window)')
            ax.set_xlabel('Date')
            ylabel = 'Annualized Volatility' if self.annualized else 'Daily Volatility'
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            
            # Add mean line
            mean_vol = volatility[col].mean()
            ax.axhline(y=mean_vol, color='blue', linestyle='--', 
                      label=f'Mean: {mean_vol:.4f}', alpha=0.7)
            ax.legend()
            
            # Plot returns if requested
            if show_returns:
                ax = axes[idx, 1]
                returns_data = self.returns[col].dropna()
                ax.plot(returns_data.index, returns_data.values, 
                       linewidth=0.5, alpha=0.6, color='black')
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax.set_title(f'{col} - Returns')
                ax.set_xlabel('Date')
                ax.set_ylabel('Returns')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_multiple_windows(
        self,
        windows_dict: dict,
        figsize: tuple = (14, 6)
    ) -> None:
        """
        Plot volatility for multiple window sizes.
        
        Parameters:
        -----------
        windows_dict : dict
            Dictionary from compute_multiple_windows()
        figsize : tuple, default=(14, 6)
            Figure size
        """
        if not windows_dict:
            raise ValueError("Empty windows dictionary")
        
        # Get first window to determine number of columns
        first_window = list(windows_dict.values())[0]
        n_cols = first_window.shape[1]
        
        fig, axes = plt.subplots(n_cols, 1, figsize=figsize)
        if n_cols == 1:
            axes = [axes]
        
        for idx, col in enumerate(first_window.columns):
            ax = axes[idx]
            
            for window, vol_df in windows_dict.items():
                ax.plot(vol_df.index, vol_df[col], 
                       linewidth=1.5, label=f'{window}-day', alpha=0.8)
            
            ax.set_title(f'{col} - Rolling Volatility (Multiple Windows)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Volatility')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_statistics(
        self,
        volatility: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Get descriptive statistics for volatility.
        
        Parameters:
        -----------
        volatility : pd.DataFrame, optional
            Volatility data
            
        Returns:
        --------
        pd.DataFrame
            Statistics (mean, std, min, max, etc.)
        """
        if volatility is None:
            if self.volatility is None:
                raise ValueError("No volatility data available")
            volatility = self.volatility
        
        stats = pd.DataFrame({
            'mean': volatility.mean(),
            'std': volatility.std(),
            'min': volatility.min(),
            'max': volatility.max(),
            'median': volatility.median(),
            'current': volatility.iloc[-1] if len(volatility) > 0 else np.nan
        })
        
        return stats
    
    def save_volatility(self, filepath: str) -> None:
        """
        Save volatility to CSV.
        
        Parameters:
        -----------
        filepath : str
            Output file path
        """
        if self.volatility is None:
            raise ValueError("No volatility to save")
        
        self.volatility.to_csv(filepath)
        print(f"✓ Volatility saved to: {filepath}")


# Convenience functions
def compute_rolling_volatility(
    returns: pd.DataFrame,
    window: int = 20,
    annualize: bool = False
) -> pd.DataFrame:
    """
    Convenience function to compute rolling volatility.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Returns data
    window : int, default=20
        Rolling window size
    annualize : bool, default=False
        Whether to annualize
        
    Returns:
    --------
    pd.DataFrame
        Rolling volatility
    """
    calc = RollingVolatility(returns)
    vol = calc.compute_volatility(window=window)
    
    if annualize:
        vol = calc.annualize(vol)
    
    return vol
