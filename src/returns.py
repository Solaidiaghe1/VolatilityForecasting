"""
Returns Calculation Module

Computes various types of returns from price data for volatility analysis.

Key Features:
- Log returns (preferred for volatility modeling)
- Simple returns
- Percent returns
- Return statistics and visualization
- Handle missing data gracefully
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, Tuple
import warnings


class ReturnsCalculator:
    """
    Calculate and analyze returns from price data.
    
    Attributes:
        prices (pd.DataFrame): Price data with DatetimeIndex
        returns (pd.DataFrame): Computed returns
        return_type (str): Type of returns computed ('log', 'simple', 'percent')
    """
    
    def __init__(self, prices: Optional[pd.DataFrame] = None):
        """
        Initialize ReturnsCalculator.
        
        Parameters:
        -----------
        prices : pd.DataFrame, optional
            Price data with DatetimeIndex
        """
        self.prices = prices
        self.returns = None
        self.return_type = None
    
    def compute_log_returns(
        self,
        prices: Optional[pd.DataFrame] = None,
        drop_na: bool = True
    ) -> pd.DataFrame:
        """
        Compute log returns from price data.
        
        Log returns are preferred for volatility modeling because:
        - They are time-additive
        - Approximately normal for small changes
        - Symmetric (loss and gain have same magnitude)
        
        Formula: r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
        
        Parameters:
        -----------
        prices : pd.DataFrame, optional
            Price data. If None, uses self.prices
        drop_na : bool, default=True
            Whether to drop NaN values (first row will be NaN)
            
        Returns:
        --------
        pd.DataFrame
            Log returns with same structure as input prices
            
        Raises:
        -------
        ValueError
            If prices contain zeros or negative values
        """
        if prices is None:
            if self.prices is None:
                raise ValueError("No price data provided")
            prices = self.prices
        else:
            self.prices = prices
        
        # Validate prices
        if (prices <= 0).any().any():
            zero_count = (prices <= 0).sum().sum()
            warnings.warn(
                f"Found {zero_count} zero or negative prices. "
                "These will produce NaN or inf in log returns."
            )
        
        # Compute log returns
        log_prices = np.log(prices)
        returns = log_prices.diff()
        
        if drop_na:
            returns = returns.dropna()
        
        self.returns = returns
        self.return_type = 'log'
        
        return returns
    
    def compute_simple_returns(
        self,
        prices: Optional[pd.DataFrame] = None,
        drop_na: bool = True
    ) -> pd.DataFrame:
        """
        Compute simple returns from price data.
        
        Formula: r_t = (P_t - P_{t-1}) / P_{t-1} = P_t / P_{t-1} - 1
        
        Parameters:
        -----------
        prices : pd.DataFrame, optional
            Price data. If None, uses self.prices
        drop_na : bool, default=True
            Whether to drop NaN values (first row will be NaN)
            
        Returns:
        --------
        pd.DataFrame
            Simple returns with same structure as input prices
        """
        if prices is None:
            if self.prices is None:
                raise ValueError("No price data provided")
            prices = self.prices
        else:
            self.prices = prices
        
        # Validate prices
        if (prices == 0).any().any():
            warnings.warn("Found zero prices. These will produce inf in simple returns.")
        
        # Compute simple returns
        returns = prices.pct_change()
        
        if drop_na:
            returns = returns.dropna()
        
        self.returns = returns
        self.return_type = 'simple'
        
        return returns
    
    def compute_percent_returns(
        self,
        prices: Optional[pd.DataFrame] = None,
        drop_na: bool = True
    ) -> pd.DataFrame:
        """
        Compute percent returns from price data.
        
        This is the same as simple returns but expressed as percentage.
        Formula: r_t = ((P_t - P_{t-1}) / P_{t-1}) * 100
        
        Parameters:
        -----------
        prices : pd.DataFrame, optional
            Price data. If None, uses self.prices
        drop_na : bool, default=True
            Whether to drop NaN values
            
        Returns:
        --------
        pd.DataFrame
            Percent returns (0.01 = 1%)
        """
        simple_returns = self.compute_simple_returns(prices, drop_na)
        returns = simple_returns * 100
        
        self.returns = returns
        self.return_type = 'percent'
        
        return returns
    
    def get_statistics(self, returns: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute descriptive statistics for returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame, optional
            Returns data. If None, uses self.returns
            
        Returns:
        --------
        pd.DataFrame
            Statistics including mean, std, skew, kurtosis, min, max
        """
        if returns is None:
            if self.returns is None:
                raise ValueError("No returns data available")
            returns = self.returns
        
        stats = pd.DataFrame({
            'mean': returns.mean(),
            'std': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'min': returns.min(),
            'max': returns.max(),
            'median': returns.median(),
            'count': returns.count()
        })
        
        return stats
    
    def check_stationarity(self, returns: Optional[pd.DataFrame] = None) -> dict:
        """
        Perform basic stationarity checks on returns.
        
        Returns should be stationary for volatility modeling.
        This performs simple checks:
        - Rolling mean stability
        - Rolling variance stability
        
        Parameters:
        -----------
        returns : pd.DataFrame, optional
            Returns data. If None, uses self.returns
            
        Returns:
        --------
        dict
            Dictionary with stationarity metrics
        """
        if returns is None:
            if self.returns is None:
                raise ValueError("No returns data available")
            returns = self.returns
        
        results = {}
        
        for col in returns.columns:
            series = returns[col].dropna()
            
            # Split into halves
            mid = len(series) // 2
            first_half = series.iloc[:mid]
            second_half = series.iloc[mid:]
            
            # Compare means and variances
            mean_change = abs(second_half.mean() - first_half.mean()) / first_half.std()
            var_ratio = second_half.var() / first_half.var()
            
            results[col] = {
                'mean_change_normalized': mean_change,
                'variance_ratio': var_ratio,
                'looks_stationary': mean_change < 1.0 and 0.5 < var_ratio < 2.0
            }
        
        return results
    
    def plot_returns(
        self,
        returns: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (14, 8),
        show_distribution: bool = True,
        show_qq: bool = True
    ) -> None:
        """
        Create comprehensive visualization of returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame, optional
            Returns data. If None, uses self.returns
        figsize : tuple, default=(14, 8)
            Figure size (width, height)
        show_distribution : bool, default=True
            Show distribution histogram with normal overlay
        show_qq : bool, default=True
            Show Q-Q plot for normality check
        """
        if returns is None:
            if self.returns is None:
                raise ValueError("No returns data available")
            returns = self.returns
        
        n_cols = returns.shape[1]
        
        # Create subplots
        if show_distribution and show_qq:
            fig, axes = plt.subplots(n_cols, 3, figsize=figsize)
        elif show_distribution or show_qq:
            fig, axes = plt.subplots(n_cols, 2, figsize=figsize)
        else:
            fig, axes = plt.subplots(n_cols, 1, figsize=figsize)
        
        # Handle single ticker case
        if n_cols == 1:
            axes = axes.reshape(1, -1)
        
        for idx, col in enumerate(returns.columns):
            series = returns[col].dropna()
            
            # Time series plot
            ax_idx = 0
            axes[idx, ax_idx].plot(series.index, series.values, linewidth=0.8, alpha=0.7)
            axes[idx, ax_idx].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[idx, ax_idx].set_title(f'{col} - Returns Over Time')
            axes[idx, ax_idx].set_xlabel('Date')
            axes[idx, ax_idx].set_ylabel('Returns')
            axes[idx, ax_idx].grid(True, alpha=0.3)
            
            # Distribution plot
            if show_distribution:
                ax_idx += 1
                axes[idx, ax_idx].hist(series, bins=50, density=True, alpha=0.7, edgecolor='black')
                
                # Overlay normal distribution
                mu, sigma = series.mean(), series.std()
                x = np.linspace(series.min(), series.max(), 100)
                axes[idx, ax_idx].plot(
                    x,
                    (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                    'r-',
                    linewidth=2,
                    label='Normal'
                )
                axes[idx, ax_idx].set_title(f'{col} - Distribution')
                axes[idx, ax_idx].set_xlabel('Returns')
                axes[idx, ax_idx].set_ylabel('Density')
                axes[idx, ax_idx].legend()
                axes[idx, ax_idx].grid(True, alpha=0.3)
            
            # Q-Q plot
            if show_qq:
                ax_idx += 1
                from scipy import stats
                stats.probplot(series, dist="norm", plot=axes[idx, ax_idx])
                axes[idx, ax_idx].set_title(f'{col} - Q-Q Plot')
                axes[idx, ax_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_rolling_stats(
        self,
        returns: Optional[pd.DataFrame] = None,
        window: int = 20,
        figsize: Tuple[int, int] = (14, 6)
    ) -> None:
        """
        Plot rolling mean and standard deviation to check stationarity.
        
        Parameters:
        -----------
        returns : pd.DataFrame, optional
            Returns data. If None, uses self.returns
        window : int, default=20
            Rolling window size
        figsize : tuple, default=(14, 6)
            Figure size
        """
        if returns is None:
            if self.returns is None:
                raise ValueError("No returns data available")
            returns = self.returns
        
        n_cols = returns.shape[1]
        fig, axes = plt.subplots(n_cols, 2, figsize=figsize)
        
        # Handle single ticker case
        if n_cols == 1:
            axes = axes.reshape(1, -1)
        
        for idx, col in enumerate(returns.columns):
            series = returns[col].dropna()
            
            # Rolling mean
            rolling_mean = series.rolling(window=window).mean()
            axes[idx, 0].plot(series.index, rolling_mean, linewidth=1.5)
            axes[idx, 0].axhline(y=series.mean(), color='r', linestyle='--', label='Overall Mean')
            axes[idx, 0].set_title(f'{col} - Rolling Mean ({window}-day)')
            axes[idx, 0].set_xlabel('Date')
            axes[idx, 0].set_ylabel('Mean Return')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Rolling std
            rolling_std = series.rolling(window=window).std()
            axes[idx, 1].plot(series.index, rolling_std, linewidth=1.5, color='orange')
            axes[idx, 1].axhline(y=series.std(), color='r', linestyle='--', label='Overall Std')
            axes[idx, 1].set_title(f'{col} - Rolling Std Dev ({window}-day)')
            axes[idx, 1].set_xlabel('Date')
            axes[idx, 1].set_ylabel('Std Dev')
            axes[idx, 1].legend()
            axes[idx, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_returns(self, filepath: str) -> None:
        """
        Save computed returns to CSV.
        
        Parameters:
        -----------
        filepath : str
            Output file path
        """
        if self.returns is None:
            raise ValueError("No returns to save. Compute returns first.")
        
        self.returns.to_csv(filepath)
        print(f"✓ Returns saved to: {filepath}")


# Convenience functions
def compute_log_returns(prices: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Convenience function to compute log returns.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data with DatetimeIndex
    drop_na : bool, default=True
        Whether to drop NaN values
        
    Returns:
    --------
    pd.DataFrame
        Log returns
    """
    calc = ReturnsCalculator(prices)
    return calc.compute_log_returns(drop_na=drop_na)


def compute_simple_returns(prices: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Convenience function to compute simple returns.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data with DatetimeIndex
    drop_na : bool, default=True
        Whether to drop NaN values
        
    Returns:
    --------
    pd.DataFrame
        Simple returns
    """
    calc = ReturnsCalculator(prices)
    return calc.compute_simple_returns(drop_na=drop_na)


def analyze_returns(prices: pd.DataFrame, return_type: str = 'log') -> dict:
    """
    Convenience function to compute returns and get comprehensive analysis.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data with DatetimeIndex
    return_type : str, default='log'
        Type of returns ('log', 'simple', 'percent')
        
    Returns:
    --------
    dict
        Dictionary containing returns, statistics, and stationarity checks
    """
    calc = ReturnsCalculator(prices)
    
    # Compute returns
    if return_type == 'log':
        returns = calc.compute_log_returns()
    elif return_type == 'simple':
        returns = calc.compute_simple_returns()
    elif return_type == 'percent':
        returns = calc.compute_percent_returns()
    else:
        raise ValueError(f"Invalid return_type: {return_type}")
    
    # Get statistics
    stats = calc.get_statistics()
    stationarity = calc.check_stationarity()
    
    return {
        'returns': returns,
        'statistics': stats,
        'stationarity': stationarity,
        'calculator': calc
    }
