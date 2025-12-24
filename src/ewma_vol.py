"""
EWMA Volatility Module

Implements Exponentially Weighted Moving Average (EWMA) volatility model.
Follows RiskMetrics methodology with default λ=0.94 for daily data.

Key Features:
- EWMA volatility calculation
- Configurable decay factor (lambda)
- RiskMetrics standard (λ=0.94)
- Annualization
- Forecasting capability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
import warnings


class EWMAVolatility:
    """
    Calculate EWMA volatility from returns data.
    
    EWMA gives more weight to recent observations using exponential decay.
    
    Formula: σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}
    
    Where:
    - λ (lambda) is the decay factor (0 < λ < 1)
    - Higher λ = more weight on historical data (slower adaptation)
    - Lower λ = more weight on recent data (faster adaptation)
    - RiskMetrics uses λ=0.94 for daily data
    
    Attributes:
        returns (pd.DataFrame): Returns data
        volatility (pd.DataFrame): Computed EWMA volatility
        lambda_param (float): Decay factor
        annualized (bool): Whether volatility is annualized
    """
    
    def __init__(self, returns: Optional[pd.DataFrame] = None):
        """
        Initialize EWMA volatility calculator.
        
        Parameters:
        -----------
        returns : pd.DataFrame, optional
            Returns data with DatetimeIndex
        """
        self.returns = returns
        self.volatility = None
        self.lambda_param = None
        self.annualized = False
    
    def compute_volatility(
        self,
        returns: Optional[pd.DataFrame] = None,
        lambda_param: float = 0.94,
        initial_vol: Optional[float] = None,
        min_periods: int = 20
    ) -> pd.DataFrame:
        """
        Compute EWMA volatility.
        
        Parameters:
        -----------
        returns : pd.DataFrame, optional
            Returns data. If None, uses self.returns
        lambda_param : float, default=0.94
            Decay factor (0 < λ < 1). RiskMetrics standard is 0.94 for daily data
        initial_vol : float, optional
            Initial volatility estimate. If None, uses simple std of first min_periods
        min_periods : int, default=20
            Minimum periods for initial volatility calculation
            
        Returns:
        --------
        pd.DataFrame
            EWMA volatility (daily, not annualized)
        """
        if returns is None:
            if self.returns is None:
                raise ValueError("No returns data provided")
            returns = self.returns
        else:
            self.returns = returns
        
        if not 0 < lambda_param < 1:
            raise ValueError("Lambda parameter must be between 0 and 1")
        
        self.lambda_param = lambda_param
        
        # Initialize volatility DataFrame
        volatility = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for col in returns.columns:
            series = returns[col].dropna()
            
            if len(series) < min_periods:
                warnings.warn(f"Not enough data for {col}. Need at least {min_periods} observations.")
                continue
            
            # Initialize variance
            if initial_vol is None:
                # Use simple variance of first min_periods
                initial_variance = series.iloc[:min_periods].var()
            else:
                initial_variance = initial_vol ** 2
            
            # Compute EWMA variance recursively
            variances = np.zeros(len(series))
            variances[0] = initial_variance
            
            squared_returns = series.values ** 2
            
            for t in range(1, len(series)):
                variances[t] = (lambda_param * variances[t-1] + 
                               (1 - lambda_param) * squared_returns[t-1])
            
            # Convert variance to volatility (std dev)
            vols = np.sqrt(variances)
            
            # Assign to DataFrame
            volatility.loc[series.index, col] = vols
        
        # Convert to float
        volatility = volatility.astype(float)
        
        self.volatility = volatility
        self.annualized = False
        
        return volatility
    
    def compute_volatility_pandas(
        self,
        returns: Optional[pd.DataFrame] = None,
        span: Optional[int] = None,
        alpha: Optional[float] = None,
        adjust: bool = True
    ) -> pd.DataFrame:
        """
        Compute EWMA volatility using pandas ewm method.
        
        This is an alternative implementation using pandas' built-in EWMA.
        
        Parameters:
        -----------
        returns : pd.DataFrame, optional
            Returns data
        span : int, optional
            Span parameter. Related to lambda by: λ = 1 - 2/(span+1)
            For λ=0.94, span ≈ 32
        alpha : float, optional
            Direct specification of alpha = 1 - λ
        adjust : bool, default=True
            Whether to use adjustment in beginning of series
            
        Returns:
        --------
        pd.DataFrame
            EWMA volatility
        """
        if returns is None:
            if self.returns is None:
                raise ValueError("No returns data provided")
            returns = self.returns
        
        if span is None and alpha is None:
            # Default to lambda=0.94, which means alpha=0.06, span≈32
            alpha = 0.06
        
        # Compute EWMA of squared returns, then take sqrt
        squared_returns = returns ** 2
        
        if alpha is not None:
            ewma_var = squared_returns.ewm(alpha=alpha, adjust=adjust).mean()
        else:
            ewma_var = squared_returns.ewm(span=span, adjust=adjust).mean()
        
        volatility = np.sqrt(ewma_var)
        
        self.volatility = volatility
        self.lambda_param = 1 - alpha if alpha else 1 - 2/(span+1)
        self.annualized = False
        
        return volatility
    
    def forecast_volatility(
        self,
        steps: int = 1,
        volatility: Optional[pd.DataFrame] = None,
        returns: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Forecast future volatility using EWMA model.
        
        For EWMA, the forecast is the current volatility (constant forecast).
        Multi-step forecasts converge to long-run average.
        
        Parameters:
        -----------
        steps : int, default=1
            Number of steps to forecast
        volatility : pd.DataFrame, optional
            Current volatility
        returns : pd.DataFrame, optional
            Returns data for long-run average calculation
            
        Returns:
        --------
        pd.DataFrame
            Forecasted volatility
        """
        if volatility is None:
            if self.volatility is None:
                raise ValueError("No volatility data available")
            volatility = self.volatility
        
        if steps == 1:
            # One-step ahead forecast = current volatility
            forecast = volatility.iloc[-1:].copy()
            return forecast
        else:
            # Multi-step forecast (simplified: use current vol)
            # More sophisticated: converge to long-run average
            if returns is None:
                returns = self.returns
            
            current_vol = volatility.iloc[-1]
            forecasts = pd.DataFrame(index=range(steps), columns=volatility.columns)
            
            for step in range(steps):
                forecasts.iloc[step] = current_vol
            
            return forecasts
    
    def annualize(
        self,
        volatility: Optional[pd.DataFrame] = None,
        periods_per_year: int = 252
    ) -> pd.DataFrame:
        """
        Annualize volatility.
        
        Parameters:
        -----------
        volatility : pd.DataFrame, optional
            Daily volatility
        periods_per_year : int, default=252
            Number of periods per year
            
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
    
    def plot_volatility(
        self,
        volatility: Optional[pd.DataFrame] = None,
        figsize: tuple = (14, 6),
        compare_to_rolling: bool = False,
        rolling_window: int = 20
    ) -> None:
        """
        Plot EWMA volatility with optional rolling volatility comparison.
        
        Parameters:
        -----------
        volatility : pd.DataFrame, optional
            Volatility data
        figsize : tuple, default=(14, 6)
            Figure size
        compare_to_rolling : bool, default=False
            Whether to overlay rolling volatility for comparison
        rolling_window : int, default=20
            Window size for rolling volatility comparison
        """
        if volatility is None:
            if self.volatility is None:
                raise ValueError("No volatility data to plot")
            volatility = self.volatility
        
        n_cols = volatility.shape[1]
        fig, axes = plt.subplots(n_cols, 1, figsize=figsize)
        
        if n_cols == 1:
            axes = [axes]
        
        for idx, col in enumerate(volatility.columns):
            ax = axes[idx]
            
            # Plot EWMA volatility
            ax.plot(volatility.index, volatility[col], 
                   linewidth=1.5, label=f'EWMA (λ={self.lambda_param:.2f})', 
                   color='darkred')
            
            # Optionally compare to rolling volatility
            if compare_to_rolling and self.returns is not None:
                rolling_vol = self.returns[col].rolling(window=rolling_window).std()
                if self.annualized:
                    rolling_vol = rolling_vol * np.sqrt(252)
                ax.plot(rolling_vol.index, rolling_vol, 
                       linewidth=1.5, label=f'Rolling ({rolling_window}-day)', 
                       color='blue', alpha=0.7)
            
            ax.set_title(f'{col} - EWMA Volatility')
            ax.set_xlabel('Date')
            ylabel = 'Annualized Volatility' if self.annualized else 'Daily Volatility'
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compare_lambdas(
        self,
        returns: Optional[pd.DataFrame] = None,
        lambdas: list = [0.90, 0.94, 0.97],
        figsize: tuple = (14, 6)
    ) -> dict:
        """
        Compare EWMA volatility with different lambda values.
        
        Parameters:
        -----------
        returns : pd.DataFrame, optional
            Returns data
        lambdas : list, default=[0.90, 0.94, 0.97]
            List of lambda values to compare
        figsize : tuple, default=(14, 6)
            Figure size
            
        Returns:
        --------
        dict
            Dictionary with lambda values as keys and volatility DataFrames as values
        """
        if returns is None:
            if self.returns is None:
                raise ValueError("No returns data provided")
            returns = self.returns
        
        results = {}
        
        for lambda_val in lambdas:
            calc = EWMAVolatility(returns)
            vol = calc.compute_volatility(lambda_param=lambda_val)
            results[lambda_val] = vol
        
        # Plot comparison
        n_cols = returns.shape[1]
        fig, axes = plt.subplots(n_cols, 1, figsize=figsize)
        
        if n_cols == 1:
            axes = [axes]
        
        for idx, col in enumerate(returns.columns):
            ax = axes[idx]
            
            for lambda_val, vol_df in results.items():
                ax.plot(vol_df.index, vol_df[col], 
                       linewidth=1.5, label=f'λ={lambda_val}', alpha=0.8)
            
            ax.set_title(f'{col} - EWMA Volatility (Different λ)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Volatility')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return results
    
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
            Statistics
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
        print(f"✓ EWMA volatility saved to: {filepath}")


# Convenience functions
def compute_ewma_volatility(
    returns: pd.DataFrame,
    lambda_param: float = 0.94,
    annualize: bool = False
) -> pd.DataFrame:
    """
    Convenience function to compute EWMA volatility.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Returns data
    lambda_param : float, default=0.94
        Decay factor (RiskMetrics standard)
    annualize : bool, default=False
        Whether to annualize
        
    Returns:
    --------
    pd.DataFrame
        EWMA volatility
    """
    calc = EWMAVolatility(returns)
    vol = calc.compute_volatility(lambda_param=lambda_param)
    
    if annualize:
        vol = calc.annualize(vol)
    
    return vol
