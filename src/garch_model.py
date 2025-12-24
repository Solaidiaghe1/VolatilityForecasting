"""
GARCH Model Module

Implements GARCH(1,1) volatility forecasting model.

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models
volatility clustering and time-varying volatility.

Key Features:
- GARCH(1,1) model fitting
- Volatility forecasting (1-step and multi-step)
- Model diagnostics and convergence checks
- Multiple distribution assumptions (normal, Student-t, skewed-t)
- Annualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple
import warnings

try:
    from arch import arch_model
    from arch.univariate import Normal, StudentsT, SkewStudent
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    warnings.warn(
        "arch package not available. Install with: pip install arch",
        ImportWarning
    )


class GARCHModel:
    """
    GARCH(1,1) volatility model.
    
    GARCH(1,1) equation:
    σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
    
    Where:
    - ω (omega) = long-run variance component
    - α (alpha) = ARCH coefficient (reaction to shocks)
    - β (beta) = GARCH coefficient (persistence)
    - Persistence = α + β (should be < 1 for stationarity)
    
    Attributes:
        returns (pd.DataFrame): Returns data
        models (dict): Fitted GARCH models for each ticker
        forecasts (dict): Volatility forecasts
        fitted (bool): Whether models have been fitted
    """
    
    def __init__(self, returns: Optional[pd.DataFrame] = None):
        """
        Initialize GARCH model.
        
        Parameters:
        -----------
        returns : pd.DataFrame, optional
            Returns data with DatetimeIndex
        """
        if not ARCH_AVAILABLE:
            raise ImportError("arch package required. Install with: pip install arch")
        
        self.returns = returns
        self.models = {}
        self.forecasts = {}
        self.fitted = False
    
    def fit(
        self,
        returns: Optional[pd.DataFrame] = None,
        p: int = 1,
        q: int = 1,
        mean: str = 'Constant',
        vol: str = 'GARCH',
        dist: str = 'normal',
        show_summary: bool = False,
        **kwargs
    ) -> dict:
        """
        Fit GARCH model(s) to returns data.
        
        Parameters:
        -----------
        returns : pd.DataFrame, optional
            Returns data. If None, uses self.returns
        p : int, default=1
            GARCH lag order
        q : int, default=1
            ARCH lag order
        mean : str, default='Constant'
            Mean model: 'Constant', 'Zero', 'AR', 'ARX', 'HAR', 'LS'
        vol : str, default='GARCH'
            Volatility model: 'GARCH', 'EGARCH', 'FIGARCH', 'HARCH'
        dist : str, default='normal'
            Error distribution: 'normal', 't', 'skewt', 'ged'
        show_summary : bool, default=False
            Whether to print model summary
        **kwargs
            Additional arguments passed to model.fit()
            
        Returns:
        --------
        dict
            Dictionary of fitted model results for each ticker
        """
        if returns is None:
            if self.returns is None:
                raise ValueError("No returns data provided")
            returns = self.returns
        else:
            self.returns = returns
        
        # Convert percentage returns to decimal if needed
        # GARCH works better with returns scaled appropriately
        if returns.abs().mean().mean() > 1:
            warnings.warn("Returns appear to be in percentage form. Converting to decimal.")
            returns = returns / 100
        
        results = {}
        
        for col in returns.columns:
            series = returns[col].dropna() * 100  # Scale to percentage for fitting
            
            if len(series) < 100:
                warnings.warn(f"Insufficient data for {col}. Need at least 100 observations.")
                continue
            
            try:
                # Create GARCH model
                model = arch_model(
                    series,
                    mean=mean,
                    vol=vol,
                    p=p,
                    q=q,
                    dist=dist
                )
                
                # Fit model
                result = model.fit(disp='off', **kwargs)
                
                results[col] = result
                
                if show_summary:
                    print(f"\n{'='*70}")
                    print(f"GARCH({p},{q}) Model Summary - {col}")
                    print(f"{'='*70}")
                    print(result.summary())
                    
                    # Check convergence
                    if hasattr(result, 'convergence_flag'):
                        if result.convergence_flag != 0:
                            warnings.warn(f"Model for {col} may not have converged properly")
                
            except Exception as e:
                warnings.warn(f"Failed to fit GARCH model for {col}: {e}")
                continue
        
        self.models = results
        self.fitted = True
        
        return results
    
    def forecast(
        self,
        horizon: int = 1,
        method: str = 'analytic',
        reindex: bool = False
    ) -> pd.DataFrame:
        """
        Forecast future volatility using fitted GARCH models.
        
        Parameters:
        -----------
        horizon : int, default=1
            Forecast horizon (number of steps ahead)
        method : str, default='analytic'
            Forecasting method: 'analytic', 'simulation', 'bootstrap'
        reindex : bool, default=False
            Whether to reindex forecasts to match returns index
            
        Returns:
        --------
        pd.DataFrame
            Forecasted volatility (variance, not std dev)
        """
        if not self.fitted or not self.models:
            raise ValueError("Models must be fitted before forecasting. Call fit() first.")
        
        forecasts = {}
        
        for ticker, model_result in self.models.items():
            try:
                # Forecast variance
                forecast_result = model_result.forecast(horizon=horizon, method=method)
                
                # Extract variance forecast
                variance_forecast = forecast_result.variance.iloc[-1]
                
                forecasts[ticker] = variance_forecast
                
            except Exception as e:
                warnings.warn(f"Failed to forecast for {ticker}: {e}")
                continue
        
        # Convert to DataFrame
        forecast_df = pd.DataFrame(forecasts)
        
        # Convert variance to standard deviation (volatility)
        forecast_df = np.sqrt(forecast_df / 10000)  # Undo the *100 scaling
        
        self.forecasts = forecast_df
        
        return forecast_df
    
    def forecast_volatility(
        self,
        horizon: int = 1,
        annualize: bool = False
    ) -> pd.DataFrame:
        """
        Forecast volatility (standard deviation) instead of variance.
        
        Parameters:
        -----------
        horizon : int, default=1
            Forecast horizon
        annualize : bool, default=False
            Whether to annualize the forecast
            
        Returns:
        --------
        pd.DataFrame
            Forecasted volatility (std dev)
        """
        forecast_var = self.forecast(horizon=horizon)
        
        if annualize:
            forecast_var = forecast_var * np.sqrt(252)
        
        return forecast_var
    
    def get_conditional_volatility(self) -> pd.DataFrame:
        """
        Extract conditional volatility from fitted models.
        
        Returns:
        --------
        pd.DataFrame
            Conditional volatility time series for each ticker
        """
        if not self.fitted or not self.models:
            raise ValueError("Models must be fitted first")
        
        cond_vol = pd.DataFrame()
        
        for ticker, model_result in self.models.items():
            # Extract conditional volatility
            vol = model_result.conditional_volatility / 100  # Undo scaling
            cond_vol[ticker] = vol
        
        return cond_vol
    
    def get_parameters(self) -> pd.DataFrame:
        """
        Extract GARCH model parameters.
        
        Returns:
        --------
        pd.DataFrame
            Model parameters (omega, alpha, beta) for each ticker
        """
        if not self.fitted or not self.models:
            raise ValueError("Models must be fitted first")
        
        params = {}
        
        for ticker, model_result in self.models.items():
            param_dict = model_result.params.to_dict()
            
            # Extract key GARCH parameters
            garch_params = {
                'omega': param_dict.get('omega', np.nan),
                'alpha[1]': param_dict.get('alpha[1]', np.nan),
                'beta[1]': param_dict.get('beta[1]', np.nan)
            }
            
            # Calculate persistence
            alpha = garch_params['alpha[1]']
            beta = garch_params['beta[1]']
            if not np.isnan(alpha) and not np.isnan(beta):
                garch_params['persistence'] = alpha + beta
            else:
                garch_params['persistence'] = np.nan
            
            params[ticker] = garch_params
        
        return pd.DataFrame(params).T
    
    def check_stationarity(self) -> pd.DataFrame:
        """
        Check if GARCH models are stationary (persistence < 1).
        
        Returns:
        --------
        pd.DataFrame
            Stationarity check results
        """
        params = self.get_parameters()
        
        checks = pd.DataFrame({
            'persistence': params['persistence'],
            'is_stationary': params['persistence'] < 1.0,
            'long_run_var': params['omega'] / (1 - params['persistence'])
        })
        
        return checks
    
    def plot_conditional_volatility(
        self,
        figsize: tuple = (14, 6),
        show_forecast: bool = True
    ) -> None:
        """
        Plot conditional volatility from fitted models.
        
        Parameters:
        -----------
        figsize : tuple, default=(14, 6)
            Figure size
        show_forecast : bool, default=True
            Whether to show forecast on plot
        """
        cond_vol = self.get_conditional_volatility()
        
        n_cols = len(cond_vol.columns)
        fig, axes = plt.subplots(n_cols, 1, figsize=figsize)
        
        if n_cols == 1:
            axes = [axes]
        
        for idx, col in enumerate(cond_vol.columns):
            ax = axes[idx]
            
            # Plot conditional volatility
            ax.plot(cond_vol.index, cond_vol[col], 
                   linewidth=1.5, label='Conditional Volatility', color='darkred')
            
            # Optionally show forecast
            if show_forecast and col in self.forecasts.columns:
                forecast_val = self.forecasts[col].iloc[0]
                ax.axhline(y=forecast_val, color='blue', linestyle='--',
                          label=f'1-step Forecast: {forecast_val:.4f}', alpha=0.7)
            
            ax.set_title(f'{col} - GARCH(1,1) Conditional Volatility')
            ax.set_xlabel('Date')
            ax.set_ylabel('Volatility')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_forecast(
        self,
        figsize: tuple = (12, 6)
    ) -> None:
        """
        Plot volatility forecasts.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 6)
            Figure size
        """
        if self.forecasts.empty:
            raise ValueError("No forecasts available. Call forecast() first.")
        
        # Plot as bar chart
        self.forecasts.T.plot(kind='bar', figsize=figsize, rot=0)
        plt.title('GARCH Volatility Forecasts')
        plt.xlabel('Ticker')
        plt.ylabel('Forecasted Volatility')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    
    def diagnostic_plot(self, ticker: str) -> None:
        """
        Create diagnostic plots for a specific ticker's GARCH model.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol to plot
        """
        if ticker not in self.models:
            raise ValueError(f"No model fitted for {ticker}")
        
        model_result = self.models[ticker]
        
        # Create diagnostic plots
        fig = model_result.plot(annualize='D')
        plt.tight_layout()
        plt.show()
    
    def save_forecasts(self, filepath: str) -> None:
        """
        Save forecasts to CSV.
        
        Parameters:
        -----------
        filepath : str
            Output file path
        """
        if self.forecasts.empty:
            raise ValueError("No forecasts to save")
        
        self.forecasts.to_csv(filepath)
        print(f"✓ GARCH forecasts saved to: {filepath}")


# Convenience functions
def fit_garch(
    returns: pd.DataFrame,
    p: int = 1,
    q: int = 1,
    show_summary: bool = False
) -> dict:
    """
    Convenience function to fit GARCH model.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Returns data
    p : int, default=1
        GARCH lag order
    q : int, default=1
        ARCH lag order
    show_summary : bool, default=False
        Whether to print summary
        
    Returns:
    --------
    dict
        Fitted model results
    """
    model = GARCHModel(returns)
    results = model.fit(p=p, q=q, show_summary=show_summary)
    return results


def forecast_garch(
    returns: pd.DataFrame,
    horizon: int = 1,
    annualize: bool = False
) -> pd.DataFrame:
    """
    Convenience function to fit GARCH and forecast.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Returns data
    horizon : int, default=1
        Forecast horizon
    annualize : bool, default=False
        Whether to annualize
        
    Returns:
    --------
    pd.DataFrame
        Forecasted volatility
    """
    model = GARCHModel(returns)
    model.fit()
    forecast = model.forecast_volatility(horizon=horizon, annualize=annualize)
    return forecast
