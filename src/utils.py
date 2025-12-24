"""
Utility Functions and Helpers

Common utility functions for volatility forecasting and analysis:
- Annualization helpers
- Date/time utilities
- Parameter validation
- Plotting utilities
- Data transformation helpers
- Statistical utilities

Author: Volatility Forecasting Project
Date: December 24, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, List, Tuple, Dict, Any
from datetime import datetime, timedelta
import warnings
from pathlib import Path


# ============================================================================
# ANNUALIZATION HELPERS
# ============================================================================

def annualize_volatility(
    volatility: Union[float, pd.Series, np.ndarray],
    periods_per_year: int = 252,
    current_period: str = 'daily'
) -> Union[float, pd.Series, np.ndarray]:
    """
    Annualize volatility from any frequency.
    
    Parameters:
    -----------
    volatility : float, Series, or ndarray
        Volatility to annualize
    periods_per_year : int, default=252
        Trading periods per year (252 for daily, 52 for weekly, etc.)
    current_period : str, default='daily'
        Current frequency ('daily', 'weekly', 'monthly', 'intraday')
        
    Returns:
    --------
    Annualized volatility (same type as input)
    """
    if current_period == 'daily':
        scaling_factor = np.sqrt(252)
    elif current_period == 'weekly':
        scaling_factor = np.sqrt(52)
    elif current_period == 'monthly':
        scaling_factor = np.sqrt(12)
    elif current_period == 'intraday':
        # Assume hourly, need custom scaling
        scaling_factor = np.sqrt(periods_per_year)
    else:
        scaling_factor = np.sqrt(periods_per_year)
    
    return volatility * scaling_factor


def deannualize_volatility(
    volatility: Union[float, pd.Series, np.ndarray],
    target_period: str = 'daily'
) -> Union[float, pd.Series, np.ndarray]:
    """
    Convert annualized volatility to target period.
    
    Parameters:
    -----------
    volatility : float, Series, or ndarray
        Annualized volatility
    target_period : str, default='daily'
        Target frequency ('daily', 'weekly', 'monthly')
        
    Returns:
    --------
    De-annualized volatility
    """
    if target_period == 'daily':
        scaling_factor = np.sqrt(252)
    elif target_period == 'weekly':
        scaling_factor = np.sqrt(52)
    elif target_period == 'monthly':
        scaling_factor = np.sqrt(12)
    else:
        scaling_factor = np.sqrt(252)
    
    return volatility / scaling_factor


def annualize_returns(
    returns: Union[float, pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> Union[float, pd.Series, np.ndarray]:
    """
    Annualize returns.
    
    Parameters:
    -----------
    returns : float, Series, or ndarray
        Returns to annualize (arithmetic mean)
    periods_per_year : int, default=252
        Trading periods per year
        
    Returns:
    --------
    Annualized returns
    """
    return returns * periods_per_year


# ============================================================================
# DATE/TIME UTILITIES
# ============================================================================

def get_trading_days(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    freq: str = 'B'
) -> pd.DatetimeIndex:
    """
    Generate trading days between dates.
    
    Parameters:
    -----------
    start_date : str or datetime
        Start date
    end_date : str or datetime
        End date
    freq : str, default='B'
        Frequency ('B' for business days, 'D' for daily)
        
    Returns:
    --------
    pd.DatetimeIndex
        Trading days
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def align_timestamps(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    method: str = 'inner'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two DataFrames by timestamp index.
    
    Parameters:
    -----------
    df1, df2 : pd.DataFrame
        DataFrames to align
    method : str, default='inner'
        Join method ('inner', 'outer', 'left', 'right')
        
    Returns:
    --------
    tuple
        (aligned_df1, aligned_df2)
    """
    common_index = df1.index.intersection(df2.index)
    
    if method == 'inner':
        return df1.loc[common_index], df2.loc[common_index]
    elif method == 'outer':
        all_index = df1.index.union(df2.index)
        return df1.reindex(all_index), df2.reindex(all_index)
    elif method == 'left':
        return df1, df2.reindex(df1.index)
    elif method == 'right':
        return df1.reindex(df2.index), df2
    else:
        raise ValueError(f"Invalid method: {method}")


def format_timestamp(
    timestamp: Union[str, datetime, pd.Timestamp],
    format_str: str = '%Y-%m-%d %H:%M:%S'
) -> str:
    """
    Format timestamp to string.
    
    Parameters:
    -----------
    timestamp : str, datetime, or pd.Timestamp
        Timestamp to format
    format_str : str
        Format string
        
    Returns:
    --------
    str
        Formatted timestamp
    """
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
    return timestamp.strftime(format_str)


# ============================================================================
# PARAMETER VALIDATION
# ============================================================================

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    name: str = "DataFrame"
) -> None:
    """
    Validate DataFrame structure.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : list, optional
        Required column names
    name : str
        Name for error messages
        
    Raises:
    -------
    ValueError
        If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError(f"{name} is empty")
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        warnings.warn(f"{name} index is not DatetimeIndex")


def validate_parameters(
    params: Dict[str, Any],
    required: List[str],
    valid_values: Optional[Dict[str, List[Any]]] = None
) -> None:
    """
    Validate parameter dictionary.
    
    Parameters:
    -----------
    params : dict
        Parameters to validate
    required : list
        Required parameter names
    valid_values : dict, optional
        Valid values for specific parameters
        
    Raises:
    -------
    ValueError
        If validation fails
    """
    # Check required parameters
    missing = set(required) - set(params.keys())
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")
    
    # Check valid values
    if valid_values:
        for param, valid in valid_values.items():
            if param in params:
                if params[param] not in valid:
                    raise ValueError(
                        f"Invalid value for {param}: {params[param]}. "
                        f"Valid values: {valid}"
                    )


def validate_numeric_range(
    value: Union[int, float],
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    name: str = "Value"
) -> None:
    """
    Validate numeric value is in range.
    
    Parameters:
    -----------
    value : int or float
        Value to validate
    min_value : float, optional
        Minimum allowed value
    max_value : float, optional
        Maximum allowed value
    name : str
        Name for error messages
        
    Raises:
    -------
    ValueError
        If validation fails
    """
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value}")


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================

def setup_plot_style(style: str = 'seaborn-v0_8-darkgrid'):
    """
    Set up matplotlib plotting style.
    
    Parameters:
    -----------
    style : str
        Matplotlib style name
    """
    try:
        plt.style.use(style)
    except:
        plt.style.use('default')
    
    # Set default parameters
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9


def save_figure(
    fig: plt.Figure,
    filepath: Union[str, Path],
    dpi: int = 300,
    bbox_inches: str = 'tight'
) -> None:
    """
    Save matplotlib figure to file.
    
    Parameters:
    -----------
    fig : plt.Figure
        Figure to save
    filepath : str or Path
        Output filepath
    dpi : int, default=300
        Resolution
    bbox_inches : str, default='tight'
        Bounding box setting
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    print(f"✓ Figure saved to: {filepath}")


def create_figure_grid(
    n_plots: int,
    ncols: int = 3,
    figsize: Optional[Tuple[int, int]] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a grid of subplots.
    
    Parameters:
    -----------
    n_plots : int
        Number of plots needed
    ncols : int, default=3
        Number of columns
    figsize : tuple, optional
        Figure size (if None, auto-calculated)
        
    Returns:
    --------
    tuple
        (fig, axes)
    """
    nrows = int(np.ceil(n_plots / ncols))
    
    if figsize is None:
        figsize = (ncols * 5, nrows * 4)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Flatten axes array for easier iteration
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    # Hide extra subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    return fig, axes


def add_recession_shading(
    ax: plt.Axes,
    start_dates: List[datetime],
    end_dates: List[datetime],
    alpha: float = 0.2
) -> None:
    """
    Add shaded regions to plot (e.g., for recessions).
    
    Parameters:
    -----------
    ax : plt.Axes
        Axes to add shading to
    start_dates : list
        Start dates for shaded regions
    end_dates : list
        End dates for shaded regions
    alpha : float, default=0.2
        Transparency
    """
    for start, end in zip(start_dates, end_dates):
        ax.axvspan(start, end, alpha=alpha, color='gray', label='Recession')


# ============================================================================
# DATA TRANSFORMATION
# ============================================================================

def winsorize(
    data: Union[pd.Series, np.ndarray],
    lower_pct: float = 1,
    upper_pct: float = 99
) -> Union[pd.Series, np.ndarray]:
    """
    Winsorize data by percentiles.
    
    Parameters:
    -----------
    data : Series or ndarray
        Data to winsorize
    lower_pct : float, default=1
        Lower percentile cutoff
    upper_pct : float, default=99
        Upper percentile cutoff
        
    Returns:
    --------
    Winsorized data (same type as input)
    """
    lower_val = np.percentile(data, lower_pct)
    upper_val = np.percentile(data, upper_pct)
    
    if isinstance(data, pd.Series):
        return data.clip(lower=lower_val, upper=upper_val)
    else:
        return np.clip(data, lower_val, upper_val)


def standardize(
    data: Union[pd.Series, pd.DataFrame],
    method: str = 'zscore'
) -> Union[pd.Series, pd.DataFrame]:
    """
    Standardize data.
    
    Parameters:
    -----------
    data : Series or DataFrame
        Data to standardize
    method : str, default='zscore'
        Standardization method ('zscore', 'minmax', 'robust')
        
    Returns:
    --------
    Standardized data
    """
    if method == 'zscore':
        return (data - data.mean()) / data.std()
    
    elif method == 'minmax':
        return (data - data.min()) / (data.max() - data.min())
    
    elif method == 'robust':
        median = data.median()
        mad = (data - median).abs().median()
        return (data - median) / mad
    
    else:
        raise ValueError(f"Invalid method: {method}")


def fill_missing_data(
    data: pd.DataFrame,
    method: str = 'ffill',
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Fill missing data in DataFrame.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with missing values
    method : str, default='ffill'
        Fill method ('ffill', 'bfill', 'interpolate', 'mean', 'median')
    limit : int, optional
        Maximum number of consecutive NaN to fill
        
    Returns:
    --------
    pd.DataFrame
        Data with missing values filled
    """
    if method == 'ffill':
        return data.fillna(method='ffill', limit=limit)
    elif method == 'bfill':
        return data.fillna(method='bfill', limit=limit)
    elif method == 'interpolate':
        return data.interpolate(method='linear', limit=limit)
    elif method == 'mean':
        return data.fillna(data.mean())
    elif method == 'median':
        return data.fillna(data.median())
    else:
        raise ValueError(f"Invalid method: {method}")


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

def calculate_rolling_stats(
    data: pd.Series,
    window: int = 20,
    stats: List[str] = ['mean', 'std']
) -> pd.DataFrame:
    """
    Calculate rolling statistics.
    
    Parameters:
    -----------
    data : pd.Series
        Input data
    window : int, default=20
        Rolling window size
    stats : list, default=['mean', 'std']
        Statistics to calculate
        
    Returns:
    --------
    pd.DataFrame
        Rolling statistics
    """
    result = pd.DataFrame(index=data.index)
    
    rolling = data.rolling(window=window)
    
    if 'mean' in stats:
        result['mean'] = rolling.mean()
    if 'std' in stats:
        result['std'] = rolling.std()
    if 'min' in stats:
        result['min'] = rolling.min()
    if 'max' in stats:
        result['max'] = rolling.max()
    if 'median' in stats:
        result['median'] = rolling.median()
    if 'skew' in stats:
        result['skew'] = rolling.skew()
    if 'kurt' in stats:
        result['kurt'] = rolling.kurt()
    
    return result


def calculate_correlation_matrix(
    data: pd.DataFrame,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Calculate correlation matrix.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    method : str, default='pearson'
        Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    return data.corr(method=method)


def detect_outliers(
    data: Union[pd.Series, np.ndarray],
    method: str = 'iqr',
    threshold: float = 3.0
) -> np.ndarray:
    """
    Detect outliers in data.
    
    Parameters:
    -----------
    data : Series or ndarray
        Input data
    method : str, default='iqr'
        Detection method ('iqr', 'zscore', 'mad')
    threshold : float, default=3.0
        Threshold for outlier detection
        
    Returns:
    --------
    np.ndarray
        Boolean array (True = outlier)
    """
    if method == 'iqr':
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (data < lower) | (data > upper)
    
    elif method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return z_scores > threshold
    
    elif method == 'mad':
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z = 0.6745 * (data - median) / mad
        return np.abs(modified_z) > threshold
    
    else:
        raise ValueError(f"Invalid method: {method}")


# ============================================================================
# FILE I/O UTILITIES
# ============================================================================

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not.
    
    Parameters:
    -----------
    path : str or Path
        Directory path
        
    Returns:
    --------
    Path
        Directory path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    format: str = 'csv',
    **kwargs
) -> None:
    """
    Save DataFrame to file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    filepath : str or Path
        Output filepath
    format : str, default='csv'
        File format ('csv', 'parquet', 'excel')
    **kwargs
        Additional arguments for save method
    """
    filepath = Path(filepath)
    ensure_directory(filepath.parent)
    
    if format == 'csv':
        df.to_csv(filepath, **kwargs)
    elif format == 'parquet':
        df.to_parquet(filepath, **kwargs)
    elif format == 'excel':
        df.to_excel(filepath, **kwargs)
    else:
        raise ValueError(f"Invalid format: {format}")
    
    print(f"✓ Saved DataFrame to: {filepath}")


def load_dataframe(
    filepath: Union[str, Path],
    format: str = 'csv',
    **kwargs
) -> pd.DataFrame:
    """
    Load DataFrame from file.
    
    Parameters:
    -----------
    filepath : str or Path
        Input filepath
    format : str, default='csv'
        File format ('csv', 'parquet', 'excel')
    **kwargs
        Additional arguments for load method
        
    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if format == 'csv':
        return pd.read_csv(filepath, **kwargs)
    elif format == 'parquet':
        return pd.read_parquet(filepath, **kwargs)
    elif format == 'excel':
        return pd.read_excel(filepath, **kwargs)
    else:
        raise ValueError(f"Invalid format: {format}")


# ============================================================================
# PERFORMANCE UTILITIES
# ============================================================================

def calculate_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.
    
    Parameters:
    -----------
    returns : Series or ndarray
        Return series
    risk_free_rate : float, default=0.0
        Risk-free rate (annualized)
    periods_per_year : int, default=252
        Periods per year for annualization
        
    Returns:
    --------
    float
        Sharpe ratio
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio.
    
    Parameters:
    -----------
    returns : Series or ndarray
        Return series
    risk_free_rate : float, default=0.0
        Risk-free rate (annualized)
    periods_per_year : int, default=252
        Periods per year for annualization
        
    Returns:
    --------
    float
        Sortino ratio
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    
    if downside_std == 0:
        return 0.0
    
    return (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)


def calculate_max_drawdown(
    returns: Union[pd.Series, np.ndarray]
) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown.
    
    Parameters:
    -----------
    returns : Series or ndarray
        Return series
        
    Returns:
    --------
    tuple
        (max_drawdown, start_idx, end_idx)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    max_dd = drawdown.min()
    end_idx = drawdown.idxmin() if isinstance(drawdown, pd.Series) else drawdown.argmin()
    
    # Find start of drawdown
    if isinstance(drawdown, pd.Series):
        start_idx = cumulative[:end_idx].idxmax()
    else:
        start_idx = cumulative[:end_idx].argmax()
    
    return max_dd, start_idx, end_idx


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Volatility Forecasting - Utilities Module")
    print("=" * 60)
    print("\nAvailable functions:")
    print("  - annualize_volatility()")
    print("  - validate_dataframe()")
    print("  - setup_plot_style()")
    print("  - winsorize()")
    print("  - calculate_sharpe_ratio()")
    print("  - and more...")
    print("\nSee module docstrings for details.")
