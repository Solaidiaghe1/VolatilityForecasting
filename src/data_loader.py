"""
Data Loader Module

Handles loading price data from CSV files or Yahoo Finance API.
Implements data cleaning, validation, and preprocessing for volatility analysis.

Key Features:
- Load data from CSV or yfinance
- Handle missing data with forward fill and interpolation
- Detect and handle zero/negative prices
- Validate data integrity
- Support multiple tickers
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Union, List, Optional
from pathlib import Path
import warnings


class DataLoader:
    """
    Load and clean price data for volatility forecasting.
    
    Attributes:
        data (pd.DataFrame): Loaded price data with DatetimeIndex
        tickers (List[str]): List of ticker symbols
    """
    
    def __init__(self):
        """Initialize DataLoader."""
        self.data = None
        self.tickers = []
    
    def load_from_csv(
        self, 
        filepath: Union[str, Path],
        date_column: str = 'Date',
        price_column: str = 'Close',
        parse_dates: bool = True
    ) -> pd.DataFrame:
        """
        Load price data from a CSV file.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the CSV file
        date_column : str, default='Date'
            Name of the date column
        price_column : str, default='Close'
            Name of the price column (or columns if multiple tickers)
        parse_dates : bool, default=True
            Whether to parse dates automatically
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with DatetimeIndex and price columns
            
        Raises:
        -------
        FileNotFoundError
            If the CSV file doesn't exist
        ValueError
            If required columns are missing
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        # Load CSV
        try:
            df = pd.read_csv(filepath, parse_dates=[date_column] if parse_dates else None)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
        
        # Validate columns
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in CSV")
        
        # Set date as index
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
        df.sort_index(inplace=True)
        
        # Identify price columns (all columns except date)
        price_cols = [col for col in df.columns if col != date_column]
        
        if len(price_cols) == 0:
            raise ValueError("No price columns found in CSV")
        
        # Store data and ticker info
        self.data = df[price_cols]
        self.tickers = price_cols
        
        print(f"✓ Loaded data from CSV: {filepath.name}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Tickers: {', '.join(self.tickers)}")
        print(f"  Total rows: {len(df)}")
        
        return self.data
    
    def load_from_yfinance(
        self,
        tickers: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "5y",
        interval: str = "1d",
        auto_adjust: bool = True
    ) -> pd.DataFrame:
        """
        Load price data from Yahoo Finance API.
        
        Parameters:
        -----------
        tickers : str or List[str]
            Single ticker or list of tickers (e.g., 'AAPL' or ['AAPL', 'MSFT'])
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        period : str, default='5y'
            Period to download (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '5y', 'max')
            Ignored if start_date is specified
        interval : str, default='1d'
            Data interval (e.g., '1m', '5m', '1h', '1d', '1wk', '1mo')
        auto_adjust : bool, default=True
            Automatically adjust OHLC prices for splits and dividends
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with DatetimeIndex and adjusted close prices
            
        Raises:
        -------
        ValueError
            If data download fails or tickers are invalid
        """
        # Convert single ticker to list
        if isinstance(tickers, str):
            tickers = [tickers]
        
        self.tickers = tickers
        
        print(f"⏳ Downloading data from Yahoo Finance...")
        print(f"  Tickers: {', '.join(tickers)}")
        
        try:
            # Download data
            if start_date:
                data = yf.download(
                    tickers,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    progress=False
                )
            else:
                data = yf.download(
                    tickers,
                    period=period,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    progress=False
                )
            
            if data.empty:
                raise ValueError(f"No data downloaded for tickers: {tickers}")
            
            # Extract close prices
            if len(tickers) == 1:
                # Single ticker - yfinance returns flat structure
                if 'Close' in data.columns:
                    self.data = data[['Close']].copy()
                    self.data.columns = tickers
                else:
                    # Already flat structure
                    self.data = data.copy()
            else:
                # Multiple tickers - extract Close prices
                if 'Close' in data.columns.get_level_values(0):
                    self.data = data['Close'].copy()
                else:
                    self.data = data.copy()
            
            # Ensure DatetimeIndex
            self.data.index = pd.to_datetime(self.data.index)
            
            print(f"✓ Successfully downloaded data")
            print(f"  Date range: {self.data.index.min()} to {self.data.index.max()}")
            print(f"  Total rows: {len(self.data)}")
            
            return self.data
            
        except Exception as e:
            raise ValueError(f"Error downloading data from Yahoo Finance: {e}")
    
    def clean_data(
        self,
        handle_missing: str = 'ffill',
        handle_zeros: str = 'ffill',
        drop_na_threshold: float = 0.5,
        interpolate_limit: int = 5
    ) -> pd.DataFrame:
        """
        Clean price data by handling missing values and zero prices.
        
        Parameters:
        -----------
        handle_missing : str, default='ffill'
            Method to handle missing values:
            - 'ffill': Forward fill
            - 'bfill': Backward fill
            - 'interpolate': Linear interpolation
            - 'drop': Drop rows with NaN
        handle_zeros : str, default='ffill'
            Method to handle zero prices:
            - 'ffill': Forward fill from last valid price
            - 'nan': Replace with NaN (then handle with handle_missing)
            - 'keep': Keep zeros (not recommended)
        drop_na_threshold : float, default=0.5
            Drop columns with more than this fraction of missing data (0-1)
        interpolate_limit : int, default=5
            Maximum number of consecutive NaNs to interpolate
            
        Returns:
        --------
        pd.DataFrame
            Cleaned price data
            
        Raises:
        -------
        ValueError
            If no data has been loaded
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_from_csv() or load_from_yfinance() first.")
        
        df = self.data.copy()
        initial_rows = len(df)
        initial_nans = df.isna().sum().sum()
        
        print(f"\n🧹 Cleaning data...")
        print(f"  Initial rows: {initial_rows}")
        print(f"  Initial NaN count: {initial_nans}")
        
        # Handle zero or negative prices
        zero_mask = (df <= 0)
        zero_count = zero_mask.sum().sum()
        
        if zero_count > 0:
            warnings.warn(f"Found {zero_count} zero or negative prices")
            
            if handle_zeros == 'ffill':
                df = df.mask(zero_mask).ffill()
            elif handle_zeros == 'nan':
                df = df.mask(zero_mask)
            elif handle_zeros != 'keep':
                raise ValueError(f"Invalid handle_zeros option: {handle_zeros}")
        
        # Drop columns with too many missing values
        missing_fraction = df.isna().sum() / len(df)
        cols_to_drop = missing_fraction[missing_fraction > drop_na_threshold].index.tolist()
        
        if cols_to_drop:
            warnings.warn(f"Dropping columns with >{drop_na_threshold*100}% missing data: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
            self.tickers = [t for t in self.tickers if t not in cols_to_drop]
        
        # Handle remaining missing values
        if handle_missing == 'ffill':
            df = df.ffill()
        elif handle_missing == 'bfill':
            df = df.bfill()
        elif handle_missing == 'interpolate':
            df = df.interpolate(method='linear', limit=interpolate_limit, limit_area='inside')
            df = df.ffill().bfill()  # Handle edges
        elif handle_missing == 'drop':
            df = df.dropna()
        else:
            raise ValueError(f"Invalid handle_missing option: {handle_missing}")
        
        # Final check for any remaining NaNs
        final_nans = df.isna().sum().sum()
        if final_nans > 0:
            warnings.warn(f"Still have {final_nans} NaN values after cleaning. Dropping rows...")
            df = df.dropna()
        
        final_rows = len(df)
        rows_dropped = initial_rows - final_rows
        
        print(f"✓ Cleaning complete")
        print(f"  Final rows: {final_rows} ({rows_dropped} dropped)")
        print(f"  Remaining NaN count: {df.isna().sum().sum()}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        
        self.data = df
        return self.data
    
    def validate_data(self) -> dict:
        """
        Validate data integrity and return summary statistics.
        
        Returns:
        --------
        dict
            Dictionary containing validation results and statistics
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        validation = {
            'is_valid': True,
            'issues': [],
            'stats': {}
        }
        
        df = self.data
        
        # Check for NaN values
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            validation['is_valid'] = False
            validation['issues'].append(f"Contains {nan_count} NaN values")
        
        # Check for zero or negative prices
        zero_count = (df <= 0).sum().sum()
        if zero_count > 0:
            validation['is_valid'] = False
            validation['issues'].append(f"Contains {zero_count} zero or negative prices")
        
        # Check for duplicate dates
        if df.index.duplicated().any():
            validation['is_valid'] = False
            validation['issues'].append("Contains duplicate dates")
        
        # Check date gaps (for daily data)
        date_diffs = df.index.to_series().diff()
        large_gaps = date_diffs[date_diffs > pd.Timedelta(days=7)]
        if len(large_gaps) > 0:
            validation['issues'].append(f"Found {len(large_gaps)} gaps >7 days")
        
        # Summary statistics
        validation['stats'] = {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'date_range': (df.index.min(), df.index.max()),
            'tickers': self.tickers,
            'price_min': df.min().to_dict(),
            'price_max': df.max().to_dict(),
            'price_mean': df.mean().to_dict()
        }
        
        return validation
    
    def save_to_csv(self, filepath: Union[str, Path]) -> None:
        """
        Save cleaned data to CSV.
        
        Parameters:
        -----------
        filepath : str or Path
            Output CSV file path
        """
        if self.data is None:
            raise ValueError("No data to save.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.data.to_csv(filepath)
        print(f"✓ Data saved to: {filepath}")
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the loaded and cleaned data.
        
        Returns:
        --------
        pd.DataFrame
            Price data with DatetimeIndex
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        return self.data.copy()


# Convenience functions
def load_prices(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Convenience function to load prices from CSV.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded and cleaned price data
    """
    loader = DataLoader()
    loader.load_from_csv(filepath)
    loader.clean_data()
    return loader.get_data()


def fetch_prices(
    tickers: Union[str, List[str]],
    start_date: Optional[str] = None,
    period: str = "5y"
) -> pd.DataFrame:
    """
    Convenience function to fetch prices from Yahoo Finance.
    
    Parameters:
    -----------
    tickers : str or List[str]
        Ticker symbol(s)
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format
    period : str, default='5y'
        Period to download if start_date not specified
        
    Returns:
    --------
    pd.DataFrame
        Downloaded and cleaned price data
    """
    loader = DataLoader()
    loader.load_from_yfinance(tickers, start_date=start_date, period=period)
    loader.clean_data()
    return loader.get_data()
