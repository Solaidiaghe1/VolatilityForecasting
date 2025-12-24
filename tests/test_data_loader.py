"""
Unit tests for data_loader module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import DataLoader, load_prices, fetch_prices


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = {
        'Date': dates,
        'AAPL': np.random.uniform(150, 200, len(dates)),
        'MSFT': np.random.uniform(300, 400, len(dates))
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(tmp_path, sample_csv_data):
    """Create a temporary CSV file for testing."""
    csv_path = tmp_path / "test_prices.csv"
    sample_csv_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def data_with_missing_values():
    """Create data with missing values for testing cleaning."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'AAPL': np.random.uniform(150, 200, 100),
        'MSFT': np.random.uniform(300, 400, 100)
    }, index=dates)
    
    # Insert missing values
    data.iloc[10:15, 0] = np.nan
    data.iloc[50, 1] = np.nan
    
    return data


@pytest.fixture
def data_with_zeros():
    """Create data with zero prices for testing."""
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    data = pd.DataFrame({
        'AAPL': np.random.uniform(150, 200, 50),
    }, index=dates)
    
    # Insert zeros
    data.iloc[20:23] = 0
    
    return data


class TestDataLoader:
    """Test suite for DataLoader class."""
    
    def test_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader.data is None
        assert loader.tickers == []
    
    def test_load_from_csv(self, sample_csv_file):
        """Test loading data from CSV."""
        loader = DataLoader()
        data = loader.load_from_csv(sample_csv_file)
        
        assert isinstance(data, pd.DataFrame)
        assert isinstance(data.index, pd.DatetimeIndex)
        assert len(loader.tickers) == 2
        assert 'AAPL' in loader.tickers
        assert 'MSFT' in loader.tickers
        assert len(data) > 0
    
    def test_load_from_csv_file_not_found(self):
        """Test loading from non-existent CSV."""
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_from_csv("nonexistent_file.csv")
    
    def test_load_from_csv_custom_columns(self, tmp_path):
        """Test loading CSV with custom column names."""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'price': np.random.uniform(100, 200, 10)
        })
        csv_path = tmp_path / "custom.csv"
        df.to_csv(csv_path, index=False)
        
        loader = DataLoader()
        data = loader.load_from_csv(csv_path, date_column='date', price_column='price')
        
        assert len(data) == 10
        assert isinstance(data.index, pd.DatetimeIndex)
    
    def test_clean_data_ffill(self, data_with_missing_values):
        """Test cleaning data with forward fill."""
        loader = DataLoader()
        loader.data = data_with_missing_values
        loader.tickers = ['AAPL', 'MSFT']
        
        cleaned = loader.clean_data(handle_missing='ffill')
        
        assert cleaned.isna().sum().sum() == 0
        assert len(cleaned) == len(data_with_missing_values)
    
    def test_clean_data_interpolate(self, data_with_missing_values):
        """Test cleaning data with interpolation."""
        loader = DataLoader()
        loader.data = data_with_missing_values
        loader.tickers = ['AAPL', 'MSFT']
        
        cleaned = loader.clean_data(handle_missing='interpolate')
        
        assert cleaned.isna().sum().sum() == 0
    
    def test_clean_data_zeros(self, data_with_zeros):
        """Test handling of zero prices."""
        loader = DataLoader()
        loader.data = data_with_zeros
        loader.tickers = ['AAPL']
        
        cleaned = loader.clean_data(handle_zeros='ffill')
        
        # Should have no zeros
        assert (cleaned <= 0).sum().sum() == 0
    
    def test_clean_data_drop_threshold(self):
        """Test dropping columns with too many missing values."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'AAPL': np.random.uniform(150, 200, 100),
            'BAD': [np.nan] * 100  # All NaN
        }, index=dates)
        
        loader = DataLoader()
        loader.data = data
        loader.tickers = ['AAPL', 'BAD']
        
        cleaned = loader.clean_data(drop_na_threshold=0.5)
        
        assert 'BAD' not in cleaned.columns
        assert 'AAPL' in cleaned.columns
    
    def test_validate_data_clean(self, sample_csv_file):
        """Test validation of clean data."""
        loader = DataLoader()
        loader.load_from_csv(sample_csv_file)
        
        validation = loader.validate_data()
        
        assert validation['is_valid'] == True
        assert len(validation['issues']) == 0
        assert 'stats' in validation
        assert validation['stats']['n_rows'] > 0
    
    def test_validate_data_with_issues(self, data_with_missing_values):
        """Test validation of data with issues."""
        loader = DataLoader()
        loader.data = data_with_missing_values
        loader.tickers = ['AAPL', 'MSFT']
        
        validation = loader.validate_data()
        
        assert validation['is_valid'] == False
        assert len(validation['issues']) > 0
    
    def test_save_to_csv(self, tmp_path, sample_csv_file):
        """Test saving data to CSV."""
        loader = DataLoader()
        loader.load_from_csv(sample_csv_file)
        
        output_path = tmp_path / "output.csv"
        loader.save_to_csv(output_path)
        
        assert output_path.exists()
        
        # Verify saved data
        saved_data = pd.read_csv(output_path, index_col=0, parse_dates=True)
        assert len(saved_data) > 0
    
    def test_get_data(self, sample_csv_file):
        """Test getting data from loader."""
        loader = DataLoader()
        loader.load_from_csv(sample_csv_file)
        
        data = loader.get_data()
        
        assert isinstance(data, pd.DataFrame)
        assert data is not loader.data  # Should be a copy (different object)
        assert data.equals(loader.data)  # But values should be the same
    
    def test_get_data_no_data(self):
        """Test getting data when none is loaded."""
        loader = DataLoader()
        with pytest.raises(ValueError):
            loader.get_data()


class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    def test_load_prices(self, sample_csv_file):
        """Test load_prices convenience function."""
        data = load_prices(sample_csv_file)
        
        assert isinstance(data, pd.DataFrame)
        assert isinstance(data.index, pd.DatetimeIndex)
        assert len(data) > 0
    
    @pytest.mark.skipif(
        True,  # Skip by default to avoid API calls
        reason="Requires internet connection and API access"
    )
    def test_fetch_prices(self):
        """Test fetch_prices convenience function (requires internet)."""
        data = fetch_prices('AAPL', period='1mo')
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_row_data(self):
        """Test handling of single row data."""
        dates = pd.date_range(start='2023-01-01', periods=1, freq='D')
        data = pd.DataFrame({'AAPL': [150.0]}, index=dates)
        
        loader = DataLoader()
        loader.data = data
        loader.tickers = ['AAPL']
        
        cleaned = loader.clean_data()
        assert len(cleaned) == 1
    
    def test_negative_prices(self):
        """Test handling of negative prices."""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({'AAPL': [150, -100, 160, 170, 180, 190, 200, 210, 220, 230]}, index=dates)
        
        loader = DataLoader()
        loader.data = data
        loader.tickers = ['AAPL']
        
        cleaned = loader.clean_data(handle_zeros='ffill')
        
        # Negative price should be replaced
        assert (cleaned <= 0).sum().sum() == 0
    
    def test_all_missing_data(self):
        """Test handling when all data is missing."""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({'AAPL': [np.nan] * 10}, index=dates)
        
        loader = DataLoader()
        loader.data = data
        loader.tickers = ['AAPL']
        
        cleaned = loader.clean_data(drop_na_threshold=0.5)
        
        # Should drop the column
        assert len(cleaned.columns) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
