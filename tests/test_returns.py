"""
Unit tests for returns module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from returns import (
    ReturnsCalculator,
    compute_log_returns,
    compute_simple_returns,
    analyze_returns
)


@pytest.fixture
def sample_prices():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    # Create prices with known pattern
    prices = pd.DataFrame({
        'AAPL': 100 * np.exp(np.random.randn(100).cumsum() * 0.01),
        'MSFT': 200 * np.exp(np.random.randn(100).cumsum() * 0.01)
    }, index=dates)
    return prices


@pytest.fixture
def simple_prices():
    """Create simple price data with known returns."""
    dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
    prices = pd.DataFrame({
        'STOCK': [100, 110, 105, 115, 120]
    }, index=dates)
    return prices


class TestReturnsCalculator:
    """Test suite for ReturnsCalculator class."""
    
    def test_initialization(self):
        """Test ReturnsCalculator initialization."""
        calc = ReturnsCalculator()
        assert calc.prices is None
        assert calc.returns is None
        assert calc.return_type is None
    
    def test_initialization_with_prices(self, sample_prices):
        """Test initialization with price data."""
        calc = ReturnsCalculator(sample_prices)
        assert calc.prices is not None
        assert calc.returns is None
    
    def test_compute_log_returns(self, sample_prices):
        """Test log returns calculation."""
        calc = ReturnsCalculator(sample_prices)
        returns = calc.compute_log_returns()
        
        assert isinstance(returns, pd.DataFrame)
        assert len(returns) == len(sample_prices) - 1  # First row is NaN and dropped
        assert returns.isna().sum().sum() == 0
        assert calc.return_type == 'log'
    
    def test_compute_log_returns_values(self, simple_prices):
        """Test log returns with known values."""
        calc = ReturnsCalculator(simple_prices)
        returns = calc.compute_log_returns()
        
        # Verify first return: ln(110/100) ≈ 0.0953
        assert abs(returns.iloc[0, 0] - np.log(110/100)) < 1e-10
        
        # Verify second return: ln(105/110) ≈ -0.0465
        assert abs(returns.iloc[1, 0] - np.log(105/110)) < 1e-10
    
    def test_compute_log_returns_no_drop_na(self, sample_prices):
        """Test log returns without dropping NaN."""
        calc = ReturnsCalculator(sample_prices)
        returns = calc.compute_log_returns(drop_na=False)
        
        assert len(returns) == len(sample_prices)
        assert returns.iloc[0].isna().all()  # First row should be NaN
    
    def test_compute_simple_returns(self, sample_prices):
        """Test simple returns calculation."""
        calc = ReturnsCalculator(sample_prices)
        returns = calc.compute_simple_returns()
        
        assert isinstance(returns, pd.DataFrame)
        assert len(returns) == len(sample_prices) - 1
        assert calc.return_type == 'simple'
    
    def test_compute_simple_returns_values(self, simple_prices):
        """Test simple returns with known values."""
        calc = ReturnsCalculator(simple_prices)
        returns = calc.compute_simple_returns()
        
        # Verify first return: (110-100)/100 = 0.10
        assert abs(returns.iloc[0, 0] - 0.10) < 1e-10
        
        # Verify second return: (105-110)/110 ≈ -0.04545
        assert abs(returns.iloc[1, 0] - (-5/110)) < 1e-10
    
    def test_compute_percent_returns(self, simple_prices):
        """Test percent returns calculation."""
        calc = ReturnsCalculator(simple_prices)
        returns = calc.compute_percent_returns()
        
        # Should be 100x simple returns
        assert abs(returns.iloc[0, 0] - 10.0) < 1e-8  # 10%
        assert calc.return_type == 'percent'
    
    def test_log_vs_simple_returns(self, sample_prices):
        """Test relationship between log and simple returns."""
        calc = ReturnsCalculator(sample_prices)
        
        log_ret = calc.compute_log_returns()
        simple_ret = calc.compute_simple_returns()
        
        # For small returns: log(1+r) ≈ r
        # Check that they're close for small returns
        small_returns = simple_ret[abs(simple_ret) < 0.05]
        if len(small_returns) > 0:
            log_equiv = np.log(1 + small_returns)
            diff = abs(log_equiv - log_ret.loc[small_returns.index])
            assert (diff < 0.001).all().all()
    
    def test_returns_no_prices_error(self):
        """Test error when computing returns without prices."""
        calc = ReturnsCalculator()
        with pytest.raises(ValueError):
            calc.compute_log_returns()
    
    def test_zero_prices_warning(self):
        """Test warning for zero prices."""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        prices = pd.DataFrame({'STOCK': [100, 0, 105, 110, 115]}, index=dates)
        
        calc = ReturnsCalculator(prices)
        with pytest.warns(UserWarning):
            calc.compute_log_returns()
    
    def test_get_statistics(self, sample_prices):
        """Test statistics calculation."""
        calc = ReturnsCalculator(sample_prices)
        calc.compute_log_returns()
        
        stats = calc.get_statistics()
        
        assert isinstance(stats, pd.DataFrame)
        assert 'mean' in stats.columns
        assert 'std' in stats.columns
        assert 'skewness' in stats.columns
        assert 'kurtosis' in stats.columns
        assert len(stats) == sample_prices.shape[1]
    
    def test_get_statistics_no_returns_error(self):
        """Test error when getting statistics without returns."""
        calc = ReturnsCalculator()
        with pytest.raises(ValueError):
            calc.get_statistics()
    
    def test_check_stationarity(self, sample_prices):
        """Test stationarity check."""
        calc = ReturnsCalculator(sample_prices)
        calc.compute_log_returns()
        
        stationarity = calc.check_stationarity()
        
        assert isinstance(stationarity, dict)
        for ticker in sample_prices.columns:
            assert ticker in stationarity
            assert 'mean_change_normalized' in stationarity[ticker]
            assert 'variance_ratio' in stationarity[ticker]
            assert 'looks_stationary' in stationarity[ticker]
    
    def test_save_returns(self, tmp_path, sample_prices):
        """Test saving returns to CSV."""
        calc = ReturnsCalculator(sample_prices)
        calc.compute_log_returns()
        
        output_path = tmp_path / "returns.csv"
        calc.save_returns(str(output_path))
        
        assert output_path.exists()
        
        # Verify saved data
        loaded = pd.read_csv(output_path, index_col=0, parse_dates=True)
        assert len(loaded) == len(calc.returns)
    
    def test_save_returns_no_data_error(self):
        """Test error when saving without computing returns."""
        calc = ReturnsCalculator()
        with pytest.raises(ValueError):
            calc.save_returns("test.csv")


class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    def test_compute_log_returns_function(self, sample_prices):
        """Test compute_log_returns convenience function."""
        returns = compute_log_returns(sample_prices)
        
        assert isinstance(returns, pd.DataFrame)
        assert len(returns) == len(sample_prices) - 1
    
    def test_compute_simple_returns_function(self, sample_prices):
        """Test compute_simple_returns convenience function."""
        returns = compute_simple_returns(sample_prices)
        
        assert isinstance(returns, pd.DataFrame)
        assert len(returns) == len(sample_prices) - 1
    
    def test_analyze_returns_log(self, sample_prices):
        """Test analyze_returns with log returns."""
        result = analyze_returns(sample_prices, return_type='log')
        
        assert 'returns' in result
        assert 'statistics' in result
        assert 'stationarity' in result
        assert 'calculator' in result
        
        assert isinstance(result['returns'], pd.DataFrame)
        assert isinstance(result['statistics'], pd.DataFrame)
        assert isinstance(result['stationarity'], dict)
    
    def test_analyze_returns_simple(self, sample_prices):
        """Test analyze_returns with simple returns."""
        result = analyze_returns(sample_prices, return_type='simple')
        
        assert result['calculator'].return_type == 'simple'
    
    def test_analyze_returns_invalid_type(self, sample_prices):
        """Test analyze_returns with invalid return type."""
        with pytest.raises(ValueError):
            analyze_returns(sample_prices, return_type='invalid')


class TestReturnsProperties:
    """Test mathematical properties of returns."""
    
    def test_returns_sum_to_log_price_change(self, simple_prices):
        """Test that log returns sum to log price change."""
        calc = ReturnsCalculator(simple_prices)
        returns = calc.compute_log_returns()
        
        # Sum of log returns should equal log(P_final / P_initial)
        total_return = returns.sum().values[0]
        expected = np.log(simple_prices.iloc[-1, 0] / simple_prices.iloc[0, 0])
        
        assert abs(total_return - expected) < 1e-10
    
    def test_returns_approximately_zero_mean(self, sample_prices):
        """Test that returns have approximately zero mean (random walk)."""
        calc = ReturnsCalculator(sample_prices)
        returns = calc.compute_log_returns()
        
        # Mean should be close to zero (within 1 std error)
        for col in returns.columns:
            mean = returns[col].mean()
            std = returns[col].std()
            std_error = std / np.sqrt(len(returns))
            
            # Should be within 3 standard errors of zero
            assert abs(mean) < 3 * std_error
    
    def test_returns_symmetry(self):
        """Test that returns are symmetric for up/down movements."""
        # Create symmetric price movements
        dates = pd.date_range(start='2023-01-01', periods=3, freq='D')
        prices = pd.DataFrame({'STOCK': [100, 110, 100]}, index=dates)
        
        calc = ReturnsCalculator(prices)
        log_ret = calc.compute_log_returns()
        
        # Log returns should be symmetric (r_up = -r_down)
        assert abs(log_ret.iloc[0, 0] + log_ret.iloc[1, 0]) < 1e-10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_price(self):
        """Test with single price (no returns possible)."""
        dates = pd.date_range(start='2023-01-01', periods=1, freq='D')
        prices = pd.DataFrame({'STOCK': [100]}, index=dates)
        
        calc = ReturnsCalculator(prices)
        returns = calc.compute_log_returns()
        
        assert len(returns) == 0
    
    def test_two_prices(self):
        """Test with two prices (one return)."""
        dates = pd.date_range(start='2023-01-01', periods=2, freq='D')
        prices = pd.DataFrame({'STOCK': [100, 110]}, index=dates)
        
        calc = ReturnsCalculator(prices)
        returns = calc.compute_log_returns()
        
        assert len(returns) == 1
        assert abs(returns.iloc[0, 0] - np.log(1.1)) < 1e-10
    
    def test_constant_prices(self):
        """Test with constant prices (zero returns)."""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        prices = pd.DataFrame({'STOCK': [100] * 10}, index=dates)
        
        calc = ReturnsCalculator(prices)
        returns = calc.compute_log_returns()
        
        assert (abs(returns) < 1e-10).all().all()
    
    def test_multiple_tickers(self, sample_prices):
        """Test with multiple tickers."""
        calc = ReturnsCalculator(sample_prices)
        returns = calc.compute_log_returns()
        
        assert returns.shape[1] == sample_prices.shape[1]
        assert all(col in returns.columns for col in sample_prices.columns)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
