"""
Unit Tests for Volatility Regimes Module

Tests cover:
- Regime classification (percentile and fixed methods)
- Regime statistics calculation
- Transition matrix analysis
- Persistence calculations
- Data filtering by regime
- Current regime detection
- Edge cases and error handling
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from volatility_regimes import (
    VolatilityRegimes,
    classify_volatility_regimes,
    analyze_regime_performance
)


class TestVolatilityRegimes(unittest.TestCase):
    """Test cases for VolatilityRegimes class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic volatility data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        
        # Single ticker with clear regimes
        self.volatility_single = pd.DataFrame({
            'AAPL': np.concatenate([
                np.random.uniform(0.10, 0.15, 84),   # Low volatility
                np.random.uniform(0.15, 0.25, 84),   # Medium volatility
                np.random.uniform(0.25, 0.40, 84)    # High volatility
            ])
        }, index=dates)
        
        # Multiple tickers
        self.volatility_multi = pd.DataFrame({
            'AAPL': np.random.uniform(0.10, 0.30, 252),
            'MSFT': np.random.uniform(0.08, 0.25, 252),
            'GOOGL': np.random.uniform(0.12, 0.35, 252)
        }, index=dates)
        
        # Returns data for performance analysis
        self.returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'MSFT': np.random.normal(0.0008, 0.015, 252)
        }, index=dates)
        
        self.classifier = VolatilityRegimes()
    
    def test_init(self):
        """Test initialization."""
        classifier = VolatilityRegimes()
        self.assertIsNone(classifier.volatility)
        self.assertIsNone(classifier.regimes)
        self.assertEqual(classifier.thresholds, {})
        self.assertEqual(classifier.percentiles, (33, 66))
        
        # Initialize with data
        classifier_with_data = VolatilityRegimes(self.volatility_single)
        self.assertIsNotNone(classifier_with_data.volatility)
        pd.testing.assert_frame_equal(classifier_with_data.volatility, self.volatility_single)
    
    def test_classify_regimes_percentile(self):
        """Test percentile-based regime classification."""
        regimes = self.classifier.classify_regimes(
            self.volatility_single,
            method='percentile',
            percentiles=(33, 66)
        )
        
        # Check output shape
        self.assertEqual(regimes.shape, self.volatility_single.shape)
        self.assertTrue(all(regimes.index == self.volatility_single.index))
        
        # Check regime values (0, 1, 2)
        unique_regimes = regimes['AAPL'].dropna().unique()
        self.assertTrue(all(r in [0, 1, 2] for r in unique_regimes))
        
        # Check percentile distribution (approximately 33%, 33%, 33%)
        counts = regimes['AAPL'].value_counts(normalize=True) * 100
        for regime in [0, 1, 2]:
            self.assertGreater(counts[regime], 20)  # At least 20%
            self.assertLess(counts[regime], 45)     # At most 45%
        
        # Check thresholds are stored
        self.assertIn('AAPL', self.classifier.thresholds)
        self.assertIn('low', self.classifier.thresholds['AAPL'])
        self.assertIn('high', self.classifier.thresholds['AAPL'])
        self.assertEqual(self.classifier.thresholds['AAPL']['method'], 'percentile')
    
    def test_classify_regimes_fixed(self):
        """Test fixed threshold regime classification."""
        regimes = self.classifier.classify_regimes(
            self.volatility_single,
            method='fixed',
            low_threshold=0.15,
            high_threshold=0.25
        )
        
        # Check output shape
        self.assertEqual(regimes.shape, self.volatility_single.shape)
        
        # Check that thresholds are correctly applied
        vol_series = self.volatility_single['AAPL']
        regime_series = regimes['AAPL']
        
        # Low regime: volatility <= 0.15
        low_mask = vol_series <= 0.15
        self.assertTrue(all(regime_series[low_mask] == 0))
        
        # High regime: volatility > 0.25
        high_mask = vol_series > 0.25
        self.assertTrue(all(regime_series[high_mask] == 2))
        
        # Medium regime: 0.15 < volatility <= 0.25
        med_mask = (vol_series > 0.15) & (vol_series <= 0.25)
        self.assertTrue(all(regime_series[med_mask] == 1))
        
        # Check thresholds are stored
        self.assertEqual(self.classifier.thresholds['AAPL']['low'], 0.15)
        self.assertEqual(self.classifier.thresholds['AAPL']['high'], 0.25)
        self.assertEqual(self.classifier.thresholds['AAPL']['method'], 'fixed')
    
    def test_classify_regimes_multi_ticker(self):
        """Test regime classification with multiple tickers."""
        regimes = self.classifier.classify_regimes(self.volatility_multi)
        
        # Check all tickers are classified
        self.assertEqual(regimes.shape, self.volatility_multi.shape)
        self.assertEqual(list(regimes.columns), list(self.volatility_multi.columns))
        
        # Check each ticker has thresholds
        for ticker in self.volatility_multi.columns:
            self.assertIn(ticker, self.classifier.thresholds)
    
    def test_classify_regimes_custom_percentiles(self):
        """Test regime classification with custom percentiles."""
        regimes = self.classifier.classify_regimes(
            self.volatility_single,
            percentiles=(25, 75)
        )
        
        # Check that percentiles are stored
        self.assertEqual(self.classifier.percentiles, (25, 75))
        self.assertEqual(
            self.classifier.thresholds['AAPL']['percentiles'],
            (25, 75)
        )
        
        # Low regime should be ~25%, high regime ~25%, medium ~50%
        counts = regimes['AAPL'].value_counts(normalize=True) * 100
        self.assertGreater(counts[1], 40)  # Medium should be largest
    
    def test_classify_regimes_errors(self):
        """Test error handling in regime classification."""
        # No data provided
        with self.assertRaises(ValueError):
            self.classifier.classify_regimes()
        
        # Fixed method without thresholds
        with self.assertRaises(ValueError):
            self.classifier.classify_regimes(
                self.volatility_single,
                method='fixed'
            )
        
        # Invalid method
        with self.assertRaises(ValueError):
            self.classifier.classify_regimes(
                self.volatility_single,
                method='invalid'
            )
    
    def test_get_regime_labels(self):
        """Test regime label retrieval."""
        # String labels
        labels = self.classifier.get_regime_labels(numeric=False)
        self.assertEqual(labels[0], 'Low')
        self.assertEqual(labels[1], 'Medium')
        self.assertEqual(labels[2], 'High')
        
        # Numeric labels
        numeric_labels = self.classifier.get_regime_labels(numeric=True)
        self.assertEqual(numeric_labels[0], 0)
        self.assertEqual(numeric_labels[1], 1)
        self.assertEqual(numeric_labels[2], 2)
    
    def test_get_regime_statistics(self):
        """Test regime statistics calculation."""
        regimes = self.classifier.classify_regimes(self.volatility_single)
        stats = self.classifier.get_regime_statistics()
        
        # Check output structure
        self.assertIsInstance(stats, pd.DataFrame)
        expected_columns = ['ticker', 'regime', 'regime_code', 'count', 
                          'percentage', 'avg_duration', 'max_duration']
        self.assertTrue(all(col in stats.columns for col in expected_columns))
        
        # Check we have stats for all 3 regimes
        self.assertEqual(len(stats), 3)  # Single ticker, 3 regimes
        
        # Check percentages sum to ~100%
        total_pct = stats['percentage'].sum()
        self.assertAlmostEqual(total_pct, 100.0, places=1)
        
        # Check counts match
        total_count = stats['count'].sum()
        self.assertEqual(total_count, len(self.volatility_single))
        
        # Check duration values are reasonable
        self.assertTrue(all(stats['avg_duration'] >= 0))
        self.assertTrue(all(stats['max_duration'] >= stats['avg_duration']))
    
    def test_get_regime_statistics_multi_ticker(self):
        """Test regime statistics with multiple tickers."""
        regimes = self.classifier.classify_regimes(self.volatility_multi)
        stats = self.classifier.get_regime_statistics()
        
        # Should have 3 regimes × 3 tickers = 9 rows
        self.assertEqual(len(stats), 9)
        
        # Check each ticker has 3 regimes
        for ticker in self.volatility_multi.columns:
            ticker_stats = stats[stats['ticker'] == ticker]
            self.assertEqual(len(ticker_stats), 3)
    
    def test_analyze_transitions(self):
        """Test transition matrix analysis."""
        regimes = self.classifier.classify_regimes(self.volatility_single)
        transitions = self.classifier.analyze_transitions()
        
        # Check output structure
        self.assertIsInstance(transitions, dict)
        self.assertIn('AAPL', transitions)
        self.assertIn('counts', transitions['AAPL'])
        self.assertIn('percentages', transitions['AAPL'])
        
        # Check matrix shape (3x3)
        counts_matrix = transitions['AAPL']['counts']
        pct_matrix = transitions['AAPL']['percentages']
        self.assertEqual(counts_matrix.shape, (3, 3))
        self.assertEqual(pct_matrix.shape, (3, 3))
        
        # Check row sums of percentage matrix equal 100% (or 0 for unused regimes)
        for idx, row_sum in enumerate(pct_matrix.sum(axis=1)):
            if row_sum > 0:
                self.assertAlmostEqual(row_sum, 100.0, places=1)
        
        # Check all values are non-negative
        self.assertTrue((counts_matrix >= 0).all().all())
        self.assertTrue((pct_matrix >= 0).all().all())
    
    def test_analyze_transitions_multi_ticker(self):
        """Test transition matrix with multiple tickers."""
        regimes = self.classifier.classify_regimes(self.volatility_multi)
        transitions = self.classifier.analyze_transitions()
        
        # Check all tickers have transition matrices
        for ticker in self.volatility_multi.columns:
            self.assertIn(ticker, transitions)
            self.assertIn('counts', transitions[ticker])
            self.assertIn('percentages', transitions[ticker])
    
    def test_calculate_persistence(self):
        """Test persistence calculation."""
        regimes = self.classifier.classify_regimes(self.volatility_single)
        persistence = self.classifier.calculate_persistence()
        
        # Check output structure
        self.assertIsInstance(persistence, pd.DataFrame)
        self.assertIn('ticker', persistence.columns)
        self.assertIn('regime', persistence.columns)
        self.assertIn('persistence_pct', persistence.columns)
        
        # Should have 3 rows (one per regime)
        self.assertEqual(len(persistence), 3)
        
        # Persistence should be between 0 and 100
        self.assertTrue(all(persistence['persistence_pct'] >= 0))
        self.assertTrue(all(persistence['persistence_pct'] <= 100))
    
    def test_calculate_persistence_multi_ticker(self):
        """Test persistence with multiple tickers."""
        regimes = self.classifier.classify_regimes(self.volatility_multi)
        persistence = self.classifier.calculate_persistence()
        
        # Should have 3 regimes × 3 tickers = 9 rows
        self.assertEqual(len(persistence), 9)
    
    def test_get_current_regime(self):
        """Test current regime retrieval."""
        regimes = self.classifier.classify_regimes(self.volatility_single)
        current = self.classifier.get_current_regime()
        
        # Check output is a Series
        self.assertIsInstance(current, pd.Series)
        
        # Check it has string labels
        self.assertIn(current['AAPL'], ['Low', 'Medium', 'High'])
        
        # Check it matches the last regime
        last_regime_code = regimes['AAPL'].iloc[-1]
        expected_label = self.classifier.get_regime_labels()[last_regime_code]
        self.assertEqual(current['AAPL'], expected_label)
    
    def test_get_current_regime_multi_ticker(self):
        """Test current regime with multiple tickers."""
        regimes = self.classifier.classify_regimes(self.volatility_multi)
        current = self.classifier.get_current_regime()
        
        # Check all tickers have current regime
        self.assertEqual(len(current), 3)
        for ticker in self.volatility_multi.columns:
            self.assertIn(ticker, current.index)
            self.assertIn(current[ticker], ['Low', 'Medium', 'High'])
    
    def test_filter_by_regime_numeric(self):
        """Test filtering data by regime (numeric code)."""
        regimes = self.classifier.classify_regimes(self.volatility_single)
        
        # Filter for low volatility regime (code 0)
        filtered = self.classifier.filter_by_regime(
            self.volatility_single,
            regime=0,
            ticker='AAPL'
        )
        
        # Check that filtered data only contains low regime periods
        self.assertIsInstance(filtered, pd.DataFrame)
        self.assertTrue(len(filtered) > 0)
        
        # Verify all filtered periods have low regime
        for date in filtered.index:
            self.assertEqual(regimes.loc[date, 'AAPL'], 0)
    
    def test_filter_by_regime_string(self):
        """Test filtering data by regime (string label)."""
        regimes = self.classifier.classify_regimes(self.volatility_single)
        
        # Filter for high volatility regime
        filtered = self.classifier.filter_by_regime(
            self.volatility_single,
            regime='High',
            ticker='AAPL'
        )
        
        # Check output
        self.assertIsInstance(filtered, pd.DataFrame)
        self.assertTrue(len(filtered) > 0)
        
        # Verify all filtered periods have high regime
        for date in filtered.index:
            self.assertEqual(regimes.loc[date, 'AAPL'], 2)
    
    def test_filter_by_regime_default_ticker(self):
        """Test filtering with default ticker selection."""
        regimes = self.classifier.classify_regimes(self.volatility_multi)
        
        # Don't specify ticker, should use first column
        filtered = self.classifier.filter_by_regime(
            self.volatility_multi,
            regime=1
        )
        
        self.assertIsInstance(filtered, pd.DataFrame)
        self.assertTrue(len(filtered) > 0)
    
    def test_filter_by_regime_errors(self):
        """Test error handling in filter_by_regime."""
        # No regimes classified
        with self.assertRaises(ValueError):
            self.classifier.filter_by_regime(self.volatility_single, regime=0)
    
    def test_regime_classification_empty_data(self):
        """Test handling of empty/insufficient data."""
        # Very small dataset
        small_data = pd.DataFrame({
            'AAPL': [0.1, 0.2, 0.3]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        with self.assertWarns(UserWarning):
            regimes = self.classifier.classify_regimes(small_data)
    
    def test_regime_with_nan_values(self):
        """Test regime classification with NaN values."""
        # Add NaN values
        vol_with_nan = self.volatility_single.copy()
        vol_with_nan.iloc[10:20, 0] = np.nan
        
        regimes = self.classifier.classify_regimes(vol_with_nan)
        
        # NaN positions should remain NaN in regimes
        self.assertTrue(regimes.iloc[10:20, 0].isna().all())
        
        # Non-NaN positions should be classified
        self.assertFalse(regimes.iloc[:10, 0].isna().any())
        self.assertFalse(regimes.iloc[20:, 0].isna().any())
    
    def test_regime_statistics_consistency(self):
        """Test that statistics are internally consistent."""
        regimes = self.classifier.classify_regimes(self.volatility_single)
        stats = self.classifier.get_regime_statistics()
        
        # Duration checks
        for _, row in stats.iterrows():
            self.assertGreaterEqual(row['max_duration'], row['avg_duration'])
            self.assertGreater(row['count'], 0)
            self.assertGreater(row['percentage'], 0)
    
    def test_convenience_function_classify(self):
        """Test convenience function for classification."""
        regimes = classify_volatility_regimes(
            self.volatility_single,
            percentiles=(33, 66)
        )
        
        # Should return classified regimes
        self.assertIsInstance(regimes, pd.DataFrame)
        self.assertEqual(regimes.shape, self.volatility_single.shape)
        self.assertTrue(all(regimes['AAPL'].dropna().isin([0, 1, 2])))
    
    def test_convenience_function_performance(self):
        """Test convenience function for performance analysis."""
        # First classify regimes
        classifier = VolatilityRegimes(self.volatility_single)
        regimes = classifier.classify_regimes()
        
        # Analyze performance
        performance = analyze_regime_performance(
            self.returns,
            regimes,
            ticker='AAPL'
        )
        
        # Check output structure
        self.assertIsInstance(performance, pd.DataFrame)
        expected_columns = ['regime', 'count', 'mean_return', 'std_return', 
                          'sharpe', 'min_return', 'max_return']
        self.assertTrue(all(col in performance.columns for col in expected_columns))
        
        # Should have 3 regimes
        self.assertEqual(len(performance), 3)
        
        # Check values are reasonable
        self.assertTrue(all(performance['count'] > 0))
        self.assertTrue(all(performance['std_return'] >= 0))
    
    def test_save_regimes(self):
        """Test saving regimes to CSV."""
        import tempfile
        
        regimes = self.classifier.classify_regimes(self.volatility_single)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = f.name
        
        try:
            self.classifier.save_regimes(temp_path)
            
            # Load and verify
            loaded = pd.read_csv(temp_path, index_col=0, parse_dates=True)
            
            # Check shape
            self.assertEqual(loaded.shape, regimes.shape)
            
            # Check values are string labels
            self.assertTrue(all(loaded['AAPL'].isin(['Low', 'Medium', 'High'])))
        
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_save_regimes_error(self):
        """Test error handling when saving without regimes."""
        with self.assertRaises(ValueError):
            self.classifier.save_regimes('dummy.csv')
    
    def test_regime_boundaries(self):
        """Test exact boundary cases in classification."""
        # Create data with exact threshold values - need at least 10 observations
        dates = pd.date_range('2023-01-01', periods=10)
        vol_data = pd.DataFrame({
            'TEST': [0.10, 0.15, 0.20, 0.25, 0.30, 0.12, 0.18, 0.22, 0.28, 0.35]
        }, index=dates)
        
        classifier = VolatilityRegimes()
        regimes = classifier.classify_regimes(
            vol_data,
            method='fixed',
            low_threshold=0.15,
            high_threshold=0.25
        )
        
        # Check boundary handling
        # <= threshold for low, > threshold for transitions
        self.assertEqual(int(regimes.loc[dates[0], 'TEST']), 0)  # 0.10 <= 0.15
        self.assertEqual(int(regimes.loc[dates[1], 'TEST']), 0)  # 0.15 <= 0.15
        self.assertEqual(int(regimes.loc[dates[2], 'TEST']), 1)  # 0.15 < 0.20 <= 0.25
        self.assertEqual(int(regimes.loc[dates[3], 'TEST']), 1)  # 0.15 < 0.25 <= 0.25
        self.assertEqual(int(regimes.loc[dates[4], 'TEST']), 2)  # 0.30 > 0.25
    
    def test_transition_matrix_single_regime(self):
        """Test transition matrix when data is relatively stable."""
        # Create data with gradual changes (more stable)
        dates = pd.date_range('2023-01-01', periods=100)
        # Use a more stable pattern
        stable_vol = pd.DataFrame({
            'TEST': np.concatenate([
                np.repeat(0.10, 40),  # Stay low
                np.repeat(0.15, 30),  # Stay medium
                np.repeat(0.25, 30)   # Stay high
            ])
        }, index=dates)
        
        classifier = VolatilityRegimes()
        regimes = classifier.classify_regimes(
            stable_vol,
            method='fixed',
            low_threshold=0.12,
            high_threshold=0.20
        )
        transitions = classifier.analyze_transitions()
        
        # Check transition matrix structure exists
        pct_matrix = transitions['TEST']['percentages']
        self.assertEqual(pct_matrix.shape, (3, 3))
        
        # With stable data, diagonal (persistence) should be high
        # Sum of diagonal elements should be much higher than off-diagonal
        diagonal_sum = (pct_matrix.iloc[0, 0] + 
                       pct_matrix.iloc[1, 1] + 
                       pct_matrix.iloc[2, 2])
        
        # At least one regime should have very high persistence (>90%)
        max_persistence = max([pct_matrix.iloc[i, i] for i in range(3)])
        self.assertGreater(max_persistence, 90.0)
    
    def test_percentile_calculation_accuracy(self):
        """Test that percentile thresholds are calculated correctly."""
        # Create data with known distribution
        np.random.seed(123)
        vol_data = pd.DataFrame({
            'TEST': np.linspace(0.10, 0.50, 1000)  # Uniform distribution
        }, index=pd.date_range('2020-01-01', periods=1000))
        
        classifier = VolatilityRegimes()
        regimes = classifier.classify_regimes(
            vol_data,
            percentiles=(25, 75)
        )
        
        # Check that thresholds are close to expected values
        low_threshold = classifier.thresholds['TEST']['low']
        high_threshold = classifier.thresholds['TEST']['high']
        
        # For uniform [0.1, 0.5], 25th percentile ≈ 0.2, 75th ≈ 0.4
        self.assertAlmostEqual(low_threshold, 0.20, places=2)
        self.assertAlmostEqual(high_threshold, 0.40, places=2)
        
        # Check regime distribution
        counts = regimes['TEST'].value_counts(normalize=True) * 100
        self.assertAlmostEqual(counts[0], 25.0, delta=2)  # Low regime
        self.assertAlmostEqual(counts[1], 50.0, delta=2)  # Medium regime
        self.assertAlmostEqual(counts[2], 25.0, delta=2)  # High regime


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_single_observation(self):
        """Test with single observation."""
        single_obs = pd.DataFrame({
            'TEST': [0.15]
        }, index=[pd.Timestamp('2023-01-01')])
        
        classifier = VolatilityRegimes()
        
        with self.assertWarns(UserWarning):
            regimes = classifier.classify_regimes(single_obs)
    
    def test_all_same_volatility(self):
        """Test with constant volatility."""
        constant = pd.DataFrame({
            'TEST': [0.20] * 100
        }, index=pd.date_range('2023-01-01', periods=100))
        
        classifier = VolatilityRegimes()
        regimes = classifier.classify_regimes(constant)
        
        # All should be classified the same
        unique_regimes = regimes['TEST'].unique()
        self.assertEqual(len(unique_regimes), 1)
    
    def test_extreme_percentiles(self):
        """Test with extreme percentile values."""
        dates = pd.date_range('2023-01-01', periods=100)
        vol_data = pd.DataFrame({
            'TEST': np.random.uniform(0.10, 0.30, 100)
        }, index=dates)
        
        classifier = VolatilityRegimes()
        
        # Very skewed percentiles (10, 90)
        regimes = classifier.classify_regimes(
            vol_data,
            percentiles=(10, 90)
        )
        
        # Check distribution
        counts = regimes['TEST'].value_counts(normalize=True) * 100
        self.assertLess(counts[0], 15)   # Low regime < 15%
        self.assertGreater(counts[1], 70)  # Medium regime > 70%
        self.assertLess(counts[2], 15)   # High regime < 15%
    
    def test_negative_volatility(self):
        """Test handling of negative volatility values."""
        # Volatility should never be negative, but test robustness
        vol_data = pd.DataFrame({
            'TEST': [-0.1, 0.1, 0.2, 0.3]
        }, index=pd.date_range('2023-01-01', periods=4))
        
        classifier = VolatilityRegimes()
        
        # Should still classify without errors
        regimes = classifier.classify_regimes(vol_data)
        self.assertEqual(regimes.shape, vol_data.shape)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple operations."""
    
    def test_full_workflow(self):
        """Test complete workflow from classification to analysis."""
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        volatility = pd.DataFrame({
            'AAPL': np.random.uniform(0.10, 0.40, 252)
        }, index=dates)
        
        returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252)
        }, index=dates)
        
        # Step 1: Classify regimes
        classifier = VolatilityRegimes(volatility)
        regimes = classifier.classify_regimes()
        
        # Step 2: Get statistics
        stats = classifier.get_regime_statistics()
        self.assertEqual(len(stats), 3)
        
        # Step 3: Analyze transitions
        transitions = classifier.analyze_transitions()
        self.assertIn('AAPL', transitions)
        
        # Step 4: Calculate persistence
        persistence = classifier.calculate_persistence()
        self.assertEqual(len(persistence), 3)
        
        # Step 5: Get current regime
        current = classifier.get_current_regime()
        self.assertIn('AAPL', current.index)
        
        # Step 6: Filter by regime
        filtered = classifier.filter_by_regime(volatility, regime='High')
        self.assertGreater(len(filtered), 0)
        
        # Step 7: Analyze performance
        performance = analyze_regime_performance(returns, regimes)
        self.assertEqual(len(performance), 3)
    
    def test_workflow_with_missing_data(self):
        """Test workflow with missing data."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # Create volatility with gaps
        volatility = pd.DataFrame({
            'AAPL': np.random.uniform(0.10, 0.40, 252)
        }, index=dates)
        volatility.iloc[50:60] = np.nan  # Add gap
        
        returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252)
        }, index=dates)
        
        # Should handle NaN values gracefully
        classifier = VolatilityRegimes(volatility)
        regimes = classifier.classify_regimes()
        
        # NaN positions should remain NaN
        self.assertTrue(regimes.iloc[50:60, 0].isna().all())
        
        # But should still produce valid statistics
        stats = classifier.get_regime_statistics()
        self.assertEqual(len(stats), 3)
        
        # And performance analysis should work
        performance = analyze_regime_performance(returns, regimes)
        self.assertEqual(len(performance), 3)


def run_tests():
    """Run all tests and display results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestVolatilityRegimes))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
