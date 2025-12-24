"""
Demo script for returns calculation module

This script demonstrates:
1. Computing log returns from price data
2. Statistical analysis of returns
3. Visual checks (distributions, Q-Q plots, stationarity)
4. Comparison of log vs simple returns
"""

import sys
from pathlib import Path
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader import fetch_prices
from src.returns import ReturnsCalculator, analyze_returns

# Suppress pandas warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 70)
    print("VOLATILITY FORECASTING - RETURNS CALCULATION DEMO")
    print("=" * 70)
    
    # Load price data
    print("\n📊 Step 1: Loading price data")
    print("-" * 70)
    
    try:
        prices = fetch_prices(['AAPL', 'MSFT', 'SPY'], period='2y')
        print(f"\n✓ Loaded {len(prices)} days of price data")
        print(f"\nPrice data sample:")
        print(prices.head())
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Compute log returns
    print("\n\n📈 Step 2: Computing log returns")
    print("-" * 70)
    
    calc = ReturnsCalculator(prices)
    returns = calc.compute_log_returns()
    
    print(f"\n✓ Computed log returns")
    print(f"  Shape: {returns.shape}")
    print(f"\nLog returns sample:")
    print(returns.head())
    print(f"\nLog returns tail:")
    print(returns.tail())
    
    # Get statistics
    print("\n\n📊 Step 3: Return statistics")
    print("-" * 70)
    
    stats = calc.get_statistics()
    print("\nDescriptive Statistics:")
    print(stats.round(6))
    
    # Annualized statistics
    print("\nAnnualized Statistics (252 trading days):")
    annualized = stats.copy()
    annualized['annual_mean'] = stats['mean'] * 252
    annualized['annual_std'] = stats['std'] * (252 ** 0.5)
    print(annualized[['annual_mean', 'annual_std']].round(6))
    
    # Check stationarity
    print("\n\n✅ Step 4: Stationarity checks")
    print("-" * 70)
    
    stationarity = calc.check_stationarity()
    
    print("\nStationarity Analysis:")
    for ticker, results in stationarity.items():
        print(f"\n{ticker}:")
        print(f"  Mean change (normalized): {results['mean_change_normalized']:.4f}")
        print(f"  Variance ratio: {results['variance_ratio']:.4f}")
        print(f"  Looks stationary: {'✓' if results['looks_stationary'] else '✗'}")
    
    # Compare log vs simple returns
    print("\n\n🔄 Step 5: Log vs Simple returns comparison")
    print("-" * 70)
    
    simple_returns = calc.compute_simple_returns(prices)
    
    print("\nFirst 5 returns for AAPL:")
    comparison = pd.DataFrame({
        'Log Returns': returns['AAPL'].head(),
        'Simple Returns': simple_returns['AAPL'].head(),
        'Difference': (returns['AAPL'] - simple_returns['AAPL']).head()
    })
    print(comparison)
    
    # Analyze returns with convenience function
    print("\n\n🔍 Step 6: Comprehensive analysis")
    print("-" * 70)
    
    analysis = analyze_returns(prices, return_type='log')
    
    print("\nAnalysis complete!")
    print(f"  Returns computed: ✓")
    print(f"  Statistics calculated: ✓")
    print(f"  Stationarity checked: ✓")
    
    # Check for outliers
    print("\n\n⚠️  Step 7: Outlier detection")
    print("-" * 70)
    
    for col in returns.columns:
        series = returns[col]
        mean = series.mean()
        std = series.std()
        
        # Define outliers as > 3 standard deviations
        outliers = series[abs(series - mean) > 3 * std]
        
        if len(outliers) > 0:
            print(f"\n{col}: Found {len(outliers)} outliers (>3σ)")
            print(f"  Largest positive: {outliers.max():.4f} ({outliers.idxmax().date()})")
            print(f"  Largest negative: {outliers.min():.4f} ({outliers.idxmin().date()})")
        else:
            print(f"\n{col}: No outliers detected")
    
    # Save returns
    print("\n\n💾 Step 8: Saving returns to CSV")
    print("-" * 70)
    
    output_path = Path(__file__).parent / 'data' / 'processed' / 'returns.csv'
    calc.save_returns(str(output_path))
    
    print("\n" + "=" * 70)
    print("✨ DEMO COMPLETED!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review the statistics above")
    print("  2. Run with plot=True to see visualizations:")
    print("     calc.plot_returns()")
    print("     calc.plot_rolling_stats()")
    print("  3. Proceed to volatility modeling")
    print("=" * 70)


if __name__ == '__main__':
    import pandas as pd
    main()
