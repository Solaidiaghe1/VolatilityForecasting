"""
Visual demo for returns module

This script creates visualizations of returns:
- Time series plots
- Distribution histograms with normal overlay
- Q-Q plots for normality testing
- Rolling statistics for stationarity checks
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader import fetch_prices
from src.returns import ReturnsCalculator


def main():
    print("=" * 70)
    print("RETURNS CALCULATION - VISUAL DEMO")
    print("=" * 70)
    
    # Load data
    print("\n📊 Loading price data...")
    prices = fetch_prices(['AAPL'], period='2y')
    
    # Compute returns
    print("\n📈 Computing returns...")
    calc = ReturnsCalculator(prices)
    returns = calc.compute_log_returns()
    
    print(f"✓ Computed {len(returns)} returns for {returns.shape[1]} ticker(s)")
    
    # Show statistics
    print("\n📊 Statistics:")
    stats = calc.get_statistics()
    print(stats.round(6))
    
    # Create visualizations
    print("\n\n🎨 Creating visualizations...")
    print("\n1. Returns time series, distribution, and Q-Q plot")
    print("   (Close the plot window to continue)")
    calc.plot_returns(figsize=(15, 5))
    
    print("\n2. Rolling statistics (stationarity check)")
    print("   (Close the plot window to complete demo)")
    calc.plot_rolling_stats(window=20, figsize=(15, 5))
    
    print("\n✨ Visual demo complete!")


if __name__ == '__main__':
    main()
