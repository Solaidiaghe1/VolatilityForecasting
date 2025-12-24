"""
Visual Demonstration of Volatility Regimes Module

This script showcases the visualization capabilities of the volatility
regimes classifier, including:
1. Regime time series plots with thresholds
2. Regime distribution charts
3. Transition matrix heatmaps
4. Persistence analysis
5. Performance by regime

Run this after running demo_volatility_models.py to generate volatility data.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from volatility_regimes import VolatilityRegimes, analyze_regime_performance
from data_loader import DataLoader
from returns import ReturnsCalculator

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')


def main():
    """Run visual demonstrations of volatility regimes."""
    
    print("=" * 80)
    print("VOLATILITY REGIMES - VISUAL DEMONSTRATION")
    print("=" * 80)
    
    # ============================================================
    # 1. Load or Generate Data
    # ============================================================
    print("\n1. LOADING DATA")
    print("-" * 80)
    
    # Try to load existing volatility data
    volatility_path = 'data/processed/volatility_comparison.csv'
    
    if os.path.exists(volatility_path):
        print(f"✓ Loading volatility data from: {volatility_path}")
        volatility = pd.read_csv(volatility_path, index_col=0, parse_dates=True)
        
        # Use GARCH or EWMA forecast if available
        if 'GARCH_Forecast' in volatility.columns:
            vol_series = volatility[['GARCH_Forecast']].rename(columns={'GARCH_Forecast': 'AAPL'})
        elif 'EWMA_lambda_0.94' in volatility.columns:
            vol_series = volatility[['EWMA_lambda_0.94']].rename(columns={'EWMA_lambda_0.94': 'AAPL'})
        else:
            vol_series = volatility.iloc[:, [0]]
            vol_series.columns = ['AAPL']
    else:
        print("⚠ Volatility data not found. Generating synthetic data...")
        print("  (Run demo_volatility_models.py first for real data)")
        
        # Generate synthetic volatility with clear regimes
        dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
        
        # Create regime patterns: low → medium → high → medium → low
        vol_values = np.concatenate([
            np.random.uniform(0.10, 0.15, 100),  # Low
            np.random.uniform(0.15, 0.25, 100),  # Medium
            np.random.uniform(0.25, 0.40, 100),  # High
            np.random.uniform(0.15, 0.25, 100),  # Medium
            np.random.uniform(0.10, 0.15, 100)   # Low
        ])
        
        vol_series = pd.DataFrame({'AAPL': vol_values}, index=dates)
    
    print(f"✓ Volatility data shape: {vol_series.shape}")
    print(f"  Date range: {vol_series.index[0]} to {vol_series.index[-1]}")
    print(f"  Mean volatility: {vol_series['AAPL'].mean():.4f}")
    print(f"  Volatility range: [{vol_series['AAPL'].min():.4f}, {vol_series['AAPL'].max():.4f}]")
    
    # ============================================================
    # 2. Classify Volatility Regimes
    # ============================================================
    print("\n2. CLASSIFYING VOLATILITY REGIMES")
    print("-" * 80)
    
    classifier = VolatilityRegimes(vol_series)
    
    # Classify using percentile method (default 33/66)
    regimes = classifier.classify_regimes(percentiles=(33, 66))
    
    print("✓ Regime classification complete")
    print(f"\nThresholds for AAPL:")
    print(f"  Low/Medium boundary:  {classifier.thresholds['AAPL']['low']:.4f}")
    print(f"  Medium/High boundary: {classifier.thresholds['AAPL']['high']:.4f}")
    
    # Regime statistics
    stats = classifier.get_regime_statistics()
    print(f"\nRegime Distribution:")
    for _, row in stats.iterrows():
        print(f"  {row['regime']:8s}: {row['percentage']:5.1f}% "
              f"(avg duration: {row['avg_duration']:.1f} days, "
              f"max: {int(row['max_duration'])} days)")
    
    # ============================================================
    # 3. VISUALIZATION 1: Regime Time Series
    # ============================================================
    print("\n3. VISUALIZATION 1: Regime Time Series")
    print("-" * 80)
    
    print("Creating regime time series plot...")
    classifier.plot_regimes(
        volatility=vol_series,
        regimes=regimes,
        figsize=(16, 8),
        show_thresholds=True
    )
    print("✓ Plot displayed (close window to continue)")
    
    # ============================================================
    # 4. VISUALIZATION 2: Regime Distribution
    # ============================================================
    print("\n4. VISUALIZATION 2: Regime Distribution")
    print("-" * 80)
    
    print("Creating regime distribution bar chart...")
    classifier.plot_regime_distribution(figsize=(10, 6))
    print("✓ Plot displayed (close window to continue)")
    
    # ============================================================
    # 5. Transition Matrix Analysis
    # ============================================================
    print("\n5. TRANSITION MATRIX ANALYSIS")
    print("-" * 80)
    
    transitions = classifier.analyze_transitions()
    trans_pct = transitions['AAPL']['percentages']
    
    print("Transition Matrix (%):")
    print(trans_pct.round(1))
    
    print("\nInterpretation:")
    print("  - Rows = 'From' regime")
    print("  - Columns = 'To' regime")
    print("  - Diagonal = Persistence (stay in same regime)")
    
    # ============================================================
    # 6. VISUALIZATION 3: Transition Heatmap
    # ============================================================
    print("\n6. VISUALIZATION 3: Transition Matrix Heatmap")
    print("-" * 80)
    
    print("Creating transition matrix heatmap...")
    classifier.plot_transition_matrix(ticker='AAPL', figsize=(10, 8))
    print("✓ Plot displayed (close window to continue)")
    
    # ============================================================
    # 7. Persistence Analysis
    # ============================================================
    print("\n7. PERSISTENCE ANALYSIS")
    print("-" * 80)
    
    persistence = classifier.calculate_persistence()
    print("Regime Persistence (probability of staying in same regime):")
    for _, row in persistence.iterrows():
        print(f"  {row['regime']:8s}: {row['persistence_pct']:.1f}%")
    
    # ============================================================
    # 8. Current Regime
    # ============================================================
    print("\n8. CURRENT REGIME DETECTION")
    print("-" * 80)
    
    current = classifier.get_current_regime()
    print(f"Current volatility regime for AAPL: {current['AAPL']}")
    print(f"Latest volatility value: {vol_series['AAPL'].iloc[-1]:.4f}")
    
    # ============================================================
    # 9. Filter Data by Regime
    # ============================================================
    print("\n9. FILTERING DATA BY REGIME")
    print("-" * 80)
    
    for regime_name in ['Low', 'Medium', 'High']:
        filtered = classifier.filter_by_regime(vol_series, regime=regime_name, ticker='AAPL')
        print(f"\n{regime_name} volatility regime:")
        print(f"  Observations: {len(filtered)}")
        print(f"  Mean: {filtered['AAPL'].mean():.4f}")
        print(f"  Std:  {filtered['AAPL'].std():.4f}")
        print(f"  Range: [{filtered['AAPL'].min():.4f}, {filtered['AAPL'].max():.4f}]")
    
    # ============================================================
    # 10. Performance Analysis (if returns available)
    # ============================================================
    print("\n10. PERFORMANCE ANALYSIS BY REGIME")
    print("-" * 80)
    
    returns_path = 'data/processed/returns.csv'
    
    if os.path.exists(returns_path):
        print(f"✓ Loading returns from: {returns_path}")
        returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        
        # Align returns with regimes
        aligned_returns = returns.reindex(regimes.index)
        
        # Analyze performance
        performance = analyze_regime_performance(
            aligned_returns,
            regimes,
            ticker='AAPL'
        )
        
        print("\nReturn Performance by Volatility Regime:")
        print("-" * 60)
        for _, row in performance.iterrows():
            print(f"\n{row['regime']} Volatility:")
            print(f"  Observations:    {int(row['count'])}")
            print(f"  Mean Return:     {row['mean_return']*100:.4f}%")
            print(f"  Std Return:      {row['std_return']*100:.4f}%")
            print(f"  Annualized Sharpe: {row['sharpe']:.3f}")
            print(f"  Min Return:      {row['min_return']*100:.4f}%")
            print(f"  Max Return:      {row['max_return']*100:.4f}%")
        
        # Create custom performance plot
        print("\n11. VISUALIZATION 4: Performance by Regime")
        print("-" * 80)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Return Performance by Volatility Regime', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Mean Returns
        ax = axes[0, 0]
        colors = ['green', 'orange', 'red']
        ax.bar(performance['regime'], performance['mean_return']*100, color=colors, alpha=0.7)
        ax.set_title('Mean Daily Return by Regime', fontweight='bold')
        ax.set_ylabel('Return (%)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot 2: Volatility
        ax = axes[0, 1]
        ax.bar(performance['regime'], performance['std_return']*100, color=colors, alpha=0.7)
        ax.set_title('Return Volatility by Regime', fontweight='bold')
        ax.set_ylabel('Standard Deviation (%)')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Sharpe Ratio
        ax = axes[1, 0]
        bars = ax.bar(performance['regime'], performance['sharpe'], color=colors, alpha=0.7)
        ax.set_title('Annualized Sharpe Ratio by Regime', fontweight='bold')
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot 4: Observation Count
        ax = axes[1, 1]
        ax.bar(performance['regime'], performance['count'], color=colors, alpha=0.7)
        ax.set_title('Number of Observations by Regime', fontweight='bold')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Performance plot displayed")
        
    else:
        print("⚠ Returns data not found. Skipping performance analysis.")
        print("  (Run demo_returns.py first to generate returns)")
    
    # ============================================================
    # 11. Save Results
    # ============================================================
    print("\n12. SAVING RESULTS")
    print("-" * 80)
    
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save regimes
    regimes_path = os.path.join(output_dir, 'volatility_regimes.csv')
    classifier.save_regimes(regimes_path)
    
    # Save statistics
    stats_path = os.path.join(output_dir, 'regime_statistics.csv')
    stats.to_csv(stats_path, index=False)
    print(f"✓ Regime statistics saved to: {stats_path}")
    
    # Save transition matrix
    trans_path = os.path.join(output_dir, 'regime_transitions.csv')
    trans_pct.to_csv(trans_path)
    print(f"✓ Transition matrix saved to: {trans_path}")
    
    # Save persistence
    persist_path = os.path.join(output_dir, 'regime_persistence.csv')
    persistence.to_csv(persist_path, index=False)
    print(f"✓ Persistence metrics saved to: {persist_path}")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    
    print("\n📊 Visualizations Created:")
    print("  1. ✓ Regime time series with threshold lines")
    print("  2. ✓ Regime distribution stacked bar chart")
    print("  3. ✓ Transition matrix heatmap")
    if os.path.exists(returns_path):
        print("  4. ✓ Performance metrics by regime")
    
    print("\n💾 Files Saved:")
    print(f"  - {regimes_path}")
    print(f"  - {stats_path}")
    print(f"  - {trans_path}")
    print(f"  - {persist_path}")
    
    print("\n🎯 Key Insights:")
    print("  - Volatility regimes provide context for risk management")
    print("  - Transition matrices reveal regime dynamics")
    print("  - Persistence metrics indicate regime stability")
    if os.path.exists(returns_path):
        print("  - Performance varies significantly across regimes")
    
    print("\n📖 Next Steps:")
    print("  - Use regimes to adjust position sizing")
    print("  - Filter strategies by favorable regimes")
    print("  - Monitor current regime for risk alerts")
    print("  - Build regime-adaptive trading systems")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
