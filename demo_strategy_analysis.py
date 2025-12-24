"""
Demo: Strategy Regime Analysis

Demonstrates integration of VWAP/MRS trading signals with volatility regimes.
Shows how strategy performance varies across different volatility environments.

Run this after:
1. Running VWAP/MRS backtest to generate trades
2. Running volatility regime classification
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from strategy_analysis import StrategyRegimeAnalyzer, quick_regime_analysis


def find_latest_trades_file(results_dir: Path) -> Path:
    """Find the most recent trades CSV file."""
    trades_files = list(results_dir.glob('trades_*.csv'))
    if not trades_files:
        raise FileNotFoundError(f"No trades files found in {results_dir}")
    
    # Sort by modification time
    latest = max(trades_files, key=lambda p: p.stat().st_mtime)
    return latest


def main():
    """Run strategy regime analysis demonstration."""
    
    print("=" * 80)
    print("STRATEGY REGIME ANALYSIS - DEMONSTRATION")
    print("=" * 80)
    
    # ============================================================
    # 1. Locate Data Files
    # ============================================================
    print("\n1. LOCATING DATA FILES")
    print("-" * 80)
    
    # Path to VWAPmrs results
    vwapmrs_results = Path('../VWAPmrs/results')
    
    # Path to VolatilityForecasting processed data
    vol_processed = Path('data/processed')
    
    # Find trades file
    try:
        trades_file = find_latest_trades_file(vwapmrs_results)
        print(f"✓ Found trades file: {trades_file.name}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n  Please run VWAP/MRS backtest first:")
        print("  cd ../VWAPmrs && python test_backtest.py")
        return
    
    # Find regimes file
    regimes_file = vol_processed / 'volatility_regimes.csv'
    if not regimes_file.exists():
        print(f"❌ Regimes file not found: {regimes_file}")
        print("\n  Please run volatility regime classification first:")
        print("  python demo_regimes.py")
        return
    
    print(f"✓ Found regimes file: {regimes_file.name}")
    
    # ============================================================
    # 2. Load Data
    # ============================================================
    print("\n2. LOADING DATA")
    print("-" * 80)
    
    # Load trades
    analyzer = StrategyRegimeAnalyzer()
    trades = analyzer.load_vwapmrs_trades(
        trades_file,
        date_columns=['entry_time', 'exit_time']
    )
    
    print(f"\nTrade Summary:")
    print(f"  Total trades: {len(trades)}")
    print(f"  Symbols: {trades['symbol'].unique().tolist()}")
    print(f"  Date range: {trades['entry_time'].min()} to {trades['exit_time'].max()}")
    
    if 'realized_pnl' in trades.columns:
        total_pnl = trades['realized_pnl'].sum()
        winning = len(trades[trades['realized_pnl'] > 0])
        win_rate = (winning / len(trades)) * 100
        print(f"  Total P&L: ${total_pnl:.2f}")
        print(f"  Win Rate: {win_rate:.1f}%")
    
    # Load regimes
    regimes = pd.read_csv(regimes_file, index_col=0, parse_dates=True)
    
    # Convert string labels back to numeric if needed
    if regimes.iloc[0, 0] in ['Low', 'Medium', 'High']:
        label_map = {'Low': 0, 'Medium': 1, 'High': 2}
        regimes = regimes.applymap(lambda x: label_map.get(x, x))
    
    analyzer.regimes = regimes
    print(f"\n✓ Loaded regimes data")
    print(f"  Shape: {regimes.shape}")
    print(f"  Date range: {regimes.index.min()} to {regimes.index.max()}")
    
    # ============================================================
    # 3. Align Trades with Regimes
    # ============================================================
    print("\n3. ALIGNING TRADES WITH REGIMES")
    print("-" * 80)
    
    aligned_trades = analyzer.align_trades_with_regimes(
        alignment_method='entry'
    )
    
    print(f"\n✓ Alignment complete")
    print(f"  Trades with regime info: {len(aligned_trades)}")
    
    # ============================================================
    # 4. Performance Analysis by Regime
    # ============================================================
    print("\n4. PERFORMANCE ANALYSIS BY REGIME")
    print("-" * 80)
    
    performance = analyzer.analyze_performance_by_regime()
    
    print("\nPerformance Metrics by Regime:")
    print("=" * 80)
    
    for _, row in performance.iterrows():
        print(f"\n{row['regime'].upper()} VOLATILITY REGIME:")
        print("-" * 60)
        print(f"  Total Trades:        {int(row['total_trades'])}")
        print(f"  Win Rate:            {row['win_rate']:.1f}%")
        print(f"  Total P&L:           ${row['total_pnl']:.2f}")
        print(f"  Average P&L:         ${row['avg_pnl']:.2f}")
        print(f"  Profit Factor:       {row['profit_factor']:.2f}")
        print(f"  Sharpe Ratio:        {row['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:       {row['sortino_ratio']:.2f}")
        print(f"  Max Drawdown:        {row['max_drawdown_pct']:.2f}%")
        
        if row['avg_holding_minutes'] is not None:
            print(f"  Avg Holding Time:    {row['avg_holding_minutes']:.1f} minutes")
    
    # ============================================================
    # 5. Signal Characteristics Analysis
    # ============================================================
    print("\n5. SIGNAL CHARACTERISTICS BY REGIME")
    print("-" * 80)
    
    characteristics = analyzer.analyze_signal_characteristics()
    
    print("\nSignal Characteristics:")
    print("=" * 80)
    
    for _, row in characteristics.iterrows():
        print(f"\n{row['regime'].upper()} VOLATILITY:")
        print("-" * 60)
        
        for col, val in row.items():
            if col != 'regime' and val is not None and not pd.isna(val):
                if isinstance(val, float):
                    print(f"  {col:30s}: {val:.4f}")
                else:
                    print(f"  {col:30s}: {val}")
    
    # ============================================================
    # 6. Generate Recommendations
    # ============================================================
    print("\n6. STRATEGY RECOMMENDATIONS")
    print("-" * 80)
    
    recommendations = analyzer.generate_recommendations(performance)
    
    print("\nActionable Recommendations:")
    print("=" * 80)
    
    for category, recs in recommendations.items():
        if recs:
            print(f"\n{category.upper().replace('_', ' ')}:")
            print("-" * 60)
            for rec in recs:
                print(f"  • {rec}")
    
    if not any(recommendations.values()):
        print("\n  ✓ Strategy performance is consistent across regimes")
        print("    No immediate adjustments recommended")
    
    # ============================================================
    # 7. Visualizations
    # ============================================================
    print("\n7. GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    print("\nCreating performance comparison charts...")
    analyzer.plot_performance_comparison(performance)
    print("✓ Performance comparison displayed (close window to continue)")
    
    print("\nCreating P&L distribution plots...")
    analyzer.plot_pnl_distributions()
    print("✓ P&L distributions displayed (close window to continue)")
    
    print("\nCreating equity curve with regime overlay...")
    analyzer.plot_equity_curves()
    print("✓ Equity curve displayed (close window to continue)")
    
    # ============================================================
    # 8. Export Results
    # ============================================================
    print("\n8. EXPORTING RESULTS")
    print("-" * 80)
    
    output_dir = Path('data/strategy_analysis')
    files = analyzer.export_analysis(
        output_dir=output_dir,
        prefix='vwapmrs_regime'
    )
    
    print(f"\n✓ Analysis exported to: {output_dir}")
    print(f"  Files created:")
    for file_type, filepath in files.items():
        print(f"    - {filepath.name}")
    
    # ============================================================
    # 9. Key Insights Summary
    # ============================================================
    print("\n9. KEY INSIGHTS")
    print("=" * 80)
    
    # Best and worst regime
    best_idx = performance['sharpe_ratio'].idxmax()
    worst_idx = performance['sharpe_ratio'].idxmin()
    
    best_regime = performance.loc[best_idx, 'regime']
    best_sharpe = performance.loc[best_idx, 'sharpe_ratio']
    best_trades = int(performance.loc[best_idx, 'total_trades'])
    
    worst_regime = performance.loc[worst_idx, 'regime']
    worst_sharpe = performance.loc[worst_idx, 'sharpe_ratio']
    worst_trades = int(performance.loc[worst_idx, 'total_trades'])
    
    print(f"\n✓ Best Performance:")
    print(f"  {best_regime} volatility regime")
    print(f"  Sharpe Ratio: {best_sharpe:.2f}")
    print(f"  Trades: {best_trades}")
    
    print(f"\n✓ Worst Performance:")
    print(f"  {worst_regime} volatility regime")
    print(f"  Sharpe Ratio: {worst_sharpe:.2f}")
    print(f"  Trades: {worst_trades}")
    
    # Performance spread
    sharpe_spread = best_sharpe - worst_sharpe
    print(f"\n✓ Performance Spread:")
    print(f"  Sharpe difference: {sharpe_spread:.2f}")
    
    if sharpe_spread > 1.0:
        print(f"  ⚠ Significant variation across regimes!")
        print(f"    Consider regime-adaptive position sizing")
    else:
        print(f"  ✓ Relatively consistent performance")
    
    # Win rate analysis
    win_rates = performance[['regime', 'win_rate']].sort_values('win_rate', ascending=False)
    print(f"\n✓ Win Rate Rankings:")
    for _, row in win_rates.iterrows():
        print(f"  {row['regime']:8s}: {row['win_rate']:.1f}%")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    
    print("\n📊 Analysis Complete:")
    print(f"  ✓ {len(aligned_trades)} trades analyzed")
    print(f"  ✓ Performance metrics calculated")
    print(f"  ✓ Signal characteristics evaluated")
    print(f"  ✓ Recommendations generated")
    print(f"  ✓ Visualizations created")
    print(f"  ✓ Results exported")
    
    print("\n💡 Next Steps:")
    print("  1. Review regime-specific performance metrics")
    print("  2. Implement recommended adjustments")
    print("  3. Backtest with regime-adaptive parameters")
    print("  4. Monitor regime transitions in live trading")
    
    print("\n📁 Output Files:")
    print(f"  {output_dir}/")
    for filepath in files.values():
        print(f"    - {filepath.name}")
    
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
