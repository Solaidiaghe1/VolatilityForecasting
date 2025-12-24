"""
Demo script for volatility regimes module

Demonstrates:
1. Regime classification (low, medium, high)
2. Regime statistics and distribution
3. Transition analysis
4. Persistence metrics
5. Performance by regime
"""

import sys
from pathlib import Path
import warnings
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader import fetch_prices
from src.returns import compute_log_returns
from src.garch_model import GARCHModel
from src.volatility_regimes import VolatilityRegimes, classify_volatility_regimes, analyze_regime_performance

warnings.filterwarnings('ignore')


def main():
    print("=" * 70)
    print("VOLATILITY REGIMES - CLASSIFICATION DEMO")
    print("=" * 70)
    
    # Load data
    print("\n📊 Step 1: Loading data")
    print("-" * 70)
    
    prices = fetch_prices(['AAPL', 'MSFT'], period='2y')
    returns = compute_log_returns(prices)
    print(f"✓ Loaded {len(returns)} returns for {returns.shape[1]} tickers")
    
    # Compute GARCH volatility for regime classification
    print("\n\n📈 Step 2: Computing GARCH volatility")
    print("-" * 70)
    
    garch_model = GARCHModel(returns)
    print("Fitting GARCH models...")
    garch_model.fit()
    
    # Get conditional volatility (annualized)
    cond_vol = garch_model.get_conditional_volatility()
    cond_vol_ann = cond_vol * (252 ** 0.5)
    
    print(f"✓ GARCH volatility computed")
    print(f"\nVolatility statistics (annualized):")
    print(cond_vol_ann.describe().round(4))
    
    # Classify regimes
    print("\n\n🎯 Step 3: Classifying volatility regimes")
    print("-" * 70)
    
    classifier = VolatilityRegimes(cond_vol_ann)
    regimes = classifier.classify_regimes(percentiles=(33, 66))
    
    print("✓ Regimes classified using percentile method")
    print(f"  Thresholds:")
    for ticker, thresholds in classifier.thresholds.items():
        print(f"  {ticker}:")
        print(f"    Low/Medium:   {thresholds['low']:.4f}")
        print(f"    Medium/High:  {thresholds['high']:.4f}")
    
    # Get regime statistics
    print("\n\n📊 Step 4: Regime statistics")
    print("-" * 70)
    
    stats = classifier.get_regime_statistics()
    print("\nRegime Distribution:")
    print(stats[['ticker', 'regime', 'count', 'percentage', 'avg_duration']].to_string(index=False))
    
    # Current regime
    print("\n\n🔍 Step 5: Current regime")
    print("-" * 70)
    
    current = classifier.get_current_regime()
    print("\nCurrent volatility regime:")
    for ticker, regime in current.items():
        vol_value = cond_vol_ann[ticker].iloc[-1]
        print(f"  {ticker}: {regime} ({vol_value:.4f})")
    
    # Transition analysis
    print("\n\n🔄 Step 6: Regime transitions")
    print("-" * 70)
    
    transitions = classifier.analyze_transitions()
    
    for ticker, trans_dict in transitions.items():
        print(f"\n{ticker} - Transition Matrix (%):")
        print(trans_dict['percentages'].round(1))
    
    # Persistence
    print("\n\n⏱️  Step 7: Regime persistence")
    print("-" * 70)
    
    persistence = classifier.calculate_persistence()
    print("\nRegime Persistence (probability of staying in same regime):")
    print(persistence.to_string(index=False))
    
    # Performance by regime
    print("\n\n📈 Step 8: Performance by regime")
    print("-" * 70)
    
    for ticker in returns.columns:
        print(f"\n{ticker} - Return Statistics by Regime:")
        perf = analyze_regime_performance(returns, regimes, ticker)
        print(perf.round(6).to_string(index=False))
    
    # Save results
    print("\n\n💾 Step 9: Saving results")
    print("-" * 70)
    
    output_dir = Path(__file__).parent / 'data' / 'processed'
    
    classifier.save_regimes(str(output_dir / 'volatility_regimes.csv'))
    stats.to_csv(output_dir / 'regime_statistics.csv', index=False)
    persistence.to_csv(output_dir / 'regime_persistence.csv', index=False)
    
    # Save transition matrices
    for ticker, trans_dict in transitions.items():
        filename = f'regime_transitions_{ticker}.csv'
        trans_dict['percentages'].to_csv(output_dir / filename)
    
    print(f"✓ All results saved to: {output_dir}")
    
    print("\n" + "=" * 70)
    print("✨ DEMO COMPLETED!")
    print("=" * 70)
    
    print("\nKey Insights:")
    print("  - Regimes based on GARCH conditional volatility")
    print("  - Percentile method ensures balanced distribution")
    print("  - High persistence = stable regimes")
    print("  - Transitions show regime dynamics")
    print("\nNext steps:")
    print("  1. Run demo_regimes_visual.py for visualizations")
    print("  2. Use regimes for strategy analysis")
    print("  3. Compare performance across regimes")
    print("=" * 70)


if __name__ == '__main__':
    main()
