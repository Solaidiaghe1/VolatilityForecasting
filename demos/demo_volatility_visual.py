"""
Visual comparison of all volatility models

Creates comprehensive visualizations comparing:
- Rolling volatility (20-day)
- EWMA volatility (λ=0.94)
- GARCH conditional volatility
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader import fetch_prices
from src.returns import compute_log_returns
from src.rolling_vol import RollingVolatility
from src.ewma_vol import EWMAVolatility
from src.garch_model import GARCHModel

warnings.filterwarnings('ignore')


def main():
    print("=" * 70)
    print("VOLATILITY MODELS - VISUAL COMPARISON")
    print("=" * 70)
    
    # Load data
    print("\n📊 Loading data and computing returns...")
    prices = fetch_prices(['AAPL'], period='2y')
    returns = compute_log_returns(prices)
    print(f"✓ Ready with {len(returns)} returns")
    
    # Compute all three models
    print("\n🔄 Computing Rolling Volatility...")
    rolling_calc = RollingVolatility(returns)
    rolling_vol = rolling_calc.compute_volatility(window=20)
    rolling_vol_ann = rolling_calc.annualize(rolling_vol)
    
    print("📉 Computing EWMA Volatility...")
    ewma_calc = EWMAVolatility(returns)
    ewma_vol = ewma_calc.compute_volatility(lambda_param=0.94)
    ewma_vol_ann = ewma_calc.annualize(ewma_vol)
    
    print("📊 Fitting GARCH Model...")
    garch_model = GARCHModel(returns)
    garch_model.fit()
    garch_cond_vol = garch_model.get_conditional_volatility()
    garch_cond_vol_ann = garch_cond_vol * (252 ** 0.5)
    
    print("\n✓ All models computed successfully!")
    
    # Create comparison plot
    print("\n🎨 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Top plot: All three volatility models
    ax1 = axes[0]
    ax1.plot(rolling_vol_ann.index, rolling_vol_ann.values, 
             linewidth=1.5, label='Rolling (20-day)', alpha=0.8)
    ax1.plot(ewma_vol_ann.index, ewma_vol_ann.values, 
             linewidth=1.5, label='EWMA (λ=0.94)', alpha=0.8)
    ax1.plot(garch_cond_vol_ann.index, garch_cond_vol_ann.values, 
             linewidth=1.5, label='GARCH(1,1)', alpha=0.8)
    
    ax1.set_title('Volatility Model Comparison - AAPL', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Annualized Volatility')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Returns with volatility overlay
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    # Returns on primary axis
    ax2.plot(returns.index, returns.values, linewidth=0.5, alpha=0.6, 
             color='black', label='Returns')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Returns', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # GARCH volatility on secondary axis
    ax2_twin.plot(garch_cond_vol_ann.index, garch_cond_vol_ann.values, 
                  linewidth=2, color='darkred', alpha=0.7, label='GARCH Volatility')
    ax2_twin.set_ylabel('Annualized Volatility', color='darkred')
    ax2_twin.tick_params(axis='y', labelcolor='darkred')
    
    ax2.set_title('Returns vs GARCH Conditional Volatility', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Show individual model comparisons
    print("\n📈 Showing Rolling Volatility with multiple windows...")
    rolling_calc2 = RollingVolatility(returns)
    multi_windows = rolling_calc2.compute_multiple_windows(
        windows=[20, 60, 252],
        annualize=True
    )
    rolling_calc2.plot_multiple_windows(multi_windows)
    
    print("\n📉 Showing EWMA with different lambdas...")
    ewma_calc2 = EWMAVolatility(returns)
    ewma_calc2.compare_lambdas(lambdas=[0.90, 0.94, 0.97])
    
    print("\n📊 Showing GARCH diagnostics...")
    garch_model.plot_conditional_volatility()
    
    print("\n✨ Visualization complete!")
    print("\nKey Observations:")
    print("  - Rolling volatility is smoothest (equal weights)")
    print("  - EWMA adapts faster to recent changes")
    print("  - GARCH captures volatility clustering best")


if __name__ == '__main__':
    main()
