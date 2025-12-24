"""
Demo script for all volatility models

This script demonstrates:
1. Rolling volatility (simple moving window)
2. EWMA volatility (exponentially weighted)
3. GARCH(1,1) volatility forecasting
4. Comparison of all three approaches
"""

import sys
from pathlib import Path
import warnings
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader import fetch_prices
from src.returns import compute_log_returns
from src.rolling_vol import RollingVolatility
from src.ewma_vol import EWMAVolatility
from src.garch_model import GARCHModel

warnings.filterwarnings('ignore')


def main():
    print("=" * 70)
    print("VOLATILITY FORECASTING - ALL MODELS DEMO")
    print("=" * 70)
    
    # Load data
    print("\n📊 Step 1: Loading price data")
    print("-" * 70)
    
    try:
        prices = fetch_prices(['AAPL'], period='2y')
        print(f"✓ Loaded {len(prices)} days of price data")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Compute returns
    print("\n\n📈 Step 2: Computing log returns")
    print("-" * 70)
    
    returns = compute_log_returns(prices)
    print(f"✓ Computed {len(returns)} returns")
    print(f"\nReturn statistics:")
    print(f"  Mean: {returns.mean().values[0]:.6f}")
    print(f"  Std:  {returns.std().values[0]:.6f}")
    
    # Rolling Volatility
    print("\n\n🔄 Step 3: Rolling Volatility (20-day window)")
    print("-" * 70)
    
    rolling_calc = RollingVolatility(returns)
    rolling_vol = rolling_calc.compute_volatility(window=20)
    rolling_vol_ann = rolling_calc.annualize(rolling_vol)
    
    stats = rolling_calc.get_statistics()
    print("\nRolling Volatility Statistics (Annualized):")
    print(stats.round(4))
    
    # EWMA Volatility
    print("\n\n📉 Step 4: EWMA Volatility (λ=0.94)")
    print("-" * 70)
    
    ewma_calc = EWMAVolatility(returns)
    ewma_vol = ewma_calc.compute_volatility(lambda_param=0.94)
    ewma_vol_ann = ewma_calc.annualize(ewma_vol)
    
    stats = ewma_calc.get_statistics()
    print("\nEWMA Volatility Statistics (Annualized):")
    print(stats.round(4))
    
    # GARCH Model
    print("\n\n📊 Step 5: GARCH(1,1) Model")
    print("-" * 70)
    
    try:
        garch_model = GARCHModel(returns)
        print("Fitting GARCH(1,1) model... (this may take a moment)")
        
        results = garch_model.fit(show_summary=False)
        
        print("\n✓ GARCH model fitted successfully")
        
        # Get parameters
        params = garch_model.get_parameters()
        print("\nGARCH Parameters:")
        print(params.round(6))
        
        # Check stationarity
        stationarity = garch_model.check_stationarity()
        print("\nStationarity Check:")
        print(stationarity.round(6))
        
        # Forecast
        forecast = garch_model.forecast_volatility(horizon=1, annualize=True)
        print("\nGARCH 1-step Forecast (Annualized):")
        print(forecast.round(4))
        
    except Exception as e:
        print(f"⚠️  GARCH model failed: {e}")
        forecast = None
    
    # Compare all models
    print("\n\n🔍 Step 6: Model Comparison")
    print("-" * 70)
    
    comparison = pd.DataFrame({
        'Rolling (20-day)': rolling_vol_ann.iloc[-1],
        'EWMA (λ=0.94)': ewma_vol_ann.iloc[-1],
    })
    
    if forecast is not None:
        comparison['GARCH Forecast'] = forecast.iloc[0]
    
    print("\nCurrent/Forecasted Volatility (Annualized):")
    print(comparison.round(4))
    
    # Compute percentage differences
    if 'GARCH Forecast' in comparison.columns:
        print("\nDifferences from GARCH:")
        rolling_diff = (comparison['Rolling (20-day)'] - comparison['GARCH Forecast']) / comparison['GARCH Forecast'] * 100
        ewma_diff = (comparison['EWMA (λ=0.94)'] - comparison['GARCH Forecast']) / comparison['GARCH Forecast'] * 100
        
        print(f"  Rolling vs GARCH: {rolling_diff.values[0]:.2f}%")
        print(f"  EWMA vs GARCH:    {ewma_diff.values[0]:.2f}%")
    
    # Save results
    print("\n\n💾 Step 7: Saving results")
    print("-" * 70)
    
    output_dir = Path(__file__).parent / 'data' / 'processed'
    
    rolling_calc.save_volatility(str(output_dir / 'rolling_volatility.csv'))
    ewma_calc.save_volatility(str(output_dir / 'ewma_volatility.csv'))
    
    if forecast is not None:
        garch_model.save_forecasts(str(output_dir / 'garch_forecast.csv'))
    
    # Save comparison
    comparison.to_csv(output_dir / 'volatility_comparison.csv')
    print(f"✓ Comparison saved to: {output_dir / 'volatility_comparison.csv'}")
    
    print("\n" + "=" * 70)
    print("✨ DEMO COMPLETED!")
    print("=" * 70)
    print("\nModel Comparison Summary:")
    print("  Rolling:  Simple, intuitive, equally weighted")
    print("  EWMA:     Adaptive, more weight on recent data")
    print("  GARCH:    Sophisticated, captures volatility clustering")
    print("\nNext steps:")
    print("  1. Compare multiple window sizes for rolling volatility")
    print("  2. Try different λ values for EWMA (0.90, 0.94, 0.97)")
    print("  3. Use GARCH forecasts for regime classification")
    print("=" * 70)


if __name__ == '__main__':
    main()
