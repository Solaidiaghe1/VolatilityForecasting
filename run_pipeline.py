#!/usr/bin/env python3
"""
End-to-End Volatility Forecasting Pipeline

Complete workflow from data loading through volatility forecasting,
regime classification, and strategy analysis.

Usage:
    python run_pipeline.py [--ticker TICKER] [--period PERIOD] [--output OUTPUT]

Example:
    python run_pipeline.py --ticker AAPL --period 5y --output results/

Pipeline Steps:
1. Data Loading & Cleaning
2. Returns Calculation
3. Volatility Modeling (Rolling, EWMA, GARCH)
4. Regime Classification
5. Strategy Analysis (optional)
6. Report Generation

Author: Volatility Forecasting Project
Date: December 24, 2025
"""

import sys
import os
from pathlib import Path
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import warnings

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import all modules
from data_loader import DataLoader, fetch_prices
from returns import ReturnsCalculator, compute_log_returns
from rolling_vol import RollingVolatility
from ewma_vol import EWMAVolatility
from garch_model import GARCHModel
from volatility_regimes import VolatilityRegimes, analyze_regime_performance
from utils import (
    setup_plot_style, ensure_directory, save_dataframe,
    validate_dataframe, annualize_volatility
)

warnings.filterwarnings('ignore')


class VolatilityPipeline:
    """
    End-to-end volatility forecasting pipeline.
    
    Orchestrates all modules to perform complete analysis from
    raw data to regime-based insights.
    """
    
    def __init__(
        self,
        ticker: str = 'AAPL',
        period: str = '2y',
        output_dir: str = 'results',
        verbose: bool = True
    ):
        """
        Initialize pipeline.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        period : str
            Data period ('1y', '2y', '5y', etc.)
        output_dir : str
            Output directory for results
        verbose : bool
            Print progress messages
        """
        self.ticker = ticker
        self.period = period
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # Results storage
        self.prices = None
        self.returns = None
        self.volatility = None
        self.regimes = None
        self.performance = None
        
        # Create output directory
        ensure_directory(self.output_dir)
        
        # Setup plotting
        setup_plot_style()
    
    def log(self, message: str, level: str = 'INFO'):
        """Print log message if verbose."""
        if self.verbose:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] {level}: {message}")
    
    def run(self, include_strategy: bool = False):
        """
        Run complete pipeline.
        
        Parameters:
        -----------
        include_strategy : bool
            Whether to include strategy analysis
        """
        self.log("=" * 80)
        self.log("VOLATILITY FORECASTING PIPELINE - START")
        self.log("=" * 80)
        
        try:
            # Step 1: Load Data
            self.step1_load_data()
            
            # Step 2: Calculate Returns
            self.step2_calculate_returns()
            
            # Step 3: Model Volatility
            self.step3_model_volatility()
            
            # Step 4: Classify Regimes
            self.step4_classify_regimes()
            
            # Step 5: Strategy Analysis (optional)
            if include_strategy:
                self.step5_strategy_analysis()
            
            # Step 6: Generate Report
            self.step6_generate_report()
            
            self.log("=" * 80)
            self.log("PIPELINE COMPLETED SUCCESSFULLY")
            self.log("=" * 80)
            
        except Exception as e:
            self.log(f"PIPELINE FAILED: {e}", level='ERROR')
            raise
    
    def step1_load_data(self):
        """Step 1: Load and clean price data."""
        self.log("\nSTEP 1: DATA LOADING & CLEANING")
        self.log("-" * 80)
        
        # Load data
        loader = DataLoader()
        self.log(f"Loading {self.ticker} data for period: {self.period}")
        
        try:
            prices = loader.load_from_yfinance(
                tickers=self.ticker,
                period=self.period,
                interval='1d'
            )
        except Exception as e:
            self.log(f"Failed to load from Yahoo Finance: {e}", level='ERROR')
            raise
        
        self.log(f"✓ Loaded {len(prices)} days of price data")
        self.log(f"  Date range: {prices.index[0]} to {prices.index[-1]}")
        
        # Clean data
        self.log("Cleaning data...")
        clean_prices = loader.clean_data(
            handle_missing='ffill',
            handle_zeros='ffill'
        )
        
        # Validate
        validation = loader.validate_data()
        self.log(f"✓ Data validated: {len(clean_prices)} rows")
        
        # Save
        output_path = self.output_dir / 'data' / f'{self.ticker}_prices.csv'
        save_dataframe(clean_prices, output_path)
        
        self.prices = clean_prices
        self.log("✓ Step 1 complete")
    
    def step2_calculate_returns(self):
        """Step 2: Calculate returns and statistics."""
        self.log("\nSTEP 2: RETURNS CALCULATION")
        self.log("-" * 80)
        
        calc = ReturnsCalculator()
        
        # Calculate log returns
        self.log("Calculating log returns...")
        returns = calc.compute_log_returns(self.prices)
        
        self.log(f"✓ Calculated returns for {len(returns)} periods")
        
        # Calculate basic statistics
        self.log(f"✓ Return statistics:")
        for col in returns.columns:
            mean_ret = returns[col].mean()
            std_ret = returns[col].std()
            ann_ret = mean_ret * 252
            ann_vol = std_ret * np.sqrt(252)
            self.log(f"  {col}:")
            self.log(f"    Mean: {mean_ret*100:.4f}% daily")
            self.log(f"    Std:  {std_ret*100:.4f}% daily")
            self.log(f"    Annualized: {ann_ret*100:.2f}%")
            self.log(f"    Ann. Volatility: {ann_vol*100:.2f}%")
        
        # Check stationarity
        self.log("Checking stationarity...")
        # Skip detailed stationarity check for now
        self.log("  ✓ Stationarity check skipped in pipeline")
        
        # Save
        output_path = self.output_dir / 'data' / f'{self.ticker}_returns.csv'
        save_dataframe(returns, output_path)
        
        self.returns = returns
        self.log("✓ Step 2 complete")
    
    def step3_model_volatility(self):
        """Step 3: Calculate volatility using multiple models."""
        self.log("\nSTEP 3: VOLATILITY MODELING")
        self.log("-" * 80)
        
        volatility_results = pd.DataFrame(index=self.returns.index)
        
        # 3a: Rolling Volatility
        self.log("Calculating rolling volatility...")
        rolling_vol = RollingVolatility()
        
        for window in [20, 60]:
            vol = rolling_vol.compute_volatility(self.returns, window=window)
            vol_ann = rolling_vol.annualize(vol)
            col_name = f'Rolling_{window}d'
            volatility_results[col_name] = vol_ann.iloc[:, 0]
            self.log(f"  ✓ {col_name}: mean={vol_ann.iloc[:, 0].mean():.4f}")
        
        # 3b: EWMA Volatility
        self.log("Calculating EWMA volatility...")
        ewma_vol = EWMAVolatility()
        
        for lambda_val in [0.94]:
            vol = ewma_vol.compute_volatility(self.returns, lambda_param=lambda_val)
            vol_ann = ewma_vol.annualize(vol)
            col_name = f'EWMA_λ{lambda_val}'
            volatility_results[col_name] = vol_ann.iloc[:, 0]
            self.log(f"  ✓ {col_name}: mean={vol_ann.iloc[:, 0].mean():.4f}")
        
        # 3c: GARCH Model
        self.log("Fitting GARCH(1,1) model...")
        try:
            garch = GARCHModel()
            garch.fit(self.returns)
            
            # Extract parameters (returns DataFrame with ticker as index)
            params_df = garch.get_parameters()
            params = params_df.loc[self.ticker]
            self.log(f"  ✓ GARCH parameters:")
            self.log(f"    ω (omega): {params['omega']:.6f}")
            self.log(f"    α (alpha[1]): {params['alpha[1]']:.6f}")
            self.log(f"    β (beta[1]):  {params['beta[1]']:.6f}")
            self.log(f"    Persistence: {params['persistence']:.6f}")
            
            # Conditional volatility
            cond_vol = garch.get_conditional_volatility()
            cond_vol_ann = cond_vol * np.sqrt(252)  # Annualize
            volatility_results['GARCH_ConditionalVol'] = cond_vol_ann.iloc[:, 0]
            self.log(f"    Mean conditional vol: {cond_vol_ann.iloc[:, 0].mean():.4f}")
            
            # Forecast
            self.log("  Generating GARCH forecasts...")
            forecast = garch.forecast_volatility(horizon=1, annualize=True)
            # Broadcast forecast to all dates
            volatility_results['GARCH_Forecast'] = forecast.iloc[0, 0]
            self.log(f"    1-step forecast: {forecast.iloc[0, 0]:.4f}")
            
        except Exception as e:
            self.log(f"  ✗ GARCH fitting failed: {e}", level='WARNING')
        
        # Save
        output_path = self.output_dir / 'data' / f'{self.ticker}_volatility.csv'
        save_dataframe(volatility_results, output_path)
        
        self.volatility = volatility_results
        self.log("✓ Step 3 complete")
    
    def step4_classify_regimes(self):
        """Step 4: Classify volatility regimes."""
        self.log("\nSTEP 4: REGIME CLASSIFICATION")
        self.log("-" * 80)
        
        # Use GARCH forecast if available, else EWMA
        if 'GARCH_Forecast' in self.volatility.columns:
            vol_for_regimes = self.volatility[['GARCH_Forecast']].copy()
            vol_for_regimes.columns = [self.ticker]
            self.log("Using GARCH forecast for regime classification")
        elif 'EWMA_λ0.94' in self.volatility.columns:
            vol_for_regimes = self.volatility[['EWMA_λ0.94']].copy()
            vol_for_regimes.columns = [self.ticker]
            self.log("Using EWMA (λ=0.94) for regime classification")
        else:
            vol_for_regimes = self.volatility.iloc[:, [0]].copy()
            vol_for_regimes.columns = [self.ticker]
            self.log(f"Using {self.volatility.columns[0]} for regime classification")
        
        # Classify regimes
        classifier = VolatilityRegimes(vol_for_regimes)
        regimes = classifier.classify_regimes(percentiles=(33, 66))
        
        self.log("✓ Regime classification complete")
        
        # Get statistics
        stats = classifier.get_regime_statistics()
        self.log("✓ Regime distribution:")
        for _, row in stats.iterrows():
            self.log(f"  {row['regime']:8s}: {row['percentage']:5.1f}% "
                    f"(avg duration: {row['avg_duration']:.1f} periods)")
        
        # Analyze transitions
        transitions = classifier.analyze_transitions()
        trans_pct = transitions[self.ticker]['percentages']
        self.log("✓ Transition matrix (%):")
        self.log(f"\n{trans_pct.round(1)}")
        
        # Persistence
        persistence = classifier.calculate_persistence()
        self.log("✓ Regime persistence:")
        for _, row in persistence.iterrows():
            self.log(f"  {row['regime']:8s}: {row['persistence_pct']:.1f}%")
        
        # Current regime
        current = classifier.get_current_regime()
        self.log(f"✓ Current regime: {current[self.ticker]}")
        
        # Save
        output_path = self.output_dir / 'data' / f'{self.ticker}_regimes.csv'
        classifier.save_regimes(output_path)
        
        self.regimes = regimes
        self.classifier = classifier
        self.log("✓ Step 4 complete")
    
    def step5_strategy_analysis(self):
        """Step 5: Strategy performance analysis (optional)."""
        self.log("\nSTEP 5: STRATEGY ANALYSIS")
        self.log("-" * 80)
        
        # Check if strategy trades are available
        vwapmrs_trades_path = Path('../VWAPmrs/results')
        
        if not vwapmrs_trades_path.exists():
            self.log("VWAPmrs trades not found. Skipping strategy analysis.", level='WARNING')
            return
        
        # Find most recent trades file
        trade_files = list(vwapmrs_trades_path.glob('trades_*.csv'))
        
        if not trade_files:
            self.log("No trade files found. Skipping strategy analysis.", level='WARNING')
            return
        
        latest_trades = max(trade_files, key=lambda p: p.stat().st_mtime)
        self.log(f"Loading trades from: {latest_trades}")
        
        try:
            from strategy_analysis import StrategyRegimeAnalyzer
            
            analyzer = StrategyRegimeAnalyzer()
            analyzer.load_vwapmrs_trades(latest_trades)
            analyzer.regimes = self.regimes
            
            # Align and analyze
            analyzer.align_trades_with_regimes()
            performance = analyzer.analyze_performance_by_regime()
            
            self.log("✓ Strategy performance by regime:")
            for _, row in performance.iterrows():
                self.log(f"  {row['regime']:8s}:")
                self.log(f"    Trades: {int(row['total_trades'])}")
                self.log(f"    Win Rate: {row['win_rate']:.1f}%")
                self.log(f"    Avg P&L: ${row['avg_pnl']:.2f}")
                self.log(f"    Sharpe: {row['sharpe_ratio']:.2f}")
            
            # Save
            output_path = self.output_dir / 'strategy' / f'{self.ticker}_strategy_performance.csv'
            ensure_directory(output_path.parent)
            save_dataframe(performance, output_path)
            
            self.performance = performance
            self.log("✓ Step 5 complete")
            
        except Exception as e:
            self.log(f"Strategy analysis failed: {e}", level='WARNING')
    
    def step6_generate_report(self):
        """Step 6: Generate summary report."""
        self.log("\nSTEP 6: REPORT GENERATION")
        self.log("-" * 80)
        
        report_path = self.output_dir / 'reports' / f'{self.ticker}_report.txt'
        ensure_directory(report_path.parent)
        
        with open(report_path, 'w') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("VOLATILITY FORECASTING & REGIME ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Ticker: {self.ticker}\n")
            f.write(f"Period: {self.period}\n")
            f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Points: {len(self.prices)}\n")
            f.write(f"Date Range: {self.prices.index[0]} to {self.prices.index[-1]}\n")
            f.write("\n")
            
            # Returns Summary
            f.write("-" * 80 + "\n")
            f.write("RETURNS SUMMARY\n")
            f.write("-" * 80 + "\n")
            
            calc = ReturnsCalculator()
            stats = calc.get_statistics(self.returns)
            
            for col in stats.index:
                row = stats.loc[col]
                f.write(f"\n{col}:\n")
                f.write(f"  Mean Return (daily): {row['mean']*100:.4f}%\n")
                f.write(f"  Std Dev (daily): {row['std']*100:.4f}%\n")
                f.write(f"  Annualized Return: {row['mean']*252*100:.2f}%\n")
                f.write(f"  Annualized Volatility: {row['std']*np.sqrt(252)*100:.2f}%\n")
                f.write(f"  Skewness: {row['skewness']:.3f}\n")
                f.write(f"  Kurtosis: {row['kurtosis']:.3f}\n")
            
            # Volatility Models
            f.write("\n" + "-" * 80 + "\n")
            f.write("VOLATILITY MODELS\n")
            f.write("-" * 80 + "\n\n")
            
            for col in self.volatility.columns:
                mean_vol = self.volatility[col].mean()
                std_vol = self.volatility[col].std()
                min_vol = self.volatility[col].min()
                max_vol = self.volatility[col].max()
                
                f.write(f"{col}:\n")
                f.write(f"  Mean: {mean_vol:.4f}\n")
                f.write(f"  Std:  {std_vol:.4f}\n")
                f.write(f"  Range: [{min_vol:.4f}, {max_vol:.4f}]\n\n")
            
            # Regime Analysis
            f.write("-" * 80 + "\n")
            f.write("VOLATILITY REGIME ANALYSIS\n")
            f.write("-" * 80 + "\n\n")
            
            stats = self.classifier.get_regime_statistics()
            f.write("Regime Distribution:\n")
            for _, row in stats.iterrows():
                f.write(f"  {row['regime']:8s}: {row['percentage']:5.1f}% ")
                f.write(f"({int(row['count'])} periods, ")
                f.write(f"avg duration: {row['avg_duration']:.1f})\n")
            
            f.write("\nTransition Matrix (%):\n")
            transitions = self.classifier.analyze_transitions()
            trans_pct = transitions[self.ticker]['percentages']
            f.write(trans_pct.round(1).to_string())
            f.write("\n\n")
            
            persistence = self.classifier.calculate_persistence()
            f.write("Regime Persistence:\n")
            for _, row in persistence.iterrows():
                f.write(f"  {row['regime']:8s}: {row['persistence_pct']:.1f}%\n")
            
            current = self.classifier.get_current_regime()
            f.write(f"\nCurrent Regime: {current[self.ticker]}\n")
            
            # Strategy Performance (if available)
            if self.performance is not None:
                f.write("\n" + "-" * 80 + "\n")
                f.write("STRATEGY PERFORMANCE BY REGIME\n")
                f.write("-" * 80 + "\n\n")
                
                for _, row in self.performance.iterrows():
                    f.write(f"{row['regime']} Volatility:\n")
                    f.write(f"  Total Trades: {int(row['total_trades'])}\n")
                    f.write(f"  Win Rate: {row['win_rate']:.1f}%\n")
                    f.write(f"  Avg P&L: ${row['avg_pnl']:.2f}\n")
                    f.write(f"  Total P&L: ${row['total_pnl']:.2f}\n")
                    f.write(f"  Sharpe Ratio: {row['sharpe_ratio']:.2f}\n")
                    f.write(f"  Profit Factor: {row['profit_factor']:.2f}\n")
                    f.write(f"  Max Drawdown: {row['max_drawdown_pct']:.2f}%\n\n")
            
            # Footer
            f.write("-" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("-" * 80 + "\n")
        
        self.log(f"✓ Report saved to: {report_path}")
        self.log("✓ Step 6 complete")


def main():
    """Main entry point for pipeline."""
    parser = argparse.ArgumentParser(
        description='End-to-end volatility forecasting pipeline'
    )
    
    parser.add_argument(
        '--ticker',
        type=str,
        default='AAPL',
        help='Stock ticker symbol (default: AAPL)'
    )
    
    parser.add_argument(
        '--period',
        type=str,
        default='2y',
        help='Data period (default: 2y)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory (default: results)'
    )
    
    parser.add_argument(
        '--strategy',
        action='store_true',
        help='Include strategy analysis'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = VolatilityPipeline(
        ticker=args.ticker,
        period=args.period,
        output_dir=args.output,
        verbose=not args.quiet
    )
    
    pipeline.run(include_strategy=args.strategy)
    
    print(f"\n✅ Pipeline completed successfully!")
    print(f"   Results saved to: {pipeline.output_dir}")
    print(f"   View report: {pipeline.output_dir}/reports/{args.ticker}_report.txt")


if __name__ == '__main__':
    main()
