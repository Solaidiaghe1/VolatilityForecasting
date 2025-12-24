"""
Strategy Overlay Analysis Module

Integrates trading strategy signals (VWAP/MRS) with volatility regime classification
to analyze performance characteristics across different volatility environments.

Key Features:
- Load and align strategy signals with volatility regimes
- Calculate performance metrics segmented by regime
- Risk-adjusted returns analysis (Sharpe, Sortino, Calmar)
- Signal quality and characteristics by regime
- Trade distribution and frequency analysis
- Visualization of regime-dependent performance
- Actionable recommendations for strategy optimization

Author: Volatility Forecasting Project
Date: December 24, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import warnings
from datetime import datetime, timedelta

try:
    from .volatility_regimes import VolatilityRegimes
except ImportError:
    from volatility_regimes import VolatilityRegimes


class StrategyRegimeAnalyzer:
    """
    Analyze trading strategy performance across volatility regimes.
    
    This class integrates strategy signals/trades with volatility regime
    classifications to provide insights into regime-dependent performance.
    
    Attributes:
        trades (pd.DataFrame): Strategy trade history
        regimes (pd.DataFrame): Volatility regime classifications
        prices (pd.DataFrame): Price data for context
        aligned_trades (pd.DataFrame): Trades aligned with regimes
    """
    
    def __init__(
        self,
        trades: Optional[pd.DataFrame] = None,
        regimes: Optional[pd.DataFrame] = None,
        prices: Optional[pd.DataFrame] = None
    ):
        """
        Initialize Strategy Regime Analyzer.
        
        Parameters:
        -----------
        trades : pd.DataFrame, optional
            Trade history with columns:
            - entry_time, exit_time, entry_price, exit_price
            - realized_pnl, direction, symbol, size, etc.
        regimes : pd.DataFrame, optional
            Volatility regime classifications
        prices : pd.DataFrame, optional
            Price data for additional context
        """
        self.trades = trades
        self.regimes = regimes
        self.prices = prices
        self.aligned_trades = None
        
        if trades is not None and regimes is not None:
            self.align_trades_with_regimes()
    
    def load_vwapmrs_trades(
        self,
        filepath: Union[str, Path],
        date_columns: List[str] = ['entry_time', 'exit_time']
    ) -> pd.DataFrame:
        """
        Load VWAP/MRS trade history from CSV.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to trades CSV file
        date_columns : list
            Column names to parse as dates
            
        Returns:
        --------
        pd.DataFrame
            Loaded trade history
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Trades file not found: {filepath}")
        
        # Load trades
        trades = pd.read_csv(filepath, parse_dates=date_columns)
        
        print(f"✓ Loaded {len(trades)} trades from: {filepath}")
        print(f"  Columns: {list(trades.columns)}")
        print(f"  Date range: {trades['entry_time'].min()} to {trades['exit_time'].max()}")
        
        self.trades = trades
        return trades
    
    def align_trades_with_regimes(
        self,
        trades: Optional[pd.DataFrame] = None,
        regimes: Optional[pd.DataFrame] = None,
        alignment_method: str = 'entry'
    ) -> pd.DataFrame:
        """
        Align trades with volatility regimes.
        
        Parameters:
        -----------
        trades : pd.DataFrame, optional
            Trade history
        regimes : pd.DataFrame, optional
            Regime classifications
        alignment_method : str, default='entry'
            How to align trades with regimes:
            - 'entry': Use entry_time
            - 'exit': Use exit_time
            - 'average': Average regime during trade
            
        Returns:
        --------
        pd.DataFrame
            Trades with regime information added
        """
        if trades is None:
            if self.trades is None:
                raise ValueError("No trades data provided")
            trades = self.trades
        else:
            self.trades = trades
        
        if regimes is None:
            if self.regimes is None:
                raise ValueError("No regimes data provided")
            regimes = self.regimes
        else:
            self.regimes = regimes
        
        # Create a copy for alignment
        aligned = trades.copy()
        
        # Ensure we have the required columns
        required_cols = ['entry_time', 'symbol']
        if not all(col in aligned.columns for col in required_cols):
            raise ValueError(f"Trades must have columns: {required_cols}")
        
        # Add regime column(s)
        regime_cols = []
        
        for symbol in regimes.columns:
            regime_col = f'regime_{symbol}'
            regime_cols.append(regime_col)
            aligned[regime_col] = None
            
            # Filter trades for this symbol
            symbol_mask = aligned['symbol'] == symbol
            symbol_trades = aligned[symbol_mask]
            
            if len(symbol_trades) == 0:
                continue
            
            # Align based on method
            if alignment_method == 'entry':
                # Map entry time to regime
                for idx, trade in symbol_trades.iterrows():
                    entry_time = trade['entry_time']
                    
                    # Find closest regime timestamp
                    if entry_time in regimes.index:
                        aligned.loc[idx, regime_col] = regimes.loc[entry_time, symbol]
                    else:
                        # Use forward fill to get most recent regime
                        regime_series = regimes[symbol]
                        earlier = regime_series[regime_series.index <= entry_time]
                        if len(earlier) > 0:
                            aligned.loc[idx, regime_col] = earlier.iloc[-1]
            
            elif alignment_method == 'exit':
                # Map exit time to regime
                if 'exit_time' not in aligned.columns:
                    raise ValueError("exit_time column required for 'exit' alignment")
                
                for idx, trade in symbol_trades.iterrows():
                    exit_time = trade['exit_time']
                    
                    if exit_time in regimes.index:
                        aligned.loc[idx, regime_col] = regimes.loc[exit_time, symbol]
                    else:
                        regime_series = regimes[symbol]
                        earlier = regime_series[regime_series.index <= exit_time]
                        if len(earlier) > 0:
                            aligned.loc[idx, regime_col] = earlier.iloc[-1]
            
            elif alignment_method == 'average':
                # Use average regime during trade lifetime
                if 'exit_time' not in aligned.columns:
                    raise ValueError("exit_time column required for 'average' alignment")
                
                for idx, trade in symbol_trades.iterrows():
                    entry_time = trade['entry_time']
                    exit_time = trade['exit_time']
                    
                    # Get regimes during trade
                    trade_regimes = regimes[symbol][
                        (regimes.index >= entry_time) & (regimes.index <= exit_time)
                    ]
                    
                    if len(trade_regimes) > 0:
                        # Use mode (most frequent regime)
                        aligned.loc[idx, regime_col] = trade_regimes.mode()[0]
        
        # Add a primary regime column (for single symbol or first symbol)
        if len(regime_cols) > 0:
            aligned['regime'] = aligned[regime_cols[0]]
        
        # Convert regime codes to labels
        regime_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
        if 'regime' in aligned.columns:
            aligned['regime_label'] = aligned['regime'].map(regime_labels)
        
        # Remove trades without regime information
        initial_count = len(aligned)
        aligned = aligned.dropna(subset=['regime'])
        removed_count = initial_count - len(aligned)
        
        if removed_count > 0:
            warnings.warn(f"Removed {removed_count} trades without regime information")
        
        print(f"\n✓ Aligned {len(aligned)} trades with volatility regimes")
        print(f"  Alignment method: {alignment_method}")
        
        if 'regime_label' in aligned.columns:
            regime_dist = aligned['regime_label'].value_counts()
            print(f"  Regime distribution:")
            for regime, count in regime_dist.items():
                pct = (count / len(aligned)) * 100
                print(f"    {regime}: {count} ({pct:.1f}%)")
        
        self.aligned_trades = aligned
        return aligned
    
    def analyze_performance_by_regime(
        self,
        trades: Optional[pd.DataFrame] = None,
        metrics: List[str] = ['all']
    ) -> pd.DataFrame:
        """
        Calculate comprehensive performance metrics by regime.
        
        Parameters:
        -----------
        trades : pd.DataFrame, optional
            Aligned trades (if None, use self.aligned_trades)
        metrics : list, default=['all']
            Metrics to calculate. Options:
            - 'all': Calculate all metrics
            - 'basic': win_rate, avg_pnl, total_pnl
            - 'risk': sharpe, sortino, max_drawdown
            - 'distribution': skewness, kurtosis, percentiles
            
        Returns:
        --------
        pd.DataFrame
            Performance metrics by regime
        """
        if trades is None:
            if self.aligned_trades is None:
                raise ValueError("No aligned trades available. Run align_trades_with_regimes() first")
            trades = self.aligned_trades
        
        if 'regime_label' not in trades.columns:
            raise ValueError("Trades must have 'regime_label' column")
        
        results = []
        
        for regime in ['Low', 'Medium', 'High']:
            regime_trades = trades[trades['regime_label'] == regime]
            
            if len(regime_trades) == 0:
                continue
            
            # Basic metrics
            total_trades = len(regime_trades)
            winning_trades = len(regime_trades[regime_trades['realized_pnl'] > 0])
            losing_trades = len(regime_trades[regime_trades['realized_pnl'] <= 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            pnl_series = regime_trades['realized_pnl']
            total_pnl = pnl_series.sum()
            avg_pnl = pnl_series.mean()
            median_pnl = pnl_series.median()
            std_pnl = pnl_series.std()
            
            # Profit/Loss metrics
            gross_profit = regime_trades[regime_trades['realized_pnl'] > 0]['realized_pnl'].sum()
            gross_loss = abs(regime_trades[regime_trades['realized_pnl'] <= 0]['realized_pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
            avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
            
            # Risk-adjusted metrics
            if std_pnl > 0:
                sharpe = (avg_pnl / std_pnl) * np.sqrt(252)  # Annualized (assuming daily)
            else:
                sharpe = 0
            
            # Sortino ratio (downside deviation)
            downside_returns = pnl_series[pnl_series < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else std_pnl
            sortino = (avg_pnl / downside_std) * np.sqrt(252) if downside_std > 0 else 0
            
            # Drawdown analysis
            cumulative_pnl = pnl_series.cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = cumulative_pnl - running_max
            max_drawdown = drawdown.min()
            max_drawdown_pct = (max_drawdown / running_max.max() * 100) if running_max.max() > 0 else 0
            
            # Calmar ratio (return / max drawdown)
            calmar = abs(total_pnl / max_drawdown) if max_drawdown < 0 else 0
            
            # Distribution metrics
            skewness = pnl_series.skew()
            kurtosis = pnl_series.kurtosis()
            
            # Percentiles
            pct_25 = pnl_series.quantile(0.25)
            pct_75 = pnl_series.quantile(0.75)
            
            # Min/Max
            min_pnl = pnl_series.min()
            max_pnl = pnl_series.max()
            
            # Trade characteristics
            if 'holding_minutes' in regime_trades.columns:
                avg_holding = regime_trades['holding_minutes'].mean()
                median_holding = regime_trades['holding_minutes'].median()
            else:
                avg_holding = None
                median_holding = None
            
            # Direction analysis
            if 'direction' in regime_trades.columns:
                long_trades = len(regime_trades[regime_trades['direction'] == 'LONG'])
                short_trades = len(regime_trades[regime_trades['direction'] == 'SHORT'])
                long_pct = (long_trades / total_trades) * 100 if total_trades > 0 else 0
            else:
                long_trades = None
                short_trades = None
                long_pct = None
            
            # Compile results
            result = {
                'regime': regime,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'median_pnl': median_pnl,
                'std_pnl': std_pnl,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'win_loss_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown_pct,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'min_pnl': min_pnl,
                'max_pnl': max_pnl,
                'pct_25': pct_25,
                'pct_75': pct_75,
                'avg_holding_minutes': avg_holding,
                'median_holding_minutes': median_holding,
                'long_trades': long_trades,
                'short_trades': short_trades,
                'long_pct': long_pct
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def analyze_signal_characteristics(
        self,
        trades: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Analyze signal and trade characteristics by regime.
        
        Parameters:
        -----------
        trades : pd.DataFrame, optional
            Aligned trades
            
        Returns:
        --------
        pd.DataFrame
            Signal characteristics by regime
        """
        if trades is None:
            if self.aligned_trades is None:
                raise ValueError("No aligned trades available")
            trades = self.aligned_trades
        
        results = []
        
        for regime in ['Low', 'Medium', 'High']:
            regime_trades = trades[trades['regime_label'] == regime]
            
            if len(regime_trades) == 0:
                continue
            
            result = {
                'regime': regime,
                'trade_count': len(regime_trades)
            }
            
            # Entry characteristics
            if 'entry_z' in regime_trades.columns:
                result['avg_entry_z'] = regime_trades['entry_z'].mean()
                result['median_entry_z'] = regime_trades['entry_z'].median()
                result['std_entry_z'] = regime_trades['entry_z'].std()
            
            # Price deviation from VWAP
            if 'entry_vwap' in regime_trades.columns and 'entry_price' in regime_trades.columns:
                result['avg_entry_deviation_pct'] = (
                    ((regime_trades['entry_price'] - regime_trades['entry_vwap']) / 
                     regime_trades['entry_vwap']) * 100
                ).mean()
            
            # Stop loss characteristics
            if 'stop_loss' in regime_trades.columns and 'entry_price' in regime_trades.columns:
                stop_distance = abs(regime_trades['entry_price'] - regime_trades['stop_loss'])
                result['avg_stop_distance'] = stop_distance.mean()
                result['avg_stop_distance_pct'] = (
                    (stop_distance / regime_trades['entry_price']) * 100
                ).mean()
            
            # Exit characteristics
            if 'exit_reason' in regime_trades.columns:
                exit_reasons = regime_trades['exit_reason'].value_counts()
                for reason, count in exit_reasons.items():
                    result[f'exit_{reason}_count'] = count
                    result[f'exit_{reason}_pct'] = (count / len(regime_trades)) * 100
            
            # Size characteristics
            if 'size' in regime_trades.columns:
                result['avg_position_size'] = regime_trades['size'].mean()
                result['median_position_size'] = regime_trades['size'].median()
            
            # Time-of-day analysis (if applicable)
            if 'entry_time' in regime_trades.columns:
                regime_trades['hour'] = pd.to_datetime(regime_trades['entry_time']).dt.hour
                result['most_common_entry_hour'] = regime_trades['hour'].mode().iloc[0] if len(regime_trades['hour'].mode()) > 0 else None
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def compare_to_baseline(
        self,
        baseline_returns: pd.Series,
        trades: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compare strategy performance to buy-and-hold baseline by regime.
        
        Parameters:
        -----------
        baseline_returns : pd.Series
            Buy-and-hold returns (aligned with regimes)
        trades : pd.DataFrame, optional
            Aligned trades
            
        Returns:
        --------
        pd.DataFrame
            Comparison metrics by regime
        """
        if trades is None:
            if self.aligned_trades is None:
                raise ValueError("No aligned trades available")
            trades = self.aligned_trades
        
        if self.regimes is None:
            raise ValueError("No regimes available")
        
        # Align baseline returns with regimes
        regime_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
        
        results = []
        
        for regime_code, regime_label in regime_labels.items():
            # Filter baseline returns by regime
            regime_mask = self.regimes.iloc[:, 0] == regime_code
            regime_baseline = baseline_returns[regime_mask]
            
            # Strategy returns for this regime
            regime_trades = trades[trades['regime_label'] == regime_label]
            
            if len(regime_trades) == 0 or len(regime_baseline) == 0:
                continue
            
            # Calculate metrics
            strategy_total_return = regime_trades['realized_pnl'].sum()
            baseline_total_return = regime_baseline.sum()
            
            strategy_sharpe = (
                (regime_trades['realized_pnl'].mean() / regime_trades['realized_pnl'].std()) 
                * np.sqrt(252)
            ) if regime_trades['realized_pnl'].std() > 0 else 0
            
            baseline_sharpe = (
                (regime_baseline.mean() / regime_baseline.std()) * np.sqrt(252)
            ) if regime_baseline.std() > 0 else 0
            
            results.append({
                'regime': regime_label,
                'strategy_total_return': strategy_total_return,
                'baseline_total_return': baseline_total_return,
                'excess_return': strategy_total_return - baseline_total_return,
                'strategy_sharpe': strategy_sharpe,
                'baseline_sharpe': baseline_sharpe,
                'sharpe_difference': strategy_sharpe - baseline_sharpe,
                'strategy_trades': len(regime_trades),
                'baseline_periods': len(regime_baseline)
            })
        
        return pd.DataFrame(results)
    
    def generate_recommendations(
        self,
        performance: Optional[pd.DataFrame] = None
    ) -> Dict[str, List[str]]:
        """
        Generate actionable recommendations based on regime analysis.
        
        Parameters:
        -----------
        performance : pd.DataFrame, optional
            Performance metrics by regime
            
        Returns:
        --------
        dict
            Recommendations categorized by type
        """
        if performance is None:
            performance = self.analyze_performance_by_regime()
        
        recommendations = {
            'position_sizing': [],
            'regime_filtering': [],
            'risk_management': [],
            'entry_conditions': [],
            'general': []
        }
        
        # Analyze performance differences across regimes
        best_regime = performance.loc[performance['sharpe_ratio'].idxmax(), 'regime']
        worst_regime = performance.loc[performance['sharpe_ratio'].idxmin(), 'regime']
        
        best_sharpe = performance.loc[performance['sharpe_ratio'].idxmax(), 'sharpe_ratio']
        worst_sharpe = performance.loc[performance['sharpe_ratio'].idxmin(), 'sharpe_ratio']
        
        # Recommendation 1: Regime filtering
        if worst_sharpe < 0:
            recommendations['regime_filtering'].append(
                f"Consider avoiding trading in {worst_regime} volatility regime (Sharpe: {worst_sharpe:.2f})"
            )
        
        if best_sharpe > 1.5:
            recommendations['regime_filtering'].append(
                f"Focus trading on {best_regime} volatility regime (Sharpe: {best_sharpe:.2f})"
            )
        
        # Recommendation 2: Position sizing
        for _, row in performance.iterrows():
            regime = row['regime']
            sharpe = row['sharpe_ratio']
            
            if sharpe > 1.5:
                recommendations['position_sizing'].append(
                    f"Increase position size in {regime} vol regime (strong risk-adjusted returns)"
                )
            elif sharpe < 0.5:
                recommendations['position_sizing'].append(
                    f"Reduce position size in {regime} vol regime (weak risk-adjusted returns)"
                )
        
        # Recommendation 3: Risk management
        for _, row in performance.iterrows():
            regime = row['regime']
            max_dd_pct = row['max_drawdown_pct']
            
            if abs(max_dd_pct) > 10:
                recommendations['risk_management'].append(
                    f"Tighten stop losses in {regime} vol regime (max drawdown: {max_dd_pct:.1f}%)"
                )
        
        # Recommendation 4: Win rate analysis
        for _, row in performance.iterrows():
            regime = row['regime']
            win_rate = row['win_rate']
            profit_factor = row['profit_factor']
            
            if win_rate < 50 and profit_factor < 1.5:
                recommendations['entry_conditions'].append(
                    f"Review entry conditions for {regime} vol regime (Win rate: {win_rate:.1f}%, PF: {profit_factor:.2f})"
                )
        
        # General recommendations
        total_trades = performance['total_trades'].sum()
        if total_trades < 100:
            recommendations['general'].append(
                f"Collect more data ({total_trades} total trades) for robust regime analysis"
            )
        
        return recommendations
    
    # Visualization methods continue...
    
    def plot_performance_comparison(
        self,
        performance: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (16, 10)
    ) -> None:
        """
        Plot comprehensive performance comparison across regimes.
        
        Parameters:
        -----------
        performance : pd.DataFrame, optional
            Performance metrics by regime
        figsize : tuple
            Figure size
        """
        if performance is None:
            performance = self.analyze_performance_by_regime()
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Strategy Performance by Volatility Regime', 
                    fontsize=16, fontweight='bold')
        
        colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        regime_colors = [colors[r] for r in performance['regime']]
        
        # Plot 1: Win Rate
        ax = axes[0, 0]
        ax.bar(performance['regime'], performance['win_rate'], color=regime_colors, alpha=0.7)
        ax.set_title('Win Rate', fontweight='bold')
        ax.set_ylabel('Win Rate (%)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=50, color='black', linestyle='--', linewidth=0.5, label='50%')
        ax.legend()
        
        # Plot 2: Average P&L
        ax = axes[0, 1]
        ax.bar(performance['regime'], performance['avg_pnl'], color=regime_colors, alpha=0.7)
        ax.set_title('Average P&L per Trade', fontweight='bold')
        ax.set_ylabel('Average P&L ($)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot 3: Sharpe Ratio
        ax = axes[0, 2]
        ax.bar(performance['regime'], performance['sharpe_ratio'], color=regime_colors, alpha=0.7)
        ax.set_title('Sharpe Ratio (Annualized)', fontweight='bold')
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.5, label='1.0')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.legend()
        
        # Plot 4: Profit Factor
        ax = axes[1, 0]
        ax.bar(performance['regime'], performance['profit_factor'].clip(upper=5), 
               color=regime_colors, alpha=0.7)
        ax.set_title('Profit Factor', fontweight='bold')
        ax.set_ylabel('Profit Factor')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.5, label='1.0')
        ax.legend()
        
        # Plot 5: Max Drawdown
        ax = axes[1, 1]
        ax.bar(performance['regime'], performance['max_drawdown_pct'], 
               color=regime_colors, alpha=0.7)
        ax.set_title('Maximum Drawdown', fontweight='bold')
        ax.set_ylabel('Max Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Trade Count
        ax = axes[1, 2]
        ax.bar(performance['regime'], performance['total_trades'], color=regime_colors, alpha=0.7)
        ax.set_title('Number of Trades', fontweight='bold')
        ax.set_ylabel('Trade Count')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_pnl_distributions(
        self,
        trades: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (15, 5)
    ) -> None:
        """
        Plot P&L distributions by regime.
        
        Parameters:
        -----------
        trades : pd.DataFrame, optional
            Aligned trades
        figsize : tuple
            Figure size
        """
        if trades is None:
            if self.aligned_trades is None:
                raise ValueError("No aligned trades available")
            trades = self.aligned_trades
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('P&L Distribution by Volatility Regime', 
                    fontsize=14, fontweight='bold')
        
        colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        
        for idx, regime in enumerate(['Low', 'Medium', 'High']):
            ax = axes[idx]
            regime_trades = trades[trades['regime_label'] == regime]
            
            if len(regime_trades) > 0:
                pnl = regime_trades['realized_pnl']
                
                # Histogram
                ax.hist(pnl, bins=30, color=colors[regime], alpha=0.6, edgecolor='black')
                
                # Add vertical lines for mean and median
                ax.axvline(pnl.mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: ${pnl.mean():.2f}')
                ax.axvline(pnl.median(), color='blue', linestyle='--', 
                          linewidth=2, label=f'Median: ${pnl.median():.2f}')
                ax.axvline(0, color='black', linestyle='-', linewidth=1)
                
                ax.set_title(f'{regime} Volatility', fontweight='bold')
                ax.set_xlabel('P&L ($)')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_equity_curves(
        self,
        trades: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (15, 6)
    ) -> None:
        """
        Plot cumulative P&L (equity curve) colored by regime.
        
        Parameters:
        -----------
        trades : pd.DataFrame, optional
            Aligned trades
        figsize : tuple
            Figure size
        """
        if trades is None:
            if self.aligned_trades is None:
                raise ValueError("No aligned trades available")
            trades = self.aligned_trades
        
        # Sort by exit time
        trades_sorted = trades.sort_values('exit_time')
        
        # Calculate cumulative P&L
        trades_sorted['cumulative_pnl'] = trades_sorted['realized_pnl'].cumsum()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        
        # Plot equity curve segments colored by regime
        for regime in ['Low', 'Medium', 'High']:
            regime_mask = trades_sorted['regime_label'] == regime
            regime_data = trades_sorted[regime_mask]
            
            if len(regime_data) > 0:
                ax.scatter(regime_data['exit_time'], regime_data['cumulative_pnl'],
                          c=colors[regime], label=regime, alpha=0.6, s=20)
        
        # Plot overall line
        ax.plot(trades_sorted['exit_time'], trades_sorted['cumulative_pnl'],
               color='black', alpha=0.3, linewidth=1)
        
        ax.set_title('Equity Curve by Volatility Regime', fontweight='bold', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative P&L ($)')
        ax.legend(title='Volatility Regime')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def export_analysis(
        self,
        output_dir: Union[str, Path],
        prefix: str = 'strategy_regime_analysis'
    ) -> Dict[str, Path]:
        """
        Export all analysis results to files.
        
        Parameters:
        -----------
        output_dir : str or Path
            Output directory
        prefix : str
            Filename prefix
            
        Returns:
        --------
        dict
            Dictionary of file type -> filepath
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        files = {}
        
        # Export aligned trades
        if self.aligned_trades is not None:
            filepath = output_dir / f'{prefix}_trades_{timestamp}.csv'
            self.aligned_trades.to_csv(filepath, index=False)
            files['aligned_trades'] = filepath
            print(f"✓ Exported aligned trades to: {filepath}")
        
        # Export performance analysis
        performance = self.analyze_performance_by_regime()
        filepath = output_dir / f'{prefix}_performance_{timestamp}.csv'
        performance.to_csv(filepath, index=False)
        files['performance'] = filepath
        print(f"✓ Exported performance analysis to: {filepath}")
        
        # Export signal characteristics
        characteristics = self.analyze_signal_characteristics()
        filepath = output_dir / f'{prefix}_characteristics_{timestamp}.csv'
        characteristics.to_csv(filepath, index=False)
        files['characteristics'] = filepath
        print(f"✓ Exported signal characteristics to: {filepath}")
        
        # Export recommendations
        recommendations = self.generate_recommendations(performance)
        filepath = output_dir / f'{prefix}_recommendations_{timestamp}.txt'
        with open(filepath, 'w') as f:
            f.write("Strategy Regime Analysis - Recommendations\n")
            f.write("=" * 60 + "\n\n")
            for category, recs in recommendations.items():
                f.write(f"\n{category.upper().replace('_', ' ')}:\n")
                f.write("-" * 40 + "\n")
                for rec in recs:
                    f.write(f"  • {rec}\n")
        files['recommendations'] = filepath
        print(f"✓ Exported recommendations to: {filepath}")
        
        return files


# Convenience functions
def quick_regime_analysis(
    trades_filepath: Union[str, Path],
    regimes_filepath: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Quick analysis workflow.
    
    Parameters:
    -----------
    trades_filepath : str or Path
        Path to trades CSV
    regimes_filepath : str or Path
        Path to regimes CSV
    output_dir : str or Path, optional
        Directory to save results
        
    Returns:
    --------
    tuple
        (performance DataFrame, recommendations dict)
    """
    # Load data
    analyzer = StrategyRegimeAnalyzer()
    analyzer.load_vwapmrs_trades(trades_filepath)
    
    regimes = pd.read_csv(regimes_filepath, index_col=0, parse_dates=True)
    analyzer.regimes = regimes
    
    # Align and analyze
    analyzer.align_trades_with_regimes()
    performance = analyzer.analyze_performance_by_regime()
    recommendations = analyzer.generate_recommendations(performance)
    
    # Visualize
    analyzer.plot_performance_comparison(performance)
    analyzer.plot_pnl_distributions()
    analyzer.plot_equity_curves()
    
    # Export if requested
    if output_dir:
        analyzer.export_analysis(output_dir)
    
    return performance, recommendations


if __name__ == "__main__":
    print("Strategy Regime Analysis Module")
    print("=" * 60)
    print("\nUsage:")
    print("  analyzer = StrategyRegimeAnalyzer()")
    print("  analyzer.load_vwapmrs_trades('path/to/trades.csv')")
    print("  analyzer.regimes = pd.read_csv('path/to/regimes.csv')")
    print("  analyzer.align_trades_with_regimes()")
    print("  performance = analyzer.analyze_performance_by_regime()")
    print("  analyzer.plot_performance_comparison()")
