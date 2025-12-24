"""
Volatility Regimes Module

Classifies market conditions into volatility regimes (low, medium, high).
Uses percentile-based thresholds for robust classification.

Key Features:
- Percentile-based regime classification (33rd, 66th percentiles)
- Configurable thresholds
- Regime transition analysis
- Persistence metrics
- Visualization tools
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple, Dict
import warnings


class VolatilityRegimes:
    """
    Classify and analyze volatility regimes.
    
    Regimes are classified based on percentiles:
    - Low: 0 to 33rd percentile
    - Medium: 33rd to 66th percentile
    - High: 66th to 100th percentile
    
    Attributes:
        volatility (pd.DataFrame): Volatility time series
        regimes (pd.DataFrame): Classified regimes
        thresholds (dict): Threshold values for each ticker
        percentiles (tuple): Percentile cutoffs (default: 33, 66)
    """
    
    def __init__(self, volatility: Optional[pd.DataFrame] = None):
        """
        Initialize VolatilityRegimes classifier.
        
        Parameters:
        -----------
        volatility : pd.DataFrame, optional
            Volatility time series with DatetimeIndex
        """
        self.volatility = volatility
        self.regimes = None
        self.thresholds = {}
        self.percentiles = (33, 66)
    
    def classify_regimes(
        self,
        volatility: Optional[pd.DataFrame] = None,
        method: str = 'percentile',
        low_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        percentiles: Tuple[float, float] = (33, 66)
    ) -> pd.DataFrame:
        """
        Classify volatility into regimes.
        
        Parameters:
        -----------
        volatility : pd.DataFrame, optional
            Volatility time series
        method : str, default='percentile'
            Classification method:
            - 'percentile': Use percentile-based thresholds (recommended)
            - 'fixed': Use fixed threshold values
        low_threshold : float, optional
            Fixed threshold for low/medium boundary (only for method='fixed')
        high_threshold : float, optional
            Fixed threshold for medium/high boundary (only for method='fixed')
        percentiles : tuple, default=(33, 66)
            Percentile cutoffs for low/medium and medium/high boundaries
            
        Returns:
        --------
        pd.DataFrame
            Regime classification (0=Low, 1=Medium, 2=High)
        """
        if volatility is None:
            if self.volatility is None:
                raise ValueError("No volatility data provided")
            volatility = self.volatility
        else:
            self.volatility = volatility
        
        self.percentiles = percentiles
        regimes = pd.DataFrame(index=volatility.index, columns=volatility.columns)
        
        for col in volatility.columns:
            series = volatility[col].dropna()
            
            if len(series) < 10:
                warnings.warn(f"Insufficient data for {col}. Need at least 10 observations.")
                continue
            
            if method == 'percentile':
                # Calculate percentile thresholds
                low_pct, high_pct = percentiles
                threshold_low = np.percentile(series, low_pct)
                threshold_high = np.percentile(series, high_pct)
                
                # Store thresholds
                self.thresholds[col] = {
                    'low': threshold_low,
                    'high': threshold_high,
                    'method': 'percentile',
                    'percentiles': percentiles
                }
                
            elif method == 'fixed':
                if low_threshold is None or high_threshold is None:
                    raise ValueError("Fixed thresholds required for method='fixed'")
                
                threshold_low = low_threshold
                threshold_high = high_threshold
                
                self.thresholds[col] = {
                    'low': threshold_low,
                    'high': threshold_high,
                    'method': 'fixed'
                }
            
            else:
                raise ValueError(f"Invalid method: {method}. Use 'percentile' or 'fixed'")
            
            # Classify regimes
            regime = pd.Series(index=series.index, dtype=int)
            regime[series <= threshold_low] = 0  # Low
            regime[(series > threshold_low) & (series <= threshold_high)] = 1  # Medium
            regime[series > threshold_high] = 2  # High
            
            regimes.loc[series.index, col] = regime
        
        # Convert to integer type
        regimes = regimes.astype('Int64')
        
        self.regimes = regimes
        return regimes
    
    def get_regime_labels(self, numeric: bool = False) -> Dict[int, str]:
        """
        Get regime labels.
        
        Parameters:
        -----------
        numeric : bool, default=False
            If True, return numeric labels, else string labels
            
        Returns:
        --------
        dict
            Mapping of regime codes to labels
        """
        if numeric:
            return {0: 0, 1: 1, 2: 2}
        else:
            return {0: 'Low', 1: 'Medium', 2: 'High'}
    
    def get_regime_statistics(
        self,
        regimes: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate statistics for each regime.
        
        Parameters:
        -----------
        regimes : pd.DataFrame, optional
            Regime classifications
            
        Returns:
        --------
        pd.DataFrame
            Statistics per regime (count, percentage, avg duration)
        """
        if regimes is None:
            if self.regimes is None:
                raise ValueError("No regime classifications available")
            regimes = self.regimes
        
        stats_list = []
        
        for col in regimes.columns:
            series = regimes[col].dropna()
            
            for regime_code in [0, 1, 2]:
                regime_label = self.get_regime_labels()[regime_code]
                
                # Count occurrences
                count = (series == regime_code).sum()
                percentage = (count / len(series)) * 100
                
                # Calculate average duration (consecutive periods)
                durations = []
                current_duration = 0
                
                for val in series:
                    if val == regime_code:
                        current_duration += 1
                    else:
                        if current_duration > 0:
                            durations.append(current_duration)
                        current_duration = 0
                
                if current_duration > 0:
                    durations.append(current_duration)
                
                avg_duration = np.mean(durations) if durations else 0
                
                stats_list.append({
                    'ticker': col,
                    'regime': regime_label,
                    'regime_code': regime_code,
                    'count': count,
                    'percentage': percentage,
                    'avg_duration': avg_duration,
                    'max_duration': max(durations) if durations else 0
                })
        
        return pd.DataFrame(stats_list)
    
    def analyze_transitions(
        self,
        regimes: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze regime transitions (transition matrix).
        
        Parameters:
        -----------
        regimes : pd.DataFrame, optional
            Regime classifications
            
        Returns:
        --------
        dict
            Dictionary with transition matrices for each ticker
        """
        if regimes is None:
            if self.regimes is None:
                raise ValueError("No regime classifications available")
            regimes = self.regimes
        
        transition_matrices = {}
        
        for col in regimes.columns:
            series = regimes[col].dropna()
            
            # Create transition matrix
            transitions = np.zeros((3, 3), dtype=int)
            
            for i in range(len(series) - 1):
                from_regime = series.iloc[i]
                to_regime = series.iloc[i + 1]
                transitions[from_regime, to_regime] += 1
            
            # Convert to DataFrame
            labels = ['Low', 'Medium', 'High']
            transition_df = pd.DataFrame(
                transitions,
                index=labels,
                columns=labels
            )
            
            # Calculate percentages
            row_sums = transition_df.sum(axis=1)
            transition_pct = transition_df.div(row_sums, axis=0) * 100
            transition_pct = transition_pct.fillna(0)
            
            transition_matrices[col] = {
                'counts': transition_df,
                'percentages': transition_pct
            }
        
        return transition_matrices
    
    def calculate_persistence(
        self,
        regimes: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate regime persistence (probability of staying in same regime).
        
        Parameters:
        -----------
        regimes : pd.DataFrame, optional
            Regime classifications
            
        Returns:
        --------
        pd.DataFrame
            Persistence probabilities for each regime
        """
        transitions = self.analyze_transitions(regimes)
        
        persistence_list = []
        
        for ticker, trans_dict in transitions.items():
            trans_pct = trans_dict['percentages']
            
            for regime_code, regime_label in enumerate(['Low', 'Medium', 'High']):
                # Persistence = diagonal element (same regime to same regime)
                persistence = trans_pct.iloc[regime_code, regime_code]
                
                persistence_list.append({
                    'ticker': ticker,
                    'regime': regime_label,
                    'persistence_pct': persistence
                })
        
        return pd.DataFrame(persistence_list)
    
    def get_current_regime(
        self,
        regimes: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Get the most recent regime classification.
        
        Parameters:
        -----------
        regimes : pd.DataFrame, optional
            Regime classifications
            
        Returns:
        --------
        pd.Series
            Current regime for each ticker
        """
        if regimes is None:
            if self.regimes is None:
                raise ValueError("No regime classifications available")
            regimes = self.regimes
        
        current = regimes.iloc[-1]
        
        # Convert to string labels
        labels = self.get_regime_labels()
        current_labeled = current.map(labels)
        
        return current_labeled
    
    def filter_by_regime(
        self,
        data: pd.DataFrame,
        regime: Union[int, str],
        ticker: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filter data by specific regime.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to filter (must have same index as regimes)
        regime : int or str
            Regime to filter (0/'Low', 1/'Medium', 2/'High')
        ticker : str, optional
            Specific ticker to filter. If None, uses first column
            
        Returns:
        --------
        pd.DataFrame
            Filtered data for specified regime
        """
        if self.regimes is None:
            raise ValueError("No regime classifications available")
        
        # Convert string regime to numeric
        if isinstance(regime, str):
            reverse_labels = {v: k for k, v in self.get_regime_labels().items()}
            regime = reverse_labels[regime]
        
        # Select ticker
        if ticker is None:
            ticker = self.regimes.columns[0]
        
        # Filter
        regime_mask = self.regimes[ticker] == regime
        filtered = data[regime_mask]
        
        return filtered
    
    def plot_regimes(
        self,
        volatility: Optional[pd.DataFrame] = None,
        regimes: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (15, 8),
        show_thresholds: bool = True
    ) -> None:
        """
        Plot volatility with regime classifications.
        
        Parameters:
        -----------
        volatility : pd.DataFrame, optional
            Volatility time series
        regimes : pd.DataFrame, optional
            Regime classifications
        figsize : tuple, default=(15, 8)
            Figure size
        show_thresholds : bool, default=True
            Whether to show threshold lines
        """
        if volatility is None:
            if self.volatility is None:
                raise ValueError("No volatility data available")
            volatility = self.volatility
        
        if regimes is None:
            if self.regimes is None:
                raise ValueError("No regime classifications available")
            regimes = self.regimes
        
        n_cols = volatility.shape[1]
        fig, axes = plt.subplots(n_cols, 1, figsize=figsize)
        
        if n_cols == 1:
            axes = [axes]
        
        colors = {0: 'green', 1: 'orange', 2: 'red'}
        labels = {0: 'Low', 1: 'Medium', 2: 'High'}
        
        for idx, col in enumerate(volatility.columns):
            ax = axes[idx]
            
            vol_series = volatility[col].dropna()
            regime_series = regimes[col].dropna()
            
            # Plot volatility colored by regime
            for regime_code in [0, 1, 2]:
                mask = regime_series == regime_code
                if mask.any():
                    ax.scatter(
                        vol_series[mask].index,
                        vol_series[mask].values,
                        c=colors[regime_code],
                        label=labels[regime_code],
                        alpha=0.6,
                        s=10
                    )
            
            # Plot threshold lines
            if show_thresholds and col in self.thresholds:
                ax.axhline(
                    y=self.thresholds[col]['low'],
                    color='blue',
                    linestyle='--',
                    alpha=0.7,
                    label=f"Low/Med: {self.thresholds[col]['low']:.4f}"
                )
                ax.axhline(
                    y=self.thresholds[col]['high'],
                    color='purple',
                    linestyle='--',
                    alpha=0.7,
                    label=f"Med/High: {self.thresholds[col]['high']:.4f}"
                )
            
            ax.set_title(f'{col} - Volatility Regimes', fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Volatility')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_regime_distribution(
        self,
        regimes: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot regime distribution as stacked bar chart.
        
        Parameters:
        -----------
        regimes : pd.DataFrame, optional
            Regime classifications
        figsize : tuple, default=(12, 6)
            Figure size
        """
        if regimes is None:
            if self.regimes is None:
                raise ValueError("No regime classifications available")
            regimes = self.regimes
        
        stats = self.get_regime_statistics()
        
        # Pivot for plotting
        pivot = stats.pivot(index='ticker', columns='regime', values='percentage')
        
        # Plot stacked bar chart
        ax = pivot.plot(
            kind='bar',
            stacked=True,
            color=['green', 'orange', 'red'],
            figsize=figsize,
            rot=0
        )
        
        ax.set_title('Volatility Regime Distribution', fontweight='bold', fontsize=14)
        ax.set_xlabel('Ticker')
        ax.set_ylabel('Percentage (%)')
        ax.legend(title='Regime', loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def plot_transition_matrix(
        self,
        ticker: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Plot regime transition matrix as heatmap.
        
        Parameters:
        -----------
        ticker : str, optional
            Ticker to plot. If None, uses first ticker
        figsize : tuple, default=(8, 6)
            Figure size
        """
        if self.regimes is None:
            raise ValueError("No regime classifications available")
        
        if ticker is None:
            ticker = self.regimes.columns[0]
        
        transitions = self.analyze_transitions()
        trans_pct = transitions[ticker]['percentages']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(trans_pct.values, cmap='YlOrRd', aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(trans_pct.columns)))
        ax.set_yticks(np.arange(len(trans_pct.index)))
        ax.set_xticklabels(trans_pct.columns)
        ax.set_yticklabels(trans_pct.index)
        
        # Add text annotations
        for i in range(len(trans_pct.index)):
            for j in range(len(trans_pct.columns)):
                text = ax.text(
                    j, i, f'{trans_pct.iloc[i, j]:.1f}%',
                    ha="center", va="center", color="black", fontsize=12
                )
        
        ax.set_title(f'{ticker} - Regime Transition Matrix', fontweight='bold', fontsize=14)
        ax.set_xlabel('To Regime')
        ax.set_ylabel('From Regime')
        
        plt.colorbar(im, ax=ax, label='Probability (%)')
        plt.tight_layout()
        plt.show()
    
    def save_regimes(self, filepath: str) -> None:
        """
        Save regime classifications to CSV.
        
        Parameters:
        -----------
        filepath : str
            Output file path
        """
        if self.regimes is None:
            raise ValueError("No regime classifications to save")
        
        # Convert numeric to string labels
        labeled_regimes = self.regimes.copy()
        labels = self.get_regime_labels()
        
        for col in labeled_regimes.columns:
            labeled_regimes[col] = labeled_regimes[col].map(labels)
        
        labeled_regimes.to_csv(filepath)
        print(f"✓ Regime classifications saved to: {filepath}")


# Convenience functions
def classify_volatility_regimes(
    volatility: pd.DataFrame,
    percentiles: Tuple[float, float] = (33, 66)
) -> pd.DataFrame:
    """
    Convenience function to classify volatility regimes.
    
    Parameters:
    -----------
    volatility : pd.DataFrame
        Volatility time series
    percentiles : tuple, default=(33, 66)
        Percentile cutoffs
        
    Returns:
    --------
    pd.DataFrame
        Regime classifications (0=Low, 1=Medium, 2=High)
    """
    classifier = VolatilityRegimes(volatility)
    regimes = classifier.classify_regimes(percentiles=percentiles)
    return regimes


def analyze_regime_performance(
    returns: pd.DataFrame,
    regimes: pd.DataFrame,
    ticker: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyze return performance by volatility regime.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Returns data
    regimes : pd.DataFrame
        Regime classifications
    ticker : str, optional
        Ticker to analyze. If None, uses first column
        
    Returns:
    --------
    pd.DataFrame
        Performance statistics by regime
    """
    if ticker is None:
        ticker = returns.columns[0]
    
    results = []
    labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    
    for regime_code in [0, 1, 2]:
        mask = regimes[ticker] == regime_code
        regime_returns = returns[ticker][mask]
        
        if len(regime_returns) > 0:
            results.append({
                'regime': labels[regime_code],
                'count': len(regime_returns),
                'mean_return': regime_returns.mean(),
                'std_return': regime_returns.std(),
                'sharpe': (regime_returns.mean() / regime_returns.std()) * np.sqrt(252) if regime_returns.std() > 0 else 0,
                'min_return': regime_returns.min(),
                'max_return': regime_returns.max()
            })
    
    return pd.DataFrame(results)
