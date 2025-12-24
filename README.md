# Volatility Forecasting and Regime Analysis

A quantitative toolkit for forecasting equity volatility and identifying volatility regimes for risk management and strategy analysis.

## Quick Start

### Run Complete Pipeline
```bash
# Basic usage (AAPL, 2-year data)
python run_pipeline.py

# Custom ticker and period
python run_pipeline.py --ticker MSFT --period 5y --output results/msft

# Include strategy analysis
python run_pipeline.py --ticker SPY --strategy
```

### Use Individual Modules
```python
from src.data_loader import fetch_prices
from src.returns import compute_log_returns
from src.rolling_vol import compute_rolling_volatility

# Load data
prices = fetch_prices('AAPL', period='2y')

# Calculate returns
returns = compute_log_returns(prices)

# Calculate volatility
volatility = compute_rolling_volatility(returns, window=20, annualize=True)
```

## Project Status

###  Completed Components

#### 1. Data Handling (`data_loader.py`)
- Load price data from CSV files or Yahoo Finance API
- Robust data cleaning with multiple strategies for missing values
- Handle zero/negative prices with forward fill or interpolation
- Data validation with comprehensive checks
- Support for single or multiple tickers
- Export cleaned data to CSV

**Features:**
- Forward fill, backward fill, and interpolation for missing data
- Configurable thresholds for dropping bad data
- Automatic detection of data quality issues
- Summary statistics and validation reports

**Usage:**
```python
from src.data_loader import DataLoader, fetch_prices

# Load from Yahoo Finance
loader = DataLoader()
data = loader.load_from_yfinance(['AAPL', 'MSFT'], period='5y')
clean_data = loader.clean_data()

# Or use convenience function
prices = fetch_prices('AAPL', start_date='2020-01-01')

# Validate data quality
validation = loader.validate_data()
print(validation)
```

#### 2. Returns Calculation (`returns.py`)
- Compute log returns (preferred for volatility modeling)
- Compute simple and percent returns
- Statistical analysis (mean, std, skewness, kurtosis)
- Stationarity checks for time series validation
- Comprehensive visualizations (distributions, Q-Q plots, rolling stats)
- Outlier detection

**Usage:**
```python
from src.returns import ReturnsCalculator, compute_log_returns

returns = compute_log_returns(prices)
calc = ReturnsCalculator(prices)
calc.plot_returns()
```

#### 3. Rolling Volatility (`rolling_vol.py`)
- Simple moving window volatility calculation
- Multiple window sizes (20, 60, 252 days)
- Annualization with configurable periods
- Statistical analysis and visualization

**Usage:**
```python
from src.rolling_vol import RollingVolatility, compute_rolling_volatility

# Quick calculation
vol = compute_rolling_volatility(returns, window=20, annualize=True)

# Detailed analysis
calc = RollingVolatility(returns)
vol = calc.compute_volatility(window=20)
vol_ann = calc.annualize(vol)
calc.plot_volatility()
```

#### 4. EWMA Volatility (`ewma_vol.py`)
- Exponentially weighted moving average volatility
- RiskMetrics standard (λ=0.94 for daily data)
- Configurable decay factors
- Comparison of multiple λ values
- 1-step ahead forecasting

**Usage:**
```python
from src.ewma_vol import EWMAVolatility, compute_ewma_volatility

# RiskMetrics standard
vol = compute_ewma_volatility(returns, lambda_param=0.94, annualize=True)

# Compare different lambdas
calc = EWMAVolatility(returns)
results = calc.compare_lambdas(lambdas=[0.90, 0.94, 0.97])
```

#### 5. GARCH Model (`garch_model.py`)
- GARCH(1,1) model fitting and forecasting
- Conditional volatility estimation
- Multi-step forecasting
- Model diagnostics and stationarity checks
- Parameter extraction (ω, α, β)

**Usage:**
```python
from src.garch_model import GARCHModel, forecast_garch

# Fit and forecast
model = GARCHModel(returns)
model.fit(show_summary=True)
forecast = model.forecast_volatility(horizon=1, annualize=True)

# Get parameters
params = model.get_parameters()
print(f"Persistence: {params['persistence']}")

# Plot conditional volatility
model.plot_conditional_volatility()
```

#### 6. Volatility Regimes (`volatility_regimes.py`) 
- Percentile-based regime classification (Low/Medium/High)
- Fixed threshold classification option
- Transition matrix analysis
- Persistence metrics
- Filter data by regime
- Performance analysis by regime
- Comprehensive visualizations

**Usage:**
```python
from src.volatility_regimes import VolatilityRegimes, analyze_regime_performance

# Classify regimes
classifier = VolatilityRegimes(volatility)
regimes = classifier.classify_regimes(percentiles=(33, 66))

# Analyze transitions
transitions = classifier.analyze_transitions()
persistence = classifier.calculate_persistence()

# Filter by regime
high_vol_data = classifier.filter_by_regime(data, regime='High')

# Visualize
classifier.plot_regimes()
classifier.plot_transition_matrix()
```

#### 7. Strategy Overlay Analysis (`strategy_analysis.py`) 
- Integrate VWAP/MRS signals with volatility regimes
- Performance metrics by regime (Sharpe, Sortino, Calmar)
- Signal characteristics and quality analysis
- Risk-adjusted returns by regime
- Regime-specific recommendations
- Comprehensive visualizations

**Usage:**
```python
from src.strategy_analysis import StrategyRegimeAnalyzer

# Load and analyze
analyzer = StrategyRegimeAnalyzer()
analyzer.load_vwapmrs_trades('path/to/trades.csv')
analyzer.regimes = pd.read_csv('path/to/regimes.csv')

# Align and analyze
analyzer.align_trades_with_regimes()
performance = analyzer.analyze_performance_by_regime()

# Visualize
analyzer.plot_performance_comparison()
analyzer.plot_equity_curves()

# Get recommendations
recommendations = analyzer.generate_recommendations()
```

#### 8. Utilities Module (`utils.py`) 
- Annualization and deannualization helpers
- Date/time alignment utilities
- Parameter validation functions
- Plotting utilities with recession shading
- Data transformation (winsorization, standardization)
- Statistical utilities (rolling stats, correlation matrices)
- File I/O utilities
- Performance metrics (Sharpe, Sortino, max drawdown)

**Usage:**
```python
from src.utils import (
    annualize_volatility,
    calculate_sharpe_ratio,
    setup_plot_style,
    validate_dataframe
)

# Annualize daily volatility
ann_vol = annualize_volatility(daily_vol)

# Calculate Sharpe ratio
sharpe = calculate_sharpe_ratio(returns)

# Setup plotting
setup_plot_style()

# Validate data
validate_dataframe(df, required_columns=['Close'])
```

#### 9. End-to-End Pipeline (`run_pipeline.py`) 
- Complete workflow from data loading to report generation
- Command-line interface with argparse
- 6-step automated process:
  1. Data loading & cleaning
  2. Returns calculation
  3. Volatility modeling (Rolling, EWMA, GARCH)
  4. Regime classification
  5. Strategy analysis (optional)
  6. Report generation
- Comprehensive logging and error handling
- Organized output structure

**Usage:**
```bash
# Basic usage
python run_pipeline.py

# Custom parameters
python run_pipeline.py --ticker MSFT --period 5y --output results/msft

# Include strategy analysis
python run_pipeline.py --ticker SPY --strategy

# Quiet mode
python run_pipeline.py --ticker AAPL --period 1y --quiet
```

**Output Structure:**
```
results/{output_name}/
├── data/
│   ├── {ticker}_prices.csv
│   ├── {ticker}_returns.csv
│   ├── {ticker}_volatility.csv
│   └── {ticker}_regimes.csv
├── strategy/
│   └── {ticker}_strategy_performance.csv
└── reports/
    └── {ticker}_report.txt
```

### 🔄 Pending (Optional)
- Jupyter notebooks for exploratory analysis
- Web dashboard for interactive visualization

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
VolatilityForecasting/
├─ data/
│   ├─ raw/              # Original price data
│   └─ processed/        # Cleaned returns data
├─ notebooks/            # Jupyter notebooks ✅
│   ├─ 01_introduction_quickstart.ipynb  ✅ Complete
│   └─ 02_data_analysis.ipynb           ✅ Complete
├─ src/                  # Python modules
│   ├─ data_loader.py       Complete
│   ├─ returns.py           Complete
│   ├─ rolling_vol.py       Complete
│   ├─ ewma_vol.py          Complete
│   ├─ garch_model.py       Complete
│   ├─ volatility_regimes.py    Complete
│   ├─ strategy_analysis.py     Complete
│   ├─ utils.py                Complete
│   └─ run_pipeline.py         Complete
├─ tests/                      # Unit tests
│   ├─ test_data_loader.py      Complete (18 tests)
│   ├─ test_returns.py          Complete (28 tests)
│   └─ test_volatility_regimes.py  Complete (35 tests)
├─ demo_strategy_analysis.py    Complete
├─ requirements.txt             Complete
├─ README.md                    Complete
└─ TESTING_SUMMARY.md           Complete
```

## Jupyter Notebooks

Interactive notebooks for exploration and learning:

### 📓 01_introduction_quickstart.ipynb
Quick start guide covering:
- Data loading from Yahoo Finance
- Returns calculation
- Volatility modeling (Rolling, EWMA, GARCH)
- Regime classification
- Performance analysis

### 📓 02_data_analysis.ipynb
Comprehensive data analysis:
- Multi-ticker data loading
- Data cleaning and validation
- Statistical analysis
- Correlation analysis
- Outlier detection

**Run notebooks:**
```bash
cd notebooks
jupyter notebook
# Or use VS Code with Jupyter extension
```

See [NOTEBOOKS_COMPLETION_REPORT.md](NOTEBOOKS_COMPLETION_REPORT.md) for details.

## Testing

### Test Coverage: 81 Tests (99% Passing) 

The project includes comprehensive unit tests covering:
- **Data Loader**: 18 tests (data loading, cleaning, validation)
- **Returns**: 28 tests (calculations, statistics, stationarity)
- **Volatility Regimes**: 35 tests (classification, transitions, filtering)

**Run tests:**

```bash
# Run all tests
PYTHONPATH=. python3 -m pytest tests/ -v

# Run specific test module
PYTHONPATH=. python3 tests/test_data_loader.py
PYTHONPATH=. python3 tests/test_returns.py
PYTHONPATH=. python3 tests/test_volatility_regimes.py

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

**Test Results:**
```
Module                     Tests    Passed   Coverage
--------------------------------------------------------
test_data_loader.py          18       17      95%
test_returns.py              28       28      100%
test_volatility_regimes.py   35       35      100%
--------------------------------------------------------
TOTAL                        81       80      99%
```

See [TESTING_SUMMARY.md](TESTING_SUMMARY.md) for detailed test documentation.

## Key Concepts

### Volatility
Measures the magnitude of price movements over time, quantifying market risk for position sizing and strategy robustness.

### Volatility Models
- **Rolling Volatility**: Simple moving window standard deviation
- **EWMA**: Exponentially weighted moving average (RiskMetrics λ=0.94)
- **GARCH(1,1)**: Autoregressive conditional heteroskedasticity model

### Volatility Regimes
Segments market conditions into low, medium, and high volatility periods using percentile-based thresholds for adaptive risk management.

## Roadmap

- [x] Project structure setup
- [x] Requirements and dependencies