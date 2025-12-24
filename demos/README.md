# Demo Scripts

This folder contains demonstration scripts showcasing the capabilities of the Volatility Forecasting toolkit.

## 📁 Available Demos

### 1. Data Loading
**`demo_data_loader.py`**
- Load data from Yahoo Finance
- Clean and validate data
- Handle missing values
- Save to CSV

```bash
python demos/demo_data_loader.py
```

---

### 2. Returns Calculation
**`demo_returns.py`**
- Calculate log returns
- Compute statistics
- Compare return types

```bash
python demos/demo_returns.py
```

**`demo_returns_visual.py`**
- Returns time series plots
- Distribution histograms
- Q-Q plots for normality

```bash
python demos/demo_returns_visual.py
```

---

### 3. Volatility Models
**`demo_volatility_models.py`**
- Rolling volatility
- EWMA volatility
- GARCH(1,1) model
- Model comparison

```bash
python demos/demo_volatility_models.py
```

**`demo_volatility_visual.py`**
- Volatility charts
- Model comparisons
- Parameter analysis

```bash
python demos/demo_volatility_visual.py
```

---

### 4. Regime Classification
**`demo_regimes.py`**
- Classify volatility regimes
- Transition analysis
- Persistence metrics

```bash
python demos/demo_regimes.py
```

**`demo_regimes_visual.py`**
- Regime visualization
- Transition matrices
- Performance by regime

```bash
python demos/demo_regimes_visual.py
```

---

### 5. Strategy Analysis
**`demo_strategy_analysis.py`**
- Load VWAP/MRS trades
- Align with regimes
- Performance metrics
- Generate recommendations

```bash
python demos/demo_strategy_analysis.py
```

---

## 🚀 Quick Start

### Run All Demos
```bash
# From project root
cd demos

# Run each demo
python demo_data_loader.py
python demo_returns.py
python demo_volatility_models.py
python demo_regimes.py
python demo_strategy_analysis.py
```

### Visual Demos
```bash
# Run visual demonstrations
python demo_returns_visual.py
python demo_volatility_visual.py
python demo_regimes_visual.py
```

---

## 📊 Demo Categories

| Category | Scripts | Visualizations |
|----------|---------|----------------|
| **Data** | demo_data_loader.py | - |
| **Returns** | demo_returns.py | demo_returns_visual.py |
| **Volatility** | demo_volatility_models.py | demo_volatility_visual.py |
| **Regimes** | demo_regimes.py | demo_regimes_visual.py |
| **Strategy** | demo_strategy_analysis.py | ✓ (built-in) |

---

## 💡 Tips

1. **Start with basics**: Run demos in order (data → returns → volatility → regimes → strategy)
2. **Visual demos**: Add `_visual` suffix for chart demonstrations
3. **Customize**: Edit ticker and period parameters in each script
4. **Learn by doing**: Modify scripts to experiment with different settings

---

## 📚 Related Resources

- **Full Pipeline**: `../run_pipeline.py` - Automated end-to-end workflow
- **Notebooks**: `../notebooks/` - Interactive Jupyter notebooks
- **Documentation**: `../README.md` - Complete project guide
- **Tests**: `../tests/` - Unit test examples

---

## 🔧 Requirements

All demos require:
- Python 3.8+
- Dependencies installed: `pip install -r ../requirements.txt`
- Internet connection (for Yahoo Finance data)

---

**Total Demos**: 8 scripts  
**Last Updated**: December 24, 2024
