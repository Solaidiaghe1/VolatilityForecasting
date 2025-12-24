"""
Demo script for data_loader module

This script demonstrates how to use the DataLoader class to:
1. Download data from Yahoo Finance
2. Clean and validate the data
3. Save to CSV
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader import DataLoader


def main():
    print("=" * 60)
    print("VOLATILITY FORECASTING - DATA LOADER DEMO")
    print("=" * 60)
    
    # Initialize loader
    loader = DataLoader()
    
    # Example 1: Download data from Yahoo Finance
    print("\n📊 Example 1: Downloading data from Yahoo Finance")
    print("-" * 60)
    
    try:
        data = loader.load_from_yfinance(
            tickers=['AAPL', 'MSFT', 'SPY'],
            period='2y',  # 2 years of data
            interval='1d'
        )
        
        print(f"\n✓ Downloaded {len(data)} days of data")
        print(f"\nFirst few rows:")
        print(data.head())
        print(f"\nLast few rows:")
        print(data.tail())
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return
    
    # Example 2: Clean the data
    print("\n\n🧹 Example 2: Cleaning the data")
    print("-" * 60)
    
    cleaned_data = loader.clean_data(
        handle_missing='ffill',
        handle_zeros='ffill'
    )
    
    # Example 3: Validate the data
    print("\n\n✅ Example 3: Validating data quality")
    print("-" * 60)
    
    validation = loader.validate_data()
    
    print(f"\nData is valid: {validation['is_valid']}")
    if validation['issues']:
        print(f"Issues found: {validation['issues']}")
    
    print(f"\nStatistics:")
    print(f"  Total rows: {validation['stats']['n_rows']}")
    print(f"  Tickers: {', '.join(validation['stats']['tickers'])}")
    print(f"  Date range: {validation['stats']['date_range'][0].date()} to {validation['stats']['date_range'][1].date()}")
    
    print(f"\nPrice ranges:")
    for ticker in validation['stats']['tickers']:
        min_price = validation['stats']['price_min'][ticker]
        max_price = validation['stats']['price_max'][ticker]
        avg_price = validation['stats']['price_mean'][ticker]
        print(f"  {ticker}: ${min_price:.2f} - ${max_price:.2f} (avg: ${avg_price:.2f})")
    
    # Example 4: Save to CSV
    print("\n\n💾 Example 4: Saving data to CSV")
    print("-" * 60)
    
    output_path = Path(__file__).parent / 'data' / 'raw' / 'sample_prices.csv'
    loader.save_to_csv(output_path)
    
    print(f"\n✓ Data saved successfully!")
    print(f"  You can now use this file for analysis")
    
    # Example 5: Load from CSV
    print("\n\n📂 Example 5: Loading data from CSV")
    print("-" * 60)
    
    loader2 = DataLoader()
    loaded_data = loader2.load_from_csv(output_path)
    
    print(f"\n✓ Loaded {len(loaded_data)} rows from CSV")
    print(f"  Tickers: {', '.join(loader2.tickers)}")
    
    print("\n" + "=" * 60)
    print("✨ Demo completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
