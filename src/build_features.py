# src/build_features.py

import pandas as pd
import numpy as np
from pathlib import Path
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import warnings
warnings.filterwarnings('ignore')

def load_stock_data(ticker):
    """Load raw stock data with proper CSV structure handling"""
    raw_path = Path('data/raw') / f'{ticker}.csv'
    
    # Read CSV skipping the first 2 rows (headers and ticker row)
    df = pd.read_csv(
        raw_path,
        skiprows=2,  # Skip "Price,Close,High..." and "Ticker,KO,KO..."
        parse_dates=[0],  # First column is date
        index_col=0
    )
    
    # Rename columns properly
    df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    
    # Convert to numeric (in case there are string values)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows with NaN values
    df = df.dropna()
    
    # Sort by date ascending
    df = df.sort_index()
    
    return df

def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Price-based features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving Averages
    df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['EMA_12'] = EMAIndicator(close=df['Close'], window=12).ema_indicator()
    df['EMA_26'] = EMAIndicator(close=df['Close'], window=26).ema_indicator()
    
    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    # RSI
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    # Bollinger Bands
    bollinger = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid']
    
    # Volatility
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Price position indicators
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    
    return df

def create_target_variable(df, horizon=5):
    """Create target variable for prediction"""
    df = df.copy()
    df['Target'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
    return df

def process_stock(ticker):
    """Process a single stock: load data, add features, save"""
    try:
        # Load data
        df = load_stock_data(ticker)
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Create target variable (predict 5 days ahead)
        df = create_target_variable(df, horizon=5)
        
        # Drop rows with NaN (from indicator calculations)
        df = df.dropna()
        
        # Save processed data
        processed_path = Path('data/processed') / f'{ticker}_features.csv'
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path)
        
        return True, f"✓ Processed {len(df)} rows"
        
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def main():
    """Main function to process all stocks"""
    print("=" * 60)
    print("STOCKLENS AI - FEATURE BUILDER")
    print("=" * 60)
    
    # List of stock tickers
    tickers = [
        'AAPL', 'AMZN', 'BA', 'BAC', 'COST', 'CVX', 'DIS', 'GOOGL',
        'JNJ', 'JPM', 'KO', 'META', 'MSFT', 'NKE', 'NVDA', 'PFE',
        'TSLA', 'UNH', 'V', 'WMT', 'XOM'
    ]
    
    print(f"\nProcessing {len(tickers)} stocks...\n")
    
    results = {}
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}")
        success, message = process_stock(ticker)
        results[ticker] = (success, message)
        print(f"  {message}")
    
    # Summary
    print("\n" + "=" * 60)
    successful = sum(1 for s, _ in results.values() if s)
    print(f"✓ Successful: {successful}/{len(tickers)}")
    print(f"❌ Failed: {len(tickers) - successful}")
    print("=" * 60)
    
    if successful > 0:
        print("\n📊 Next step: python -m src.make_dataset")

if __name__ == '__main__':
    main()