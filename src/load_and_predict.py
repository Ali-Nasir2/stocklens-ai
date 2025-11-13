import yfinance as yf
import pandas as pd
import pandas_ta as ta
import joblib
import numpy as np
import json
import tensorflow as tf

# Configuration
LOOKBACK = 60

# Load trained hybrid model and scaler
print("[INIT] Loading hybrid model and scaler...")
MODEL = tf.keras.models.load_model("models/stocklens_hybrid_best.keras", compile=False)
SCALER = joblib.load("models/scaler.pkl")

# Load feature list and split info
with open("models/feature_list.json", "r") as f:
    FEATURE_LIST = json.load(f)

with open("models/feature_split.json", "r") as f:
    FEATURE_SPLIT = json.load(f)

price_indices = FEATURE_SPLIT['price_indices']
indicator_indices = FEATURE_SPLIT['indicator_indices']

print(f"[INIT] Hybrid model ready.")
print(f"  Price features: {FEATURE_SPLIT['price_features']}")
print(f"  Indicator features: {FEATURE_SPLIT['indicator_features']}")

# Helper: add technical indicators
def _add_indicators(df):
    """Add technical indicators exactly as in training"""
    # RSI (14-period)
    df["RSI"] = ta.rsi(df["Close"], length=14)
    
    # Simple Moving Averages
    df["SMA50"] = ta.sma(df["Close"], length=50)
    df["SMA200"] = ta.sma(df["Close"], length=200)
    
    # MACD (12, 26, 9)
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd is not None and "MACD_12_26_9" in macd.columns:
        df["MACD_12_26_9"] = macd["MACD_12_26_9"]
    
    # Daily Return %
    df["Daily_Return_%"] = df["Close"].pct_change() * 100
    
    # Drop NaN rows
    df.dropna(inplace=True)
    
    return df


# Predict next-day closing price
def predict_next(symbol):
    print(f"\n{'='*60}")
    print(f"PREDICTION FOR {symbol}")
    print(f"{'='*60}")
    print(f"[INFO] Fetching latest data for {symbol}...")
    
    # Download 2 years of data
    df = yf.download(symbol, period="2y", interval="1d", progress=False)
    if df.empty:
        print(f"[ERROR] No data found for {symbol}.")
        return None

    # Reset index
    df.reset_index(inplace=True)
    
    # Flatten multi-level columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns.values]
    
    # Rename columns
    column_mapping = {
        'Close': 'Close',
        f'Close_{symbol}': 'Close',
        'High': 'High',
        f'High_{symbol}': 'High',
        'Low': 'Low',
        f'Low_{symbol}': 'Low',
        'Open': 'Open',
        f'Open_{symbol}': 'Open',
        'Volume': 'Volume',
        f'Volume_{symbol}': 'Volume'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
    
    # Store current price
    current_price = df["Close"].iloc[-1]
    
    # Add indicators
    df = _add_indicators(df)
    
    # Check for missing features
    missing_features = [f for f in FEATURE_LIST if f not in df.columns]
    if missing_features:
        print(f"[ERROR] Missing features: {missing_features}")
        print(f"[ERROR] Available columns: {df.columns.tolist()}")
        return None
    
    # Select features in correct order
    df = df[FEATURE_LIST]
    
    # Check we have enough data
    if len(df) < LOOKBACK:
        print(f"[ERROR] Not enough data. Need {LOOKBACK} rows, got {len(df)}")
        return None

    print(f"[INFO] Using {len(df)} rows of historical data")
    
    # Scale features
    arr = SCALER.transform(df)
    
    # Take last LOOKBACK timesteps
    X = arr[-LOOKBACK:]
    
    # Split into price and indicator streams
    X_price = X[:, price_indices].reshape(1, LOOKBACK, len(price_indices))
    X_indicators = X[:, indicator_indices].reshape(1, LOOKBACK, len(indicator_indices))

    # Predict with both inputs (hybrid model)
    pred_norm = MODEL.predict([X_price, X_indicators], verbose=0)[0][0]

    # Denormalize
    close_index = FEATURE_LIST.index("Close")
    close_min = SCALER.data_min_[close_index]
    close_max = SCALER.data_max_[close_index]
    pred_price = pred_norm * (close_max - close_min) + close_min

    # Show comparison
    print(f"\n[RESULT] {symbol} - Next Day Prediction (Hybrid Model)")
    print(f"-"*60)
    print(f" Current Close Price  : ${current_price:.2f}")
    print(f" Predicted Next Close : ${pred_price:.2f}")
    print(f" Normalized Output    : {pred_norm:.4f}")

    # Direction indicator
    diff = pred_price - current_price
    direction = "📈 UP" if diff > 0 else "📉 DOWN"
    pct_change = (diff / current_price) * 100
    
    print(f"\n[TREND] Expected Movement: {direction}")
    print(f"        Price Change: {diff:+.2f} USD ({pct_change:+.2f}%)")
    print(f"{'='*60}\n")

    return {
        "symbol": symbol,
        "current_price": float(current_price),
        "predicted_price": float(pred_price),
        "direction": direction,
        "diff": float(diff),
        "pct_change": float(pct_change)
    }


# Run predictions
if __name__ == "__main__":
    # Test on all training stocks
    stocks = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]
    
    print("\n" + "="*60)
    print("STOCKLENS AI - LIVE PREDICTIONS (HYBRID MODEL)")
    print("="*60)
    
    results = []
    for ticker in stocks:
        result = predict_next(ticker)
        if result:
            results.append(result)
    
    # Summary table
    if results:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"{'Stock':<10} {'Current':<12} {'Predicted':<12} {'Change':<15} {'Direction'}")
        print("-"*60)
        for r in results:
            print(f"{r['symbol']:<10} ${r['current_price']:<11.2f} ${r['predicted_price']:<11.2f} {r['pct_change']:>6.2f}%{' '*7} {r['direction']}")
        print("="*60)