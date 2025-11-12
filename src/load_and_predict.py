import yfinance as yf
import pandas as pd
import pandas_ta as ta
import joblib
import numpy as np
import json
import tensorflow as tf

# Configuration
LOOKBACK = 60

# -----------------------------
# Load trained model and scaler
# -----------------------------
print("[INIT] Loading model and scaler...")
MODEL = tf.keras.models.load_model("models/stocklens_lstm_gru_best.h5", compile=False)
SCALER = joblib.load("models/scaler.pkl")

# Load feature list to ensure correct order
with open("models/feature_list.json", "r") as f:
    FEATURE_LIST = json.load(f)

print(f"[INIT] Model ready. Expected features: {FEATURE_LIST}")

# -----------------------------
# Helper: add technical indicators (MATCHING TRAINING DATA)
# -----------------------------
def _add_indicators(df):
    """Add technical indicators exactly as in training"""
    # Basic price features already exist: Close, High, Low, Open, Volume
    
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
    
    # Drop NaN rows created by indicators
    df.dropna(inplace=True)
    
    return df


# -----------------------------
# Predict next-day closing price
# -----------------------------
def predict_next(symbol):
    print(f"\n{'='*60}")
    print(f"PREDICTION FOR {symbol}")
    print(f"{'='*60}")
    print(f"[INFO] Fetching latest data for {symbol}...")
    
    # Download 2 years to ensure enough data for SMA200
    df = yf.download(symbol, period="2y", interval="1d", progress=False)
    if df.empty:
        print(f"[ERROR] No data found for {symbol}.")
        return None

    # Reset index to make Date a column
    df.reset_index(inplace=True)
    
    # Flatten multi-level columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns.values]
    
    # Rename columns to match expected format
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
    
    # Store current price before processing
    current_price = df["Close"].iloc[-1]
    
    # Add indicators
    df = _add_indicators(df)
    
    # Ensure all required features exist
    missing_features = [f for f in FEATURE_LIST if f not in df.columns]
    if missing_features:
        print(f"[ERROR] Missing features: {missing_features}")
        print(f"[ERROR] Available columns: {df.columns.tolist()}")
        return None
    
    # Select only the features used in training (in correct order)
    df = df[FEATURE_LIST]
    
    # Check we have enough data for lookback window
    if len(df) < LOOKBACK:
        print(f"[ERROR] Not enough data. Need {LOOKBACK} rows, got {len(df)}")
        return None

    print(f"[INFO] Using {len(df)} rows of historical data")
    
    # Scale features using the same scaler from training
    arr = SCALER.transform(df)
    
    # Take last LOOKBACK timesteps
    X = arr[-LOOKBACK:].reshape(1, LOOKBACK, arr.shape[1])

    # Predict normalized value
    pred_norm = MODEL.predict(X, verbose=0)[0][0]

    # Denormalize (convert to USD)
    close_index = FEATURE_LIST.index("Close")
    close_min = SCALER.data_min_[close_index]
    close_max = SCALER.data_max_[close_index]
    pred_price = pred_norm * (close_max - close_min) + close_min

    # Show comparison
    print(f"\n[RESULT] {symbol} - Next Day Prediction")
    print(f"─"*60)
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


# -----------------------------
# Run direct test
# -----------------------------
if __name__ == "__main__":
    # Test on all training stocks
    stocks = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]
    
    print("\n" + "="*60)
    print("STOCKLENS AI - LIVE PREDICTIONS")
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
        print("─"*60)
        for r in results:
            print(f"{r['symbol']:<10} ${r['current_price']:<11.2f} ${r['predicted_price']:<11.2f} {r['pct_change']:>6.2f}%{' '*7} {r['direction']}")
        print("="*60)