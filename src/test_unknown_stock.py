import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import json
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings
import time
warnings.filterwarnings('ignore')

print("="*60)
print("STOCKLENS AI - UNKNOWN STOCK TEST")
print("Testing model on stock it has NEVER seen!")
print("="*60)

# Configuration - CHANGE THIS TO TEST DIFFERENT STOCKS
UNKNOWN_TICKER = "AAPL"  # 🔧 Changed from NVDA (now truly unknown!)

MODEL_PATH = "models/stocklens_hybrid_best.keras"
SCALER_PATH = "models/scaler.pkl"
FEATURE_LIST_PATH = "models/feature_list.json"
FEATURE_SPLIT_PATH = "models/feature_split.json"

# Load trained model and config
print(f"\n🎯 Target Stock: {UNKNOWN_TICKER}")

# Show training stocks
try:
    from config import TRAIN_TICKERS
    print(f"   Training stocks: {', '.join(TRAIN_TICKERS)}")
    if UNKNOWN_TICKER in TRAIN_TICKERS:
        print(f"   ⚠️  WARNING: {UNKNOWN_TICKER} WAS in training set!")
        print(f"   Try one of these instead: NFLX, AMD, CRM, PYPL, INTC")
except:
    pass

print(f"   {UNKNOWN_TICKER} should be completely unknown to the model!")

# [1/6] Load model
print(f"\n[1/6] Loading model...")
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURE_LIST_PATH, "r") as f:
    feature_list = json.load(f)

with open(FEATURE_SPLIT_PATH, "r") as f:
    feature_split = json.load(f)

print(f"   ✓ Model loaded")
print(f"   ✓ Expected features: {feature_list}")

# [2/6] Download stock data
print(f"\n[2/6] Downloading {UNKNOWN_TICKER} data...")

max_retries = 3
df = None

for attempt in range(max_retries):
    try:
        print(f"   Attempt {attempt + 1}/{max_retries}...")
        df = yf.download(UNKNOWN_TICKER, start="2018-01-01", progress=False)
        
        if df.empty:
            raise ValueError("No data returned")
        
        # 🔧 FIX: Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        print(f"   ✓ Downloaded {len(df)} days")
        print(f"   ✓ Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"   ✓ Columns: {df.columns.tolist()}")
        break
        
    except Exception as e:
        print(f"   ❌ Attempt {attempt + 1} failed: {str(e)}")
        if attempt < max_retries - 1:
            time.sleep(2)
        else:
            raise Exception(f"Failed to download {UNKNOWN_TICKER} after {max_retries} attempts")

# [3/6] Calculate technical indicators (match build_features.py)
print(f"\n[3/6] Calculating technical indicators...")

from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# 🔧 FIX: Ensure Series, not DataFrame columns
close = df['Close'].squeeze() if hasattr(df['Close'], 'squeeze') else df['Close']
high = df['High'].squeeze() if hasattr(df['High'], 'squeeze') else df['High']
low = df['Low'].squeeze() if hasattr(df['Low'], 'squeeze') else df['Low']
open_ = df['Open'].squeeze() if hasattr(df['Open'], 'squeeze') else df['Open']
volume = df['Volume'].squeeze() if hasattr(df['Volume'], 'squeeze') else df['Volume']

# Price-based features
df['Returns'] = close.pct_change()
df['Log_Returns'] = np.log(close / close.shift(1))

# Moving Averages
df['SMA_20'] = SMAIndicator(close=close, window=20).sma_indicator()
df['SMA_50'] = SMAIndicator(close=close, window=50).sma_indicator()
df['EMA_12'] = EMAIndicator(close=close, window=12).ema_indicator()
df['EMA_26'] = EMAIndicator(close=close, window=26).ema_indicator()

# MACD
macd = MACD(close=close)
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()
df['MACD_Diff'] = macd.macd_diff()

# RSI
df['RSI'] = RSIIndicator(close=close, window=14).rsi()

# Bollinger Bands
bollinger = BollingerBands(close=close, window=20, window_dev=2)
df['BB_High'] = bollinger.bollinger_hband()
df['BB_Low'] = bollinger.bollinger_lband()
df['BB_Mid'] = bollinger.bollinger_mavg()
df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid']

# Volatility
df['Volatility'] = df['Returns'].rolling(window=20).std()

# Volume indicators
df['Volume_SMA'] = volume.rolling(window=20).mean()
df['Volume_Ratio'] = volume / df['Volume_SMA']

# Price position indicators
df['High_Low_Ratio'] = high / low
df['Close_Open_Ratio'] = close / open_

print(f"   ✓ Indicators calculated")
print(f"   ✓ Available features: {df.columns.tolist()}")

# [4/6] Select and scale features
print(f"\n[4/6] Preparing data...")

# Check which features exist
missing_features = [f for f in feature_list if f not in df.columns]
if missing_features:
    print(f"   ⚠️  Missing features: {missing_features}")
    print(f"   Available: {df.columns.tolist()}")
    raise ValueError(f"Missing required features: {missing_features}")

# Select only the features used in training
df_features = df[feature_list].copy()
df_features = df_features.dropna()

print(f"   ✓ Selected {len(feature_list)} features")
print(f"   ✓ Data shape after dropna: {df_features.shape}")

# Scale the data
scaled_data = scaler.transform(df_features.values)
print(f"   ✓ Data scaled")

# [5/6] Create sequences
print(f"\n[5/6] Creating sequences...")

LOOKBACK = 60
X_price_list = []
X_indicator_list = []
y_list = []
dates = []

price_indices = feature_split['price_indices']
indicator_indices = feature_split['indicator_indices']

for i in range(LOOKBACK, len(scaled_data)):
    sequence = scaled_data[i-LOOKBACK:i, :]
    
    X_price_list.append(sequence[:, price_indices])
    X_indicator_list.append(sequence[:, indicator_indices])
    y_list.append(scaled_data[i, 0])  # Close price
    dates.append(df_features.index[i])

X_price = np.array(X_price_list)
X_indicator = np.array(X_indicator_list)
y = np.array(y_list)

print(f"   ✓ Created {len(X_price)} sequences")
print(f"   ✓ X_price shape: {X_price.shape}")
print(f"   ✓ X_indicator shape: {X_indicator.shape}")

# [6/6] Make predictions
print(f"\n[6/6] Making predictions...")

y_pred = model.predict([X_price, X_indicator], verbose=0)
y_pred = y_pred.flatten()

print(f"   ✓ Predictions complete")

# Denormalize predictions
close_idx = feature_list.index('Close')
close_min = scaler.data_min_[close_idx]
close_max = scaler.data_max_[close_idx]

def denorm(arr):
    return arr * (close_max - close_min) + close_min

y_real = denorm(y)
y_pred_real = denorm(y_pred)

# Calculate metrics
mse = mean_squared_error(y_real, y_pred_real)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_real, y_pred_real)
r2 = r2_score(y_real, y_pred_real)

# Directional accuracy
actual_direction = np.diff(y_real) > 0
pred_direction = np.diff(y_pred_real) > 0
directional_accuracy = np.mean(actual_direction == pred_direction) * 100

# UP/DOWN accuracy
up_mask = actual_direction == True
down_mask = actual_direction == False

up_correct = np.sum((actual_direction == pred_direction) & up_mask)
down_correct = np.sum((actual_direction == pred_direction) & down_mask)

up_total = np.sum(up_mask)
down_total = np.sum(down_mask)

up_accuracy = (up_correct / up_total * 100) if up_total > 0 else 0
down_accuracy = (down_correct / down_total * 100) if down_total > 0 else 0

# Print results
print("\n" + "="*60)
print(f"TEST RESULTS - {UNKNOWN_TICKER} (UNKNOWN STOCK)")
print("="*60)

print(f"\n📊 Price Prediction Metrics:")
print(f"   RMSE:  ${rmse:.2f}")
print(f"   MAE:   ${mae:.2f}")
print(f"   R²:    {r2:.4f}")

print(f"\n🎯 Directional Accuracy (MOST IMPORTANT!):")
print(f"   Overall Accuracy: {directional_accuracy:.2f}%")
print(f"   UP days:          {up_accuracy:.2f}% ({up_correct}/{up_total})")
print(f"   DOWN days:        {down_accuracy:.2f}% ({down_correct}/{down_total})")

print(f"\n📈 Price Statistics:")
print(f"   Actual prices:     ${y_real.min():.2f} - ${y_real.max():.2f}")
print(f"   Predicted prices:  ${y_pred_real.min():.2f} - ${y_pred_real.max():.2f}")
print(f"   Average actual:    ${y_real.mean():.2f}")
print(f"   Average predicted: ${y_pred_real.mean():.2f}")

# Show last 10 predictions
print(f"\n📅 Last 10 Predictions:")
print(f"{'Date':<12} {'Actual':<12} {'Predicted':<12} {'Error':<12} {'Direction':<15}")
print("-" * 70)

for i in range(-10, 0):
    date = dates[i].strftime('%Y-%m-%d')
    actual = y_real[i]
    pred = y_pred_real[i]
    error = pred - actual
    
    if i < len(y_real) - 1:
        actual_dir = "↑ UP" if y_real[i+1] > actual else "↓ DOWN"
        pred_dir = "↑" if y_pred_real[i+1] > pred else "↓"
        correct = "✓" if (y_real[i+1] > actual) == (y_pred_real[i+1] > pred) else "✗"
        direction_str = f"{actual_dir} {pred_dir} {correct}"
    else:
        direction_str = "N/A"
    
    print(f"{date:<12} ${actual:<11.2f} ${pred:<11.2f} ${error:<11.2f} {direction_str:<15}")

print("\n" + "="*60)
print("INTERPRETATION:")
print("="*60)

if directional_accuracy >= 70:
    print("✅ EXCELLENT! Model predicts direction very well!")
elif directional_accuracy >= 60:
    print("✓ GOOD! Model has decent directional accuracy")
elif directional_accuracy >= 55:
    print("⚠ FAIR - Better than random, but room for improvement")
else:
    print("❌ POOR - Model struggles with this stock")

if abs(up_accuracy - down_accuracy) < 15:
    print("✅ BALANCED predictions (no UP/DOWN bias)")
else:
    print(f"⚠ BIAS detected: Favors {'UP' if up_accuracy > down_accuracy else 'DOWN'} predictions")

print("="*60)