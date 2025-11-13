import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import json
import joblib

print("="*60)
print("STOCKLENS AI - DATASET CREATION (21 Stocks)")
print("="*60)

# Configuration
LOOKBACK = 60  # 60-day sequences
PROCESSED_DIR = Path("data/processed")
INTERIM_DIR = Path("data/interim")
MODELS_DIR = Path("models")

# Create directories
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Features to use for training
FEATURE_COLS = [
    'Close', 'High', 'Low', 'Open',  # Price features
    'RSI', 'SMA_20', 'SMA_50',       # Momentum & Trend
    'MACD', 'MACD_Signal',           # MACD indicators
    'Returns'                         # Returns
]

print(f"\n[1/5] Loading processed stock data...")
print(f"   Features to use: {len(FEATURE_COLS)}")

# Load all processed stocks
all_data = []
stock_names = []

processed_files = sorted(PROCESSED_DIR.glob("*_features.csv"))

if not processed_files:
    raise FileNotFoundError("No processed files found! Run: python -m src.build_features")

print(f"   Found {len(processed_files)} stock files")

for file_path in processed_files:
    ticker = file_path.stem.replace("_features", "")
    stock_names.append(ticker)
    
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Check if all features exist
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        print(f"   ⚠ {ticker}: Missing columns {missing_cols}, skipping...")
        continue
    
    # Select only required features
    df_features = df[FEATURE_COLS].copy()
    
    # Drop any remaining NaN
    df_features = df_features.dropna()
    
    all_data.append(df_features.values)
    print(f"   ✓ {ticker}: {len(df_features)} rows")

print(f"\n   ✓ Loaded {len(all_data)} stocks successfully")

# Combine all stocks
combined_data = np.vstack(all_data)
print(f"   ✓ Combined shape: {combined_data.shape}")

# [2/5] Normalize data
print(f"\n[2/5] Normalizing data...")
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(combined_data)
print(f"   ✓ Data scaled to [0, 1]")

# Save scaler
scaler_path = MODELS_DIR / "scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"   ✓ Scaler saved: {scaler_path}")

# [3/5] Create sequences
print(f"\n[3/5] Creating {LOOKBACK}-day sequences...")

X = []
y = []

# Create sequences for each stock separately to avoid mixing stocks
current_idx = 0
for stock_idx, stock_data in enumerate(all_data):
    stock_scaled = scaler.transform(stock_data)
    
    for i in range(LOOKBACK, len(stock_scaled)):
        X.append(stock_scaled[i-LOOKBACK:i, :])  # 60 days of features
        y.append(stock_scaled[i, 0])  # Next day's Close price (index 0)
    
    print(f"   ✓ {stock_names[stock_idx]}: Created {len(stock_scaled) - LOOKBACK} sequences")

X = np.array(X)
y = np.array(y)

print(f"\n   ✓ Final dataset:")
print(f"      X shape: {X.shape} (samples, timesteps, features)")
print(f"      y shape: {y.shape}")

# [4/5] Check balance
print(f"\n[4/5] Checking data balance...")
price_changes = np.diff(y)
up_moves = np.sum(price_changes > 0)
down_moves = np.sum(price_changes < 0)
neutral = np.sum(price_changes == 0)

total = len(price_changes)
up_pct = (up_moves / total) * 100
down_pct = (down_moves / total) * 100

print(f"   UP movements:    {up_moves:,} ({up_pct:.1f}%)")
print(f"   DOWN movements:  {down_moves:,} ({down_pct:.1f}%)")
print(f"   Neutral:         {neutral:,} ({(neutral/total)*100:.1f}%)")

if abs(up_pct - down_pct) < 10:
    print(f"   ✓ Good balance! (difference: {abs(up_pct - down_pct):.1f}%)")
else:
    print(f"   ⚠ Imbalance detected: {abs(up_pct - down_pct):.1f}% difference")

# [5/5] Save dataset
print(f"\n[5/5] Saving dataset...")

X_path = INTERIM_DIR / "X.npy"
y_path = INTERIM_DIR / "y.npy"

np.save(X_path, X)
np.save(y_path, y)

print(f"   ✓ X saved: {X_path}")
print(f"   ✓ y saved: {y_path}")

# Save feature list
feature_list_path = MODELS_DIR / "feature_list.json"
with open(feature_list_path, "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)
print(f"   ✓ Feature list saved: {feature_list_path}")

# Save metadata
metadata = {
    "lookback": LOOKBACK,
    "num_stocks": len(stock_names),
    "stock_names": stock_names,
    "total_samples": int(len(X)),
    "features": FEATURE_COLS,
    "num_features": len(FEATURE_COLS),
    "up_pct": float(up_pct),
    "down_pct": float(down_pct),
}

metadata_path = INTERIM_DIR / "dataset_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"   ✓ Metadata saved: {metadata_path}")

print("\n" + "="*60)
print("DATASET CREATION COMPLETE!")
print("="*60)
print(f"\n📊 Dataset Statistics:")
print(f"   Stocks:          {len(stock_names)}")
print(f"   Total samples:   {len(X):,}")
print(f"   Lookback:        {LOOKBACK} days")
print(f"   Features:        {len(FEATURE_COLS)}")
print(f"   Balance:         {up_pct:.1f}% UP / {down_pct:.1f}% DOWN")

print("\n" + "="*60)
print("✅ Ready for training!")
print("="*60)
print("\n📊 Next step: python -m src.train_model")