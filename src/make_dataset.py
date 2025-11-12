import os, glob, json
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from sklearn.preprocessing import MinMaxScaler

# Configuration
PREFERRED_FEATURES = ["Close", "High", "Low", "Open", "Volume", "RSI", "SMA50", "SMA200", "MACD_12_26_9", "Daily_Return_%"]
ESSENTIAL_TARGET = "Close"
LOOKBACK = 60

# -------------------------------------------
# Fetch company fundamentals (non-time-series)
# -------------------------------------------
def get_fundamentals(ticker):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        market_cap = info.get("marketCap", np.nan)
        dividend_yield = info.get("dividendYield", np.nan)
        pe_ratio = info.get("trailingPE", np.nan)
        roe = np.nan
        try:
            fin = tk.financials
            net_income = fin.loc["Net Income"].iloc[0] if "Net Income" in fin.index else np.nan
            equity = fin.loc["Total Stockholder Equity"].iloc[0] if "Total Stockholder Equity" in fin.index else np.nan
            if not np.isnan(net_income) and not np.isnan(equity) and equity != 0:
                roe = (net_income / equity) * 100
        except Exception:
            pass
        return market_cap, dividend_yield, pe_ratio, roe
    except Exception as e:
        print(f"[WARN] Fundamentals unavailable for {ticker}: {e}")
        return np.nan, np.nan, np.nan, np.nan


# -------------------------------------------
# Build technical + fundamental feature set
# -------------------------------------------
def build_features(path, outdir="data/processed"):
    ticker = os.path.basename(path).replace("_ind.csv", "").replace(".csv", "")

    # ---- Simply read the CSV normally (it already has proper headers) ----
    df = pd.read_csv(path)
    
    # Convert Date to datetime and set as index
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df[df["Date"].notna()]  # Drop rows with invalid dates
    df.set_index("Date", inplace=True)
    
    # Keep only columns that exist in PREFERRED_FEATURES
    cols = [c for c in PREFERRED_FEATURES if c in df.columns]
    
    # Make sure target exists
    if ESSENTIAL_TARGET not in cols:
        raise ValueError(f"[ERR] '{ESSENTIAL_TARGET}' not found in {path}. Available: {df.columns.tolist()}")
    
    df = df[cols]
    
    # Drop rows where essential price columns are NaN
    essential_price_cols = [c for c in ["Close", "High", "Low", "Open"] if c in df.columns]
    if essential_price_cols:
        df = df.dropna(subset=essential_price_cols, how='any')
    
    # Forward-fill and backward-fill technical indicators with NaN warmup periods
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Final check: drop any remaining rows with NaN
    df = df.dropna()
    
    # Sort by date
    df = df.sort_index()
    
    # Add a Ticker column for debugging
    df["Ticker"] = ticker
    
    return df


def load_all(folder="data/processed"):
    files = glob.glob(os.path.join(folder, "*_ind.csv"))
    if not files:
        raise FileNotFoundError(f"[ERR] No processed files found in {folder}. Run build_features first.")
    dfs = []
    for f in files:
        try:
            df = build_features(f)
            dfs.append(df)
            print(f"[load] {os.path.basename(f)} rows={len(df)}")
        except Exception as e:
            print(f"[skip] {os.path.basename(f)}: {e}")
    if not dfs:
        raise RuntimeError("[ERR] No valid processed files after filtering.")
    # Vertically stack all tickers (same schema)
    merged = pd.concat(dfs, axis=0, copy=False).sort_index()
    return merged


def make_windows(arr, target_idx, lookback=60):
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i-lookback:i, :])      # window of features
        y.append(arr[i, target_idx])        # next-step target (Close)
    return np.array(X), np.array(y)


def main():
    os.makedirs("data/interim", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = load_all("data/processed")
    print(f"[dataset] Total rows loaded: {len(df)}")

    # Final feature list (without the non-numeric 'Ticker')
    feat_cols = [c for c in df.columns if c != "Ticker"]
    # Ensure target is present
    if ESSENTIAL_TARGET not in feat_cols:
        raise ValueError(f"[ERR] Target '{ESSENTIAL_TARGET}' missing from final features {feat_cols}")

    # Scale all features column-wise
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feat_cols].values)
    # Index of target within the scaled array
    target_idx = feat_cols.index(ESSENTIAL_TARGET)

    # Build windows
    X, y = make_windows(scaled, target_idx, lookback=LOOKBACK)

    # Save artifacts
    np.save("data/interim/X.npy", X)
    np.save("data/interim/y.npy", y)
    joblib.dump(scaler, "models/scaler.pkl")
    with open("models/feature_list.json", "w") as f:
        json.dump(feat_cols, f)

    print(f"[dataset] features={len(feat_cols)} target='{ESSENTIAL_TARGET}' lookback={LOOKBACK}")
    print(f"[dataset] X.shape={X.shape}  y.shape={y.shape}")
    print("[OK] Saved: data/interim/X.npy, data/interim/y.npy, models/scaler.pkl, models/feature_list.json")


if __name__ == "__main__":
    main()