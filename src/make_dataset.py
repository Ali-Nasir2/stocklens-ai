import glob, numpy as np, pandas as pd, joblib, os
from sklearn.preprocessing import MinMaxScaler
from .config import LOOKBACK

def load_merge():
    files = glob.glob("data/processed/*_ind.csv")
    dfs = [pd.read_csv(f, parse_dates=["Date"], index_col="Date") for f in files]
    df = pd.concat(dfs, axis=0).dropna().sort_index()
    return df

def make_dataset(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X, y = [], []
    target_col = df.columns.get_loc("Close")
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i-LOOKBACK:i])
        y.append(scaled[i, target_col])
    X, y = np.array(X), np.array(y)
    os.makedirs("data/interim", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    np.save("data/interim/X.npy", X)
    np.save("data/interim/y.npy", y)
    joblib.dump(scaler, "models/scaler.pkl")
    print(f"[OK] Dataset ready: X={X.shape}, y={y.shape}")

if __name__ == "__main__":
    df = load_merge()
    make_dataset(df)
