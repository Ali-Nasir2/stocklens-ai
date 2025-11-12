import os, yfinance as yf
from .config import TRAIN_TICKERS, START_DATE, END_DATE, INTERVAL

def fetch_one(ticker, outdir="data/raw"):
    os.makedirs(outdir, exist_ok=True)
    df = yf.download(ticker, start=START_DATE, end=END_DATE, interval=INTERVAL)
    df.to_csv(os.path.join(outdir, f"{ticker}.csv"))
    print(f"[OK] {ticker} saved.")
    return df

if __name__ == "__main__":
    for t in TRAIN_TICKERS:
        fetch_one(t)
