import os, pandas as pd, pandas_ta as ta, glob, yfinance as yf
import numpy as np

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

def build_features(path, outdir="data/processed"):
    ticker = os.path.basename(path).replace(".csv", "")
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")

    # ---- Technical Indicators ----
    df["RSI"] = ta.rsi(df["Close"], 14)
    df["SMA50"] = ta.sma(df["Close"], 50)
    df["SMA200"] = ta.sma(df["Close"], 200)
    macd = ta.macd(df["Close"])
    df = pd.concat([df, macd], axis=1)

    # ---- Daily Gain/Loss ----
    df["Daily_Return_%"] = df["Close"].pct_change() * 100
    df["Gain_Loss"] = np.where(df["Daily_Return_%"] >= 0, "Gain", "Loss")

    # ---- Upside / Downside ----
    window = 14
    df["Upside_Potential_%"] = df["Daily_Return_%"].rolling(window).apply(lambda x: x[x > 0].mean(), raw=True)
    df["Downside_Risk_%"] = df["Daily_Return_%"].rolling(window).apply(lambda x: x[x < 0].mean(), raw=True)

    # ---- Fundamentals ----
    market_cap, dividend_yield, pe_ratio, roe = get_fundamentals(ticker)
    df["Market_Cap"] = market_cap
    df["Dividend_Yield"] = dividend_yield
    df["PE_Ratio"] = pe_ratio
    df["ROE"] = roe

    df.dropna(inplace=True)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{ticker}_ind.csv")
    df.to_csv(outpath)
    print(f"[OK] Full features built for {ticker}")
    return outpath

if __name__ == "__main__":
    for f in glob.glob("data/raw/*.csv"):
        build_features(f)
