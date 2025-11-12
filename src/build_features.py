import os, pandas as pd, pandas_ta as ta, glob, yfinance as yf
import numpy as np

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
    ticker = os.path.basename(path).replace(".csv", "")

    # ---- Read and normalize CSV with messy top metadata ----
    raw = pd.read_csv(path, header=None, dtype=str)

    # First row contains the column names (but missing "Date")
    header_row = raw.iloc[0].tolist()  # e.g. ['Price','Close','High','Low','Open','Volume']
    
    # Drop metadata rows (ticker row and "Date" row)
    rows_to_drop = []
    if len(raw) > 1 and str(raw.iloc[1, 0]).strip().lower().startswith("ticker"):
        rows_to_drop.append(1)
    if len(raw) > 2 and str(raw.iloc[2, 0]).strip().lower() == "date":
        rows_to_drop.append(2)
    
    if rows_to_drop:
        raw = raw.drop(index=rows_to_drop).reset_index(drop=True)

    # Build final columns: prepend "Date" to the header row.
    cols = ["Date"] + [c for c in header_row if str(c) and str(c).strip() != ""]
    
    # Slice data rows (everything after the original header row)
    data_start_index = 1  # everything after the first row (which was header_row)
    df = raw.iloc[data_start_index:].copy()
    df.columns = cols[: df.shape[1]]  # align column count with actual data columns

    # Convert types
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    # Convert other columns to numeric (coerce errors to NaN)
    for c in df.columns:
        if c != "Date":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows where Date couldn't be parsed
    df = df[df["Date"].notna()]
    
    df.set_index("Date", inplace=True)
    
    # Debug: print actual columns
    print(f"[DEBUG] {ticker} columns: {df.columns.tolist()}")
    
    # Verify Close column exists
    if "Close" not in df.columns:
        raise ValueError(f"'Close' column not found after parsing {path}. Columns: {df.columns.tolist()}")

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
    df["Upside_Potential_%"] = df["Daily_Return_%"].rolling(window).apply(
        lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else np.nan, raw=True
    )
    df["Downside_Risk_%"] = df["Daily_Return_%"].rolling(window).apply(
        lambda x: x[x < 0].mean() if len(x[x < 0]) > 0 else np.nan, raw=True
    )

    # ---- Fundamentals ----
    market_cap, dividend_yield, pe_ratio, roe = get_fundamentals(ticker)
    df["Market_Cap"] = market_cap
    df["Dividend_Yield"] = dividend_yield
    df["PE_Ratio"] = pe_ratio
    df["ROE"] = roe

    # ---- Final cleanup ----
    # Only drop rows where essential price columns are NaN
    # Check which columns actually exist before trying to drop NaNs
    essential_cols = [c for c in ["Price", "Close", "High", "Low", "Open", "Volume"] if c in df.columns]
    if essential_cols:
        df.dropna(subset=essential_cols, inplace=True)
    
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{ticker}_ind.csv")
    df.to_csv(outpath)

    print(f"[OK] Full features built for {ticker} → saved to {outpath} ({len(df)} rows)")
    return outpath


# -------------------------------------------
# Run the feature builder for all raw files
# -------------------------------------------
if __name__ == "__main__":
    for f in glob.glob("data/raw/*.csv"):
        build_features(f)