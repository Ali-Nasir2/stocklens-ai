import os
import yfinance as yf
from datetime import datetime
import time

try:
    from .config import TRAIN_TICKERS, START_DATE, END_DATE, INTERVAL
except ImportError:
    from config import TRAIN_TICKERS, START_DATE, END_DATE, INTERVAL

def fetch_one(ticker, outdir="data/raw", max_retries=3):
    """
    Download stock data with retry logic
    """
    os.makedirs(outdir, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            print(f"[{ticker}] Downloading... (attempt {attempt + 1}/{max_retries})")
            
            df = yf.download(
                ticker, 
                start=START_DATE, 
                end=END_DATE or datetime.now().strftime("%Y-%m-%d"),
                interval=INTERVAL,
                progress=False,
                timeout=30
            )
            
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            # Save to CSV
            output_path = os.path.join(outdir, f"{ticker}.csv")
            df.to_csv(output_path)
            
            print(f"[OK] {ticker} saved: {len(df)} days → {output_path}")
            return df
            
        except Exception as e:
            print(f"[ERROR] {ticker} attempt {attempt + 1} failed: {str(e)[:100]}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"[RETRY] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"[FAILED] {ticker} - all {max_retries} attempts failed")
                return None

if __name__ == "__main__":
    print("="*60)
    print("STOCKLENS AI - DATA FETCHER (20 Diverse Stocks)")
    print("="*60)
    print(f"\nFetching {len(TRAIN_TICKERS)} stocks from {START_DATE}")
    print(f"Sectors: Tech, Retail, Finance, Healthcare, Energy, Consumer\n")
    
    successful = []
    failed = []
    
    for i, ticker in enumerate(TRAIN_TICKERS, 1):
        print(f"\n[{i}/{len(TRAIN_TICKERS)}] Processing {ticker}...")
        print("-" * 60)
        
        result = fetch_one(ticker)
        
        if result is not None:
            successful.append(ticker)
        else:
            failed.append(ticker)
        
        # Small delay between requests to avoid rate limiting
        if i < len(TRAIN_TICKERS):
            time.sleep(0.5)
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"✓ Successful: {len(successful)}/{len(TRAIN_TICKERS)}")
    print(f"  {', '.join(successful)}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        print(f"  {', '.join(failed)}")
        print(f"\n  Tip: Run script again to retry failed tickers")
    
    print(f"\n✅ Data saved to: data/raw/")
    print("="*60)
    print("\n📊 Next step: python -m src.build_features")