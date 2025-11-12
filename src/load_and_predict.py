import yfinance as yf, pandas as pd, pandas_ta as ta, joblib, numpy as np, tensorflow as tf
from .config import LOOKBACK

MODEL = tf.keras.models.load_model("models/stocklens_lstm_gru.h5")
SCALER = joblib.load("models/scaler.pkl")

def _add_indicators(df):
    df["RSI"] = ta.rsi(df["Close"], 14)
    df["EMA20"] = ta.ema(df["Close"], 20)
    df["EMA50"] = ta.ema(df["Close"], 50)
    macd = ta.macd(df["Close"])
    df = pd.concat([df, macd], axis=1)
    df.dropna(inplace=True)
    return df

def predict_next(symbol):
    df = yf.download(symbol, period="1y", interval="1d")
    df = _add_indicators(df)
    cols = SCALER.feature_names_in_
    df = df.reindex(columns=cols).dropna()
    arr = SCALER.transform(df)
    X = arr[-LOOKBACK:].reshape(1, LOOKBACK, arr.shape[1])
    pred = MODEL.predict(X)[0][0]
    print(f"[Predict] {symbol} → {pred:.2f}")
    return pred

if __name__ == "__main__":
    predict_next("TSLA")
