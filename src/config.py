import os
from dotenv import load_dotenv
load_dotenv()

TRAIN_TICKERS = ["AAPL","MSFT","TSLA","GOOGL","AMZN"]
START_DATE = "2018-01-01"
END_DATE = None
INTERVAL = "1d"
LOOKBACK = 60
