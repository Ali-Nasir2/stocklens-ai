import os
from dotenv import load_dotenv
load_dotenv()

# 20 Diverse stocks across sectors for better generalization
TRAIN_TICKERS = [
    # Tech Giants (5)
    "AAPL",   # Apple - Consumer Electronics
    "MSFT",   # Microsoft - Enterprise Software
    "GOOGL",  # Google - Digital Advertising
    "NVDA",   # NVIDIA - Semiconductors
    "META",   # Meta - Social Media
    
    # E-commerce & Retail (3)
    "AMZN",   # Amazon - E-commerce
    "WMT",    # Walmart - Traditional Retail
    "COST",   # Costco - Warehouse Retail
    
    # Finance (3)
    "JPM",    # JPMorgan - Banking
    "BAC",    # Bank of America - Banking
    "V",      # Visa - Payment Processing
    
    # Healthcare (3)
    "JNJ",    # Johnson & Johnson - Pharmaceuticals
    "UNH",    # UnitedHealth - Health Insurance
    "PFE",    # Pfizer - Biotech
    
    # Energy & Industrials (3)
    "XOM",    # ExxonMobil - Oil & Gas
    "CVX",    # Chevron - Energy
    "BA",     # Boeing - Aerospace
    
    # Consumer & Entertainment (3)
    "KO",     # Coca-Cola - Beverages
    "NKE",    # Nike - Apparel
    "DIS",    # Disney - Entertainment
]

START_DATE = "2018-01-01"
END_DATE = None  # Will use current date
INTERVAL = "1d"
LOOKBACK = 60

# Optional: Test stocks (not in training)
TEST_TICKERS = ["TSLA", "NFLX", "AMD", "CRM", "PYPL"]