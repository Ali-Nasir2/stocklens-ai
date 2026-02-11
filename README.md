# 📈 StockLens AI

> **AI-powered stock price prediction using hybrid deep learning architecture**

StockLens AI is a sophisticated machine learning system that predicts stock prices using a dual-branch neural network trained on 21 diverse stocks across multiple sectors. The model combines LSTM and GRU architectures to analyze both price patterns and technical indicators, achieving **77% directional accuracy** in predicting market movements.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ Key Features

- **🎯 Hybrid Neural Network**: Dual-branch architecture separating price (LSTM) and indicators (GRU) for superior pattern recognition
- **📊 77% Directional Accuracy**: Correctly predicts market direction (UP/DOWN) 77% of the time
- **🌐 Multi-Sector Training**: Trained on 21 diverse stocks across tech, finance, healthcare, energy, and consumer sectors
- **🔮 Unknown Stock Prediction**: Generalizes well to stocks never seen during training
- **📈 Advanced Technical Indicators**: Incorporates RSI, MACD, Bollinger Bands, moving averages, and volatility metrics
- **⚡ Real-time Predictions**: Fetch live data from Yahoo Finance and get instant predictions
- **📉 Comprehensive Evaluation**: Detailed metrics including RMSE, MAE, R², and directional accuracy

---

## 🏗️ Architecture

The model uses a **hybrid dual-branch architecture** to process different types of features:

```
┌─────────────────────┐         ┌──────────────────────┐
│   Price Stream      │         │  Indicator Stream    │
│  (OHLC Data)        │         │  (RSI, MACD, etc.)   │
└──────────┬──────────┘         └──────────┬───────────┘
           │                               │
      ┌────▼────┐                    ┌─────▼─────┐
      │  LSTM   │                    │    GRU    │
      │ Layers  │                    │  Layers   │
      └────┬────┘                    └─────┬─────┘
           │                               │
      ┌────▼────┐                    ┌─────▼─────┐
      │Attention│                    │Bidirectional│
      └────┬────┘                    └─────┬─────┘
           │                               │
           └───────────┬───────────────────┘
                       │
                  ┌────▼────┐
                  │  Merge  │
                  └────┬────┘
                       │
                  ┌────▼────┐
                  │  Dense  │
                  │ Layers  │
                  └────┬────┘
                       │
                  ┌────▼────┐
                  │ Output  │
                  │(Price)  │
                  └─────────┘
```

**Key Components:**
- **Bidirectional LSTM** for price sequence learning
- **Bidirectional GRU** for indicator pattern recognition  
- **Attention mechanism** for important temporal feature extraction
- **Layer Normalization** for stable training
- **L2 Regularization** to prevent overfitting
- **Dropout layers** for better generalization

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ali-Nasir2/stocklens-ai.git
   cd stocklens-ai
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional)
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

---

## 🚀 Quick Start

### Train the Model

```bash
# Download data for 21 stocks
python -m src.fetch_data

# Build features and prepare datasets
python -m src.build_features

# Train the hybrid model
python -m src.train_model
```

### Make Predictions

#### Test on Unknown Stock
```bash
python -m src.test_unknown_stock
```

#### Quick Prediction Script
```bash
python scripts/quick_predict.py
```

### Evaluate the Model

```bash
# Generate evaluation metrics and plots
python -m src.eval_plots
```

---

## 📊 Performance Metrics

### Test Set Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Directional Accuracy** | 77.10% | ✅ Excellent trend prediction |
| **RMSE** | $93.44 | Average prediction error |
| **MAE** | $78.33 | Mean absolute error |
| **R² Score** | 0.2351 | Moderate correlation |
| **MAPE** | 29.38% | Mean absolute percentage error |

### Training Data
- **21 diverse stocks** across 5+ sectors
- **~39,000 training samples** (2018-present)
- **60-day lookback window**
- **9 technical features** per timestamp

### Trained Stocks
- **Tech Giants**: AAPL, MSFT, GOOGL, NVDA, META
- **E-commerce**: AMZN, WMT, COST
- **Finance**: JPM, BAC, V
- **Healthcare**: JNJ, UNH, PFE
- **Energy**: XOM, CVX, BA
- **Consumer**: KO, NKE, DIS

---

## 🗂️ Project Structure

```
stocklens-ai/
├── src/                          # Source code
│   ├── config.py                 # Configuration (tickers, dates, etc.)
│   ├── fetch_data.py             # Download stock data from Yahoo Finance
│   ├── build_features.py         # Calculate technical indicators
│   ├── make_dataset.py           # Create train/test sequences
│   ├── train_model.py            # Train hybrid neural network
│   ├── test_unknown_stock.py     # Test on unseen stocks
│   ├── eval_plots.py             # Generate evaluation visualizations
│   ├── load_and_predict.py       # Load model and predict
│   └── utils_time.py             # Time series utilities
│
├── scripts/                      # Utility scripts
│   ├── quick_predict.py          # Quick prediction script
│   └── run_all.sh                # Run entire pipeline
│
├── models/                       # Saved models and artifacts
│   ├── stocklens_hybrid_best.keras
│   ├── stocklens_hybrid_final.keras
│   ├── scaler.pkl                # Feature scaler
│   ├── feature_list.json         # Feature names
│   ├── feature_split.json        # Price vs indicator split
│   ├── registry.json             # Model metadata
│   └── training_history.json     # Training metrics
│
├── data/                         # Data directory (gitignored)
│   ├── raw/                      # Raw CSV files
│   ├── interim/                  # Processed numpy arrays
│   └── processed/                # Final datasets
│
├── reports/                      # Evaluation reports
│   ├── evaluation_metrics.json
│   ├── evaluation_report.txt
│   └── figures/                  # Visualizations
│
├── notebooks/                    # Jupyter notebooks
│   └── EDA_and_validation.ipynb
│
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## �� Usage Examples

### Example 1: Predict a Specific Stock

```python
import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import json

# Load model and scaler
model = load_model('models/stocklens_hybrid_best.keras')
scaler = joblib.load('models/scaler.pkl')

# Download stock data
df = yf.download('AAPL', start='2023-01-01')

# ... (feature engineering and preprocessing)
# ... (create sequences)

# Make prediction
prediction = model.predict([X_price, X_indicators])
print(f"Predicted next price: ${prediction[0][0]:.2f}")
```

### Example 2: Batch Predictions

```python
# Test multiple stocks
test_stocks = ['TSLA', 'NFLX', 'AMD', 'CRM']

for ticker in test_stocks:
    # Load data and predict
    result = predict_stock(ticker)
    print(f"{ticker}: {result['direction']} ({result['accuracy']:.1f}%)")
```

---

## 📈 Technical Indicators

The model uses the following technical indicators:

### Trend Indicators
- **SMA (Simple Moving Average)**: 20-day, 50-day
- **EMA (Exponential Moving Average)**: 12-day, 26-day
- **MACD**: Moving Average Convergence Divergence
- **MACD Signal**: Signal line
- **MACD Diff**: Histogram

### Momentum Indicators
- **RSI (Relative Strength Index)**: 14-day period
- **Returns**: Daily percentage change
- **Log Returns**: Logarithmic returns

### Volatility Indicators
- **Bollinger Bands**: Upper, middle, lower bands
- **BB Width**: Band width indicator
- **Volatility**: 20-day rolling standard deviation

### Volume Indicators
- **Volume SMA**: 20-day average volume
- **Volume Ratio**: Current volume vs average

### Price Indicators
- **High/Low Ratio**: Daily range indicator
- **Close/Open Ratio**: Intraday movement

---

## 🎓 Model Training Details

### Hyperparameters
- **Learning Rate**: 0.0001
- **Batch Size**: 64
- **Epochs**: 75 (with early stopping)
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Dropout Rate**: 0.15-0.25
- **L2 Regularization**: 0.003

### Training Strategy
- **Validation Split**: 25% of training data
- **Early Stopping**: Patience of 20 epochs
- **Learning Rate Reduction**: Factor 0.5, patience 7 epochs
- **Best Model Checkpoint**: Saved based on validation loss

---

## 🔬 Evaluation & Testing

### Running Evaluations

```bash
# Generate comprehensive evaluation report
python -m src.eval_plots
```

This creates:
- Evaluation metrics (JSON & TXT)
- Training history plots
- Prediction vs actual comparison charts
- Residual analysis

### Understanding Results

**Directional Accuracy** is the most important metric for trading:
- **> 70%**: Excellent - profitable trading possible
- **60-70%**: Good - positive expected value
- **55-60%**: Fair - slight edge over random
- **< 55%**: Poor - not better than guessing

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Ideas for Contributions
- Add more technical indicators
- Implement ensemble methods
- Add real-time streaming predictions
- Create web interface
- Improve documentation
- Add more test stocks
- Optimize hyperparameters

---

## ⚠️ Disclaimer

**This project is for educational purposes only.**

- Stock market predictions are inherently uncertain
- Past performance does not guarantee future results
- Do not use this model for actual trading without proper risk management
- Always consult with financial advisors before making investment decisions
- The authors are not responsible for any financial losses

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free stock data via `yfinance`
- **TensorFlow/Keras** for the deep learning framework
- **ta library** for technical analysis indicators
- The open-source community for inspiration and tools

---

## 📧 Contact

**Ali Nasir**
- GitHub: [@Ali-Nasir2](https://github.com/Ali-Nasir2)
- Project: [stocklens-ai](https://github.com/Ali-Nasir2/stocklens-ai)

---

## 🗺️ Roadmap

- [ ] Add sentiment analysis from news/social media
- [ ] Implement reinforcement learning for trading strategies
- [ ] Create REST API for predictions
- [ ] Add support for cryptocurrency predictions
- [ ] Develop web dashboard with real-time updates
- [ ] Add portfolio optimization features
- [ ] Implement backtesting framework

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by [Ali Nasir](https://github.com/Ali-Nasir2)

</div>
