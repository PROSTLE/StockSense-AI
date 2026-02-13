# 📈 StockSense AI — AI-Powered Stock Market Prediction

> **Hackathon Project** | Built in 48 hours

##🎤 Elevator Pitch (30 seconds)

*"StockSense AI uses deep learning and NLP to predict stock prices 5 days ahead.
Enter any ticker — our LSTM model analyzes 2 years of market data and technical
indicators, while FinBERT reads the latest financial news to gauge market
sentiment. You get an interactive chart with predictions, a confidence score,
a risk level, and a clear sentiment breakdown — all in seconds."*

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js (optional – only if you want a dev server for frontend)
- ~2 GB disk for FinBERT model weights (auto-downloaded)

### 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

The API will be live at **http://localhost:8000**.

### 2. Frontend Setup

Simply open `frontend/index.html` in your browser, **or** serve it:

```bash
cd frontend
python -m http.server 3000
```

Then visit **http://localhost:3000**.

---

## 🔗 API Endpoints

| Method | Endpoint                   | Description                        |
|--------|----------------------------|------------------------------------|
| GET    | `/api/stock/{ticker}`      | Historical data + indicators       |
| GET    | `/api/predict/{ticker}`    | LSTM 5-day price prediction        |
| GET    | `/api/sentiment/{ticker}`  | FinBERT news sentiment analysis    |
| GET    | `/api/summary/{ticker}`    | Combined analysis (all of above)   |

---

## 🏗️ Architecture

```
User → React/HTML Frontend → FastAPI Backend
                                ├── yfinance (stock data)
                                ├── ta (technical indicators)
                                ├── PyTorch LSTM (price prediction)
                                └── FinBERT (sentiment analysis)
```

---

## 🧠 ML Pipeline

1. **Data**: 2 years of daily OHLCV via Yahoo Finance
2. **Features**: Close price (normalized via MinMaxScaler)
3. **Model**: 2-layer LSTM (64 hidden units) → FC → 5-day output
4. **Training**: 30 epochs, Adam optimizer, MSE loss, 85/15 train/test split
5. **Inference**: Last 60-day window → next 5 closing prices




