"""
Fetches historical stock data from Yahoo Finance and computes
technical indicators: SMA, EMA, RSI, MACD, and Volume profile.
"""

import yfinance as yf
import pandas as pd
import ta
from datetime import datetime


def fetch_stock_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    df.reset_index(inplace=True)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA, EMA, RSI, MACD, and volume-based indicators."""
    close = df["Close"]
    volume = df["Volume"]

    df["SMA_20"] = ta.trend.sma_indicator(close, window=20)
    df["SMA_50"] = ta.trend.sma_indicator(close, window=50)
    df["EMA_20"] = ta.trend.ema_indicator(close, window=20)
    df["RSI"] = ta.momentum.rsi(close, window=14)

    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    df["Volume_SMA_20"] = ta.trend.sma_indicator(volume, window=20)
    df.dropna(inplace=True)
    return df


def get_stock_info(ticker: str) -> dict:
    """Return basic stock metadata."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "name": info.get("shortName", ticker),
        "sector": info.get("sector", "N/A"),
        "currency": info.get("currency", "USD"),
        "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
        "market_cap": info.get("marketCap", 0),
        "day_high": info.get("dayHigh", 0),
        "day_low": info.get("dayLow", 0),
        "prev_close": info.get("previousClose", 0),
        "open": info.get("open", 0),
        "volume": info.get("volume", 0),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh", 0),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow", 0),
    }


def fetch_live_price(ticker: str) -> dict:
    """Fetch the latest price snapshot for live ticker display."""
    stock = yf.Ticker(ticker)
    info = stock.info

    current = info.get("currentPrice", info.get("regularMarketPrice", 0))
    prev_close = info.get("previousClose", 0)

    change = round(current - prev_close, 2) if current and prev_close else 0
    change_pct = round((change / prev_close) * 100, 2) if prev_close else 0

    return {
        "ticker": ticker,
        "price": current,
        "change": change,
        "change_pct": change_pct,
        "direction": "up" if change >= 0 else "down",
        "day_high": info.get("dayHigh", 0),
        "day_low": info.get("dayLow", 0),
        "volume": info.get("volume", 0),
        "timestamp": datetime.now().isoformat(),
    }


TIMEFRAME_MAP = {
    "1d": ("1d", "5m"),
    "5d": ("5d", "15m"),
    "1wk": ("5d", "15m"),
    "1mo": ("1mo", "1h"),
    "3mo": ("3mo", "1d"),
    "6mo": ("6mo", "1d"),
    "1y": ("1y", "1d"),
    "2y": ("2y", "1wk"),
}


def fetch_chart_data(ticker: str, timeframe: str) -> list[dict]:
    """Return OHLCV records for a given timeframe."""
    period, interval = TIMEFRAME_MAP.get(timeframe, ("1mo", "1h"))
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No chart data for {ticker}")
    df.reset_index(inplace=True)

    records = []
    date_col = "Datetime" if "Datetime" in df.columns else "Date"
    for _, row in df.iterrows():
        records.append({
            "time": str(row[date_col]),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        })
    return records