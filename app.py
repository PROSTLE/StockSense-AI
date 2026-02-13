"""
StockSense AI — FastAPI Backend (v3.0)
Includes: Stock Data, Prediction, Sentiment, Charts, High Potential, AUTO TRADING
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from stock_data import fetch_stock_data, add_technical_indicators, get_stock_info, fetch_live_price, fetch_chart_data
from sentiment import analyze_sentiment
from model import train_and_predict
from trader import (
    get_portfolio, reset_portfolio, toggle_bot,
    evaluate_trade_signal, execute_sell, check_position,
)
import concurrent.futures

app = FastAPI(title="StockSense AI", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "StockSense AI API v3.0 is running 🚀"}


# ══ STOCK DATA ═══════════════════════════════════════
@app.get("/api/stock/{ticker}")
def get_stock(ticker: str):
    try:
        df = fetch_stock_data(ticker)
        df = add_technical_indicators(df)
        records = df.tail(200).to_dict(orient="records")
        for r in records:
            if "Date" in r:
                r["Date"] = str(r["Date"])
        info = get_stock_info(ticker)
        return {"info": info, "data": records}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/live/{ticker}")
def live_price(ticker: str):
    try:
        return fetch_live_price(ticker)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/chart/{ticker}")
def chart_data(ticker: str, timeframe: str = Query("1mo", regex="^(1d|5d|1wk|1mo|3mo|6mo|1y|2y)$")):
    try:
        data = fetch_chart_data(ticker, timeframe)
        return {"ticker": ticker, "timeframe": timeframe, "data": data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ══ ML PREDICTION ════════════════════════════════════
# In app.py / predict endpoint
@app.get("/api/predict/{ticker}")
def predict(ticker: str):
    try:
        df = fetch_stock_data(ticker)
        df = add_technical_indicators(df)
        result = train_and_predict(df)
        return {"ticker": ticker, **result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ══ SENTIMENT ════════════════════════════════════════
@app.get("/api/sentiment/{ticker}")
def sentiment(ticker: str):
    try:
        return analyze_sentiment(ticker)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ══ COMBINED SUMMARY ═════════════════════════════════
@app.get("/api/summary/{ticker}")
def summary(ticker: str):
    try:
        info = get_stock_info(ticker)
        df = fetch_stock_data(ticker)
        df_ind = add_technical_indicators(df)
        prediction = train_and_predict(df_ind)
        sent = analyze_sentiment(ticker)

        latest = df_ind.iloc[-1]
        indicators = {
            "RSI": round(float(latest["RSI"]), 2),
            "MACD": round(float(latest["MACD"]), 4),
            "MACD_Signal": round(float(latest["MACD_Signal"]), 4),
            "SMA_20": round(float(latest["SMA_20"]), 2),
            "SMA_50": round(float(latest["SMA_50"]), 2),
        }

        return {
            "info": info,
            "indicators": indicators,
            "prediction": prediction,
            "sentiment": sent,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ══ HIGH POTENTIAL SCANNER ═══════════════════════════
SCAN_TICKERS = [
    "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
    "NFLX", "AMD", "JPM", "V", "BA", "DIS", "PYPL", "INTC",
]


def _analyze_one(ticker: str) -> dict | None:
    try:
        info = get_stock_info(ticker)
        df = fetch_stock_data(ticker, period="3mo")
        df = add_technical_indicators(df)
        latest = df.iloc[-1]

        rsi = float(latest["RSI"])
        macd = float(latest["MACD"])
        macd_signal = float(latest["MACD_Signal"])
        sma_20 = float(latest["SMA_20"])
        price = float(latest["Close"])

        score = 50
        if rsi < 30: score += 20
        elif rsi < 40: score += 10
        elif rsi > 70: score -= 15
        elif rsi > 60: score -= 5

        if macd > macd_signal: score += 15
        else: score -= 10

        if price > sma_20: score += 10
        else: score -= 10

        vol = float(latest["Volume"])
        vol_avg = float(latest["Volume_SMA_20"])
        if vol_avg > 0 and vol > vol_avg * 1.3: score += 10

        score = max(0, min(100, score))

        if score >= 70: signal_text = "🟢 Strong Buy"
        elif score >= 55: signal_text = "🔵 Buy"
        elif score >= 40: signal_text = "🟡 Hold"
        else: signal_text = "🔴 Sell"

        prev_close = float(df.iloc[-2]["Close"])
        change_pct = round(((price - prev_close) / prev_close) * 100, 2)

        return {
            "ticker": ticker, "name": info.get("name", ticker),
            "price": round(price, 2), "change_pct": change_pct,
            "rsi": round(rsi, 2), "score": score,
            "signal": signal_text, "currency": info.get("currency", "USD"),
        }
    except Exception:
        return None


@app.get("/api/high-potential")
def high_potential():
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_analyze_one, t): t for t in SCAN_TICKERS}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"stocks": results, "count": len(results)}


# ══ AUTO TRADING ENDPOINTS ═══════════════════════════

@app.get("/api/trade/portfolio")
def trade_portfolio():
    """Get current portfolio state."""
    try:
        portfolio = get_portfolio()
        # Calculate live P&L for open positions
        total_unrealized = 0
        for ticker, pos in portfolio["positions"].items():
            try:
                live = fetch_live_price(ticker)
                current = live["price"]
                pnl = (current - pos["buy_price"]) * pos["shares"]
                pos["current_price"] = current
                pos["unrealized_pnl"] = round(pnl, 2)
                pos["pnl_pct"] = round(((current - pos["buy_price"]) / pos["buy_price"]) * 100, 2)
                total_unrealized += pnl
            except Exception:
                pos["current_price"] = pos["buy_price"]
                pos["unrealized_pnl"] = 0
                pos["pnl_pct"] = 0

        portfolio["total_unrealized_pnl"] = round(total_unrealized, 2)
        portfolio["total_value"] = round(portfolio["balance"] + total_unrealized + sum(
            p["shares"] * p.get("current_price", p["buy_price"]) for p in portfolio["positions"].values()
        ), 2)

        # Calculate total realized P&L
        portfolio["total_realized_pnl"] = round(
            sum(t["profit"] for t in portfolio["trade_history"]), 2
        )
        portfolio["total_trades"] = len(portfolio["trade_history"])
        portfolio["winning_trades"] = sum(1 for t in portfolio["trade_history"] if t["profit"] > 0)
        portfolio["losing_trades"] = sum(1 for t in portfolio["trade_history"] if t["profit"] <= 0)

        if portfolio["total_trades"] > 0:
            portfolio["win_rate"] = round(
                (portfolio["winning_trades"] / portfolio["total_trades"]) * 100, 1
            )
        else:
            portfolio["win_rate"] = 0

        return portfolio
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/trade/reset")
def trade_reset():
    """Reset portfolio to $100,000."""
    return reset_portfolio()


@app.post("/api/trade/toggle")
def trade_toggle():
    """Toggle auto-trading bot ON/OFF."""
    return toggle_bot()


@app.post("/api/trade/execute/{ticker}")
def trade_execute(ticker: str):
    """
    Run the LSTM prediction for a ticker and auto-execute
    the trade signal (BUY/SELL/HOLD/WAIT).
    """
    try:
        # Get current price
        live = fetch_live_price(ticker)
        current_price = live["price"]

        # Get LSTM prediction
        df = fetch_stock_data(ticker)
        df = add_technical_indicators(df)
        prediction = train_and_predict(df)

        # Evaluate and execute trade signal
        signal = evaluate_trade_signal(
            ticker=ticker,
            current_price=current_price,
            predicted_prices=prediction["predicted_prices"],
            confidence=prediction["confidence"],
        )

        signal["live"] = live
        signal["prediction_details"] = prediction
        return signal

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/trade/sell/{ticker}")
def trade_force_sell(ticker: str):
    """Force-sell a position at current market price."""
    try:
        live = fetch_live_price(ticker)
        result = execute_sell(ticker, live["price"], "Manual sell (forced)")
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/trade/check/{ticker}")
def trade_check(ticker: str):
    """Check if a position should be sold (stop-loss / take-profit)."""
    try:
        live = fetch_live_price(ticker)
        result = check_position(ticker, live["price"])
        result["live"] = live
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/trade/auto-scan")
def trade_auto_scan():
    """
    Scan all high-potential stocks, run predictions,
    and auto-execute trades for the best signals.
    Only works when bot is active.
    """
    try:
        portfolio = get_portfolio()
        if not portfolio["bot_active"]:
            return {"status": "inactive", "message": "Bot is OFF. Toggle it on first."}

        results = []

        # First: check existing positions for stop-loss / take-profit
        for ticker in list(portfolio["positions"].keys()):
            try:
                live = fetch_live_price(ticker)
                check = check_position(ticker, live["price"])
                if check["action"] == "sell":
                    sell_result = execute_sell(ticker, live["price"], check["reason"])
                    results.append({"ticker": ticker, "action": "SELL", "result": sell_result})
            except Exception:
                continue

        # Then: scan for new buy opportunities
        for scan_ticker in SCAN_TICKERS[:8]:  # Limit to top 8 for speed
            if scan_ticker in portfolio["positions"]:
                continue
            try:
                live = fetch_live_price(scan_ticker)
                df = fetch_stock_data(scan_ticker)
                df = add_technical_indicators(df)
                pred = train_and_predict(df, epochs=30)  # fewer epochs for speed

                signal = evaluate_trade_signal(
                    ticker=scan_ticker,
                    current_price=live["price"],
                    predicted_prices=pred["predicted_prices"],
                    confidence=pred["confidence"],
                )
                results.append(signal)
            except Exception:
                continue

        return {
            "status": "completed",
            "scanned": len(results),
            "results": results,
            "portfolio": get_portfolio(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
