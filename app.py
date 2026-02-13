
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from stock_data import fetch_stock_data, add_technical_indicators, get_stock_info, fetch_live_price, fetch_chart_data
from sentiment import analyze_sentiment
from model import train_and_predict
from intraday_model import generate_intraday_signal
from trader import (
    get_portfolio, reset_portfolio, toggle_bot,
    evaluate_trade_signal, execute_sell, execute_buy, check_position,
    compute_dynamic_levels, detect_market_regime,
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


@app.get("/api/market-indices")
def get_market_indices():
    import yfinance as yf
    indices = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN"}
    results = []
    for name, symbol in indices.items():
        try:
            stock = yf.Ticker(symbol)
            info = stock.info or {}
            hist = stock.history(period="5d")
            price = 0
            prev = 0
            if len(hist) >= 2:
                price = round(float(hist["Close"].iloc[-1]), 2)
                prev = round(float(hist["Close"].iloc[-2]), 2)
            elif len(hist) == 1:
                price = round(float(hist["Close"].iloc[-1]), 2)
                prev = info.get("previousClose", info.get("regularMarketPreviousClose", price))
            else:
                price = info.get("regularMarketPrice", info.get("previousClose", 0))
                prev = info.get("previousClose", info.get("regularMarketPreviousClose", price))
            if not price and info:
                price = info.get("regularMarketPrice", info.get("previousClose", 0))
            if not prev or prev == price:
                prev = info.get("previousClose", info.get("regularMarketPreviousClose", prev))
            change = round(price - prev, 2)
            change_pct = round((change / prev) * 100, 2) if prev else 0
            results.append({
                "name": name, "symbol": symbol, "price": price,
                "change": change, "change_pct": change_pct,
                "direction": "up" if change >= 0 else "down",
            })
        except Exception:
            results.append({"name": name, "symbol": symbol, "price": 0, "change": 0, "change_pct": 0, "direction": "up"})
    return {"indices": results}


@app.get("/api/predict/{ticker}")
def predict(ticker: str):
    try:
        df = fetch_stock_data(ticker)
        df = add_technical_indicators(df)
        result = train_and_predict(df)
        return {"ticker": ticker, **result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/sentiment/{ticker}")
def sentiment(ticker: str):
    try:
        return analyze_sentiment(ticker)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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
            "BB_Upper": round(float(latest["BB_Upper"]), 2),
            "BB_Lower": round(float(latest["BB_Lower"]), 2),
            "BB_Width": round(float(latest["BB_Width"]), 4),
            "ATR": round(float(latest["ATR"]), 2),
        }

        return {
            "info": info,
            "indicators": indicators,
            "prediction": prediction,
            "sentiment": sent,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


STOCK_DATABASE = [
    {"ticker": "RELIANCE.NS", "name": "Reliance Industries"},
    {"ticker": "TCS.NS", "name": "Tata Consultancy Services"},
    {"ticker": "INFY.NS", "name": "Infosys"},
    {"ticker": "HDFCBANK.NS", "name": "HDFC Bank"},
    {"ticker": "ICICIBANK.NS", "name": "ICICI Bank"},
    {"ticker": "BHARTIARTL.NS", "name": "Bharti Airtel"},
    {"ticker": "SBIN.NS", "name": "State Bank of India"},
    {"ticker": "HINDUNILVR.NS", "name": "Hindustan Unilever"},
    {"ticker": "ITC.NS", "name": "ITC Limited"},
    {"ticker": "LT.NS", "name": "Larsen & Toubro"},
    {"ticker": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank"},
    {"ticker": "TATAMOTORS.NS", "name": "Tata Motors"},
    {"ticker": "WIPRO.NS", "name": "Wipro"},
    {"ticker": "MARUTI.NS", "name": "Maruti Suzuki India"},
    {"ticker": "ADANIENT.NS", "name": "Adani Enterprises"},
    {"ticker": "TATASTEEL.NS", "name": "Tata Steel"},
    {"ticker": "TATASILV.NS", "name": "Tata Silver ETF"},
    {"ticker": "TATAPOWER.NS", "name": "Tata Power Company"},
    {"ticker": "TATACOMM.NS", "name": "Tata Communications"},
    {"ticker": "TATAELXSI.NS", "name": "Tata Elxsi"},
    {"ticker": "TATACHEM.NS", "name": "Tata Chemicals"},
    {"ticker": "TATACONSUM.NS", "name": "Tata Consumer Products"},
    {"ticker": "TITAN.NS", "name": "Titan Company"},
    {"ticker": "SUNPHARMA.NS", "name": "Sun Pharmaceutical"},
    {"ticker": "HCLTECH.NS", "name": "HCL Technologies"},
    {"ticker": "BAJFINANCE.NS", "name": "Bajaj Finance"},
    {"ticker": "BAJAJFINSV.NS", "name": "Bajaj Finserv"},
    {"ticker": "ASIANPAINT.NS", "name": "Asian Paints"},
    {"ticker": "AXISBANK.NS", "name": "Axis Bank"},
    {"ticker": "ULTRACEMCO.NS", "name": "UltraTech Cement"},
    {"ticker": "NESTLEIND.NS", "name": "Nestle India"},
    {"ticker": "NTPC.NS", "name": "NTPC Limited"},
    {"ticker": "POWERGRID.NS", "name": "Power Grid Corporation"},
    {"ticker": "ONGC.NS", "name": "Oil & Natural Gas Corporation"},
    {"ticker": "COALINDIA.NS", "name": "Coal India"},
    {"ticker": "JSWSTEEL.NS", "name": "JSW Steel"},
    {"ticker": "TECHM.NS", "name": "Tech Mahindra"},
    {"ticker": "DRREDDY.NS", "name": "Dr. Reddy's Laboratories"},
    {"ticker": "CIPLA.NS", "name": "Cipla"},
    {"ticker": "DIVISLAB.NS", "name": "Divi's Laboratories"},
    {"ticker": "EICHERMOT.NS", "name": "Eicher Motors"},
    {"ticker": "HEROMOTOCO.NS", "name": "Hero MotoCorp"},
    {"ticker": "BAJAJ-AUTO.NS", "name": "Bajaj Auto"},
    {"ticker": "M&M.NS", "name": "Mahindra & Mahindra"},
    {"ticker": "BRITANNIA.NS", "name": "Britannia Industries"},
    {"ticker": "DABUR.NS", "name": "Dabur India"},
    {"ticker": "GODREJCP.NS", "name": "Godrej Consumer Products"},
    {"ticker": "PIDILITIND.NS", "name": "Pidilite Industries"},
    {"ticker": "BERGEPAINT.NS", "name": "Berger Paints India"},
    {"ticker": "HAVELLS.NS", "name": "Havells India"},
    {"ticker": "SIEMENS.NS", "name": "Siemens India"},
    {"ticker": "ABB.NS", "name": "ABB India"},
    {"ticker": "ADANIPORTS.NS", "name": "Adani Ports"},
    {"ticker": "ADANIGREEN.NS", "name": "Adani Green Energy"},
    {"ticker": "GRASIM.NS", "name": "Grasim Industries"},
    {"ticker": "INDUSINDBK.NS", "name": "IndusInd Bank"},
    {"ticker": "SBILIFE.NS", "name": "SBI Life Insurance"},
    {"ticker": "HDFCLIFE.NS", "name": "HDFC Life Insurance"},
    {"ticker": "APOLLOHOSP.NS", "name": "Apollo Hospitals"},
    {"ticker": "ZOMATO.NS", "name": "Zomato"},
    {"ticker": "PAYTM.NS", "name": "One97 Communications (Paytm)"},
    {"ticker": "NYKAA.NS", "name": "FSN E-Commerce (Nykaa)"},
    {"ticker": "DMART.NS", "name": "Avenue Supermarts (DMart)"},
    {"ticker": "IRCTC.NS", "name": "IRCTC"},
    {"ticker": "HAL.NS", "name": "Hindustan Aeronautics"},
    {"ticker": "BEL.NS", "name": "Bharat Electronics"},
    {"ticker": "BANKBARODA.NS", "name": "Bank of Baroda"},
    {"ticker": "PNB.NS", "name": "Punjab National Bank"},
    {"ticker": "CANBK.NS", "name": "Canara Bank"},
    {"ticker": "IOC.NS", "name": "Indian Oil Corporation"},
    {"ticker": "BPCL.NS", "name": "Bharat Petroleum"},
    {"ticker": "HINDPETRO.NS", "name": "Hindustan Petroleum"},
    {"ticker": "VEDL.NS", "name": "Vedanta Limited"},
    {"ticker": "HINDALCO.NS", "name": "Hindalco Industries"},
    {"ticker": "GOLDIAM.NS", "name": "Goldiam International"},
    {"ticker": "GOLDBEES.NS", "name": "Nippon India Gold ETF"},
    {"ticker": "SILVEREES.NS", "name": "Nippon India Silver ETF"},
]


@app.get("/api/search")
def search_stocks(q: str = Query("", min_length=1)):
    query = q.strip().upper()
    if not query:
        return {"results": []}

    matches = []
    for stock in STOCK_DATABASE:
        ticker_upper = stock["ticker"].upper()
        name_upper = stock["name"].upper()
        if query in ticker_upper or query in name_upper:
            matches.append(stock)

    matches.sort(key=lambda s: (
        0 if s["ticker"].upper().startswith(query) else
        1 if s["name"].upper().startswith(query) else 2
    ))

    return {"results": matches[:10]}


SCAN_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "BHARTIARTL.NS", "SBIN.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS",
    "KOTAKBANK.NS", "TATAMOTORS.NS", "WIPRO.NS", "MARUTI.NS", "ADANIENT.NS",
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
            "signal": signal_text, "currency": info.get("currency", "INR"),
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


@app.get("/api/trade/portfolio")
def trade_portfolio():
    try:
        portfolio = get_portfolio()
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
    return reset_portfolio()


@app.post("/api/trade/toggle")
def trade_toggle():
    return toggle_bot()


@app.post("/api/trade/execute/{ticker}")
def trade_execute(ticker: str):
    try:
        live = fetch_live_price(ticker)
        current_price = live["price"]

        df = fetch_stock_data(ticker)
        df = add_technical_indicators(df)
        prediction = train_and_predict(df)

        latest = df.iloc[-1]
        indicators = {
            "RSI": float(latest["RSI"]),
            "MACD": float(latest["MACD"]),
            "MACD_Signal": float(latest["MACD_Signal"]),
            "SMA_20": float(latest["SMA_20"]),
            "SMA_50": float(latest["SMA_50"]),
            "price": current_price,
        }

        try:
            sent = analyze_sentiment(ticker)
            sentiment_score = sent.get("overall_score", 0.0)
        except Exception:
            sentiment_score = 0.0

        signal = evaluate_trade_signal(
            ticker=ticker,
            current_price=current_price,
            predicted_prices=prediction["predicted_prices"],
            confidence=prediction["confidence"],
            indicators=indicators,
            sentiment_score=sentiment_score,
        )

        signal["live"] = live
        signal["prediction_details"] = prediction
        return signal

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/trade/sell/{ticker}")
def trade_force_sell(ticker: str):
    try:
        live = fetch_live_price(ticker)
        result = execute_sell(ticker, live["price"], "Manual sell (forced)")
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/trade/check/{ticker}")
def trade_check(ticker: str):
    try:
        live = fetch_live_price(ticker)
        result = check_position(ticker, live["price"])
        result["live"] = live
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/trade/intraday-signal/{ticker}")
def trade_intraday_signal(ticker: str):
    """Single-stock intraday signal using XGBoost pipeline."""
    try:
        signal = generate_intraday_signal(ticker)
        return signal
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/trade/auto-scan")
def trade_auto_scan():
    """Auto-scan using intraday XGBoost pipeline with dynamic SL/TP."""
    try:
        portfolio = get_portfolio()
        if not portfolio["bot_active"]:
            return {"status": "inactive", "message": "Bot is OFF. Toggle it on first."}

        results = []

        # 1. Check existing positions for SL/TP/trailing triggers
        for ticker in list(portfolio["positions"].keys()):
            try:
                live = fetch_live_price(ticker)
                check = check_position(ticker, live["price"])
                if check["action"] == "sell":
                    sell_result = execute_sell(ticker, live["price"], check["reason"])
                    results.append({"ticker": ticker, "action": "SELL", "result": sell_result})
            except Exception:
                continue

        # 2. Scan for new entries using intraday XGBoost signals
        for scan_ticker in SCAN_TICKERS[:8]:
            if scan_ticker in portfolio["positions"]:
                continue
            try:
                signal = generate_intraday_signal(scan_ticker)

                if signal.get("action") == "BUY":
                    buy_result = execute_buy(
                        ticker=scan_ticker,
                        current_price=signal["entry_price"],
                        predicted_prices=[signal["tp1_price"]],
                        confidence=signal["confidence"],
                        sl_price=signal["sl_price"],
                        tp1_price=signal["tp1_price"],
                        tp2_price=signal.get("tp2_price"),
                        regime=signal["regime"],
                    )
                    signal["trade_result"] = buy_result

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
