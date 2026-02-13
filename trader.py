
import json
import os
from datetime import datetime
from threading import Lock

INITIAL_BALANCE = 1_000_000.00
STOP_LOSS_PCT = -3.0
TAKE_PROFIT_PCT = 5.0
TRAILING_STOP_PCT = -2.0
MAX_HOLD_DAYS = 5
POSITION_SIZE_PCT = 10.0

DATA_FILE = "portfolio.json"
_lock = Lock()


def _default_state():
    return {
        "balance": INITIAL_BALANCE,
        "initial_balance": INITIAL_BALANCE,
        "positions": {},
        "trade_history": [],
        "bot_active": False,
        "created_at": datetime.now().isoformat(),
    }


def _load_state() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return _default_state()


def _save_state(state: dict):
    with open(DATA_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def get_portfolio() -> dict:
    with _lock:
        state = _load_state()
    return state


def reset_portfolio() -> dict:
    with _lock:
        state = _default_state()
        _save_state(state)
    return state


def toggle_bot() -> dict:
    with _lock:
        state = _load_state()
        state["bot_active"] = not state["bot_active"]
        _save_state(state)
    return {"bot_active": state["bot_active"]}


def execute_buy(ticker: str, current_price: float, predicted_prices: list,
                confidence: float) -> dict:
    with _lock:
        state = _load_state()

        if ticker in state["positions"]:
            return {"status": "skipped", "reason": f"Already holding {ticker}"}

        position_value = state["balance"] * (POSITION_SIZE_PCT / 100)
        if position_value < current_price:
            return {"status": "skipped", "reason": "Insufficient balance"}

        shares = int(position_value // current_price)
        if shares == 0:
            return {"status": "skipped", "reason": "Cannot afford even 1 share"}

        cost = shares * current_price

        stop_loss = round(current_price * (1 + STOP_LOSS_PCT / 100), 2)
        take_profit = round(current_price * (1 + TAKE_PROFIT_PCT / 100), 2)

        position = {
            "ticker": ticker,
            "shares": shares,
            "buy_price": current_price,
            "cost": round(cost, 2),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "trailing_stop": stop_loss,
            "peak_price": current_price,
            "predicted_prices": predicted_prices,
            "confidence": confidence,
            "buy_time": datetime.now().isoformat(),
            "day_count": 0,
        }

        state["balance"] -= cost
        state["balance"] = round(state["balance"], 2)
        state["positions"][ticker] = position
        _save_state(state)

        return {
            "status": "bought",
            "ticker": ticker,
            "shares": shares,
            "price": current_price,
            "cost": round(cost, 2),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "balance": state["balance"],
        }


def execute_sell(ticker: str, current_price: float, reason: str) -> dict:
    with _lock:
        state = _load_state()

        if ticker not in state["positions"]:
            return {"status": "skipped", "reason": f"No position in {ticker}"}

        pos = state["positions"][ticker]
        revenue = pos["shares"] * current_price
        profit = revenue - pos["cost"]
        profit_pct = (profit / pos["cost"]) * 100

        trade = {
            "ticker": ticker,
            "shares": pos["shares"],
            "buy_price": pos["buy_price"],
            "sell_price": current_price,
            "cost": pos["cost"],
            "revenue": round(revenue, 2),
            "profit": round(profit, 2),
            "profit_pct": round(profit_pct, 2),
            "buy_time": pos["buy_time"],
            "sell_time": datetime.now().isoformat(),
            "reason": reason,
            "confidence": pos["confidence"],
        }

        state["balance"] += revenue
        state["balance"] = round(state["balance"], 2)
        state["trade_history"].append(trade)
        del state["positions"][ticker]
        _save_state(state)

        return {
            "status": "sold",
            "ticker": ticker,
            "shares": trade["shares"],
            "price": current_price,
            "profit": trade["profit"],
            "profit_pct": trade["profit_pct"],
            "reason": reason,
            "balance": state["balance"],
        }


def check_position(ticker: str, current_price: float) -> dict:
    with _lock:
        state = _load_state()

    if ticker not in state["positions"]:
        return {"action": "none", "reason": "No position"}

    pos = state["positions"][ticker]
    buy_price = pos["buy_price"]
    change_pct = ((current_price - buy_price) / buy_price) * 100

    if current_price <= pos["stop_loss"]:
        return {"action": "sell", "reason": f"Stop-loss triggered ({STOP_LOSS_PCT}%)"}

    if current_price >= pos["take_profit"]:
        return {"action": "sell", "reason": f"Take-profit reached (+{TAKE_PROFIT_PCT}%)"}

    if current_price > pos["peak_price"]:
        with _lock:
            state = _load_state()
            state["positions"][ticker]["peak_price"] = current_price
            new_trailing = round(current_price * (1 + TRAILING_STOP_PCT / 100), 2)
            if new_trailing > state["positions"][ticker]["trailing_stop"]:
                state["positions"][ticker]["trailing_stop"] = new_trailing
            _save_state(state)
    elif current_price <= pos["trailing_stop"]:
        return {"action": "sell", "reason": f"Trailing stop triggered ({TRAILING_STOP_PCT}%)"}

    pos["day_count"] += 1
    if pos["day_count"] >= MAX_HOLD_DAYS:
        return {"action": "sell", "reason": f"Max hold period ({MAX_HOLD_DAYS} days)"}

    return {
        "action": "hold",
        "reason": "Within thresholds",
        "current_pnl": round(change_pct, 2),
        "current_price": current_price,
        "stop_loss": pos["stop_loss"],
        "take_profit": pos["take_profit"],
        "trailing_stop": pos["trailing_stop"],
    }


def compute_technical_score(indicators: dict) -> float:
    score = 50.0

    rsi = indicators.get("RSI", 50)
    macd = indicators.get("MACD", 0)
    macd_signal = indicators.get("MACD_Signal", 0)
    sma_20 = indicators.get("SMA_20", 0)
    sma_50 = indicators.get("SMA_50", 0)
    price = indicators.get("price", 0)

    if rsi < 25:
        score += 20
    elif rsi < 35:
        score += 12
    elif rsi < 45:
        score += 5
    elif rsi > 75:
        score -= 20
    elif rsi > 65:
        score -= 12
    elif rsi > 55:
        score -= 5

    macd_diff = macd - macd_signal
    if macd_diff > 0:
        score += min(15, macd_diff * 100)
    else:
        score += max(-15, macd_diff * 100)

    if sma_20 > 0 and price > sma_20:
        score += 8
    elif sma_20 > 0:
        score -= 8

    if sma_50 > 0 and price > sma_50:
        score += 7
    elif sma_50 > 0:
        score -= 7

    if sma_20 > 0 and sma_50 > 0:
        if sma_20 > sma_50:
            score += 5
        else:
            score -= 5

    return max(0, min(100, score))


def compute_composite_score(
    predicted_prices: list,
    current_price: float,
    confidence: float,
    indicators: dict | None = None,
    sentiment_score: float | None = None,
) -> dict:
    pred_day5 = predicted_prices[-1]
    pred_change_pct = ((pred_day5 - current_price) / current_price) * 100
    days_above = sum(1 for p in predicted_prices if p > current_price)

    lstm_score = 50.0
    lstm_score += pred_change_pct * 5
    lstm_score += (days_above - 2.5) * 6
    lstm_score += (confidence - 50) * 0.3
    lstm_score = max(0, min(100, lstm_score))

    if indicators:
        tech_score = compute_technical_score(indicators)
    else:
        tech_score = 50.0

    if sentiment_score is not None:
        sent_score = 50 + sentiment_score * 50
        sent_score = max(0, min(100, sent_score))
    else:
        sent_score = 50.0

    LSTM_W, TECH_W, SENT_W = 0.40, 0.35, 0.25
    composite = (lstm_score * LSTM_W) + (tech_score * TECH_W) + (sent_score * SENT_W)
    composite = round(max(0, min(100, composite)), 1)

    return {
        "composite_score": composite,
        "lstm_score": round(lstm_score, 1),
        "tech_score": round(tech_score, 1),
        "sent_score": round(sent_score, 1),
        "pred_change_pct": round(pred_change_pct, 2),
        "days_above": days_above,
    }


def detect_market_regime(indicators: dict | None) -> dict:
    if not indicators:
        return {
            "regime": "NEUTRAL",
            "buy_threshold": 52,
            "sell_threshold": 35,
            "position_pct": 10,
            "min_confidence": 20,
        }

    rsi = indicators.get("RSI", 50)
    macd = indicators.get("MACD", 0)
    macd_signal = indicators.get("MACD_Signal", 0)
    sma_20 = indicators.get("SMA_20", 0)
    sma_50 = indicators.get("SMA_50", 0)
    price = indicators.get("price", 0)

    bull_signals = 0
    bear_signals = 0

    if rsi < 40:
        bull_signals += 1
    elif rsi > 65:
        bear_signals += 1

    if macd > macd_signal:
        bull_signals += 1
    else:
        bear_signals += 1

    if sma_20 > 0 and price > sma_20:
        bull_signals += 1
    elif sma_20 > 0:
        bear_signals += 1

    if sma_50 > 0 and price > sma_50:
        bull_signals += 1
    elif sma_50 > 0:
        bear_signals += 1

    if sma_20 > sma_50 > 0:
        bull_signals += 1
    elif sma_50 > sma_20 > 0:
        bear_signals += 1

    if bull_signals >= 4:
        return {
            "regime": "BULL",
            "buy_threshold": 45,
            "sell_threshold": 30,
            "position_pct": 15,
            "min_confidence": 15,
        }
    elif bear_signals >= 4:
        return {
            "regime": "BEAR",
            "buy_threshold": 60,
            "sell_threshold": 40,
            "position_pct": 5,
            "min_confidence": 30,
        }
    else:
        return {
            "regime": "NEUTRAL",
            "buy_threshold": 52,
            "sell_threshold": 35,
            "position_pct": 10,
            "min_confidence": 20,
        }


def evaluate_trade_signal(
    ticker: str,
    current_price: float,
    predicted_prices: list,
    confidence: float,
    indicators: dict | None = None,
    sentiment_score: float | None = None,
) -> dict:
    state = _load_state()

    regime = detect_market_regime(indicators)
    buy_threshold = regime["buy_threshold"]
    sell_threshold = regime["sell_threshold"]

    scores = compute_composite_score(
        predicted_prices, current_price, confidence,
        indicators, sentiment_score,
    )
    composite = scores["composite_score"]

    pred_day1 = predicted_prices[0]
    pred_day5 = predicted_prices[-1]
    pred_direction = "up" if pred_day5 > current_price else "down"

    signal = {
        "ticker": ticker,
        "current_price": current_price,
        "predicted_day1": round(pred_day1, 2),
        "predicted_day5": round(pred_day5, 2),
        "pred_change_pct": scores["pred_change_pct"],
        "pred_direction": pred_direction,
        "days_above_current": scores["days_above"],
        "confidence": confidence,
        "composite_score": composite,
        "market_regime": regime["regime"],
        "scores": scores,
    }

    if ticker in state["positions"]:
        check = check_position(ticker, current_price)
        if check["action"] == "sell":
            result = execute_sell(ticker, current_price, check["reason"])
            signal["action"] = "SELL"
            signal["trade_result"] = result
        elif composite <= sell_threshold:
            result = execute_sell(ticker, current_price,
                                 f"Bearish signal in {regime['regime']} market ({composite}/100)")
            signal["action"] = "SELL"
            signal["trade_result"] = result
        else:
            signal["action"] = "HOLD"
            signal["position"] = check
    else:
        if composite >= buy_threshold and confidence >= regime["min_confidence"]:
            result = execute_buy(ticker, current_price, predicted_prices, confidence)
            signal["action"] = "BUY"
            signal["trade_result"] = result
        elif (scores["pred_change_pct"] > 2.0 and scores["days_above"] >= 3
              and confidence >= 25):
            result = execute_buy(ticker, current_price, predicted_prices, confidence)
            signal["action"] = "BUY"
            signal["trade_result"] = result
        else:
            signal["action"] = "WAIT"
            signal["reason"] = (
                f"Score {composite}/100 ({regime['regime']} mode, "
                f"need ≥{buy_threshold})"
            )

    return signal
