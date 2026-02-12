"""
StockSense AI — Auto Trading Engine (Paper Trading)
Strategy: LSTM Prediction-Based
- BUY when model predicts price going UP in next 5 days
- SELL when stop-loss / take-profit / trailing-stop / prediction reversal
"""

import json
import os
from datetime import datetime
from threading import Lock

# ── CONFIG ────────────────────────────────���────────────
INITIAL_BALANCE = 100_000.00
STOP_LOSS_PCT = -3.0          # -3% from buy price
TAKE_PROFIT_PCT = 5.0         # +5% from buy price
TRAILING_STOP_PCT = -2.0      # -2% from peak after buy
MAX_HOLD_DAYS = 5             # auto-sell after 5 days
POSITION_SIZE_PCT = 10.0      # use 10% of balance per trade

DATA_FILE = "portfolio.json"
_lock = Lock()


# ── PORTFOLIO STATE ────────────────────────────────────
def _default_state():
    return {
        "balance": INITIAL_BALANCE,
        "initial_balance": INITIAL_BALANCE,
        "positions": {},         # ticker -> position dict
        "trade_history": [],     # list of completed trades
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
    """Return current portfolio state."""
    with _lock:
        state = _load_state()
    return state


def reset_portfolio() -> dict:
    """Reset portfolio to initial state."""
    with _lock:
        state = _default_state()
        _save_state(state)
    return state


def toggle_bot() -> dict:
    """Toggle auto-trading bot ON/OFF."""
    with _lock:
        state = _load_state()
        state["bot_active"] = not state["bot_active"]
        _save_state(state)
    return {"bot_active": state["bot_active"]}


# ── TRADING LOGIC ──────────────────────────────────────
def execute_buy(ticker: str, current_price: float, predicted_prices: list,
                confidence: float) -> dict:
    """
    Execute a BUY order (paper trade).
    """
    with _lock:
        state = _load_state()

        # Already holding this ticker?
        if ticker in state["positions"]:
            return {"status": "skipped", "reason": f"Already holding {ticker}"}

        # Calculate position size
        position_value = state["balance"] * (POSITION_SIZE_PCT / 100)
        if position_value < current_price:
            return {"status": "skipped", "reason": "Insufficient balance"}

        shares = int(position_value // current_price)
        if shares == 0:
            return {"status": "skipped", "reason": "Cannot afford even 1 share"}

        cost = shares * current_price

        # Stop-loss & take-profit prices
        stop_loss = round(current_price * (1 + STOP_LOSS_PCT / 100), 2)
        take_profit = round(current_price * (1 + TAKE_PROFIT_PCT / 100), 2)

        # Create position
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
    """
    Execute a SELL order (paper trade).
    """
    with _lock:
        state = _load_state()

        if ticker not in state["positions"]:
            return {"status": "skipped", "reason": f"No position in {ticker}"}

        pos = state["positions"][ticker]
        revenue = pos["shares"] * current_price
        profit = revenue - pos["cost"]
        profit_pct = (profit / pos["cost"]) * 100

        # Record trade history
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
    """
    Check if a position should be sold based on stop-loss,
    take-profit, or trailing stop.
    """
    with _lock:
        state = _load_state()

    if ticker not in state["positions"]:
        return {"action": "none", "reason": "No position"}

    pos = state["positions"][ticker]
    buy_price = pos["buy_price"]
    change_pct = ((current_price - buy_price) / buy_price) * 100

    # ── STOP LOSS ──
    if current_price <= pos["stop_loss"]:
        return {"action": "sell", "reason": f"Stop-loss triggered ({STOP_LOSS_PCT}%)"}

    # ── TAKE PROFIT ──
    if current_price >= pos["take_profit"]:
        return {"action": "sell", "reason": f"Take-profit reached (+{TAKE_PROFIT_PCT}%)"}

    # ── TRAILING STOP ──
    if current_price > pos["peak_price"]:
        # Update peak and trailing stop
        with _lock:
            state = _load_state()
            state["positions"][ticker]["peak_price"] = current_price
            new_trailing = round(current_price * (1 + TRAILING_STOP_PCT / 100), 2)
            if new_trailing > state["positions"][ticker]["trailing_stop"]:
                state["positions"][ticker]["trailing_stop"] = new_trailing
            _save_state(state)
    elif current_price <= pos["trailing_stop"]:
        return {"action": "sell", "reason": f"Trailing stop triggered ({TRAILING_STOP_PCT}%)"}

    # ── MAX HOLD DAYS ──
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


def evaluate_trade_signal(ticker: str, current_price: float,
                          predicted_prices: list, confidence: float) -> dict:
    """
    Core strategy: Decide BUY / SELL / HOLD based on LSTM prediction.
    """
    state = _load_state()

    # Prediction analysis
    pred_day1 = predicted_prices[0]
    pred_day5 = predicted_prices[-1]
    pred_change_pct = ((pred_day5 - current_price) / current_price) * 100
    pred_direction = "up" if pred_day5 > current_price else "down"

    # Count how many predicted days are above current price
    days_above = sum(1 for p in predicted_prices if p > current_price)

    signal = {
        "ticker": ticker,
        "current_price": current_price,
        "predicted_day1": round(pred_day1, 2),
        "predicted_day5": round(pred_day5, 2),
        "pred_change_pct": round(pred_change_pct, 2),
        "pred_direction": pred_direction,
        "days_above_current": days_above,
        "confidence": confidence,
    }

    # Already holding?
    if ticker in state["positions"]:
        check = check_position(ticker, current_price)
        if check["action"] == "sell":
            result = execute_sell(ticker, current_price, check["reason"])
            signal["action"] = "SELL"
            signal["trade_result"] = result
        elif pred_direction == "down" and days_above <= 1 and confidence > 40:
            result = execute_sell(ticker, current_price, "Prediction reversal (downtrend)")
            signal["action"] = "SELL"
            signal["trade_result"] = result
        else:
            signal["action"] = "HOLD"
            signal["position"] = check
    else:
        # BUY conditions: prediction is UP + enough confidence
        if (pred_direction == "up" and days_above >= 3 and
                confidence >= 35 and pred_change_pct > 0.5):
            result = execute_buy(ticker, current_price, predicted_prices, confidence)
            signal["action"] = "BUY"
            signal["trade_result"] = result
        else:
            signal["action"] = "WAIT"
            signal["reason"] = "Conditions not met for entry"

    return signal