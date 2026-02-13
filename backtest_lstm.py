
import os
import sys
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
# Add backend directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stock_data import fetch_stock_data, add_technical_indicators
from model import train_and_predict

def evaluate_prediction_accuracy(ticker, start_date, end_date, lookahead_days=5):
    """
    For a given ticker and date range, run the LSTM model as if in the past, then compare its 5-day forecast to actual closes.
    Returns a DataFrame with prediction, actual, and error for each window.
    """
    df = fetch_stock_data(ticker)
    df = add_technical_indicators(df)
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].reset_index(drop=True)
    results = []
    for i in range(len(df) - 60 - lookahead_days):
        window = df.iloc[i:i+60+lookahead_days]
        hist = window.iloc[:60]
        future = window.iloc[60:60+lookahead_days]
        # Skip if not enough future data or if hist is too short after dropna
        if len(future) < lookahead_days or len(hist.dropna()) < 60:
            continue
        try:
            pred = train_and_predict(hist)
            pred_prices = pred['predicted_prices']
            actual_prices = future['Close'].values.tolist()
            errors = [abs(p - a) for p, a in zip(pred_prices, actual_prices)]
            results.append({
                'date': hist.iloc[-1]['Date'],
                'predicted': pred_prices,
                'actual': actual_prices,
                'mae': np.mean(errors),
                'mse': np.mean([(p-a)**2 for p,a in zip(pred_prices, actual_prices)]),
            })
        except Exception as e:
            print(f"Skipping window at {hist.iloc[-1]['Date']} due to error: {e}")
            continue
    return pd.DataFrame(results)

if __name__ == "__main__":
    ticker = input("Enter IPO stock ticker (e.g. RELIANCE.NS): ").strip()
    # Fetch all available data for the ticker (max 2 years)
    df_all = fetch_stock_data(ticker, period="2y", interval="1d")
    df_all = add_technical_indicators(df_all)
    if len(df_all) < 70:
        print("Not enough data for backtest (need at least 70 days). Exiting.")
        sys.exit(1)
    # Find all valid window start indices (where 60-row history and 5-row future have no NaNs)
    valid_starts = []
    for i in range(len(df_all) - 65):
        hist = df_all.iloc[i:i+60]
        future = df_all.iloc[i+60:i+65]
        if len(hist.dropna()) == 60 and len(future.dropna()) == 5:
            valid_starts.append(i)
    if not valid_starts:
        print("No valid windows found in the last 2 years for this stock.")
        sys.exit(1)
    # Pick a random valid window
    start_idx = random.choice(valid_starts)
    # Pick a random end index at least 65 days ahead, but not more than 120 days or end of data
    max_end = min(start_idx + 120, len(df_all) - 1)
    end_idx = random.randint(start_idx + 65, max_end)
    start_date = str(df_all.iloc[start_idx]['Date'])[:10]
    end_date = str(df_all.iloc[end_idx]['Date'])[:10]
    print(f"Randomly selected date range: {start_date} to {end_date}")
    df = evaluate_prediction_accuracy(ticker, start_date, end_date)
    if df.empty:
        print("No valid prediction windows found in this range.")
        # Try to print the last available predicted and actual price if possible
        # Use the last 60 rows for prediction and next 5 for actual
        hist = df_all.iloc[start_idx:start_idx+60]
        future = df_all.iloc[start_idx+60:start_idx+65]
        if len(hist.dropna()) == 60 and len(future.dropna()) == 5:
            try:
                pred = train_and_predict(hist)
                pred_prices = pred['predicted_prices']
                actual_prices = future['Close'].values.tolist()
                print(f"Predicted 5-day prices: {pred_prices}")
                print(f"Actual 5-day prices:    {actual_prices}")
            except Exception as e:
                print(f"Could not compute prediction for last window: {e}")
    else:
        print(df[['date', 'mae', 'mse']])
        print("Average MAE:", df['mae'].mean())
        print("Average MSE:", df['mse'].mean())
        df.to_csv(f"trained_models/{ticker}_backtest_results.csv", index=False)
        print(f"Results saved to trained_models/{ticker}_backtest_results.csv")
