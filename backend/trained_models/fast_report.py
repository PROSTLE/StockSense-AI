import os
import sys
import time
import numpy as np
import pandas as pd

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stock_data import fetch_stock_data, add_technical_indicators
from model import train_and_predict, _prediction_cache

def fast_report(ticker):
    print(f"\n🚀 GENERATING INSTANT ACCURACY REPORT FOR: {ticker}")
    print("-" * 50)
    
    try:
        # 1. Get Data
        df = fetch_stock_data(ticker, period="1y", interval="1d")
        df = add_technical_indicators(df)
        df = df.dropna().reset_index(drop=True)
        
        # 2. Setup 1 Window (Most Recent)
        cutoff = len(df) - 5 
        hist_df = df.iloc[:cutoff].copy()
        future_df = df.iloc[cutoff:cutoff+5].copy()
        actual_prices = future_df["Close"].values.tolist()
        dates = [str(d)[:10] for d in future_df["Date"]]
        current_price = float(hist_df["Close"].iloc[-1])
        
        # 3. Predict
        _prediction_cache.clear()
        print(f"⏳ Training LSTM on latest data (takes ~40s)...")
        t0 = time.time()
        pred = train_and_predict(hist_df)
        elapsed = time.time() - t0
        
        # 4. Score
        predicted = pred["predicted_prices"]
        mape = np.mean([abs(p - a) / a * 100 for p, a in zip(predicted, actual_prices)])
        accuracy = 100 - mape
        
        print(f"\n✅ REPORT COMPLETE IN {elapsed:.1f}s")
        print(f"{'='*70}")
        print(f" STOCK: {ticker.upper()}")
        print(f"{'='*70}")
        print(f" {'DATE':<12} | {'PREDICTED':<12} | {'ACTUAL':<12} | {'DIFF %':<8}")
        print(f" {'-'*12}-|-{'-'*12}-|-{'-'*12}-|-{'-'*8}")
        
        for d, p, a in zip(dates, predicted, actual_prices):
            p_val = f"₹{p:.2f}"
            a_val = f"₹{a:.2f}"
            diff = abs(p - a) / a * 100
            print(f" {d:<12} | {p_val:<12} | {a_val:<12} | {diff:>7.2f}%")
            
        print(f"{'='*70}")
        print(f" OVERALL ACCURACY: {accuracy:.2f}%")
        print(f" CONFIDENCE     : {pred.get('confidence', 0)}%")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    fast_report(ticker)
