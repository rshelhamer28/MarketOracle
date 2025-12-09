import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_fear_gauge():
    print("Downloading live market data...")
    # Download SPY (The Market) and ^VIX (The Fear Index)
    tickers = ['SPY', '^VIX']
    data = yf.download(tickers, start="2020-01-01", progress=False)
    
    # --- ROBUST DATA EXTRACTION ---
    # We check if 'Adj Close' exists, otherwise use 'Close'
    # We DO NOT flatten the columns here, because we need the Ticker names!
    try:
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        else:
            prices = data['Close']
            
        # Extract the specific columns for SPY and VIX
        spy = prices['SPY']
        vix = prices['^VIX']
        
    except KeyError:
        # Fallback for some yfinance versions that might return different structures
        # This tries to grab them if the levels are swapped
        spy = data.xs('SPY', level=1, axis=1)['Adj Close']
        vix = data.xs('^VIX', level=1, axis=1)['Adj Close']

    # --- MATH TIME ---
    # 1. Calculate Realized Volatility (What actually happened)
    # We take a 30-day rolling window of standard deviation
    log_rets = np.log(spy / spy.shift(1))
    
    # We multiply by sqrt(252) * 100 to make it comparable to the VIX index
    realized_vol = log_rets.rolling(window=30).std() * np.sqrt(252) * 100
    
    # 2. Prepare Implied Volatility (What the market PREDICTS)
    # The VIX is already annualized, so we just use it directly
    implied_vol = vix
    
    # Drop empty data points (NaNs) so the chart lines up
    df = pd.DataFrame({'Realized': realized_vol, 'Implied': implied_vol}).dropna()
    
    return df['Realized'], df['Implied']

if __name__ == "__main__":
    try:
        realized, implied = get_fear_gauge()
        
        # Create the "Prediction Gap" Chart
        plt.figure(figsize=(12, 6))
        
        plt.plot(implied, color='red', label='Market Prediction (VIX)', alpha=0.7)
        plt.plot(realized, color='blue', label='Actual Reality (Realized Vol)', alpha=0.6)
        
        # Fill the gap to show "Fear Premium"
        # When Red is above Blue, traders are paying extra for insurance
        plt.fill_between(implied.index, realized, implied, where=(implied > realized), 
                        color='red', alpha=0.1, label='Fear Premium (Expensive Options)')
        
        plt.title("Prediction Market: What Traders Expect vs. What Actually Happens")
        plt.ylabel("Volatility Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # --- TERMINAL ANALYSIS ---
        curr_imp = implied.iloc[-1]
        curr_real = realized.iloc[-1]
        
        print(f"\n--- MARKET SENTIMENT READING ---")
        print(f"Market Prediction (VIX): {curr_imp:.2f}")
        print(f"Actual Volatility (30d): {curr_real:.2f}")
        
        diff = curr_imp - curr_real
        print(f"Spread: {diff:.2f} points")
        
        if diff > 8:
            print("STATUS: HIGH FEAR (Market is pricing in a crash; Insurance is expensive)")
        elif diff < 0:
            print("STATUS: COMPLACENCY (Market is too calm; Risk of surprise drop)")
        else:
            print("STATUS: NORMAL (Expectations match reality)")
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Try running 'pip install --upgrade yfinance' in your terminal just in case.")