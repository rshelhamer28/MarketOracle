import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

def get_stock_data(ticker, start_date='2020-01-01'):
    # Download data
    data = yf.download(ticker, start=start_date, progress=False)
    
    # --- DATA CLEANING (The Fix) ---
    # 1. Flatten MultiIndex columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # 2. Select Close price
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    else:
        prices = data['Close']
    
    # 3. FORCE it to be a single list of numbers, not a table
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]  # Just take the first column, whatever it is
        
    # Calculate Log Returns
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.dropna()
    
    return prices, log_returns

def analyze_statistics(log_returns):
    u = log_returns.mean()
    sigma = log_returns.std()
    
    # --- SAFETY VALVE ---
    # If 'u' is still a Series/Table, extract the raw number
    if isinstance(u, pd.Series):
        u = u.iloc[0]
    if isinstance(sigma, pd.Series):
        sigma = sigma.iloc[0]
    
    print(f"--- Statistics ---")
    print(f"Daily Mean: {u:.6f}")
    print(f"Daily Volatility: {sigma:.6f}")
    
    return u, sigma

if __name__ == "__main__":
    ticker = "SPY"
    prices, log_returns = get_stock_data(ticker)
    u, sigma = analyze_statistics(log_returns)

    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(prices)
    plt.title(f"{ticker} Price History")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.hist(log_returns, bins=50, density=True, alpha=0.6, color='green')
    plt.title("Log Returns Distribution")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()