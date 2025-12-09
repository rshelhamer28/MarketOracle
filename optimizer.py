import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
from data_loader import get_stock_data

def get_portfolio_data(tickers):
    """
    Fetches data for multiple stocks and combines them into one table.
    """
    df = pd.DataFrame()
    for t in tickers:
        prices, _ = get_stock_data(t)
        if isinstance(prices, pd.Series):
            df[t] = prices
        else:
             # Fallback if data shape is weird
            df[t] = prices.iloc[:, 0]
            
    # Calculate Log Returns for the whole dataframe
    log_returns = np.log(df / df.shift(1)).dropna()
    return df, log_returns

def portfolio_stats(weights, log_returns):
    """
    Calculates Portfolio Return and Portfolio Volatility using Matrix Math.
    """
    # Convert weights to array
    weights = np.array(weights)
    
    # Annualized Return = Sum of (Weight * Mean_Return) * 252
    port_return = np.sum(log_returns.mean() * weights) * 252
    
    # Portfolio Volatility = Sqrt( Weights_Transposed * Covariance_Matrix * Weights ) * Sqrt(252)
    # This is the standard Linear Algebra formula for portfolio risk
    cov_matrix = log_returns.cov() * 252
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    return port_return, port_volatility

def min_sharpe_ratio(weights, log_returns):
    """
    The function we want to minimize (Negative Sharpe Ratio).
    Because 'scipy' only finds minimums, we minimize the negative to find the maximum.
    """
    p_ret, p_vol = portfolio_stats(weights, log_returns)
    risk_free_rate = 0.04 # Assuming 4% risk-free rate (Treasury bills)
    sharpe = (p_ret - risk_free_rate) / p_vol
    return -sharpe

if __name__ == "__main__":
    # --- INPUTS ---
    # We mix Stocks (SPY), Long-Term Bonds (TLT), and Gold (GLD)
    tickers = ['SPY', 'TLT', 'GLD']
    print(f"Optimizing Portfolio for: {tickers}...")
    
    prices, log_returns = get_portfolio_data(tickers)
    
    # --- OPTIMIZATION ---
    num_assets = len(tickers)
    args = (log_returns,)
    
    # Constraints: Sum of weights must equal 1 (100% of money invested)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: Each weight must be between 0 and 1 (No short selling)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Initial Guess: Equal weights (33% each)
    initial_guess = num_assets * [1. / num_assets,]
    
    # The Magic Solver (SLSQP Algorithm)
    result = sco.minimize(min_sharpe_ratio, initial_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    
    # --- RESULTS ---
    print("\n--- OPTIMAL PORTFOLIO ALLOCATION ---")
    optimal_weights = result.x
    
    for i, ticker in enumerate(tickers):
        print(f"{ticker}: {optimal_weights[i]:.2%}")
        
    p_ret, p_vol = portfolio_stats(optimal_weights, log_returns)
    print(f"\nExpected Annual Return: {p_ret:.2%}")
    print(f"Expected Volatility:    {p_vol:.2%}")
    print(f"Sharpe Ratio:           {(p_ret - 0.04) / p_vol:.2f}")

    # --- PIE CHART ---
    plt.figure(figsize=(6, 6))
    plt.pie(optimal_weights, labels=tickers, autopct='%1.1f%%', startangle=140)
    plt.title("Mathematically Optimal Portfolio")
    plt.show()