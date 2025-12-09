import numpy as np
import matplotlib.pyplot as plt
from data_loader import get_stock_data, analyze_statistics

def run_monte_carlo(ticker, days_to_predict=252, iterations=1000):
    # 1. Get the Stats from your Data Loader
    prices, log_returns = get_stock_data(ticker)
    u, sigma = analyze_statistics(log_returns)
    
    # Get the last actual price to start the simulation
    # "iloc[-1]" means the very last item in the list
    start_price = prices.iloc[-1]
    
    # 2. Set up the Simulation
    dt = 1  # Time step (1 day)
    
    # Generate random shocks (Z) for every day and every simulation at once
    # This creates a massive matrix of random numbers (Normal Distribution)
    Z = np.random.normal(0, 1, (days_to_predict, iterations))
    
    # 3. The Math (Geometric Brownian Motion)
    # Drift term: (Mean - 0.5 * Variance)
    drift = u - (0.5 * sigma**2)
    
    # Diffusion term: Volatility * Random Shock
    diffusion = sigma * Z
    
    # Calculate daily returns
    daily_returns = np.exp(drift + diffusion)
    
    # 4. Build Price Paths
    price_paths = np.zeros((days_to_predict, iterations))
    price_paths[0] = start_price
    
    # Loop through each day, multiplying yesterday's price by today's return
    for t in range(1, days_to_predict):
        price_paths[t] = price_paths[t-1] * daily_returns[t]
        
    return price_paths

def plot_simulation(ticker, price_paths):
    plt.figure(figsize=(10, 6))
    
    # Plot first 50 paths (Blue lines) to show "alternate universes"
    plt.plot(price_paths[:, :50], alpha=0.2, color='blue')
    
    # Plot the Mean Path (Red line) - The "Statistical Average" outcome
    mean_path = np.mean(price_paths, axis=1)
    plt.plot(mean_path, color='red', linewidth=2, label='Mean Path')
    
    plt.title(f"Monte Carlo Simulation: {ticker} (Next 252 Days)")
    plt.xlabel("Days into Future")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    
    # --- Prediction Market Stats ---
    final_prices = price_paths[-1]
    current_price = price_paths[0][0]
    
    print(f"\n--- PREDICTION MARKET: {ticker} ---")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Avg Expected Price: ${np.mean(final_prices):.2f}")
    
    # Probability Calculation
    # Let's see the odds of it going up 10%
    target = current_price * 1.10
    prob = np.sum(final_prices > target) / len(final_prices)
    print(f"Probability of +10% Gain (${target:.2f}): {prob:.2%}")

if __name__ == "__main__":
    # Feel free to change this to "NVDA" or "TSLA"
    run_ticker = "SPY"
    
    # Running 5,000 simulations
    paths = run_monte_carlo(run_ticker, days_to_predict=252, iterations=5000)
    plot_simulation(run_ticker, paths)