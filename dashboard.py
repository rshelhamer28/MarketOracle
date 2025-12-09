import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco

# Import your custom modules
# (Make sure these files are in the same folder!)
from data_loader import get_stock_data, analyze_statistics
from simulation import run_monte_carlo
from market_sentiment import get_fear_gauge

st.set_page_config(page_title="The Market Oracle", layout="wide")

st.title("🔮 The Market Oracle: Quantitative Analysis Suite")
st.markdown("A stochastic pricing model, portfolio optimizer, and market sentiment analyzer.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("User Controls")
sim_ticker = st.sidebar.text_input("Simulation Ticker", "NVDA")
port_tickers = st.sidebar.text_input("Portfolio Tickers (comma separated)", "SPY,TLT,GLD")
days_pred = st.sidebar.slider("Prediction Days", 30, 365, 252)

# Create Tabs for the different tools
tab1, tab2, tab3 = st.tabs(["📉 Monte Carlo Simulation", "⚖️ Portfolio Optimizer", "😨 Fear Gauge"])

# --- TAB 1: THE SIMULATION ---
with tab1:
    st.header(f"Monte Carlo Simulation: {sim_ticker}")
    
    if st.button("Run Simulation"):
        with st.spinner("Simulating 1,000 alternate universes..."):
            # 1. Run the math
            price_paths = run_monte_carlo(sim_ticker, days_to_predict=days_pred, iterations=1000)
            
            # 2. Visualize it directly in the dashboard
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(price_paths[:, :50], alpha=0.2, color='blue') # Show 50 paths
            ax.plot(np.mean(price_paths, axis=1), color='red', linewidth=2, label='Mean Path')
            ax.set_title(f"Future Price Cone ({days_pred} Days)")
            ax.legend()
            
            # Show the plot
            st.pyplot(fig)
            
            # 3. Stats
            final_prices = price_paths[-1]
            current_price = price_paths[0][0]
            exp_price = np.mean(final_prices)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("Expected Price", f"${exp_price:.2f}", delta=f"{(exp_price-current_price)/current_price:.1%}")
            
            # Probability Math
            target = current_price * 1.20
            prob = np.sum(final_prices > target) / len(final_prices)
            col3.metric("Prob of +20% Gain", f"{prob:.1%}")

# --- TAB 2: THE OPTIMIZER ---
with tab2:
    st.header("Efficient Frontier Optimizer")
    
    # Clean up the input string
    ticker_list = [t.strip() for t in port_tickers.split(',')]
    
    if st.button("Optimize Portfolio"):
        with st.spinner("Crunching covariance matrices..."):
            # Fetch Data Manually here to ensure robustness
            df = pd.DataFrame()
            for t in ticker_list:
                prices, _ = get_stock_data(t)
                if isinstance(prices, pd.Series):
                    df[t] = prices
                else:
                    df[t] = prices.iloc[:, 0]
            
            log_returns = np.log(df / df.shift(1)).dropna()
            
            # --- RUN OPTIMIZATION (Local Logic) ---
            def portfolio_stats(weights):
                weights = np.array(weights)
                port_return = np.sum(log_returns.mean() * weights) * 252
                cov_matrix = log_returns.cov() * 252
                port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return port_return, port_vol

            def min_sharpe(weights):
                r, v = portfolio_stats(weights)
                return -((r - 0.04) / v)

            num_assets = len(ticker_list)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(num_assets))
            init_guess = num_assets * [1./num_assets,]
            
            result = sco.minimize(min_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            
            # Display Results
            weights = result.x
            
            st.subheader("Optimal Allocation")
            
            # Pie Chart
            fig2, ax2 = plt.subplots()
            ax2.pie(weights, labels=ticker_list, autopct='%1.1f%%', startangle=90)
            ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig2)

# --- TAB 3: MARKET SENTIMENT ---
with tab3:
    st.header("Market Fear Gauge (VIX vs Realized)")
    
    if st.button("Analyze Market Mood"):
        with st.spinner("Comparing option premiums to reality..."):
            realized, implied = get_fear_gauge()
            
            # Create a nice area chart
            chart_data = pd.DataFrame({
                "Market Prediction (VIX)": implied,
                "Actual Volatility": realized
            })
            
            st.line_chart(chart_data)
            
            curr_vix = implied.iloc[-1]
            curr_real = realized.iloc[-1]
            diff = curr_vix - curr_real
            
            st.metric("Current VIX (Fear Cost)", f"{curr_vix:.2f}")
            st.metric("Actual Volatility (Reality)", f"{curr_real:.2f}")
            
            if diff > 5:
                st.error("STATUS: HIGH FEAR (Options are Expensive)")
            elif diff < 0:
                st.warning("STATUS: COMPLACENCY (Risk of Crash)")
            else:
                st.success("STATUS: NORMAL")