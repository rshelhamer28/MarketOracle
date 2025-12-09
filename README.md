# 🔮 The Market Oracle

**A Quantitative Finance Dashboard built with Python, Streamlit, and Stochastic Calculus.**

## 📖 About The Project

I built **The Market Oracle** because I wanted to move beyond simple technical analysis (drawing lines on charts) and start using the same mathematical models employed by quantitative hedge funds. 

This application doesn't just show you what a stock price *did* in the past; it uses probability theory to simulate what it might do in the *future*, and it uses linear algebra to calculate exactly how to structure a portfolio to minimize risk.

It is a full-stack data engineering project that fetches live market data, processes it through complex statistical models, and visualizes the results in an interactive web dashboard.

---

## 🚀 Key Features

### 1. The "Crystal Ball" (Monte Carlo Simulation)
Most predictions are just guesses. This tool uses **Geometric Brownian Motion (GBM)** to generate 1,000+ "alternate universes" for a stock's future. 
* **The Math:** $dS_t = \mu S_t dt + \sigma S_t dW_t$
* **The Value:** Instead of a single price target, it gives you a **Probability Cone**. It answers questions like: *"What are the odds NVDA is above $150 in 30 days based on its historical volatility?"*

### 2. The "Robot Advisor" (Portfolio Optimizer)
This module takes a list of assets (e.g., Stocks, Gold, Bonds) and finds the mathematically perfect allocation.
* **The Math:** Uses `scipy.optimize` to minimize the negative Sharpe Ratio via the **SLSQP** algorithm.
* **The Value:** It calculates the **Efficient Frontier**—finding the specific mix of assets that offers the highest possible return for the lowest possible risk.

### 3. The "Fear Gauge" (Sentiment Analyzer)
This tool compares what the market *feels* (Implied Volatility via VIX) vs. what is *actually happening* (Realized Volatility).
* **The Value:** It spots market anomalies. When the spread is high, options are expensive and traders are panicked. When the spread is low, the market is complacent.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Web Dashboard)
* **Data Engine:** YFinance API (Live Market Data)
* **Linear Algebra:** NumPy (Matrix Operations)
* **Optimization:** SciPy (Minimization Algorithms)
* **Visualization:** Matplotlib (Dynamic Charting)

---

## 📦 How to Run This Locally

1.  **Clone the repository** (or download the files).
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Launch the Dashboard:**
    ```bash
    streamlit run dashboard.py
    ```
4.  **Open your browser:** The app will typically run at `http://localhost:8501`.

---

## 📂 Project Structure

* `dashboard.py`: The main entry point. Runs the Streamlit web server and handles the UI.
* `simulation.py`: Contains the Monte Carlo logic and geometric brownian motion formulas.
* `optimizer.py`: Handles the covariance matrices and portfolio allocation algorithms.
* `market_sentiment.py`: Analyzes the VIX spread to determine market regime.
* `data_loader.py`: The utility script that fetches and cleans raw financial data.

---

Built by Rowan Shelhamer as a Portfolio Project.

