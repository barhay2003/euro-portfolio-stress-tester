import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# 1. Define a simple 3-asset portfolio ("The Recipe")
# SPY = US S&P 500 ETF | VGK = European Stocks ETF | BND = Total Bond Market ETF
tickers = ["SPY", "VGK", "BND"]
weights = np.array([0.60, 0.20, 0.20])  # 60% US Stocks, 20% EU Stocks, 20% Bonds

initial_investment = 10000  # Starting with €10,000
years = 10
trading_days = 252 * years
num_simulations = 1000

print("Fetching historical market data...")
# Download 5 years of daily closing prices
data = yf.download(tickers, period="5y", auto_adjust=True)["Close"]

# 2. Calculate Daily Percentage Returns & Portfolio Volatility
daily_returns = data.pct_change().dropna()

# Expected annual return and covariance (how assets move together)
mean_daily_returns = daily_returns.mean()
cov_matrix = daily_returns.cov()

# Portfolio-level daily expected return & standard deviation (volatility)
port_daily_return = np.sum(mean_daily_returns * weights)
port_daily_volatility = np.sqrt(
    np.dot(weights.T, np.dot(cov_matrix, weights))
)

print(f"Expected Annual Return: {port_daily_return * 252:.2%}")
print(f"Annual Volatility (Risk): {port_daily_volatility * np.sqrt(252):.2%}")

# 3. Run 1,000 Monte Carlo Simulations ("Parallel Universes")
# Generate random daily shocks assuming a normal distribution
random_shocks = np.random.normal(
    loc=port_daily_return,
    scale=port_daily_volatility,
    size=(trading_days, num_simulations),
)

# Convert daily returns into cumulative portfolio value paths
portfolio_paths = initial_investment * np.cumprod(1 + random_shocks, axis=0)

# 4. Extract Percentiles for the "Probability Cone"
median_path = np.percentile(portfolio_paths, 50, axis=1)
best_case = np.percentile(portfolio_paths, 90, axis=1)  # Top 10% outcome
worst_case = np.percentile(portfolio_paths, 10, axis=1)  # Bottom 10% outcome

print(f"\n--- 10-Year Projection (Starting €{initial_investment:,}) ---")
print(f"Best Case (90th percentile):  €{best_case[-1]:,.2f}")
print(f"Median Expected Outcome:      €{median_path[-1]:,.2f}")
print(f"Worst Case (10th percentile): €{worst_case[-1]:,.2f}")

# 5. Plot the Probability Cone
plt.figure(figsize=(10, 6))
plt.plot(
    portfolio_paths[:, :50], color="gray", alpha=0.15
)  # Show 50 sample paths
plt.plot(median_path, color="#00ffcc", linewidth=2.5, label="Median (50th %)")
plt.plot(
    best_case,
    color="#2ecc71",
    linestyle="--",
    linewidth=2,
    label="Optimistic (90th %)",
)
plt.plot(
    worst_case,
    color="#e74c3c",
    linestyle="--",
    linewidth=2,
    label="Pessimistic (10th %)",
)

plt.title(
    f"10-Year Portfolio Monte Carlo Simulation ({num_simulations} Runs)",
    fontsize=14,
)
plt.xlabel("Trading Days (252 days = 1 Year)")
plt.ylabel("Portfolio Value (€)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
