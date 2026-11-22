import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# 1. Reliable Euro-Denominated UCITS ETFs on Xetra (.DE)
# VWCE.DE = Vanguard FTSE All-World | EUNL.DE = iShares MSCI World | EUNA.DE = Global Bonds EUR-Hedged
tickers = ["VWCE.DE", "EUNL.DE", "EUNA.DE"]
weights = np.array([0.50, 0.30, 0.20])

initial_investment = 5000  # Starting lump sum (€)
monthly_contribution = 200  # Recurring monthly investment (€)
years = 10
trading_days = 252 * years
num_simulations = 1000

print("Fetching historical Euro UCITS market data...")
raw_data = yf.download(tickers, period="5y", auto_adjust=True)["Close"]

# Defensive Check: Verify every ticker downloaded valid data
missing_tickers = [t for t in tickers if raw_data[t].dropna().empty]
if missing_tickers:
  raise ValueError(
      f"Failed to download data for: {missing_tickers}. Check ticker symbols."
  )

# Forward-fill minor exchange holiday gaps, then drop any leading NaNs
data = raw_data[tickers].ffill().dropna()
print(f"Loaded {len(data)} aligned Euro trading days across {tickers}.")

# 2. Calculate Daily Percentage Returns & Portfolio Volatility
daily_returns = data.pct_change().dropna()
mean_daily_returns = daily_returns.mean()
cov_matrix = daily_returns.cov()

port_daily_return = np.sum(mean_daily_returns * weights)
port_daily_volatility = np.sqrt(
    np.dot(weights.T, np.dot(cov_matrix, weights))
)

print(f"Expected Annual Return: {port_daily_return * 252:.2%}")
print(f"Annual Volatility (Risk): {port_daily_volatility * np.sqrt(252):.2%}")

# 3. Run 1,000 Monte Carlo Simulations with Monthly Contributions (DCA)
random_shocks = np.random.normal(
    loc=port_daily_return,
    scale=port_daily_volatility,
    size=(trading_days, num_simulations),
)

portfolio_paths = np.zeros((trading_days, num_simulations))
portfolio_paths[0] = initial_investment

cash_contributed = np.zeros(trading_days)
cash_contributed[0] = initial_investment

for day in range(1, trading_days):
  portfolio_paths[day] = portfolio_paths[day - 1] * (1 + random_shocks[day])
  cash_contributed[day] = cash_contributed[day - 1]

  # Every 21 trading days (~1 calendar month), add recurring contribution
  if day % 21 == 0:
    portfolio_paths[day] += monthly_contribution
    cash_contributed[day] += monthly_contribution

# 4. Extract Percentiles
median_path = np.percentile(portfolio_paths, 50, axis=1)
best_case = np.percentile(portfolio_paths, 90, axis=1)
worst_case = np.percentile(portfolio_paths, 10, axis=1)
total_cash = cash_contributed[-1]

print(f"\n--- 10-Year Euro Projection (DCA: €{monthly_contribution}/mo) ---")
print(f"Total Cash You Put In:        €{total_cash:,.2f}")
print(
    f"Best Case (90th percentile):  €{best_case[-1]:,.2f} (+€{best_case[-1] - total_cash:,.2f} gain)"
)
print(
    f"Median Expected Outcome:      €{median_path[-1]:,.2f} (+€{median_path[-1] - total_cash:,.2f} gain)"
)
print(
    f"Worst Case (10th percentile): €{worst_case[-1]:,.2f} ({worst_case[-1] - total_cash:+,.2f} net)"
)

# 5. Plot Probability Cone vs. Cash Contributed
plt.figure(figsize=(10, 6))
plt.plot(portfolio_paths[:, :50], color="gray", alpha=0.12)
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
plt.plot(
    cash_contributed,
    color="#f1c40f",
    linestyle=":",
    linewidth=2.5,
    label="Total Cash Contributed",
)

plt.title(
    f"10-Year Euro UCITS Simulation (€{initial_investment:,} + €{monthly_contribution}/mo)",
    fontsize=13,
)
plt.xlabel("Trading Days (252 days = 1 Year)")
plt.ylabel("Portfolio Value (€)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()