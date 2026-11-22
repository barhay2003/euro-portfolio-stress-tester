from engine.data_loader import fetch_aligned_prices
from engine.monte_carlo import run_rebalance_comparison
import matplotlib.pyplot as plt
import numpy as np

tickers = ["VWCE.DE", "EUNL.DE", "EUNA.DE"]
weights = np.array([0.50, 0.30, 0.20])  # Target: 80% Stocks / 20% Bonds

print("Loading Euro UCITS market data...")
prices = fetch_aligned_prices(tickers, period="5y")

print("Running 1,000 correlated Cholesky simulations...")
results = run_rebalance_comparison(
    prices=prices,
    weights=weights,
    initial_investment=5000,
    monthly_contribution=200,
    years=10,
    num_simulations=1000,
)

med_no_rebal = np.percentile(results["no_rebal_paths"], 50, axis=1)
med_rebal = np.percentile(results["rebal_paths"], 50, axis=1)
cash = results["cash_contributed"]

print("\n--- 10-Year Rebalancing Comparison (Median) ---")
print(f"Total Cash Contributed:       €{cash[-1]:,.2f}")
print(f"Unrebalanced (Buy & Drift):   €{med_no_rebal[-1]:,.2f}")
print(f"Rebalanced (Every 6 Months):  €{med_rebal[-1]:,.2f}")
print(
    f"Unrebalanced Final Stock %:   {results['median_final_equity_share']:.1%} (Target was 80.0%)"
)

plt.figure(figsize=(10, 6))
plt.plot(
    med_no_rebal,
    color="#e67e22",
    linewidth=2.5,
    label="Unrebalanced Median (Drift)",
)
plt.plot(
    med_rebal,
    color="#00ffcc",
    linewidth=2.5,
    linestyle="--",
    label="Rebalanced Every 6 Mo",
)
plt.plot(
    cash,
    color="#f1c40f",
    linestyle=":",
    linewidth=2,
    label="Total Cash Contributed",
)

plt.title("v0.3 Cholesky Multi-Asset Simulation: Drift vs. 6-Month Rebalancing")
plt.xlabel("Trading Days (252 days = 1 Year)")
plt.ylabel("Portfolio Value (€)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()