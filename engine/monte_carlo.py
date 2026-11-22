import numpy as np
import pandas as pd


def run_rebalance_comparison(
    prices: pd.DataFrame,
    weights: np.ndarray,
    initial_investment: float = 5000.0,
    monthly_contribution: float = 200.0,
    years: int = 10,
    num_simulations: int = 1000,
    rebalance_days: int = 126,
) -> dict:
  """Simulates correlated multi-asset paths with and without periodic rebalancing."""
  daily_returns = prices.pct_change().dropna()
  mean_returns = daily_returns.mean().values
  cov_matrix = daily_returns.cov().values

  num_assets = len(weights)
  trading_days = 252 * years

  # Cholesky decomposition: L @ L.T = cov_matrix
  cholesky_L = np.linalg.cholesky(cov_matrix)

  # Generate correlated daily returns: shape (trading_days, num_simulations, num_assets)
  uncorrelated_z = np.random.normal(
      0, 1, size=(trading_days, num_simulations, num_assets)
  )
  correlated_shocks = uncorrelated_z @ cholesky_L.T + mean_returns

  # Track individual asset buckets for both strategies
  buckets_no_rebal = np.zeros((trading_days, num_simulations, num_assets))
  buckets_rebal = np.zeros((trading_days, num_simulations, num_assets))

  buckets_no_rebal[0] = initial_investment * weights
  buckets_rebal[0] = initial_investment * weights

  cash_contributed = np.zeros(trading_days)
  cash_contributed[0] = initial_investment

  for day in range(1, trading_days):
    growth = 1.0 + correlated_shocks[day]
    buckets_no_rebal[day] = buckets_no_rebal[day - 1] * growth
    buckets_rebal[day] = buckets_rebal[day - 1] * growth
    cash_contributed[day] = cash_contributed[day - 1]

    # Monthly DCA contribution split according to target weights
    if day % 21 == 0:
      dca_alloc = monthly_contribution * weights
      buckets_no_rebal[day] += dca_alloc
      buckets_rebal[day] += dca_alloc
      cash_contributed[day] += monthly_contribution

    # Periodic rebalancing back to target weights
    if day % rebalance_days == 0:
      totals = buckets_rebal[day].sum(axis=1, keepdims=True)
      buckets_rebal[day] = totals * weights

  total_no_rebal = buckets_no_rebal.sum(axis=2)
  total_rebal = buckets_rebal.sum(axis=2)

  # Calculate final equity weight drift in the unrebalanced portfolio
  final_equity_share = (
      buckets_no_rebal[-1, :, :2].sum(axis=1) / total_no_rebal[-1]
  )

  return {
      "no_rebal_paths": total_no_rebal,
      "rebal_paths": total_rebal,
      "cash_contributed": cash_contributed,
      "median_final_equity_share": float(np.median(final_equity_share)),
  }