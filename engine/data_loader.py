import pandas as pd
import yfinance as yf


def fetch_aligned_prices(
    tickers: list[str], period: str = "5y"
) -> pd.DataFrame:
  """Downloads adjusted close prices, validates tickers, and aligns trading days."""
  raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)[
      "Close"
  ]

  if isinstance(raw, pd.Series):
    raw = raw.to_frame(name=tickers[0])

  missing = [t for t in tickers if t not in raw.columns or raw[t].dropna().empty]
  if missing:
    raise ValueError(f"Failed to download valid data for: {missing}")

  return raw[tickers].ffill().dropna()