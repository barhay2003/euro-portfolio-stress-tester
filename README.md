# Nordic Portfolio Stress-Tester & Monte Carlo Simulator

A quantitative portfolio simulation and risk-analysis engine built in Python. Designed to model multi-asset allocation strategies, historical market drawdowns, and long-term compounding for European retail investors using Euro-denominated UCITS ETFs and Nordic equities.

---

## Why This Project Exists
Most retail portfolio calculators rely on single-line average return projections and US-centric assumptions that ignore currency drag, volatility clustering, and allocation drift. 

This tool acts as a quantitative "flight simulator" for personal capital—allowing users to stress-test asset mixes against historical market crashes (such as the 2020 COVID shock and 2022 rate-hike cycle) and simulate thousands of correlated future market paths.

---

## Current Capabilities (v0.1)
* **Historical Market Data Ingestion:** Automated retrieval of adjusted daily closing prices via `yfinance`.
* **Portfolio Risk & Return Profiling:** Computes annualized expected returns and covariance-adjusted portfolio volatility.
* **Monte Carlo Simulation Engine:** Generates 1,000 randomized 10-year price trajectories to visualize 10th, 50th (median), and 90th percentile probability cones.

---

## Development Roadmap
- [x] **v0.1 — Core Prototype:** Single-curve Monte Carlo simulation engine and percentile visualization (`simulator.py`).
- [ ] **v0.2 — Euro Localization & DCA:** Native support for Euro-denominated UCITS ETFs (e.g., `VWCE.DE`, `EUNL.DE`, `AGGH.DE`), live `EUR/USD` FX conversion, and recurring monthly contributions (Dollar-Cost Averaging).
- [ ] **v0.3 — Multi-Asset Correlated Simulation:** Cholesky decomposition for independent asset path generation with a 6-month periodic rebalancing toggle (`ON` vs. `OFF`).
- [ ] **v0.4 — Historical Stress-Testing:** Maximum drawdown (`MaxDD`) and time-to-recovery analysis across historical crisis windows.
- [ ] **v1.0 — Interactive Web Application:** FastAPI backend with SQLite/Parquet caching paired with an interactive, mobile-responsive Plotly/Streamlit dashboard.
- [ ] **v2.0 — AI Risk & Rebalancing Copilot:** Structured LLM analysis layer that explains portfolio concentration risk, currency exposure, and drawdown sensitivity in plain language.

---

## Project Architecture
```text
├── engine/
│   ├── __init__.py
│   ├── data_loader.py       # Market data ingestion & local caching
│   └── monte_carlo.py       # Cholesky & portfolio simulation math
├── tests/
│   └── test_math.py         # Unit tests for financial & statistical functions
├── simulator.py             # Standalone v0.1 CLI prototype
├── requirements.txt         # Project dependencies
└── README.md
