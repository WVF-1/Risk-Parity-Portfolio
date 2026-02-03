# Risk-Parity-Portfolio
A risk-aware portfolio construction project, attempting to leverage FRED and Yahoo Finance data, in order to hand craft optimal risk averse investment portfolios.

# Risk Parity Portfolio — Risk-Aware Portfolio Construction

**Theme:** Risk-aware portfolio engineering  
**Level:** Intermediate–Advanced Quantitative Finance  
**Goal:** Build a professional-grade portfolio construction and analysis system based on *risk parity* principles.

---

## Project Overview

Traditional portfolios allocate capital (e.g., 60/40), but **risk parity portfolios allocate risk**. This project builds a full quantitative pipeline that:

* Estimates asset volatility and correlation
* Allocates capital so **each asset contributes equal portfolio risk**
* Benchmarks performance against:

  * 60/40 Portfolio
  * Equal-weight Portfolio
* Evaluates portfolio behavior during financial crises

This project reframes portfolio thinking from **return-focused → risk-aware**, setting the foundation for **machine learning-based volatility forecasting** in future work.

---

## Key Research Questions

* Why are equal-weight portfolios misleading?
* How do we balance **risk**, not just capital?
* How do different portfolio constructions behave during crises?
* Does risk parity reduce drawdowns and improve stability?

---

## Methodology Overview

### Data Collection

**Market Data (Yahoo Finance):**

| Asset Class | ETF Proxy |
| ----------- | --------- |
| US Equities | SPY       |
| Bonds       | TLT       |
| Gold        | GLD       |
| Commodities | DBC       |
| REITs       | VNQ       |

**Macroeconomic Data (FRED):**

| Indicator           | FRED Code |
| ------------------- | --------- |
| CPI                 | CPIAUCSL  |
| Federal Funds Rate  | FEDFUNDS  |
| 10Y Treasury Yield  | DGS10     |
| Recession Indicator | USREC     |

---

### Return & Volatility Estimation

* Daily log returns
* Rolling volatility estimation (60-day window)
* Covariance matrix estimation

---

### Risk Parity Optimization

Portfolio weights are computed by solving:

> Minimize squared deviation of asset risk contributions from equality

This ensures:

[ RC_1 = RC_2 = ... = RC_n ]

where:

[ RC_i = \frac{w_i (\Sigma w)_i}{\sigma_p} ]

Optimization is solved using **SciPy constrained minimization**.

---

### Benchmark Portfolio Construction

* **Risk Parity Portfolio** (optimized)
* **60/40 Portfolio** (stocks / bonds)
* **Equal-Weight Portfolio**

---

### Crisis Performance Evaluation

Performance is evaluated during:

| Event                   | Period       |
| ----------------------- | ------------ |
| Global Financial Crisis | 2008–2009    |
| COVID Crash             | Feb–Apr 2020 |
| Inflation / Rate Shock  | 2022         |

Metrics:

* Portfolio volatility
* Maximum drawdown
* Recovery speed

---

## Technical Stack

| Component     | Tools               |
| ------------- | ------------------- |
| Data          | yfinance, fredapi   |
| Math          | NumPy               |
| Optimization  | SciPy               |
| Time Series   | pandas              |
| Visualization | matplotlib, seaborn |

---

## Key Deliverables

* Risk contribution bar plots
* Portfolio allocation tables
* Rolling volatility comparisons
* Crisis drawdown charts
* Performance metrics tables

---

## 🗂 Project Structure

```text
risk_parity_portfolio/
│
├── data/
│   └── raw_prices.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_volatility_estimation.ipynb
│   ├── 03_risk_parity_optimization.ipynb
│   ├── 04_portfolio_backtest.ipynb
│   └── 05_crisis_analysis.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── risk_models.py
│   ├── optimizer.py
│   ├── backtester.py
│   └── visualization.py
│
└── README.md
```

---

## How To Run

### Install Dependencies

```bash
pip install numpy pandas scipy matplotlib yfinance fredapi
```

### Run Analysis

Open notebooks sequentially:

1. `01_data_exploration.ipynb`
2. `02_volatility_estimation.ipynb`
3. `03_risk_parity_optimization.ipynb`
4. `04_portfolio_backtest.ipynb`
5. `05_crisis_analysis.ipynb`

---

## Project Extensions

This project sets the foundation for:

* Machine learning volatility forecasting
* Regime-based portfolio optimization
* Adaptive asset allocation
* Stress testing under macroeconomic scenarios

---

## Why This Project Matters

Most student portfolios focus only on:

> Maximizing returns.

This project focuses on:

> Engineering stability and controlling risk.

This approach mirrors professional hedge fund and institutional portfolio design, demonstrating real-world quantitative finance skills.

---

## Acknowledgements

* FRED — Federal Reserve Economic Data
* Yahoo Finance — Market Data API
* SciPy Optimization Library

