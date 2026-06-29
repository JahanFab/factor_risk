# Factor Risk Model

Decomposes a multi-asset portfolio's returns into explainable risk factors (market, size, value, momentum, bond, gold) using OLS regression and PCA, then stress-tests the portfolio under historical market scenarios.

---

## Overview

Understanding *why* a portfolio moves is as important as knowing *how much* it moves. This project builds a Fama-French style factor model that attributes every basis point of portfolio volatility to a known risk source, and quantifies how the portfolio would have performed in historical crises.

---

## How It Works

### 1. Portfolio

15 large-cap US stocks spanning five sectors:

| Sector | Tickers |
|--------|---------|
| Tech | AAPL, MSFT, GOOGL, AMZN, NVDA, META |
| Financials | JPM, BAC |
| Healthcare/Consumer | JNJ, PG |
| Energy | XOM, CVX |
| Macro | GLD, TLT, SPY |

### 2. Factor Proxies

| Factor | Proxy | Description |
|--------|-------|-------------|
| Market | SPY | Broad US equity market |
| Size | IWM | Small-cap premium (Russell 2000) |
| Value | IVE | S&P 500 Value ETF |
| Momentum | MTUM | iShares MSCI Momentum |
| Low Vol | USMV | iShares MSCI Min Volatility |
| Bond | TLT | Long-duration Treasury bonds |
| Gold | GLD | Gold commodity |

### 3. OLS Factor Regression

```
r_portfolio = α + β_Market·r_Market + β_Size·r_Size + ... + ε
```

- Estimates factor exposures (betas), t-statistics, and p-values
- Residual = idiosyncratic return (unexplained by factors)

### 4. Variance Decomposition

Each factor's contribution to total portfolio variance:

```
contribution_i = β_i × Σ_j(β_j × Cov(f_i, f_j)) / Var(portfolio)
```

### 5. PCA Factor Analysis

Extracts statistical factors from the asset covariance matrix — no assumed factor proxies required. Identifies the dominant directions of co-movement.

### 6. Stress Testing

Applies historical shock vectors to the estimated betas:

```
portfolio_impact = Σ_i(β_i × shock_i)
```

---

## Results

### Factor Exposures

| Factor | Beta | t-stat | Significant |
|--------|------|--------|-------------|
| Market | 1.43 | 41.2 | Yes |
| Momentum | −0.18 | −10.0 | Yes |
| Size | −0.13 | −8.8 | Yes |
| Low Vol | −0.27 | −8.8 | Yes |
| Gold | +0.04 | 5.2 | Yes |
| Value | −0.03 | −0.6 | No |

**R² = 0.924** — 92.4% of portfolio variance explained by these 7 factors.
**Annualized Alpha = 6.1%**

### Variance Attribution

- Market: 136.8% of variance (dominant driver, partially offset by other factors)
- Momentum tilt: −19.8%
- Size tilt: −12.7%
- Idiosyncratic: 7.6%

### Stress Test

| Scenario | Portfolio Impact |
|----------|----------------|
| 2008 GFC | **−46.3%** |
| Tech Bubble 2000 | **−46.2%** |
| COVID Crash 2020 | −33.8% |
| 2022 Rate Shock | −19.8% |
| Inflation Surge | −16.8% |
| Risk-Off Rally | +24.5% |

---

## Output Plots

| File | Description |
|------|-------------|
| `factor_risk_model.png` | 7-panel: factor betas, variance attribution pie, stress test bar chart, cumulative returns vs factors, rolling 60-day market beta, PCA scree plot, factor loading heatmap |

---

## Usage

```bash
pip install numpy pandas scikit-learn matplotlib scipy yfinance
python factor_risk_model.py
```

Data is fetched live via `yfinance`. Requires an internet connection.

---

## Dependencies

```
numpy
pandas
scikit-learn
matplotlib
scipy
yfinance
```
