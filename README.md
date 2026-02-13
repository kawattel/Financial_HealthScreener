# Financial Health Analyzer (Tech Stocks)
**Streamlit dashboard for financial ratio diagnostics, peer comparison, and market context**

## Overview
This project is a Streamlit-based financial analysis dashboard designed to evaluate and compare the financial health of major tech companies over time. It combines **financial statement ratios (quarterly)** with **market data from yfinance** to provide a practical “internal analytics tool” style interface for screening profitability, liquidity, efficiency, and solvency.

The app supports:
- Multi-ticker comparison across multiple ratio categories
- Industry benchmarks with optional confidence intervals
- Trendline and 4-quarter moving average overlays
- Live and historical price visualizations
- Ratio ranking snapshots (latest quarter)
- Export of cleaned ratio data to CSV

**USE PROVIDED CSV**
---

## Key Features

### 1) Ratio Comparison Over Time (Quarterly)
Compare selected tickers across:

**Profitability & Liquidity**
- Net Profit Margin (`npm`)
- Return on Assets (`roa`)
- Return on Equity (`roe`)
- Current Ratio (`curr_ratio`)
- Quick Ratio (`quick_ratio`)

**Efficiency**
- Inventory Turnover (`inv_turn`)
- Asset Turnover (`at_turn`)
- Receivables Turnover (`rect_turn`)
- Payables Turnover (`pay_turn`)

**Solvency**
- Debt-to-Equity Ratio (`de_ratio`)  
  (computed as `totdebt_invcap / equity_invcap` with zero-safe handling)

---

### 2) Industry Benchmarking + Confidence Intervals
For the selected ratio, the dashboard computes an industry benchmark by quarter and can optionally overlay a confidence interval using a user-selected confidence level:
- 80% (z = 1.28)
- 90% (z = 1.645)
- 95% (z = 1.96)

Confidence intervals are computed using:
- mean, standard deviation, and count by quarter
- SEM = std / sqrt(n)
- margin of error = z * SEM

---

### 3) Trend + Smoothing Controls
Optional overlays:
- **Trend line** (linear regression via `scipy.stats.linregress`)
- **4-quarter moving average** (rolling window)

---

### 4) Ratio Rankings (Latest Quarter)
Toggleable ranking views:
- Financial ratio ranking (latest quarter by ticker)
- Efficiency ratio ranking (latest quarter by ticker)
- Solvency ratio ranking (latest quarter by ticker)

These create a quick “scoreboard” style snapshot for peer comparison.

---

### 5) Market Data Integration (yfinance)
Optional market context tools:
- **Live stock prices** with up/down indicator (↑ / ↓)
- **Historical price trend charts** over the selected date range

> Note: Live prices may not update during weekends/holidays/market-closed periods.

---

### 6) Data Cleaning + Quarterly Alignment
The pipeline standardizes time-series behavior by:
- Converting `qdate` to datetime and snapping to **quarter-end**
- Aggregating by `(TICKER, qdate)` using numeric column means
- Interpolating and forward-filling key ratios to handle missing observations
- Dedicated cleaning utility for selected efficiency ratios

It also maps **FB → META** to keep historical continuity.

---

### 7) Export
Users can download the processed dataset from the dashboard via:
- **Download Financial Data as CSV** (`financial_ratios.csv`)

---

## Tech Stack
- **Python**
- **Streamlit** (UI / dashboard)
- **Pandas / NumPy** (data wrangling)
- **Matplotlib** (charts)
- **SciPy** (trendline regression)
- **yfinance** (market prices)

---

## Dataset Requirements
This project expects a CSV containing quarterly ratio data with at least:

**Core columns**
- `TICKER`
- `qdate`

**Ratio columns used in the dashboard**
- `npm`, `roa`, `roe`, `curr_ratio`, `quick_ratio`
- `totdebt_invcap`, `equity_invcap` (for `de_ratio`)
- `inv_turn`, `at_turn`, `rect_turn`, `pay_turn` (efficiency ratios)

> The included code currently points to a local CSV path. For GitHub, replace that with a relative path (example below) and store your dataset in `/data`.

Example:
```python
csv_path = "data/Cleaned_Tech_Financial_Ratios.csv"
```
---

## Author

**Kurt Wattelet II**  
B.A. Finance – Financial Analytics  
University of Arkansas  
2024
