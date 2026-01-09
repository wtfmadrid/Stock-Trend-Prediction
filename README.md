# 📈 Stock Trend & Risk Analyzer — LSTM Forecast + Backtest (US / Canada / India)

A student-friendly, finance-focused Streamlit web app that lets you analyze stocks across **US (NYSE/NASDAQ)**, **Canada (TSX)**, and **India (NSE)** markets using interactive charts and a machine-learning model (LSTM) to forecast **next-day returns**. The app also includes a realistic **transaction-cost-aware backtest** to compare the strategy vs **Buy & Hold**.

Link for overview of project: https://drive.google.com/file/d/1PyU9y1BVSD3u-Q570-Zu8ahORXs6tRKA/view?usp=sharing

---

## 🚀 What this project does

### ✅ Stock Analysis Dashboard (Interactive)
For any ticker you enter, the app fetches historical OHLCV data from Yahoo Finance (via `yfinance`) and generates:

- Interactive **candlestick chart** (with optional overlays)
- Trend indicators (MA100 / MA200)
- Momentum indicators (RSI, MACD)
- Volatility & risk measures (Bollinger Bands, ATR, rolling volatility)
- Risk visuals (drawdown curve)
- Market sensitivity (**beta vs benchmark**) using a market ETF/index

### ✅ ML Forecasting (LSTM)
The LSTM model uses a rolling window of the **last 60 trading days** of engineered features to predict:

> **Next-day return** (not the next-day price)

This output is used to generate a simple Long/Cash signal.

### ✅ Backtest (with transaction costs)
The predicted returns are converted into a simple trading strategy:

- **LONG** if predicted next-day return > threshold  
- **CASH** otherwise  
- Costs are applied when switching positions (basis points / bps)

The app plots:
- Strategy equity curve vs Buy & Hold equity curve
- Sharpe ratio, max drawdown, turnover, exposure, total return

---

## 📊 Indicators & Risk Analytics Used (and why)

This project intentionally uses **core finance indicators** that are common in technical analysis and are also useful as ML features.

### 1) Moving Averages (MA100 / MA200)
**Why:** Smooths out price noise and helps identify trend direction.
- MA100 captures medium-term trend
- MA200 captures long-term trend and major trend breaks

**Interpretation:**
- Price above MA200 → generally bullish trend
- Price below MA200 → generally bearish trend

---

### 2) Returns (`ret`) + Lagged Returns (`ret_1`, `ret_5`, `ret_20`)
**Why:** Returns represent the core “movement” a model tries to learn.
- `ret` = daily return
- `ret_5` = weekly-ish change
- `ret_20` = monthly-ish change
- Lagged returns help detect momentum vs mean-reversion patterns

---

### 3) RSI (14) — Relative Strength Index
**Why:** Measures momentum and whether price is “overbought” or “oversold”.

- RSI > 70 → overbought (price may be stretched)
- RSI < 30 → oversold (price may be stretched downward)

RSI gives the model a momentum signal that isn’t just raw price movement.

---

### 4) MACD (12,26,9)
**Why:** Captures trend + momentum shifts using exponential moving averages.
- MACD line = EMA(12) - EMA(26)
- Signal line = EMA(9) of MACD
- Histogram shows the momentum difference

MACD is useful to detect turning points and trend acceleration/slowdown.

---

### 5) Bollinger Bands (20, 2 std) + `bb_width`
**Why:** Bollinger Bands measure volatility around a moving average.

- Upper/Lower bands expand during high volatility
- Bands contract during low volatility (often called “volatility compression”)

`bb_width` normalizes the band width:
> (Upper − Lower) / Mid

This makes it scale-free and helps the model detect risk regimes.

---

### 6) ATR(14) — Average True Range + `atr_pct`
**Why:** ATR measures the typical daily range (risk / volatility proxy).

ATR uses True Range (TR):
- |High − Low|
- |High − PrevClose|
- |Low − PrevClose|

Then ATR14 = rolling mean of TR over 14 days.

`atr_pct = ATR14 / Close` makes it comparable across different-priced stocks.

This helps interpret “how risky” the stock is currently.

---

### 7) Rolling Volatility (annualized): `vol20`, `vol60`, `vol252`
**Why:** Captures volatility regimes over different horizons.
- 20-day → short-term risk
- 60-day → medium-term risk
- 252-day → long-term risk baseline

Calculated as:
> rolling std(daily returns) × √252

Volatility regimes matter because model behavior and market behavior change during high vs low volatility periods.

---

### 8) Drawdown
**Why:** Shows peak-to-trough declines from the historical maximum.
- Helps interpret “pain” periods and crashes
- Useful when comparing strategies and risk management

---

### 9) Beta vs Benchmark
**Why:** Measures how sensitive a stock is to the market benchmark.
- β > 1 → moves more than the market (higher risk)
- β < 1 → moves less than the market (more defensive)

Benchmarks used:
- US: SPY
- Canada: XIU.TO
- India: ^NSEI

---

## 🤖 Machine Learning Approach (LSTM)

### Goal
Predict **next-day return** using sequential information:

- Inputs: last **60 days** of engineered features
- Output: predicted return for the next trading day

### Why LSTM?
Financial time series have temporal structure (momentum, volatility clustering, trends).  
LSTMs are designed to learn patterns from sequences rather than isolated rows.

### Training setup (high level)
- Multi-stock training using a cross-market universe (US/CA/IN)
- Time-based split (train before a cutoff date, test after)
- StandardScaler fit only on training data (prevents leakage)
- Metrics tracked:
  - MAE (magnitude error)
  - Directional Accuracy (sign correctness)
  - Backtest metrics (Sharpe, drawdown, turnover)

---

## 📉 Strategy Logic (Long/Cash)

We convert predicted return into a simple decision:

- If **pred > threshold** → LONG (invested)
- Else → CASH (not invested)

Transaction costs:
- Applied when position changes (in bps)

Why this strategy?
It is simple enough to explain, but realistic enough to test:
- Costs matter
- Turnover matters
- Buy & Hold is a strong baseline

---

## 🧠 Key Takeaways from Results
- Performance varies by ticker: some stocks behave very close to Buy & Hold (model stays long most days).
- The strategy is strongest when it successfully avoids some negative-return periods.
- Strategy evaluation is done with finance metrics (not just ML metrics), since low MAE does not always mean high Sharpe.

---

## 📉 Strategy Performance Metrics (How to Read the Results)

In addition to ML metrics (MAE, directional accuracy), this project evaluates the model using **finance-standard backtesting metrics**, because in trading **how you make returns matters as much as how accurate predictions are**.

Below is what each metric means and why it matters.

---

### 1) Strategy Total Return
**What it is:**  
The cumulative return of the model-driven strategy over the test period.

**How to read it:**  
- `+1.00` = +100% total return  
- `+0.25` = +25% total return  

**Why it matters:**  
Shows whether the strategy actually grows capital over time.

---

### 2) Buy & Hold Total Return
**What it is:**  
Return from simply buying the stock on the first day and holding it until the end.

**Why it matters:**  
Buy & Hold is the **baseline**.  
If a strategy cannot compete with Buy & Hold, it may not justify the extra complexity and trading costs.

---

### 3) Strategy Sharpe Ratio
**What it is:**  
Risk-adjusted return measure.

**How to interpret:**
- **Sharpe < 0.5** → weak risk-adjusted performance  
- **0.5 – 1.0** → acceptable  
- **1.0 – 1.5** → good  
- **> 1.5** → very strong (rare in practice)

**Why it matters:**  
A strategy that makes money but with huge volatility is **not desirable**.  
Sharpe balances **return vs risk**.

---

### 4) Buy & Hold Sharpe
**What it is:**  
Risk-adjusted performance of simply holding the stock.

**Key comparison:**  
- If Strategy Sharpe ≈ Buy & Hold Sharpe → model adds limited value  
- If Strategy Sharpe > Buy & Hold Sharpe → model improves risk efficiency  
- If Strategy Sharpe < Buy & Hold Sharpe → Buy & Hold dominates

---

### 5) Max Drawdown
**What it is:**  
Worst peak-to-trough loss during the period.

Example:
- `-0.15` = at some point the strategy was down **15% from its peak**

**Why it matters:**  
Drawdown measures **psychological pain and capital risk**.  
Two strategies with the same return can feel very different if one has much deeper drawdowns.

---

### 6) Avg Exposure
**What it is:**  
Average fraction of time the strategy is invested.

Examples:
- `1.0` → always invested (similar to Buy & Hold)
- `0.7` → invested ~70% of the time
- `0.3` → mostly in cash

**Why it matters:**  
Lower exposure with similar returns means **better timing** and less risk.

---

### 7) Avg Turnover
**What it is:**  
Average amount of position change per day.

- High turnover → frequent trading
- Low turnover → stable positions

**Why it matters:**  
High turnover increases:
- Transaction costs
- Slippage
- Real-world implementation risk

Lower turnover strategies are **more realistic**.

---

### 8) Transaction Costs (bps)
**What it is:**  
Cost paid when positions change (e.g., 5 bps = 0.05%).

**Why it matters:**  
Many ML strategies look good **before costs** but fail **after costs**.  
Including costs makes the backtest more realistic.

---

### 9) Equity Curve
**What it is:**  
Cumulative growth of $1 invested in:
- Strategy
- Buy & Hold

**Why it matters:**  
Shows:
- Stability of returns
- Crash behavior
- Recovery speed

A smoother equity curve is often preferable even if returns are slightly lower.

---

## ⚖️ How to Judge Strategy Quality (Summary)

A strategy is considered **meaningful** if:
- Strategy Sharpe ≥ Buy & Hold Sharpe
- Drawdown is lower or comparable
- Turnover is reasonable
- Exposure is controlled (not always 100%)

A strategy can still be valuable **even if total returns are similar**, as long as:
- Risk is lower
- Drawdowns are smaller
- Volatility is reduced

---

## ⚠️ Important Note
This project evaluates **next-day predictions**, which are inherently noisy.  
Small performance improvements are realistic and expected in short-horizon trading.

Beating Buy & Hold consistently on single stocks is extremely difficult, especially after transaction costs.

---

## 🛠️ Tech Stack
- **Python**
- **Streamlit** (web app)
- **yfinance** (market data)
- **Pandas / NumPy** (data processing)
- **Plotly** (interactive charts)
- **PyTorch** (LSTM model)
- **scikit-learn** (scaling + metrics)

---

## ▶️ How to run locally

1) Create and activate a virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```
2) Install dependencies
```bash
pip install -r requirements.txt
```
3) Run Streamlit
```bash
streamlit run app.py
```

---

## Future Improvements
If this were expanded further, improvements could include:

- Portfolio mode (rank top predicted tickers rather than single-stock backtests)

- External features (news sentiment, earnings surprises, fundamentals, macro)

- Better risk controls (volatility filters, stop-loss logic, position sizing)

- Model ensembling (LSTM + tree models)

## Disclaimer

This project is for educational purposes only and is not financial advice. Real markets include slippage, liquidity constraints, and event risk not fully captured in historical data.
