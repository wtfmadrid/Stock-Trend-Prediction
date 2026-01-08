import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Plotly (interactive charts)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# LSTM artifacts + inference
import json
import joblib
from pathlib import Path

import torch
import torch.nn as nn

st.set_page_config(page_title="Stock Trend & Risk Analyzer", layout="wide")


# WHY THIS APP STRUCTURE?
# - Multi-market (US/Canada/India) stock analyzer that is:
#   (1) interactive,
#   (2) finance-oriented (returns, risk, beta, drawdown),
#   (3) ML-ready + actually uses ML (LSTM inference + backtest).
#
# - We use yfinance directly (no CSV) so the app works just from a ticker.




# Market config

# WHY:
# Different markets use different ticker suffixes in Yahoo Finance.
# We map a dropdown market choice to a suffix and a benchmark ETF/index.
MARKETS = {
    "US (NYSE/NASDAQ)": {"suffix": "", "benchmark": "SPY", "currency_hint": "USD"},
    "Canada (TSX)": {"suffix": ".TO", "benchmark": "XIU.TO", "currency_hint": "CAD"},
    "India (NSE)": {"suffix": ".NS", "benchmark": "^NSEI", "currency_hint": "INR"},
}

DEFAULT_START = "2010-01-01"
DEFAULT_END = "2025-12-31"



# Helpers: ticker formatting + download

def format_ticker(symbol: str, market_name: str) -> str:
    """
    Convert a user-provided symbol into the correct Yahoo Finance ticker format
    for the chosen market.

    WHY:
    - US tickers: AAPL
    - Canada TSX tickers: RY.TO (so user can just type RY)
    - India NSE tickers: RELIANCE.NS (so user can just type RELIANCE)
    """
    symbol = symbol.strip().upper()
    suffix = MARKETS[market_name]["suffix"]

    # If user already provided suffix, keep it
    if suffix and symbol.endswith(suffix):
        return symbol

    # For US, allow symbols like BRK.B -> BRK-B (Yahoo often prefers '-')
    if market_name.startswith("US") and "." in symbol and "-" not in symbol:
        symbol = symbol.replace(".", "-")

    return symbol + suffix


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV price data from yfinance.

    WHY auto_adjust=True?
    - Adjusts historical prices for splits/dividends (consistent time series).
    - With auto_adjust=True, Yahoo/yfinance returns adjusted OHLC.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    df.rename(columns={"Date": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    return df



# Indicators + risk features
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators + risk metrics on top of OHLCV data.

    WHY these indicators/metrics?
    - Trend: MA100/MA200
    - Momentum: RSI
    - Trend momentum: MACD
    - Volatility: Bollinger Bands / ATR / rolling vol
    - Finance risk: returns / drawdown

    We ALSO add the exact feature columns needed by the LSTM pipeline:
    - ret_1, ret_5, ret_20, bb_width, atr_pct, target (next-day return)
    """
    out = df.copy()

    # If yfinance returned MultiIndex columns, flatten them
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    
    close = out["Close"].squeeze()


    # Moving averages (trend)
    out["MA100"] = close.rolling(100).mean()
    out["MA200"] = close.rolling(200).mean()

    # Returns (core finance feature)
    out["ret"] = close.pct_change()
    out["log_ret"] = np.log(close).diff()

    # Multi-horizon returns (useful features for ML)
    # WHY:
    # - ret: 1-day return
    # - ret_5: 5-day return (weekly-ish)
    # - ret_20: ~1-month return
    out["ret_1"] = out["ret"].shift(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_20"] = close.pct_change(20)

    # RSI (14): momentum
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    out["RSI14"] = 100 - (100 / (1 + rs))

    # MACD (12,26,9): trend momentum
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

    # Bollinger Bands (20, 2 std): volatility envelope
    m20 = close.rolling(20).mean()
    s20 = close.rolling(20).std()
    out["BB_mid"] = m20
    out["BB_up"] = m20 + 2 * s20
    out["BB_low"] = m20 - 2 * s20

    # Bollinger width feature (normalized)
    # It captures "volatility expansion vs compression" in a scale-free way.
    out["bb_width"] = (out["BB_up"] - out["BB_low"]) / out["BB_mid"]

    # ATR (14): "typical move size" (risk sizing)
    high = out["High"]
    low = out["Low"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()

    # ATR in % terms (scale-free volatility proxy)
    out["atr_pct"] = out["ATR14"] / close

    # Rolling volatility (annualized, based on daily returns)
    out["vol20"] = out["ret"].rolling(20).std() * np.sqrt(252)   # short-term
    out["vol60"] = out["ret"].rolling(60).std() * np.sqrt(252)   # medium-term
    out["vol252"] = out["ret"].rolling(252).std() * np.sqrt(252) # long-term

    # Drawdown (peak-to-trough %)
    roll_max = close.cummax()
    out["drawdown"] = (close / roll_max) - 1.0

    # Target for ML: next-day return
    # This matches the training target: predict tomorrow's return.
    out["target"] = out["Close"].pct_change().shift(-1) 

    return out


def compute_beta(asset_df: pd.DataFrame, bench_df: pd.DataFrame) -> float:
    
    # Compute beta vs benchmark using aligned daily returns.
    
    a = asset_df[["date", "ret"]].dropna()
    b = bench_df[["date", "ret"]].dropna()
    merged = pd.merge(a, b, on="date", suffixes=("_asset", "_bench"))

    if merged.shape[0] < 50:
        return float("nan")

    cov = np.cov(merged["ret_asset"], merged["ret_bench"])[0, 1]
    var = np.var(merged["ret_bench"])
    return float(cov / var) if var > 0 else float("nan")



# Plotting (INTERACTIVE Plotly)
def plot_candles_with_overlays(
    df: pd.DataFrame,
    title: str,
    show_ma100: bool,
    show_ma200: bool,
    show_bbands: bool,
    show_volume: bool
):
    """
    Finance-grade interactive chart:
    - Candlesticks (OHLC)
    - Optional overlays: MA100/MA200 + Bollinger Bands
    - Optional volume bars
    - Range slider + quick range selector buttons
    """
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
            specs=[[{"type": "candlestick"}],
                   [{"type": "bar"}]]
        )
    else:
        fig = make_subplots(
            rows=1, cols=1,
            shared_xaxes=True,
            specs=[[{"type": "candlestick"}]]
        )

    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candles"
        ),
        row=1, col=1
    )

    if show_ma100:
        fig.add_trace(go.Scatter(x=df["date"], y=df["MA100"], name="MA100"), row=1, col=1)
    if show_ma200:
        fig.add_trace(go.Scatter(x=df["date"], y=df["MA200"], name="MA200"), row=1, col=1)

    if show_bbands:
        fig.add_trace(go.Scatter(x=df["date"], y=df["BB_up"], name="BB Upper", line=dict(dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["BB_mid"], name="BB Mid", line=dict(dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["BB_low"], name="BB Lower", line=dict(dash="dot")), row=1, col=1)

    if show_volume:
        fig.add_trace(go.Bar(x=df["date"], y=df["Volume"], name="Volume"), row=2, col=1)

    # Layout 
    fig.update_layout(
        title=title,
        height=650 if show_volume else 520,
        margin=dict(l=10, r=10, t=45, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),

        xaxis=dict(
            type="date",
            rangeslider=dict(visible=False),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="Max"),
                ]
            ),
        ),
    ) 
    st.plotly_chart(fig, use_container_width=True)



def plot_rsi(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["RSI14"], name="RSI(14)"))
    fig.add_hline(y=70, line_dash="dash")
    fig.add_hline(y=30, line_dash="dash")
    fig.update_layout(
        title=title,
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(range=[0, 100]),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_macd(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["MACD"], name="MACD"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["MACD_signal"], name="Signal"))
    fig.add_trace(go.Bar(x=df["date"], y=df["MACD_hist"], name="Hist"))
    fig.update_layout(
        title=title,
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_drawdown(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["drawdown"], name="Drawdown"))
    fig.update_layout(
        title=title,
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_volatility(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["vol20"], name="Vol 20d (ann.)"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["vol60"], name="Vol 60d (ann.)"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["vol252"], name="Vol 252d (ann.)"))
    fig.update_layout(
        title=title,
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        yaxis_title="Annualized volatility"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_equity_curve(daily: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["strat_equity"], name="Strategy", mode ="lines", line=dict(color="#00C853", width=3)))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["bh_equity"], name="Buy & Hold", mode="lines", line=dict(color="#FF5252", width=3, dash="dash")))
    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=10, r=10, t=45, b=10),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)



# LSTM: load artifacts + define model
class LSTMRegressor(nn.Module):
    """
    Simple LSTM regressor (NO ticker embedding).
    This matches our updated training direction: general pattern learning without
    "only-known-tickers" limitation.
    """
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        yhat = self.head(last).squeeze(-1)
        return yhat


@st.cache_resource(show_spinner=False)
def load_lstm_artifacts():
    """
    Load model + scaler + features/config ONCE for the app.

    Why cache_resource:
    - Model/scaler are heavy-ish objects. We don't want to reload on each rerun.
    """
    art_dir = Path("D:\\Stock Trend Prediction\\artifacts")
    model_path = art_dir / "lstm_model.pt"
    scaler_path = art_dir / "lstm_scaler.pkl"
    feats_path = art_dir / "lstm_features.json"
    cfg_path = art_dir / "lstm_config.json"

    missing = [p.name for p in [model_path, scaler_path, feats_path, cfg_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifact(s) in ./artifacts: {missing}\n"
            f"Expected: lstm_model.pt, lstm_scaler.pkl, lstm_features.json, lstm_config.json"
        )

    features = json.loads(feats_path.read_text())
    cfg = json.loads(cfg_path.read_text())
    scaler = joblib.load(scaler_path)

    # Model hyperparams must match training (stored in config).
    hidden = int(cfg.get("hidden", 64))
    layers = int(cfg.get("layers", 2))
    dropout = float(cfg.get("dropout", 0.2))
    seq_len = int(cfg.get("seq_len", 60))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMRegressor(n_features=len(features), hidden=hidden, layers=layers, dropout=dropout).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, scaler, features, cfg, device, seq_len


def build_lstm_prediction_frame(df_feat: pd.DataFrame, features: list, seq_len: int, scaler) -> pd.DataFrame:
    """
    Create a per-day prediction dataset for a SINGLE ticker:
    - Requires df_feat includes 'date', 'target', and the model feature columns.
    - Returns a frame aligned to predictions with columns:
        ['date', 'pred', 'target']
    """
    d = df_feat.copy()
    d = d.sort_values("date").reset_index(drop=True)

    d["target"] = d["Close"].pct_change().shift(-1)

    d = d.dropna(subset=["target"] + features).reset_index(drop=True)
    if len(d) <= seq_len + 5:
        return pd.DataFrame()

    X = scaler.transform(d[features].values).astype(np.float32)

    # Build rolling sequences -> predict target at index i
    # (same shape logic as training: X[i-seq_len:i] predicts y at i)
    X_seq = []
    dates = []
    targets = []
    for i in range(seq_len, len(d)):
        X_seq.append(X[i - seq_len:i])
        dates.append(d.loc[i, "date"])
        targets.append(d.loc[i, "target"])

    out = pd.DataFrame({"date": dates, "target": targets})
    out["pred"] = np.nan  # fill after inference
    out["_Xseq"] = X_seq  # temporary
    return out


@torch.no_grad()
def infer_lstm(model, device: str, seq_batch: np.ndarray) -> np.ndarray:
    """
    Run LSTM inference on a batch of sequences.
    seq_batch: (N, T, F) float32
    """
    xb = torch.tensor(seq_batch, dtype=torch.float32, device=device)
    preds = model(xb).detach().cpu().numpy()
    return preds



# Backtest (model-agnostic): works for LSTM preds (and any preds array)
def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1.0
    return float(dd.min())


def sharpe_ratio(daily_returns: pd.Series, rf_daily: float = 0.0) -> float:
    r = daily_returns.dropna() - rf_daily
    if r.std() == 0 or len(r) < 2:
        return float("nan")
    return float((r.mean() / r.std()) * np.sqrt(252))


def run_backtest_long_cash(
    df_pred: pd.DataFrame,
    cost_bps: float = 5.0,
    signal_threshold: float = 0.0,
    rebalance_every: int = 1,
) -> pd.DataFrame:
    """
    Long/Cash strategy backtest for a SINGLE ticker.

    Rules:
    - Signal on rebalance days only:
        pos = 1 if pred > threshold else 0
      Between rebalance days, hold position.

    Costs:
    - cost_bps applies when position changes, like when we change from long -> cash or vice versa.

    Output:
    - daily equity curves for Strategy vs Buy&Hold.
    """
    bt = df_pred[["date", "pred", "target"]].dropna().sort_values("date").reset_index(drop=True)

    # Desired position every day (based on pred/threshold)
    bt["desired_pos"] = (bt["pred"] > signal_threshold).astype(int)

    # Rebalance control: only update pos every N days
    bt["reb_idx"] = np.arange(len(bt))
    reb_mask = (bt["reb_idx"] % rebalance_every) == 0

    bt["pos"] = np.nan
    bt.loc[reb_mask, "pos"] = bt.loc[reb_mask, "desired_pos"]
    bt["pos"] = bt["pos"].ffill().fillna(0)

    bt["pos_prev"] = bt["pos"].shift(1).fillna(0)
    bt["turnover"] = (bt["pos"] - bt["pos_prev"]).abs()

    cost = cost_bps / 10000.0
    bt["cost"] = bt["turnover"] * cost

    bt["strat_ret"] = bt["pos"] * bt["target"] - bt["cost"]
    bt["bh_ret"] = bt["target"]

    daily = bt[["date", "strat_ret", "bh_ret", "pos", "turnover"]].copy()
    daily["strat_equity"] = (1 + daily["strat_ret"]).cumprod()
    daily["bh_equity"] = (1 + daily["bh_ret"]).cumprod()

    daily["avg_exposure"] = daily["pos"]
    daily["avg_turnover"] = daily["turnover"]
    return daily


def summarize_backtest(daily: pd.DataFrame, label: str = "") -> dict:
    strat_total = float(daily["strat_equity"].iloc[-1] - 1)
    bh_total = float(daily["bh_equity"].iloc[-1] - 1)

    strat_sh = sharpe_ratio(daily["strat_ret"])
    bh_sh = sharpe_ratio(daily["bh_ret"])

    strat_mdd = max_drawdown(daily["strat_equity"])
    bh_mdd = max_drawdown(daily["bh_equity"])

    return {
        "Label": label,
        "Strategy Total Return": strat_total,
        "Buy&Hold Total Return": bh_total,
        "Strategy Sharpe": strat_sh,
        "Buy&Hold Sharpe": bh_sh,
        "Strategy Max Drawdown": strat_mdd,
        "Buy&Hold Max Drawdown": bh_mdd,
        "Avg Exposure": float(daily["avg_exposure"].mean()),
        "Avg Turnover": float(daily["avg_turnover"].mean()),
        "Days": int(len(daily)),
    }



# UI
st.title("📈 Stock Trend & Risk Analyzer (US / Canada / India)")

with st.sidebar:
    st.header("Inputs")
    market = st.selectbox("Market", list(MARKETS.keys()))
    mode = st.radio("Mode", ["Single stock", "Compare two stocks"], horizontal=False)

    start = st.text_input("Start date (YYYY-MM-DD)", DEFAULT_START)
    end = st.text_input("End date (YYYY-MM-DD)", DEFAULT_END)

    if mode == "Single stock":
        sym = st.text_input("Ticker symbol", "AAPL")
        ticker1 = format_ticker(sym, market)
        ticker2 = None
    else:
        sym1 = st.text_input("Ticker A", "AAPL")
        sym2 = st.text_input("Ticker B", "MSFT")
        ticker1 = format_ticker(sym1, market)
        ticker2 = format_ticker(sym2, market)

    st.subheader("Chart toggles")
    show_volume = st.checkbox("Show volume", value=True)
    show_ma100 = st.checkbox("Show MA100", value=True)
    show_ma200 = st.checkbox("Show MA200", value=True)
    show_bbands = st.checkbox("Show Bollinger Bands", value=True)

    st.subheader("LSTM backtest settings")
    # Why:
    # Our strategy performance can flip if costs are unrealistic.
    # Exposing cost as a slider makes the app feel finance grade and realistic.
    cost_bps = st.slider("Transaction cost (bps)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
    signal_threshold = st.selectbox("Signal threshold (pred return)", [0.0, 0.0005, 0.001, 0.002], index=0)
    rebalance_every = st.selectbox("Rebalance frequency (days)", [1, 5, 10, 21], index=0)

    st.caption(f"Benchmark used for beta: **{MARKETS[market]['benchmark']}**")
    st.caption(f"Currency hint: **{MARKETS[market]['currency_hint']}**")
    run = st.button("Run analysis", type="primary")


if not run:
    st.info("Pick a market + ticker(s), then click **Run analysis**.")
    st.stop()

bench_ticker = MARKETS[market]["benchmark"]

# Load data
df1_raw = load_price_data(ticker1, start, end)
if df1_raw.empty:
    st.error(f"No data returned for {ticker1}. Check ticker/market suffix.")
    st.stop()

df1 = add_indicators(df1_raw)

df_bench_raw = load_price_data(bench_ticker, start, end)
df_bench = add_indicators(df_bench_raw) if not df_bench_raw.empty else pd.DataFrame()

beta1 = compute_beta(df1, df_bench) if not df_bench.empty else float("nan")

# Summary cards
last = df1.dropna().iloc[-1]
colA, colB, colC, colD = st.columns(4)
colA.metric("Ticker", ticker1)
colB.metric("Last Close", f"{last['Close']:.2f}")
colC.metric("Annualized Vol (20d)", f"{(last['vol20'] * 100):.1f}%")
colD.metric("Beta vs Benchmark", "N/A" if np.isnan(beta1) else f"{beta1:.2f}")

# Main interactive candlestick chart
plot_candles_with_overlays(
    df1.dropna(),
    title=f"{ticker1} Candlestick (auto-adjusted) + Overlays",
    show_ma100=show_ma100,
    show_ma200=show_ma200,
    show_bbands=show_bbands,
    show_volume=show_volume
)

# Secondary analytics
left, right = st.columns(2)
with left:
    plot_rsi(df1, f"{ticker1} RSI(14)")
    plot_macd(df1, f"{ticker1} MACD (12,26,9)")
with right:
    plot_volatility(df1, f"{ticker1} Rolling Volatility (annualized)")
    plot_drawdown(df1, f"{ticker1} Drawdown")


# LSTM section (single stock)
if mode == "Single stock":
    st.subheader("🤖 LSTM Forecast + Backtest")

    try:
        model, scaler, lstm_features, lstm_cfg, device, seq_len = load_lstm_artifacts()
    except Exception as e:
        st.error(
            "LSTM artifacts not loaded. Make sure you have ./artifacts/"
            " with lstm_model.pt, lstm_scaler.pkl, lstm_features.json, lstm_config.json.\n\n"
            f"Error: {e}"
        )
        st.stop()

    # Build prediction dataset aligned to the model's expected features
    df_pred = build_lstm_prediction_frame(df1, lstm_features, seq_len, scaler)
    if df_pred.empty:
        st.warning(
            f"Not enough clean rows to run LSTM (need > {seq_len} rows with all features). "
            "Try expanding your date range."
        )
    else:
        # Inference in one batch for speed
        seq_batch = np.stack(df_pred["_Xseq"].values).astype(np.float32)
        preds = infer_lstm(model, device, seq_batch)
        df_pred["pred"] = preds
        df_pred = df_pred.drop(columns=["_Xseq"])

        # ---- Today/next-day forecast (latest available prediction) ----
        latest = df_pred.iloc[-1]
        pred_ret = float(latest["pred"])
        pred_pct = pred_ret * 100

        # Signal logic (matches the backtest logic)
        signal = "LONG" if pred_ret > float(signal_threshold) else "CASH"
        signal_color = "green" if signal == "LONG" else "gray"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Seq length", str(seq_len))
        c2.metric("Predicted next-day return", f"{pred_pct:+.3f}%")
        c3.metric("Signal", signal)
        # A simple “confidence proxy”: bigger |pred| = stronger conviction (still not true confidence)
        c4.metric("Conviction (|pred|)", f"{abs(pred_ret)*100:.3f}%")

        st.caption(
            "Model predicts **next-day return** (not price). Strategy converts predictions to a Long/Cash signal, "
            "then evaluates it with transaction costs."
        )

        # Backtest on this ticker
        daily = run_backtest_long_cash(
            df_pred=df_pred,
            cost_bps=float(cost_bps),
            signal_threshold=float(signal_threshold),
            rebalance_every=int(rebalance_every),
        )
        summary = summarize_backtest(daily, label=f"LSTM rb={rebalance_every}, th={signal_threshold}, cost={cost_bps}bps")
        st.dataframe(pd.DataFrame([summary]), use_container_width=True)

        plot_equity_curve(daily, title="Equity Curve — LSTM Strategy vs Buy & Hold")

        # show prediction vs target scatter / diagnostics
        with st.expander("Diagnostics"):
            da = (np.sign(df_pred["pred"].values) == np.sign(df_pred["target"].values)).mean()
            ic = np.corrcoef(df_pred["pred"].values, df_pred["target"].values)[0, 1]

            st.write(f"Directional Accuracy (DA): **{da:.3f}**")
            st.write(f"Information Coefficient (corr(pred, target)): **{ic:.3f}**")

            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=df_pred["pred"], y=df_pred["target"],
                mode="markers", name="pred vs target", opacity=0.5
            ))
            fig_sc.update_layout(
                title="Predicted return vs Actual next-day return",
                height=380,
                margin=dict(l=10, r=10, t=45, b=10),
                xaxis_title="Predicted return",
                yaxis_title="Actual next-day return",
            )
            st.plotly_chart(fig_sc, use_container_width=True)

else:
    st.info("LSTM Forecast + Backtest is shown in **Single stock** mode (to keep inference/backtest clear and fast).")


# Compare mode (unchanged, analytics only)
if ticker2:
    df2_raw = load_price_data(ticker2, start, end)
    if df2_raw.empty:
        st.error(f"No data returned for {ticker2}. Check ticker/market suffix.")
        st.stop()

    df2 = add_indicators(df2_raw)
    beta2 = compute_beta(df2, df_bench) if not df_bench.empty else float("nan")

    st.subheader("🔁 Comparison")

    merged = pd.merge(
        df1[["date", "Close"]],
        df2[["date", "Close"]],
        on="date",
        suffixes=(f"_{ticker1}", f"_{ticker2}")
    ).dropna()

    merged[f"Norm_{ticker1}"] = merged[f"Close_{ticker1}"] / merged[f"Close_{ticker1}"].iloc[0] * 100
    merged[f"Norm_{ticker2}"] = merged[f"Close_{ticker2}"] / merged[f"Close_{ticker2}"].iloc[0] * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["date"], y=merged[f"Norm_{ticker1}"], name=f"{ticker1} (start=100)"))
    fig.add_trace(go.Scatter(x=merged["date"], y=merged[f"Norm_{ticker2}"], name=f"{ticker2} (start=100)"))
    fig.update_layout(
        title="Normalized performance (start=100)",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Rolling correlation of returns
    r1 = df1[["date", "ret"]].rename(columns={"ret": "ret1"})
    r2 = df2[["date", "ret"]].rename(columns={"ret": "ret2"})
    corr_df = pd.merge(r1, r2, on="date").dropna()
    corr_df["roll_corr_60"] = corr_df["ret1"].rolling(60).corr(corr_df["ret2"])

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=corr_df["date"], y=corr_df["roll_corr_60"], name="60d Rolling Corr"))
    fig2.update_layout(
        title="Rolling correlation (60d) of daily returns",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified"
    )
    st.plotly_chart(fig2, use_container_width=True)

    def summary_row(dfx: pd.DataFrame, label: str, beta: float):
        d = dfx.dropna()
        lastp = d.iloc[-1]["Close"]
        vol20 = d.iloc[-1]["vol20"]
        mdd = d["drawdown"].min()

        rets = d["ret"].dropna()
        r_yr = (1 + rets).prod() ** (252 / max(1, rets.shape[0])) - 1

        return {
            "Ticker": label,
            "Last Close": round(float(lastp), 2),
            "Ann. Vol (20d)": round(float(vol20) * 100, 1),
            "Max Drawdown": round(float(mdd) * 100, 1),
            "Approx Ann. Return": round(float(r_yr) * 100, 1),
            "Beta": None if np.isnan(beta) else round(float(beta), 2),
        }

    table = pd.DataFrame([
        summary_row(df1, ticker1, beta1),
        summary_row(df2, ticker2, beta2),
    ])
    st.dataframe(table, use_container_width=True)

st.divider()
st.caption(
    "Note: Data is **auto-adjusted** (splits/dividends) for cleaner analysis and to avoid fake jumps in indicators/ML."
)
