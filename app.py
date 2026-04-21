"""
Quantitative Investment Toolkit
--------------------------------
A multi-module decision-support application integrating financial theory,
real market data, and quantitative modeling.

Modules:
  1. Portfolio Optimization  - Markowitz Mean-Variance with Efficient Frontier
  2. Risk Analytics          - VaR (Historical & Parametric), CVaR, Lower Partial Std Dev, Sortino
  3. Valuation Percentile    - Historical P/E percentile analysis (rich/cheap signal)
  4. Factor Model / CAPM     - Beta, Alpha, R-squared via OLS regression
  5. Monte Carlo Simulation  - Forward portfolio path simulation under GBM

Author: Student project for Investment and Finance course
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy import stats
from datetime import date, timedelta

# ----------------------------------------------------------------------------- 
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quantitative Investment Toolkit",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light custom styling
st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    h1, h2, h3 {color: #0e1117;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# DATA FETCHING (cached)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers, start, end):
    """Download adjusted close prices for a list of tickers from Yahoo Finance.

    Robust to the several column layouts yfinance can return across versions.
    Returns a DataFrame with one column per ticker (ticker symbol as column name).
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = [t.upper() for t in tickers]

    try:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    if data is None or data.empty:
        return pd.DataFrame()

    close = pd.DataFrame(index=data.index)

    if isinstance(data.columns, pd.MultiIndex):
        # Two possible layouts: (ticker, field) OR (field, ticker).
        lvl0 = set(data.columns.get_level_values(0))
        lvl1 = set(data.columns.get_level_values(1))
        # Prefer auto-adjusted "Close"; fall back to "Adj Close" if present
        for t in tickers:
            series = None
            if t in lvl0:
                # (ticker, field)
                sub = data[t]
                for field in ("Close", "Adj Close"):
                    if field in sub.columns:
                        series = sub[field]
                        break
            elif t in lvl1:
                # (field, ticker)
                for field in ("Close", "Adj Close"):
                    if (field, t) in data.columns:
                        series = data[(field, t)]
                        break
            if series is not None:
                close[t] = series
    else:
        # Flat columns — single ticker case
        for field in ("Close", "Adj Close"):
            if field in data.columns:
                close[tickers[0]] = data[field]
                break

    close = close.dropna(how="all").ffill().dropna()
    return close


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_trailing_pe(ticker, start, end):
    """
    Build an approximate historical trailing P/E series using price / trailing EPS.
    Trailing EPS is estimated from the firm's quarterly net income history.
    Because yfinance exposes only limited quarterly history (~4 quarters), we
    back-fill using the earliest known TTM EPS and scale when newer data arrives.

    Returns a daily Series of estimated P/E ratios, or None if unavailable.
    """
    try:
        t = yf.Ticker(ticker)
        prices = fetch_prices(ticker, start, end)
        if prices.empty:
            return None
        price_series = prices[ticker]

        # Try quarterly income statement first
        qis = t.quarterly_income_stmt
        if qis is not None and not qis.empty and "Net Income" in qis.index:
            ni = qis.loc["Net Income"].dropna().sort_index()
        else:
            ni = None

        shares = None
        info = {}
        try:
            info = t.info or {}
            shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        except Exception:
            info = {}
            shares = None

        eps_series = None
        if ni is not None and shares and len(ni) >= 1:
            # TTM net income at each quarter end (rolling 4q sum)
            ttm = ni.rolling(4).sum().dropna()
            if len(ttm) == 0 and len(ni) > 0:
                # not enough history for rolling 4q; use annualized last quarter * 4
                ttm = ni * 4
            ttm_eps = ttm / shares
            # Align to daily price index (forward-fill)
            ttm_eps.index = pd.to_datetime(ttm_eps.index)
            eps_series = ttm_eps.reindex(price_series.index, method="ffill")

        # Fallback: use current trailing EPS from info as constant-level baseline
        if eps_series is None or eps_series.dropna().empty:
            trailing_eps = info.get("trailingEps")
            if trailing_eps and trailing_eps > 0:
                eps_series = pd.Series(trailing_eps, index=price_series.index)
            else:
                return None

        pe = price_series / eps_series
        pe = pe.replace([np.inf, -np.inf], np.nan).dropna()
        # Filter out obvious nonsense (negative or extreme)
        pe = pe[(pe > 0) & (pe < 500)]
        if pe.empty:
            return None
        return pe
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info(ticker):
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


# -----------------------------------------------------------------------------
# CORE QUANT FUNCTIONS
# -----------------------------------------------------------------------------
def annualize_return(daily_returns, periods=252):
    return daily_returns.mean() * periods

def annualize_vol(daily_returns, periods=252):
    return daily_returns.std() * np.sqrt(periods)

def portfolio_stats(weights, mu, cov, rf=0.0):
    """Return annualized (return, vol, Sharpe) for given weights (mu & cov already annualized)."""
    w = np.asarray(weights)
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe

def neg_sharpe(weights, mu, cov, rf):
    r, v, s = portfolio_stats(weights, mu, cov, rf)
    return -s if np.isfinite(s) else 1e6

def portfolio_variance(weights, cov):
    w = np.asarray(weights)
    return float(w @ cov @ w)

def optimize_min_var(mu, cov, allow_short=False):
    n = len(mu)
    bounds = [(-1.0, 1.0) if allow_short else (0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.repeat(1.0 / n, n)
    res = minimize(portfolio_variance, x0, args=(cov,), method="SLSQP",
                   bounds=bounds, constraints=cons)
    return res.x

def optimize_max_sharpe(mu, cov, rf=0.0, allow_short=False):
    n = len(mu)
    bounds = [(-1.0, 1.0) if allow_short else (0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.repeat(1.0 / n, n)
    res = minimize(neg_sharpe, x0, args=(mu, cov, rf), method="SLSQP",
                   bounds=bounds, constraints=cons)
    return res.x

def efficient_frontier(mu, cov, n_points=40, allow_short=False):
    """Solve a series of min-variance problems subject to a target return."""
    n = len(mu)
    target_rets = np.linspace(mu.min(), mu.max(), n_points)
    bounds = [(-1.0, 1.0) if allow_short else (0.0, 1.0)] * n
    frontier_vol = []
    frontier_ret = []
    frontier_w = []
    x0 = np.repeat(1.0 / n, n)
    for target in target_rets:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        ]
        res = minimize(portfolio_variance, x0, args=(cov,), method="SLSQP",
                       bounds=bounds, constraints=cons)
        if res.success:
            r, v, _ = portfolio_stats(res.x, mu, cov)
            frontier_ret.append(r)
            frontier_vol.append(v)
            frontier_w.append(res.x)
    return np.array(frontier_ret), np.array(frontier_vol), frontier_w


def historical_var(returns, alpha=0.05):
    """Historical VaR at given confidence level. Returns a positive number = loss magnitude."""
    return -np.percentile(returns, 100 * alpha)

def historical_cvar(returns, alpha=0.05):
    """Historical Conditional VaR (Expected Shortfall)."""
    var = -np.percentile(returns, 100 * alpha)
    tail = returns[returns <= -var]
    return -tail.mean() if len(tail) > 0 else np.nan

def parametric_var(returns, alpha=0.05):
    """Parametric (Gaussian) VaR."""
    mu, sigma = returns.mean(), returns.std()
    return -(mu + sigma * stats.norm.ppf(alpha))

def lower_partial_std(returns, mar=0.0):
    """Lower Partial Standard Deviation relative to a Minimum Acceptable Return (MAR).
    This is the downside-risk analogue to standard deviation, used in the Sortino ratio."""
    downside = np.minimum(returns - mar, 0.0)
    return np.sqrt(np.mean(downside ** 2))

def sortino_ratio(returns, mar=0.0, periods=252):
    lpstd = lower_partial_std(returns, mar)
    if lpstd == 0:
        return np.nan
    excess = returns.mean() - mar
    return (excess / lpstd) * np.sqrt(periods)


def capm_regression(asset_returns, market_returns, rf_daily=0.0):
    """Run OLS: (r_i - rf) = alpha + beta * (r_m - rf) + eps. Returns beta, alpha (daily), R^2, t-stats."""
    y = asset_returns - rf_daily
    x = market_returns - rf_daily
    df = pd.concat([y, x], axis=1).dropna()
    df.columns = ["y", "x"]
    if len(df) < 30:
        return None
    X = np.column_stack([np.ones(len(df)), df["x"].values])
    yv = df["y"].values
    beta_hat, *_ = np.linalg.lstsq(X, yv, rcond=None)
    alpha, beta = beta_hat[0], beta_hat[1]
    y_pred = X @ beta_hat
    resid = yv - y_pred
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((yv - yv.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n, k = len(yv), 2
    sigma2 = ss_res / (n - k)
    cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov_beta))
    t_alpha = alpha / se[0] if se[0] > 0 else np.nan
    t_beta = beta / se[1] if se[1] > 0 else np.nan
    return {
        "alpha_daily": alpha,
        "alpha_annual": alpha * 252,
        "beta": beta,
        "r_squared": r2,
        "t_alpha": t_alpha,
        "t_beta": t_beta,
        "n_obs": n,
        "residual_vol_annual": np.sqrt(sigma2) * np.sqrt(252),
    }


def monte_carlo_paths(mu_annual, sigma_annual, S0, horizon_days=252, n_paths=1000, seed=42):
    """Simulate Geometric Brownian Motion paths."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    drift = (mu_annual - 0.5 * sigma_annual ** 2) * dt
    diffusion = sigma_annual * np.sqrt(dt)
    shocks = rng.standard_normal((horizon_days, n_paths))
    log_returns = drift + diffusion * shocks
    log_paths = np.cumsum(log_returns, axis=0)
    paths = S0 * np.exp(log_paths)
    paths = np.vstack([np.repeat(S0, n_paths), paths])
    return paths


# -----------------------------------------------------------------------------
# SIDEBAR - GLOBAL INPUTS
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Global Settings")
st.sidebar.caption("Inputs here apply across modules.")

default_end = date.today()
default_start = default_end - timedelta(days=365 * 5)

start_date = st.sidebar.date_input("Start date", value=default_start, max_value=default_end)
end_date = st.sidebar.date_input("End date", value=default_end, max_value=default_end)

rf_annual = st.sidebar.number_input(
    "Risk-free rate (annual, %)", min_value=0.0, max_value=20.0, value=4.5, step=0.25,
    help="Used for Sharpe, Sortino, CAPM. A common proxy is the 3-month T-Bill yield.",
) / 100.0

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data source: Yahoo Finance via `yfinance`. Prices are auto-adjusted for "
    "splits/dividends. Cached for 1 hour."
)

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.title("📈 Quantitative Investment Toolkit")
st.markdown(
    "A decision-support application that integrates **portfolio theory**, "
    "**risk models**, **valuation analysis**, and **factor models** on real "
    "market data."
)

tabs = st.tabs([
    "🏠 Overview",
    "📊 Portfolio Optimization",
    "⚠️ Risk Analytics",
    "💰 Valuation Percentile",
    "📐 CAPM / Factor Model",
    "🎲 Monte Carlo",
])

# =============================================================================
# TAB 0: OVERVIEW
# =============================================================================
with tabs[0]:
    st.header("What this app does")
    st.markdown(
        """
This toolkit lets you apply classical quantitative-finance methods to real
equity data. Pick a module from the tabs above:

- **Portfolio Optimization** — Build a mean-variance efficient frontier, find
  the Max-Sharpe and Min-Variance portfolios, and compare to an equal-weight
  benchmark.
- **Risk Analytics** — Compute Historical VaR, Parametric VaR, CVaR (Expected
  Shortfall), Lower Partial Standard Deviation, and the Sortino ratio for any
  asset or combination.
- **Valuation Percentile** — Rank today's P/E against the stock's own history.
  *Example reading:* "The current P/E is in the 95th percentile → richer than
  95% of the firm's history."
- **CAPM / Factor Model** — Regress an asset on a market benchmark to estimate
  β, α, and R². Outputs t-statistics so you can judge significance.
- **Monte Carlo** — Simulate thousands of forward paths under Geometric
  Brownian Motion; read off terminal-value distributions and loss probabilities.

### Data
All prices are auto-adjusted daily closes from **Yahoo Finance** (`yfinance`).
Set the date window and risk-free rate in the sidebar. Caching (1h TTL) keeps
the app responsive.

### How to use
1. Set your global date range and risk-free rate in the sidebar.
2. Open any module tab, enter ticker(s), and click the action button.
3. Outputs update dynamically: tables, interactive Plotly charts, and
   economic interpretation.
        """
    )

    st.subheader("Quick start — try these tickers")
    st.code(
        "Portfolio:   AAPL, MSFT, GOOGL, AMZN, JPM, XOM, JNJ, PG, KO, NVDA\n"
        "Valuation:   AAPL   (vs. its own P/E history)\n"
        "CAPM:        TSLA   with market = SPY\n"
        "Monte Carlo: SPY    (1-year forward simulation)",
        language="text",
    )


# =============================================================================
# TAB 1: PORTFOLIO OPTIMIZATION
# =============================================================================
with tabs[1]:
    st.header("Markowitz Mean-Variance Portfolio Optimization")
    st.markdown(
        "Classical mean-variance theory (Markowitz, 1952): given a set of "
        "assets with expected returns **μ** and covariance **Σ**, every "
        "efficient portfolio minimizes variance for a target return. We "
        "solve it numerically with SLSQP."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        tickers_input = st.text_input(
            "Tickers (comma-separated)",
            value="AAPL, MSFT, GOOGL, AMZN, JPM, XOM, JNJ, PG",
            key="port_tickers",
        )
    with col2:
        allow_short = st.checkbox("Allow short selling", value=False,
                                  help="If off, weights are constrained to [0, 1].")

    run_port = st.button("🚀 Run Optimization", type="primary", key="run_port")

    if run_port:
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        if len(tickers) < 2:
            st.error("Please enter at least 2 tickers.")
        else:
            with st.spinner("Downloading prices..."):
                prices = fetch_prices(tickers, start_date, end_date)

            if prices.empty or prices.shape[1] < 2:
                st.error("Could not retrieve sufficient price data. Check tickers or date range.")
            else:
                # Keep tickers we actually got
                tickers = list(prices.columns)
                rets = prices.pct_change().dropna()
                mu = annualize_return(rets).values
                cov = (rets.cov() * 252).values

                # Optimize
                w_ms = optimize_max_sharpe(mu, cov, rf=rf_annual, allow_short=allow_short)
                w_mv = optimize_min_var(mu, cov, allow_short=allow_short)
                w_eq = np.repeat(1.0 / len(tickers), len(tickers))

                # Frontier
                with st.spinner("Computing efficient frontier..."):
                    f_ret, f_vol, _ = efficient_frontier(mu, cov, n_points=40,
                                                         allow_short=allow_short)

                # Individual asset stats
                asset_vol = np.sqrt(np.diag(cov))

                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=f_vol, y=f_ret, mode="lines", name="Efficient Frontier",
                    line=dict(color="#1f77b4", width=3),
                ))
                fig.add_trace(go.Scatter(
                    x=asset_vol, y=mu, mode="markers+text",
                    text=tickers, textposition="top center",
                    name="Individual Assets",
                    marker=dict(size=9, color="#7f7f7f"),
                ))
                r_ms, v_ms, s_ms = portfolio_stats(w_ms, mu, cov, rf_annual)
                r_mv, v_mv, s_mv = portfolio_stats(w_mv, mu, cov, rf_annual)
                r_eq, v_eq, s_eq = portfolio_stats(w_eq, mu, cov, rf_annual)
                fig.add_trace(go.Scatter(
                    x=[v_ms], y=[r_ms], mode="markers", name="Max Sharpe",
                    marker=dict(size=16, symbol="star", color="gold",
                                line=dict(color="black", width=1)),
                ))
                fig.add_trace(go.Scatter(
                    x=[v_mv], y=[r_mv], mode="markers", name="Min Variance",
                    marker=dict(size=14, symbol="diamond", color="#2ca02c",
                                line=dict(color="black", width=1)),
                ))
                fig.add_trace(go.Scatter(
                    x=[v_eq], y=[r_eq], mode="markers", name="Equal Weight",
                    marker=dict(size=12, symbol="circle", color="#d62728",
                                line=dict(color="black", width=1)),
                ))
                # Capital Market Line from risk-free rate through max-Sharpe point
                if v_ms > 0:
                    cml_x = np.linspace(0, max(f_vol) * 1.15, 50)
                    cml_y = rf_annual + (r_ms - rf_annual) / v_ms * cml_x
                    fig.add_trace(go.Scatter(
                        x=cml_x, y=cml_y, mode="lines",
                        name="Capital Market Line",
                        line=dict(color="orange", dash="dash", width=1.5),
                    ))

                fig.update_layout(
                    title="Efficient Frontier",
                    xaxis_title="Annualized Volatility (σ)",
                    yaxis_title="Annualized Expected Return (μ)",
                    xaxis_tickformat=".1%",
                    yaxis_tickformat=".1%",
                    height=520,
                    hovermode="closest",
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Summary table of the three portfolios
                summary = pd.DataFrame({
                    "Portfolio": ["Max Sharpe", "Min Variance", "Equal Weight"],
                    "Return":    [r_ms, r_mv, r_eq],
                    "Volatility":[v_ms, v_mv, v_eq],
                    "Sharpe":    [s_ms, s_mv, s_eq],
                }).set_index("Portfolio")
                st.subheader("Portfolio comparison")
                st.dataframe(
                    summary.style.format({"Return": "{:.2%}", "Volatility": "{:.2%}",
                                          "Sharpe": "{:.3f}"}),
                    use_container_width=True,
                )

                # Weight tables
                st.subheader("Optimal weights")
                wdf = pd.DataFrame({
                    "Ticker": tickers,
                    "Max Sharpe": w_ms,
                    "Min Variance": w_mv,
                    "Equal Weight": w_eq,
                }).set_index("Ticker")
                st.dataframe(
                 wdf.style.format("{:.2%}"),   
                    use_container_width=True,
                )

                # Weight bar chart
                fig2 = go.Figure()
                for col in ["Max Sharpe", "Min Variance", "Equal Weight"]:
                    fig2.add_trace(go.Bar(name=col, x=wdf.index, y=wdf[col]))
                fig2.update_layout(
                    barmode="group",
                    title="Portfolio weights",
                    yaxis_tickformat=".0%",
                    height=400,
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Economic interpretation
                st.subheader("Economic interpretation")
                concentrated = wdf["Max Sharpe"].abs().max()
                interp = (
                    f"The **Max Sharpe portfolio** achieves an expected return of "
                    f"**{r_ms:.2%}** with volatility **{v_ms:.2%}**, giving a "
                    f"Sharpe ratio of **{s_ms:.2f}** (risk-free = {rf_annual:.2%}). "
                )
                if concentrated > 0.5:
                    interp += (
                        f"Note that the optimizer concentrates > {concentrated:.0%} in a "
                        "single name — a well-known limitation of mean-variance "
                        "optimization when expected returns are estimated from short "
                        "historical samples (the *error-maximization* problem of "
                        "Michaud, 1989). Consider using shrinkage estimators, "
                        "resampling, or adding position-size constraints."
                    )
                else:
                    interp += (
                        "The optimal portfolio is reasonably diversified across the asset "
                        "set, suggesting the historical covariance structure provides "
                        "meaningful diversification benefits."
                    )
                st.info(interp)


# =============================================================================
# TAB 2: RISK ANALYTICS
# =============================================================================
with tabs[2]:
    st.header("Risk Analytics: VaR, CVaR, and Downside-Risk Measures")
    st.markdown(
        "**Value-at-Risk (VaR)** answers: *what is the worst loss, over a given "
        "horizon, at a given confidence level?* **CVaR** (Expected Shortfall) "
        "averages losses beyond the VaR cutoff. **Lower Partial Standard "
        "Deviation** (LPSD) measures only *downside* dispersion relative to a "
        "Minimum Acceptable Return (MAR) and drives the **Sortino ratio**."
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        risk_tickers = st.text_input(
            "Tickers (comma-separated). Equal weights used if multiple.",
            value="SPY",
            key="risk_tickers",
        )
    with col2:
        conf = st.select_slider(
            "Confidence level", options=[0.90, 0.95, 0.975, 0.99], value=0.95,
            key="risk_conf",
        )
    with col3:
        mar_annual = st.number_input(
            "MAR (annual, %)", min_value=-10.0, max_value=20.0,
            value=float(rf_annual * 100), step=0.5,
            help="Minimum Acceptable Return for LPSD / Sortino. Defaults to r_f.",
        ) / 100.0

    run_risk = st.button("🚀 Run Risk Analysis", type="primary", key="run_risk")

    if run_risk:
        tickers = [t.strip().upper() for t in risk_tickers.split(",") if t.strip()]
        with st.spinner("Fetching data..."):
            prices = fetch_prices(tickers, start_date, end_date)

        if prices.empty:
            st.error("No data retrieved.")
        else:
            rets = prices.pct_change().dropna()
            if rets.shape[1] > 1:
                w = np.repeat(1.0 / rets.shape[1], rets.shape[1])
                port_ret = rets @ w
                label = f"Equal-weight portfolio of {len(tickers)} assets"
            else:
                port_ret = rets.iloc[:, 0]
                label = tickers[0]

            alpha = 1 - conf
            mar_daily = (1 + mar_annual) ** (1 / 252) - 1

            var_hist = historical_var(port_ret, alpha=alpha)
            var_param = parametric_var(port_ret, alpha=alpha)
            cvar = historical_cvar(port_ret, alpha=alpha)
            lpsd_d = lower_partial_std(port_ret, mar=mar_daily)
            lpsd_a = lpsd_d * np.sqrt(252)
            sortino = sortino_ratio(port_ret, mar=mar_daily)
            ann_vol = port_ret.std() * np.sqrt(252)
            ann_ret = port_ret.mean() * 252
            sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else np.nan

            st.subheader(f"Results — {label}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ann. Return", f"{ann_ret:.2%}")
            c2.metric("Ann. Volatility", f"{ann_vol:.2%}")
            c3.metric("Sharpe", f"{sharpe:.3f}")
            c4.metric("Sortino", f"{sortino:.3f}")

            st.markdown("#### Daily loss measures")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Historical VaR ({conf:.0%})", f"{var_hist:.2%}")
            c2.metric(f"Parametric VaR ({conf:.0%})", f"{var_param:.2%}")
            c3.metric(f"CVaR / Exp. Shortfall ({conf:.0%})", f"{cvar:.2%}")
            c4.metric("LPSD (annualized)", f"{lpsd_a:.2%}")

            # Histogram with VaR / CVaR markers
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=port_ret, nbinsx=80, name="Daily returns",
                marker_color="#1f77b4", opacity=0.75,
            ))
            fig.add_vline(x=-var_hist, line_dash="dash", line_color="red",
                          annotation_text=f"Hist VaR {conf:.0%} = {var_hist:.2%}",
                          annotation_position="top")
            fig.add_vline(x=-cvar, line_dash="dot", line_color="darkred",
                          annotation_text=f"CVaR = {cvar:.2%}",
                          annotation_position="bottom")
            fig.update_layout(
                title="Distribution of daily returns",
                xaxis_title="Daily return",
                yaxis_title="Frequency",
                xaxis_tickformat=".1%",
                height=420, bargap=0.02,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Cumulative equity curve and drawdown
            eq = (1 + port_ret).cumprod()
            rolling_max = eq.cummax()
            dd = eq / rolling_max - 1

            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 subplot_titles=("Cumulative return", "Drawdown"),
                                 row_heights=[0.6, 0.4], vertical_spacing=0.08)
            fig2.add_trace(go.Scatter(x=eq.index, y=eq, mode="lines",
                                      name="Equity", line=dict(color="#1f77b4")),
                           row=1, col=1)
            fig2.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines",
                                      name="Drawdown", fill="tozeroy",
                                      line=dict(color="#d62728")),
                           row=2, col=1)
            fig2.update_yaxes(tickformat=".1%", row=2, col=1)
            fig2.update_layout(height=520, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

            max_dd = dd.min()
            st.info(
                f"**Economic reading.** A {conf:.0%} historical VaR of "
                f"{var_hist:.2%} means that, historically, daily losses exceeded "
                f"{var_hist:.2%} on {alpha:.0%} of trading days. The CVaR of "
                f"{cvar:.2%} tells you the *average* loss on those bad days — "
                f"the conditional tail. Parametric VaR ({var_param:.2%}) assumes "
                "returns are Gaussian; when it differs materially from the "
                "historical number, fat tails are present. The worst historical "
                f"drawdown over this window was **{max_dd:.2%}**. "
                f"The Sortino ratio ({sortino:.2f}) uses LPSD rather than full "
                "volatility, rewarding managers who avoid downside but not those "
                "who simply have high upside volatility."
            )


# =============================================================================
# TAB 3: VALUATION PERCENTILE
# =============================================================================
with tabs[3]:
    st.header("Valuation Percentile: P/E vs. Own History")
    st.markdown(
        "Rank today's P/E ratio against the stock's own historical distribution. "
        "A reading in the **95th percentile** means the stock is trading at a "
        "richer multiple than on 95% of historical days — a classic "
        "comparable-valuation signal."
    )
    st.caption(
        "*Note:* Historical TTM EPS is reconstructed from quarterly net income "
        "disclosures. Yahoo Finance provides only a limited history of fundamentals "
        "(~3–5 years), so the percentile window depends on data availability."
    )

    val_ticker = st.text_input("Ticker", value="AAPL", key="val_ticker").strip().upper()
    run_val = st.button("🚀 Compute Valuation Percentile", type="primary", key="run_val")

    if run_val and val_ticker:
        with st.spinner("Fetching fundamentals and prices..."):
            pe = fetch_trailing_pe(val_ticker, start_date, end_date)
            info = fetch_info(val_ticker)

        if pe is None or pe.empty:
            st.error(
                f"Could not build a historical P/E series for **{val_ticker}**. "
                "This usually means quarterly net income or share-count data was "
                "not available. Try a large-cap US equity (e.g., AAPL, MSFT, JPM, KO)."
            )
        else:
            current_pe = float(pe.iloc[-1])
            pct_rank = float((pe <= current_pe).mean())
            mean_pe = float(pe.mean())
            median_pe = float(pe.median())
            min_pe, max_pe = float(pe.min()), float(pe.max())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current P/E", f"{current_pe:.2f}")
            c2.metric("Percentile vs history", f"{pct_rank:.1%}")
            c3.metric("Historical median", f"{median_pe:.2f}")
            c4.metric("Historical range", f"{min_pe:.1f} – {max_pe:.1f}")

            # Time-series chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pe.index, y=pe, mode="lines", name="Trailing P/E",
                line=dict(color="#1f77b4", width=2),
            ))
            fig.add_hline(y=median_pe, line_dash="dash", line_color="gray",
                          annotation_text=f"Median = {median_pe:.1f}")
            fig.add_hline(y=current_pe, line_dash="dot", line_color="red",
                          annotation_text=f"Current = {current_pe:.1f}")
            # Shade 10th–90th percentile band
            p10, p90 = np.percentile(pe, [10, 90])
            fig.add_hrect(y0=p10, y1=p90, fillcolor="lightgray", opacity=0.25,
                          line_width=0, annotation_text="10–90th pct band",
                          annotation_position="top left")
            fig.update_layout(
                title=f"{val_ticker} — Historical Trailing P/E",
                xaxis_title="Date",
                yaxis_title="Price / TTM EPS",
                height=460,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Histogram
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=pe, nbinsx=50, marker_color="#1f77b4",
                                        opacity=0.75, name="Historical P/E"))
            fig2.add_vline(x=current_pe, line_dash="dot", line_color="red",
                           annotation_text=f"Current = {current_pe:.1f}")
            fig2.update_layout(
                title="Distribution of historical P/E observations",
                xaxis_title="P/E",
                yaxis_title="Days",
                height=380, bargap=0.02,
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Signal interpretation
            if pct_rank >= 0.90:
                verdict = "🔴 **Rich**"
                reading = (
                    f"Today's P/E of **{current_pe:.1f}** sits above **{pct_rank:.0%}** "
                    "of the firm's historical observations — a classic rich-valuation "
                    "signal. This alone is not a sell recommendation: multiples "
                    "expand legitimately when growth, margins, or the discount rate "
                    "improve. But mean reversion in multiples is one of the most "
                    "robust empirical patterns in equities."
                )
            elif pct_rank <= 0.10:
                verdict = "🟢 **Cheap**"
                reading = (
                    f"Today's P/E of **{current_pe:.1f}** is below **{1-pct_rank:.0%}** "
                    "of historical observations — a value signal. Investigate whether "
                    "this reflects a durable deterioration in fundamentals (a value "
                    "trap) or a transitory dislocation."
                )
            else:
                verdict = "🟡 **Near historical norm**"
                reading = (
                    f"Today's P/E of **{current_pe:.1f}** is in the {pct_rank:.0%} "
                    "percentile of its own history — neither stretched nor depressed. "
                    "Valuation alone provides little edge; look to catalysts or "
                    "other factors."
                )
            st.markdown(f"### Verdict: {verdict}")
            st.info(reading)

            # Reference comparables (optional context)
            name = info.get("longName") or info.get("shortName") or val_ticker
            sector = info.get("sector", "n/a")
            industry = info.get("industry", "n/a")
            fwd_pe = info.get("forwardPE")
            trail_pe_info = info.get("trailingPE")
            st.markdown("#### Company context")
            st.write(
                f"**{name}** — Sector: *{sector}*, Industry: *{industry}*"
                + (f"\n\n- Yahoo trailing P/E: {trail_pe_info:.2f}" if trail_pe_info else "")
                + (f"\n- Yahoo forward P/E: {fwd_pe:.2f}" if fwd_pe else "")
            )


# =============================================================================
# TAB 4: CAPM / FACTOR MODEL
# =============================================================================
with tabs[4]:
    st.header("CAPM / Single-Factor Model")
    st.markdown(
        "Regress the asset's **excess return** on the market's **excess return**:\n\n"
        "$$r_i - r_f = \\alpha + \\beta\\,(r_m - r_f) + \\varepsilon$$\n\n"
        "- **β** — systematic (market) risk exposure.\n"
        "- **α** — average excess return unexplained by the market (Jensen's alpha).\n"
        "- **R²** — fraction of return variance explained by the market factor.\n"
        "- **t-stats** — significance of α and β."
    )

    col1, col2 = st.columns(2)
    with col1:
        asset_t = st.text_input("Asset ticker", value="TSLA", key="capm_asset").strip().upper()
    with col2:
        mkt_t = st.text_input("Market benchmark ticker", value="SPY", key="capm_mkt").strip().upper()

    run_capm = st.button("🚀 Run CAPM Regression", type="primary", key="run_capm")

    if run_capm:
        with st.spinner("Running regression..."):
            prices = fetch_prices([asset_t, mkt_t], start_date, end_date)

        if prices.empty or prices.shape[1] < 2:
            st.error("Could not retrieve both series.")
        else:
            rets = prices.pct_change().dropna()
            rf_daily = (1 + rf_annual) ** (1 / 252) - 1
            out = capm_regression(rets[asset_t], rets[mkt_t], rf_daily=rf_daily)
            if out is None:
                st.error("Not enough overlapping observations.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Beta (β)", f"{out['beta']:.3f}",
                          help="Sensitivity to the market benchmark.")
                c2.metric("Alpha (annualized)", f"{out['alpha_annual']:.2%}",
                          help="Excess return unexplained by market exposure.")
                c3.metric("R²", f"{out['r_squared']:.3f}",
                          help="Share of return variance explained by the market.")
                c4.metric("N (days)", f"{out['n_obs']}")

                c1, c2, c3 = st.columns(3)
                c1.metric("t-stat (α)", f"{out['t_alpha']:.2f}")
                c2.metric("t-stat (β)", f"{out['t_beta']:.2f}")
                c3.metric("Idiosyncratic vol (ann.)", f"{out['residual_vol_annual']:.2%}")

                # Scatter plot with fitted line
                x = (rets[mkt_t] - rf_daily).values
                y = (rets[asset_t] - rf_daily).values
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = out["alpha_daily"] + out["beta"] * x_line

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="markers", name="Daily obs.",
                    marker=dict(size=4, color="#1f77b4", opacity=0.45),
                ))
                fig.add_trace(go.Scatter(
                    x=x_line, y=y_line, mode="lines", name="CAPM fit",
                    line=dict(color="red", width=2),
                ))
                fig.update_layout(
                    title=f"{asset_t} vs {mkt_t} — excess daily returns",
                    xaxis_title=f"{mkt_t} excess return",
                    yaxis_title=f"{asset_t} excess return",
                    xaxis_tickformat=".1%",
                    yaxis_tickformat=".1%",
                    height=480,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Interpretation
                interp = []
                if out["beta"] > 1.1:
                    interp.append(
                        f"β = {out['beta']:.2f} > 1 → **aggressive** asset; amplifies "
                        "market moves."
                    )
                elif out["beta"] < 0.9:
                    interp.append(
                        f"β = {out['beta']:.2f} < 1 → **defensive** asset; dampens "
                        "market moves."
                    )
                else:
                    interp.append(
                        f"β = {out['beta']:.2f} ≈ 1 → moves roughly in line with the "
                        "market."
                    )
                if abs(out["t_alpha"]) > 2:
                    sign = "positive" if out["alpha_annual"] > 0 else "negative"
                    interp.append(
                        f"α t-stat = {out['t_alpha']:.2f} → statistically significant "
                        f"{sign} alpha of **{out['alpha_annual']:.2%}** p.a. over this "
                        "window. Treat with caution: in-sample alpha rarely survives "
                        "out-of-sample."
                    )
                else:
                    interp.append(
                        f"α t-stat = {out['t_alpha']:.2f} → alpha is not statistically "
                        "different from zero. Consistent with CAPM in this sample."
                    )
                interp.append(
                    f"R² = {out['r_squared']:.2f} → the market explains "
                    f"**{out['r_squared']:.0%}** of {asset_t}'s return variance; the "
                    f"remaining {1-out['r_squared']:.0%} is idiosyncratic (residual vol "
                    f"= {out['residual_vol_annual']:.2%} annualized)."
                )
                st.info("  \n".join(interp))


# =============================================================================
# TAB 5: MONTE CARLO
# =============================================================================
with tabs[5]:
    st.header("Monte Carlo Simulation (Geometric Brownian Motion)")
    st.markdown(
        "Simulate forward price paths under GBM:\n\n"
        "$$S_{t+\\Delta t} = S_t \\exp\\!\\left[\\left(\\mu - \\tfrac{1}{2}\\sigma^2\\right)\\Delta t + \\sigma\\sqrt{\\Delta t}\\,Z\\right], \\quad Z \\sim \\mathcal{N}(0,1)$$\n\n"
        "μ and σ are estimated from the historical window."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        mc_ticker = st.text_input("Ticker", value="SPY", key="mc_t").strip().upper()
    with col2:
        horizon = st.number_input("Horizon (trading days)", 20, 2520, 252, 20)
    with col3:
        n_paths = st.number_input("Paths", 100, 20000, 2000, 100)
    with col4:
        seed = st.number_input("Random seed", 0, 10000, 42, 1)

    run_mc = st.button("🚀 Run Monte Carlo", type="primary", key="run_mc")

    if run_mc:
        with st.spinner("Simulating..."):
            prices = fetch_prices(mc_ticker, start_date, end_date)
            if prices.empty:
                st.error("No data.")
            else:
                rets = prices.pct_change().dropna().iloc[:, 0]
                mu_a = rets.mean() * 252
                sig_a = rets.std() * np.sqrt(252)
                S0 = float(prices.iloc[-1, 0])

                paths = monte_carlo_paths(mu_a, sig_a, S0, horizon, n_paths, seed=int(seed))
                terminals = paths[-1]

                c1, c2, c3 = st.columns(3)
                c1.metric("Estimated μ (annual)", f"{mu_a:.2%}")
                c2.metric("Estimated σ (annual)", f"{sig_a:.2%}")
                c3.metric("Starting price", f"${S0:,.2f}")

                # Plot sample of paths + percentile bands
                time_axis = np.arange(horizon + 1)
                p5 = np.percentile(paths, 5, axis=1)
                p50 = np.percentile(paths, 50, axis=1)
                p95 = np.percentile(paths, 95, axis=1)

                sample_idx = np.random.default_rng(int(seed)).choice(
                    n_paths, size=min(100, n_paths), replace=False)

                fig = go.Figure()
                for i in sample_idx:
                    fig.add_trace(go.Scatter(
                        x=time_axis, y=paths[:, i], mode="lines",
                        line=dict(color="rgba(31,119,180,0.08)", width=1),
                        showlegend=False, hoverinfo="skip",
                    ))
                fig.add_trace(go.Scatter(x=time_axis, y=p95, mode="lines",
                                         name="95th pct", line=dict(color="green", dash="dash")))
                fig.add_trace(go.Scatter(x=time_axis, y=p50, mode="lines",
                                         name="Median", line=dict(color="black", width=2)))
                fig.add_trace(go.Scatter(x=time_axis, y=p5, mode="lines",
                                         name="5th pct", line=dict(color="red", dash="dash")))
                fig.update_layout(
                    title=f"{mc_ticker} — {n_paths:,} simulated paths over {horizon} trading days",
                    xaxis_title="Trading days ahead",
                    yaxis_title="Simulated price",
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Terminal distribution
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(x=terminals, nbinsx=60,
                                            marker_color="#1f77b4", opacity=0.8))
                fig2.add_vline(x=S0, line_dash="dot", line_color="black",
                               annotation_text=f"Today = ${S0:.2f}")
                fig2.add_vline(x=np.percentile(terminals, 5), line_dash="dash",
                               line_color="red",
                               annotation_text=f"5th pct = ${np.percentile(terminals,5):.2f}")
                fig2.update_layout(
                    title="Distribution of terminal prices",
                    xaxis_title="Terminal price",
                    yaxis_title="Frequency",
                    height=380, bargap=0.02,
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Scenario stats
                ret_dist = terminals / S0 - 1
                prob_loss = (terminals < S0).mean()
                prob_down10 = (ret_dist < -0.10).mean()
                prob_up20 = (ret_dist > 0.20).mean()
                expected_return = ret_dist.mean()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("P(loss)", f"{prob_loss:.1%}")
                c2.metric("P(return < -10%)", f"{prob_down10:.1%}")
                c3.metric("P(return > +20%)", f"{prob_up20:.1%}")
                c4.metric("Expected return", f"{expected_return:.2%}")

                st.info(
                    f"Over a **{horizon}-day** horizon starting from "
                    f"${S0:,.2f}, the model implies a **{prob_loss:.0%}** "
                    f"probability of ending below today's price and a "
                    f"**{prob_down10:.0%}** probability of a loss exceeding 10%. "
                    "GBM assumes log-normal returns with constant μ and σ, which "
                    "understates tail risk relative to empirical equity distributions."
                )


# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "**Disclaimer.** This is an educational tool. Outputs are model estimates on "
    "historical data and are **not** investment advice. Past performance does "
    "not guarantee future results."
)
