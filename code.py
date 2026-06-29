"""
Factor Risk Model
Decomposes portfolio returns into risk factors (momentum, value, size, market)
using PCA and OLS regression, then stress-tests the portfolio.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
from typing import Dict, List, Tuple


# Data Fetching

# Example; 15 large-cap US stocks
PORTFOLIO = {
    "AAPL": 0.10, "MSFT": 0.10, "GOOGL": 0.08, "AMZN": 0.08,
    "NVDA": 0.10, "META": 0.07, "JPM": 0.07,  "BAC": 0.05,
    "JNJ":  0.06, "PG":   0.06, "XOM": 0.06,  "CVX": 0.05,
    "GLD":  0.04, "TLT":  0.04, "SPY": 0.04,
}

# Factor proxies
FACTOR_TICKERS = {
    "Market":   "SPY",   # broad market
    "Size":     "IWM",   # small-cap (Russell 2000) → size factor
    "Value":    "IVE",   # S&P 500 Value ETF
    "Momentum": "MTUM",  # iShares MSCI Momentum
    "LowVol":   "USMV",  # iShares MSCI Min Vol
    "Bond":     "TLT",   # long-duration bonds
    "Gold":     "GLD",   # gold
}


def fetch_returns(tickers: List[str], period: str = "3y") -> pd.DataFrame:
    all_tickers = list(set(tickers))
    raw = yf.download(all_tickers, period=period, auto_adjust=True, progress=False)
    close = raw["Close"] if "Close" in raw else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(0)
    returns = close.pct_change().dropna()
    return returns[[t for t in all_tickers if t in returns.columns]]


# Factor Model 

def compute_portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    common = [t for t in weights if t in asset_returns.columns]
    w = np.array([weights[t] for t in common])
    w = w / w.sum()
    return asset_returns[common].dot(w)


def ols_factor_regression(
    portfolio_ret: pd.Series,
    factor_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Regress portfolio returns on factor returns (Fama-French style).
    Returns exposures (betas), t-stats, R², and alpha.
    """
    common_idx = portfolio_ret.index.intersection(factor_returns.index)
    y = portfolio_ret.loc[common_idx].values
    X = factor_returns.loc[common_idx].values
    X_const = np.column_stack([np.ones(len(X)), X])

    model = LinearRegression(fit_intercept=False)
    model.fit(X_const, y)
    betas = model.coef_

    # Residuals and stats
    pred = X_const @ betas
    resid = y - pred
    n, k = len(y), X_const.shape[1]
    s2 = np.dot(resid, resid) / (n - k)
    XtX_inv = np.linalg.pinv(X_const.T @ X_const)
    se = np.sqrt(s2 * np.diag(XtX_inv))
    t_stats = betas / se
    r2 = 1 - np.var(resid) / np.var(y)

    factor_names = ["Alpha"] + list(factor_returns.columns)
    result = pd.DataFrame({
        "beta": betas,
        "std_err": se,
        "t_stat": t_stats,
        "p_value": [2 * (1 - stats.t.cdf(abs(t), df=n-k)) for t in t_stats],
    }, index=factor_names)
    result.attrs["r2"] = r2
    result.attrs["alpha_annualized"] = betas[0] * 252
    return result


def pca_factor_analysis(
    asset_returns: pd.DataFrame,
    n_components: int = 5,
) -> Tuple[PCA, pd.DataFrame, pd.DataFrame]:
    """Extract statistical factors via PCA."""
    scaler = StandardScaler()
    X = scaler.fit_transform(asset_returns.dropna())
    pca = PCA(n_components=n_components)
    pca.fit(X)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=asset_returns.columns,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    explained = pd.DataFrame({
        "variance_explained": pca.explained_variance_ratio_,
        "cumulative": np.cumsum(pca.explained_variance_ratio_),
    }, index=[f"PC{i+1}" for i in range(n_components)])

    factor_returns = pd.DataFrame(
        pca.transform(X),
        index=asset_returns.dropna().index,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    return pca, loadings, explained, factor_returns


#Risk Decomposition 

def decompose_portfolio_risk(
    portfolio_ret: pd.Series,
    factor_returns: pd.DataFrame,
    regression: pd.DataFrame,
) -> Dict:
    """Attribute portfolio variance to each factor."""
    factor_names = [f for f in regression.index if f != "Alpha"]
    betas = regression.loc[factor_names, "beta"].values

    # Factor covariance matrix
    fac_cov = factor_returns.loc[portfolio_ret.index.intersection(factor_returns.index)].cov().values

    # Factor variance contribution
    total_var = portfolio_ret.var()
    factor_var = betas @ fac_cov @ betas
    idio_var = max(total_var - factor_var, 0)

    contributions = {}
    for i, name in enumerate(factor_names):


        # Marginal contribution: beta_i * sum_j(beta_j * cov(fi, fj))
        marginal = betas[i] * (fac_cov[i] @ betas)
        contributions[name] = marginal / total_var if total_var > 0 else 0

    return {
        "total_var": total_var,
        "factor_var": factor_var,
        "idiosyncratic_var": idio_var,
        "r_squared": factor_var / total_var if total_var > 0 else 0,
        "factor_contributions": contributions,
    }




#  Stress Testing 

STRESS_SCENARIOS = {
    "2020 COVID Crash":       {"Market": -0.35, "Bond": +0.15, "Gold": +0.05,
                               "Size": -0.40, "Momentum": -0.25, "Value": -0.35, "LowVol": -0.20},
    "2022 Rate Shock":        {"Market": -0.20, "Bond": -0.30, "Gold": -0.02,
                               "Size": -0.22, "Momentum": -0.15, "Value": +0.05, "LowVol": -0.10},
    "2008 GFC":               {"Market": -0.50, "Bond": +0.25, "Gold": +0.10,
                               "Size": -0.55, "Momentum": -0.40, "Value": -0.45, "LowVol": -0.35},
    "Tech Bubble Burst 2000": {"Market": -0.45, "Bond": +0.20, "Gold": +0.05,
                               "Size": -0.30, "Momentum": -0.50, "Value": +0.10, "LowVol": -0.20},
    "Inflation Surge":        {"Market": -0.15, "Bond": -0.20, "Gold": +0.20,
                               "Size": -0.10, "Momentum": -0.05, "Value": +0.15, "LowVol": -0.05},
    "Risk-Off Rally":         {"Market": +0.25, "Bond": -0.05, "Gold": -0.02,
                               "Size": +0.30, "Momentum": +0.20, "Value": +0.15, "LowVol": +0.15},
}




def stress_test(regression: pd.DataFrame, scenarios: Dict = None) -> pd.DataFrame:
    if scenarios is None:
        scenarios = STRESS_SCENARIOS
    results = {}
    for scenario, shocks in scenarios.items():
        port_impact = 0.0
        # Alpha doesn't respond to factor shocks
        for factor, shock in shocks.items():
            if factor in regression.index:
                port_impact += regression.loc[factor, "beta"] * shock
        # Add alpha * 1 year (scenario assumed to play out over ~1yr)
        port_impact += regression.loc["Alpha", "beta"] * 252 / 12   # 1-month alpha
        results[scenario] = port_impact

    return pd.Series(results, name="portfolio_impact").sort_values()







# Visualization 

def plot_factor_model(
    portfolio_ret: pd.Series,
    factor_returns: pd.DataFrame,
    regression: pd.DataFrame,
    risk_decomp: Dict,
    stress: pd.Series,
    pca_explained: pd.DataFrame,
    pca_loadings: pd.DataFrame,
    save_path: str = None,
):
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)
    fig.suptitle("Factor Risk Model — Portfolio Analysis", fontsize=14, fontweight="bold")

    # 1. Factor betas
    ax1 = fig.add_subplot(gs[0, 0])
    factors = [f for f in regression.index if f != "Alpha"]
    betas = regression.loc[factors, "beta"]
    colors = ["#2ecc71" if b > 0 else "#e74c3c" for b in betas]
    ax1.barh(factors, betas, color=colors, alpha=0.8)
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_title(f"Factor Exposures (Betas)\nR²={regression.attrs.get('r2', 0):.3f}")
    ax1.set_xlabel("Beta")

    # 2. Risk attribution pie
    ax2 = fig.add_subplot(gs[0, 1])
    contribs = risk_decomp["factor_contributions"]
    idio_frac = risk_decomp["idiosyncratic_var"] / risk_decomp["total_var"]
    labels = list(contribs.keys()) + ["Idiosyncratic"]
    sizes  = [max(v, 0) for v in contribs.values()] + [max(idio_frac, 0)]
    if sum(sizes) > 0:
        ax2.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 7})
    ax2.set_title("Variance Attribution")

    # 3. Stress test
    ax3 = fig.add_subplot(gs[0, 2])
    colors_stress = ["#2ecc71" if v > 0 else "#e74c3c" for v in stress.values]
    bars = ax3.barh(stress.index, stress.values * 100, color=colors_stress, alpha=0.8)
    ax3.axvline(0, color="black", lw=0.8)
    ax3.set_xlabel("Portfolio Impact (%)")
    ax3.set_title("Stress Test Scenarios")
    ax3.tick_params(axis="y", labelsize=7)

    # 4. Portfolio cumulative return vs factors
    ax4 = fig.add_subplot(gs[1, :2])
    port_cum = (1 + portfolio_ret).cumprod()
    ax4.plot(port_cum.index, port_cum.values, label="Portfolio", linewidth=2, color="navy")
    for fname, col in zip(["Market", "Bond", "Momentum"], ["#e74c3c", "#2ecc71", "#e67e22"]):
        if fname in factor_returns.columns:
            fac_cum = (1 + factor_returns[fname]).cumprod()
            common = port_cum.index.intersection(fac_cum.index)
            ax4.plot(fac_cum.loc[common].index, fac_cum.loc[common].values,
                     label=fname, linewidth=1, alpha=0.7, color=col)
    ax4.set_title("Cumulative Returns: Portfolio vs Key Factors")
    ax4.legend(fontsize=8); ax4.set_ylabel("Cumulative Return")



    # 5. Rolling beta to market
    ax5 = fig.add_subplot(gs[1, 2])
    common_idx = portfolio_ret.index.intersection(factor_returns.index)
    port_a = portfolio_ret.loc[common_idx]
    mkt_a  = factor_returns.loc[common_idx, "Market"] if "Market" in factor_returns.columns else None
    if mkt_a is not None:
        roll = 60
        rolling_beta = port_a.rolling(roll).cov(mkt_a) / mkt_a.rolling(roll).var()
        ax5.plot(rolling_beta.index, rolling_beta.values, color="#8e44ad", linewidth=1.2)
        ax5.axhline(1.0, color="black", lw=0.8, linestyle="--", alpha=0.4)
        ax5.set_title(f"Rolling {roll}-day Beta to Market")
        ax5.set_ylabel("Beta")


    # 6. PCA explained variance

    ax6 = fig.add_subplot(gs[2, 0])
    ax6.bar(pca_explained.index, pca_explained["variance_explained"] * 100,
            color="#3498db", alpha=0.8, label="Individual")
    ax6.plot(pca_explained.index, pca_explained["cumulative"] * 100,
             color="#e74c3c", marker="o", markersize=4, label="Cumulative")
    ax6.set_xlabel("Principal Component"); ax6.set_ylabel("Variance Explained (%)")
    ax6.set_title("PCA Scree Plot"); ax6.legend(fontsize=8)

    # 7. PCA loadings heatmap (PC1-3)

    ax7 = fig.add_subplot(gs[2, 1:])
    load3 = pca_loadings.iloc[:, :3]
    im = ax7.imshow(load3.values.T, cmap="RdBu_r", aspect="auto", vmin=-0.5, vmax=0.5)
    ax7.set_yticks(range(3)); ax7.set_yticklabels(load3.columns)
    ax7.set_xticks(range(len(load3.index)))
    ax7.set_xticklabels(load3.index, rotation=45, ha="right", fontsize=8)
    ax7.set_title("PCA Factor Loadings (PC1-3)")
    plt.colorbar(im, ax=ax7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    plt.show()


# Main 

def run(plot: bool = True):
    print("\n" + "="*60)
    print("  Factor Risk Model — Portfolio Decomposition & Stress Test")
    print("="*60 + "\n")

    all_tickers = list(PORTFOLIO.keys()) + list(FACTOR_TICKERS.values())
    print(f"► Fetching 3-year returns for {len(set(all_tickers))} tickers …")
    returns = fetch_returns(list(set(all_tickers)), period="3y")
    print(f"  {len(returns)} trading days loaded")

    # Factor returns

    factor_ret = returns[[v for v in FACTOR_TICKERS.values() if v in returns.columns]].copy()
    factor_ret.columns = [k for k, v in FACTOR_TICKERS.items() if v in returns.columns]

    # Portfolio returns

    portfolio_ret = compute_portfolio_returns(returns, PORTFOLIO)
    print(f"  Portfolio annualized return: {portfolio_ret.mean()*252:.2%}")
    print(f"  Portfolio annualized vol:    {portfolio_ret.std()*np.sqrt(252):.2%}")
    sharpe = portfolio_ret.mean() / portfolio_ret.std() * np.sqrt(252)
    print(f"  Sharpe ratio:                {sharpe:.2f}")

    print("\n► Running OLS factor regression …")
    regression = ols_factor_regression(portfolio_ret, factor_ret)
    print(f"  R²: {regression.attrs.get('r2',0):.4f}")
    print(f"  Annualized Alpha: {regression.attrs.get('alpha_annualized',0):.4%}")
    print(f"\n  Factor Exposures:")
    print(regression[["beta", "t_stat", "p_value"]].round(4).to_string())

    print("\n► Decomposing portfolio risk …")
    risk_decomp = decompose_portfolio_risk(portfolio_ret, factor_ret, regression)
    print(f"  Factor-explained variance: {risk_decomp['r_squared']:.2%}")
    print(f"  Idiosyncratic variance:    {risk_decomp['idiosyncratic_var']/risk_decomp['total_var']:.2%}")
    print(f"  Top factor contributors:")
    for k, v in sorted(risk_decomp["factor_contributions"].items(), key=lambda x: -abs(x[1])):
        print(f"    {k:<12}: {v:.4f} ({v*100:.1f}% of total var)")

    print("\n► PCA factor analysis …")
    asset_ret = returns[[t for t in PORTFOLIO if t in returns.columns]]
    pca, loadings, explained, pca_factors = pca_factor_analysis(asset_ret, n_components=5)
    print(explained.round(4).to_string())

    print("\n Stress Testing …")
    stress = stress_test(regression)
    print(f"\n  {'Scenario':<30} {'Impact':>10}")
    print("  " + "─"*42)
    for scenario, impact in stress.items():
        bar = "█" * int(abs(impact) * 100 / 5)
        
        sign = "+" if impact > 0 else ""
        print(f"  {scenario:<30} {sign}{impact:.2%}  {bar}")

    if plot:
        print("\n Generating plots …")
        plot_factor_model(portfolio_ret, factor_ret, regression, risk_decomp,
                          stress, explained, loadings,
                          save_path="factor_risk_model.png")

    return regression, risk_decomp, stress


if __name__ == "__main__":
    run(plot=True)
