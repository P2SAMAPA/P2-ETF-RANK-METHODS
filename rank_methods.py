import numpy as np
import pandas as pd

def relative_strength(prices_series, window=252):
    if len(prices_series) < window + 1:
        return np.nan
    return (prices_series.iloc[-1] / prices_series.iloc[-window]) - 1.0

def short_momentum(prices_series, window=21):
    return relative_strength(prices_series, window)

def earnings_surrogate(prices_series, window=60):
    if len(prices_series) < window:
        return np.nan
    x = np.arange(window)
    y = prices_series.iloc[-window:].values
    slope = np.polyfit(x, y, 1)[0]
    return slope

def industry_group_rank(returns_df, window=252):
    """
    For each ETF, compute its rank within its sector (using last `window` days return).
    Returns dict of normalised rank (0-1, higher better). If insufficient data, returns 0.5 for all.
    """
    if returns_df.empty or len(returns_df) < window:
        return {col: 0.5 for col in returns_df.columns}
    total_return = returns_df.iloc[-window:].sum()
    ranks = total_return.rank(pct=True)
    return ranks.to_dict()

def compute_composite_score(prices_series, returns_df, etf, window):
    """
    Compute rank score using available factors (no volume).
    """
    rs = relative_strength(prices_series, window=window)
    mom = short_momentum(prices_series, window=21)
    eps = earnings_surrogate(prices_series, window=60)
    group_rank = industry_group_rank(returns_df, window=window).get(etf, 0.5)
    # If any factor is missing (NaN), replace with neutral value
    if np.isnan(rs): rs = 0.0
    if np.isnan(mom): mom = 0.0
    if np.isnan(eps): eps = 0.0
    # For ranking, we need percentiles later; we'll return raw values
    return {"rs": rs, "mom": mom, "eps": eps, "group_rank": group_rank}
