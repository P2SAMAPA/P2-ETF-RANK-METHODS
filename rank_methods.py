import numpy as np
import pandas as pd

def relative_strength(prices_series, window=252):
    """RS = (price_now / price_n_days_ago) - 1."""
    if len(prices_series) < window + 1:
        return 0.0
    price_now = prices_series.iloc[-1]
    price_past = prices_series.iloc[-window]
    return (price_now / price_past) - 1.0

def short_momentum(prices_series, window=21):
    """1‑month return."""
    return relative_strength(prices_series, window)

def volume_surge(volume_series, window=20, threshold=1.5):
    """1 if latest volume > threshold * average volume over window."""
    if len(volume_series) < window + 1:
        return 0.0
    avg_vol = volume_series.iloc[-window:].mean()
    current_vol = volume_series.iloc[-1]
    return 1.0 if current_vol > threshold * avg_vol else 0.0

def earnings_surrogate(prices_series, window=60):
    """Price acceleration: slope of linear regression over last `window` days."""
    if len(prices_series) < window:
        return 0.0
    x = np.arange(window)
    y = prices_series.iloc[-window:].values
    slope = np.polyfit(x, y, 1)[0]
    return slope

def industry_group_rank(returns_df, window=252):
    """
    For each ETF, compute its rank within its sector (using last `window` days return).
    Returns normalised rank (0 to 1) where higher = better.
    """
    if len(returns_df) < window:
        return {etf: 0.5 for etf in returns_df.columns}
    total_return = returns_df.iloc[-window:].sum()
    ranks = total_return.rank(pct=True)
    return ranks.to_dict()

def compute_composite_score(prices_series, volume_series, returns_df, etf, window):
    """
    Compute composite rank score for a single ETF.
    Components:
        - RS (12‑month return)
        - 1‑month momentum
        - volume surge (binary)
        - earnings surrogate (price slope)
        - industry group rank
    """
    rs = relative_strength(prices_series, window=config.RS_WINDOW)
    mom = short_momentum(prices_series, window=config.RS_SHORT_WINDOW)
    vol_surge = volume_surge(volume_series, window=config.VOL_SURGE_WINDOW, threshold=config.VOL_SURGE_THRESHOLD)
    eps = earnings_surrogate(prices_series, window=config.EARNINGS_PROXY_WINDOW)
    group_rank = industry_group_rank(returns_df, window=config.RS_WINDOW).get(etf, 0.5)
    # Normalise RS and momentum (they may be unbounded) – we use rank within the universe
    # But we don't have the universe here; we'll normalise after computing all scores.
    # For now, return raw components; normalisation will be done in trainer.
    return {
        "rs": rs,
        "mom": mom,
        "vol_surge": vol_surge,
        "eps": eps,
        "group_rank": group_rank
    }
