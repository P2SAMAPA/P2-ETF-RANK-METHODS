import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from rank_methods import compute_composite_score

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Rank Methods) ===")
        prices = data_manager.prepare_price_matrix(df, tickers)
        volumes = data_manager.prepare_volume_matrix(df, tickers)
        # Returns needed for group rank
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if prices.empty or len(prices) < max(config.WINDOWS) + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(prices) < win + 10:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            prices_win = prices.iloc[-win:]
            volumes_win = volumes.iloc[-win:]
            returns_win = returns.iloc[-win:] if not returns.empty else pd.DataFrame(0, index=prices_win.index, columns=tickers)
            # Compute raw components for all ETFs
            raw_scores = {}
            for etf in tickers:
                if etf not in prices_win.columns or etf not in volumes_win.columns:
                    continue
                comp = compute_composite_score(prices_win[etf], volumes_win[etf], returns_win, etf, win)
                raw_scores[etf] = comp
            if not raw_scores:
                continue
            # Normalise RS and momentum across ETFs (convert to percentiles)
            rs_vals = [raw_scores[etf]["rs"] for etf in raw_scores]
            mom_vals = [raw_scores[etf]["mom"] for etf in raw_scores]
            eps_vals = [raw_scores[etf]["eps"] for etf in raw_scores]
            # Compute percentiles (higher = better)
            rs_pct = pd.Series(rs_vals).rank(pct=True).values
            mom_pct = pd.Series(mom_vals).rank(pct=True).values
            eps_pct = pd.Series(eps_vals).rank(pct=True).values
            # Composite score: equal weight of RS, momentum, volume surge, earnings surrogate, group rank
            composite_scores = {}
            for i, etf in enumerate(raw_scores):
                score = (rs_pct[i] + mom_pct[i] + raw_scores[etf]["vol_surge"] + eps_pct[i] + raw_scores[etf]["group_rank"]) / 5.0
                composite_scores[etf] = score
            window_results[win] = composite_scores
            for etf, score in composite_scores.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)

        if not best_per_etf:
            print("  No valid predictions – falling back to historical mean return")
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (max(mean_ret, 1e-6), 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        full_scores = {ticker: {"score": float(score), "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "rank_score": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs by rank score: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/rank_methods_{today}.json")
    with open(local_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Rank‑Based Methods Engine complete ===")

if __name__ == "__main__":
    main()
