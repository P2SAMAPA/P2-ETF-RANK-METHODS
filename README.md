# Rank‑Based Methods Engine

Implements multi‑factor ranking for ETF selection: relative strength (12‑month), 1‑month momentum, volume surge, earnings surrogate (price slope), and industry group rank. The composite score (average of normalised components) is used to rank ETFs. Multi‑window evaluation selects the best window per ETF.

- **Factors:** RS, short momentum, volume surge, earnings surrogate, industry group rank
- **Normalisation:** percentiles across the universe for continuous factors
- **Composite score:** equal‑weighted average (0–1, higher better)
- **Windows:** 63, 252, 504, 1008, 2016, 4032 days (best per ETF)
- **Output:** top 3 ETFs per universe

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
