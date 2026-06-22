# A/B Test Analysis Dashboard

End-to-end statistical experimentation framework analyzing a **100,000-user checkout optimization experiment** (50K per variant). Covers frequentist testing, Bayesian inference, CUPED variance reduction, and an interactive Streamlit dashboard.

## Experiment Results

**Test:** 1-step checkout vs. legacy 3-step checkout · **Window:** 30 days

| Metric | Control | Treatment | Change |
|--------|---------|-----------|--------|
| Users | 50,000 | 50,000 | — |
| Conversion rate | 5.12% | 6.84% | +33.7% relative |
| Absolute lift | — | — | +1.72pp |
| 95% CI | — | — | [+1.43pp, +2.02pp] |
| p-value | — | — | < 0.0001 |
| P(treatment wins) | — | — | >99.9% (Bayesian) |
| Avg revenue per user | $5.33 | $7.34 | +37.7% |

**Recommendation: SHIP IT ✅**

## Statistical Methods

| Method | Purpose |
|--------|---------|
| Two-proportion z-test | Conversion rate significance |
| Mann-Whitney U | Revenue significance (zero-inflated distribution) |
| Bayesian Beta-Binomial | P(treatment > control) — intuitive for stakeholders |
| CUPED | Variance reduction using pre-experiment purchase history |
| Power analysis | Sample size planning before launch |

## Dashboard Features

- KPI cards — conversion rates, sample sizes, revenue per user
- Daily CVR trend over 30-day window
- Revenue distribution with Mann-Whitney annotation
- Bayesian posterior visualization
- Segment breakdown by device and country
- Interactive sample size calculator
- Upload-your-own-CSV mode for any experiment

## Run Locally

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`

## Tech Stack

- **Python** — pandas, numpy, scipy.stats
- **Streamlit** — interactive dashboard
- **Plotly** — charts and visualizations
- **Custom stats library** — `utils/stats_engine.py`

## Project Structure
