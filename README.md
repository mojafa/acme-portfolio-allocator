# ACME Fund — ML Portfolio Allocator

**Live app:** [acme-portfolio-allocator.streamlit.app](https://acme-portfolio-allocator.streamlit.app/)

A supervised classifier that recommends one of four model portfolios (Conservative, Balanced, Aggressive or Thematic-Tech) for a new investor at client onboarding, served through a Streamlit adviser front end.

> **Academic context.** This repository is the code deliverable for Part 2 of the **BB7031 Machine Learning and FinTech Applications** individual assignment at Kingston University. It accompanies a separately submitted written report (1,800 words) and two BPMN 2.0 diagrams modelling (i) ACME Fund's client-onboarding business process and (ii) the ML application development workflow for this model. The fictional firm "ACME Fund" is the SME used throughout the report.

---

## What it does

ACME Fund is a mid-sized UK discretionary wealth manager. At onboarding, each new investor is allocated to one of four model portfolios based on the adviser's reading of a risk questionnaire, the client's financial profile, and the current market regime. This project automates that decision with a **Random Forest classifier** trained on 2,500 synthetic (GDPR-safe) client profiles, labelled with a rule encoding a senior adviser's allocation heuristic.

The model reads **ten features**:

- **Seven client features** from the onboarding form: `age`, `income`, `liquid_aum`, `risk_tolerance`, `horizon_years`, `esg_preference`, `income_stability`.
- **Three market-regime features** computed from VTI's trailing 252 trading days: `mkt_ann_vol`, `mkt_momentum`, `mkt_sharpe`.

Each of the four model portfolios is a fixed-weight recipe over **twelve real ETFs** spanning bonds, broad equities, sector equities, gold and REITs. The Streamlit UI renders the recommended portfolio's ticker-level composition with £-amount allocations on the client's own AUM, class probabilities, top feature drivers, and a model card.

---

## Held-out performance

Three candidate models were trained and compared on a 625-client stratified test set; the Random Forest was deployed because it matches the MLP on accuracy while remaining explainable through feature importance and SHAP.

| Model | Accuracy | Macro F1 | Macro ROC-AUC |
|---|---|---|---|
| DecisionTreeClassifier (depth 5) | 82.1 % | 0.586 | 0.836 |
| **RandomForestClassifier (200 trees)** | **87.4 %** | **0.659** | **0.885** |
| MLPClassifier (32-16) | 85.0 % | 0.682 | 0.863 |

---

## Quick start

Requires Python 3.11 or later, and [`uv`](https://github.com/astral-sh/uv) (recommended) or plain `pip`.

```bash
# Clone
git clone https://github.com/mojafa/acme-portfolio-allocator.git
cd acme-portfolio-allocator

# Environment
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Open the notebook (trains the model end-to-end, ~40 seconds)
jupyter notebook notebooks/acme_portfolio_allocator.ipynb

# Or launch the adviser UI against the already-trained model
streamlit run app/app.py
```

The notebook ends with three `joblib.dump` calls; the Streamlit app opens by loading those same three artefacts. That is the full MLOps deployment handshake — no serving code is re-defined, no model is re-trained at start-up.

---

## Project structure

```
acme-portfolio-allocator/
├── notebooks/
│   └── acme_portfolio_allocator.ipynb   # End-to-end: ingest, train DT/RF/MLP,
│                                        # compare, SHAP, joblib.dump, reload & predict
├── app/
│   └── app.py                           # Streamlit adviser UI (3 pages)
├── models/                              # Trained artefacts (committed so the app
│   ├── acme_rf_model.joblib             # works on a fresh clone without re-training)
│   ├── acme_scaler.joblib
│   └── acme_metadata.joblib
├── signavio/                            # BPMN 2.0 diagrams + simulation export
│   ├── ACME Fund — Onboarding & Portfolio Allocation (Business Process).bpmn
│   ├── ACME Portfolio Allocator — ML Application Development Workflow.bpmn
│   └── ACME Portfolio Allocator — ML Application Development Workflow Simulation.xlsx
├── requirements.txt
└── README.md
```

The two `.bpmn` files can be re-imported into Signavio to reproduce the diagrams and rerun the simulation; the `.xlsx` is the native Signavio simulation output (per-task cost, resource consumption per lane, waiting-time analysis).

---

## The model portfolios

Each of ACME's four model portfolios is a fixed-weight blend of the twelve-ETF universe:

| Portfolio | Composition |
|---|---|
| **Conservative** | 50 % AGG · 20 % TLT · 10 % VTI · 10 % XLP · 10 % GLD |
| **Balanced** | 30 % AGG · 30 % VTI · 15 % VXUS · 10 % XLV · 10 % VNQ · 5 % GLD |
| **Aggressive** | 35 % VTI · 20 % VXUS · 20 % QQQ · 10 % XLF · 10 % XLE · 5 % VNQ |
| **Thematic-Tech** | 40 % QQQ · 35 % XLK · 15 % VTI · 10 % AGG |

| Ticker | Asset |
|---|---|
| AGG | US Aggregate Bonds |
| TLT | Long-Duration Treasuries |
| VTI | Total US Equities |
| VXUS | International Equities (ex-US) |
| QQQ | NASDAQ-100 |
| XLK | Tech Sector |
| XLF | Financials Sector |
| XLE | Energy Sector |
| XLP | Consumer Staples |
| XLV | Healthcare |
| GLD | Gold |
| VNQ | US REITs |

---

## Streamlit adviser UI

Three pages, all reading from the joblib artefacts in `models/`:

1. **Recommend a Portfolio** — onboarding form + recommendation card (portfolio, confidence, class probabilities, top feature drivers, ticker-level holdings with £-amount, trailing 12-month metrics).
2. **Model Card** — FCA-ready transparency artefact: intended use, training data, held-out metrics, known limitations, feature importance.
3. **Compare Models** — DT / RF / MLP side-by-side on accuracy, F1, ROC-AUC, with the explainability-vs-accuracy trade-off explained.

---

## Notes on data

- Market data (12 ETFs) is pulled live from Yahoo Finance via `yfinance` when the notebook runs. Cached data is not committed.
- Client data is **entirely synthetic** and generated inside the notebook. Real ACME client records would be GDPR-protected and would only be used inside an FCA-sanctioned sandbox for production retraining — this is discussed in the accompanying report.
- Portfolio labels are produced by a rule encoding a senior adviser's allocation heuristic (100-minus-age equity budget scaled by risk tolerance, horizon and income stability) with 7 % noise added to mimic human variability.

---

## Acknowledgement of generative-AI contribution

Generative AI was used to scaffold standard scikit-learn patterns and the Streamlit UI. All algorithmic choices, feature definitions, hyperparameters and the decision to deploy the Random Forest over the MLP were made by the author; generated code was reviewed, edited and tested before use. Numerical results reported here come from the executed notebook.
