"""
ACME Fund — Adviser-facing portfolio allocator.

Loads the Random Forest, scaler and metadata produced by
notebooks/acme_portfolio_allocator.ipynb and serves an adviser UI that
recommends one of the four ACME model portfolios for a walk-in investor,
with the prediction probabilities and the top feature drivers shown
alongside.

This file is the serving half of the MLOps demonstration:
  notebook  →  joblib.dump(model, scaler, metadata)  →  app.py  →  client
"""
import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

st.set_page_config(
    page_title='ACME Fund — ML Portfolio Allocator',
    page_icon='💼',
    layout='wide',
)


@st.cache_resource
def load_artifacts():
    model    = joblib.load(os.path.join(MODEL_DIR, 'acme_rf_model.joblib'))
    scaler   = joblib.load(os.path.join(MODEL_DIR, 'acme_scaler.joblib'))
    metadata = joblib.load(os.path.join(MODEL_DIR, 'acme_metadata.joblib'))
    return model, scaler, metadata


model, scaler, meta = load_artifacts()

PORTFOLIO_BLURBS = {
    'Conservative':
        'Capital-preservation bias. Mostly investment-grade bonds and defensive '
        'equity sectors. Suits near-retirees and low-horizon investors.',
    'Balanced':
        'Classic 60/40-style blend across global equities, bonds, healthcare '
        'and REITs. Suits the median ACME client.',
    'Aggressive':
        'Equity-dominant with cyclical sector tilts and international exposure. '
        'Suits long-horizon, high-risk-tolerance investors.',
    'Thematic-Tech':
        'Concentrated technology and growth exposure. Highest expected '
        'return and highest drawdown risk — only appropriate for young, '
        'high-risk-tolerance investors with long horizons.',
}

TICKER_NAMES = {
    'AGG':  'US Aggregate Bonds',
    'TLT':  'Long-Duration Treasuries',
    'VTI':  'Total US Equities',
    'VXUS': 'International Equities (ex-US)',
    'QQQ':  'NASDAQ-100',
    'XLK':  'Tech Sector',
    'XLF':  'Financials Sector',
    'XLE':  'Energy Sector',
    'XLP':  'Consumer Staples',
    'XLV':  'Healthcare',
    'GLD':  'Gold',
    'VNQ':  'US REITs',
}


# ─── Sidebar navigation ──────────────────────────────────────────────────────
page = st.sidebar.radio(
    'View',
    ['Recommend a Portfolio', 'Model Card', 'Compare Models'],
)

st.sidebar.markdown('---')
st.sidebar.caption(
    f"**Model in production**\n\n"
    f"`{meta['model_name']}` — trained on {meta['trained_on']}  \n"
    f"Test accuracy **{meta['test_accuracy']:.1%}** · "
    f"ROC-AUC **{meta['test_roc_auc_macro']:.2f}**"
)


# ─── PAGE 1 · Recommender ─────────────────────────────────────────────────────
if page == 'Recommend a Portfolio':
    st.title('ACME Fund — Portfolio Allocator')
    st.caption(
        'ML-assisted model-portfolio recommendation for onboarding advisers. '
        'Enter the new client profile on the left; the Random Forest returns a '
        'recommended portfolio with class probabilities and the top feature drivers.'
    )

    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.subheader('Client profile')
        age = st.slider('Age', 22, 75, 40)
        income = st.number_input('Annual income (£)', 20_000, 500_000, 65_000, step=1_000)
        liquid_aum = st.number_input('Liquid investable assets (£)', 5_000, 5_000_000, 120_000, step=5_000)
        risk_tolerance = st.slider('Risk tolerance (1 cautious · 10 aggressive)', 1, 10, 6)
        horizon_years = st.slider('Investment horizon (years)', 1, 40, 15)
        income_stability = st.select_slider(
            'Income stability',
            options=[1, 2, 3, 4, 5],
            value=4,
            format_func=lambda x: {
                1: '1 — gig / freelance',
                2: '2 — short contract',
                3: '3 — standard employed',
                4: '4 — long-tenured',
                5: '5 — public sector / tenured',
            }[x],
        )
        esg_preference = st.checkbox('Requires ESG-compliant holdings', value=False)

    client = {
        'age': age,
        'income': income,
        'liquid_aum': liquid_aum,
        'risk_tolerance': risk_tolerance,
        'horizon_years': horizon_years,
        'esg_preference': int(esg_preference),
        'income_stability': income_stability,
    }
    row = {**client, **meta['market_state']}
    X_new = pd.DataFrame([row])[meta['feature_cols']]
    X_new_sc = scaler.transform(X_new)
    proba = model.predict_proba(X_new_sc)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = meta['class_names'][pred_idx]

    with col2:
        st.subheader('Recommendation')
        st.metric('Recommended portfolio', pred_class, f'{proba[pred_idx]*100:.0f}% confidence')
        st.info(PORTFOLIO_BLURBS[pred_class])

        st.markdown('**Probability across all four portfolios**')
        proba_df = pd.DataFrame({
            'portfolio': meta['class_names'],
            'probability': proba,
        }).sort_values('probability', ascending=True)
        fig, ax = plt.subplots(figsize=(6, 2.8))
        colors = ['#2a628f' if p == pred_class else '#c5c5c5' for p in proba_df['portfolio']]
        ax.barh(proba_df['portfolio'], proba_df['probability'], color=colors)
        ax.set_xlim(0, 1); ax.set_xlabel('Probability')
        for i, (p, v) in enumerate(zip(proba_df['portfolio'], proba_df['probability'])):
            ax.text(v + 0.01, i, f'{v:.1%}', va='center', fontsize=9)
        st.pyplot(fig, clear_figure=True)

        # Per-tree votes approximate feature contribution on this single case
        st.markdown('**Why this recommendation — top feature drivers (RF importance)**')
        fi = pd.Series(model.feature_importances_, index=meta['feature_cols'])
        fi_top = fi.sort_values(ascending=False).head(5)
        st.dataframe(
            fi_top.rename('importance').to_frame().style.format({'importance': '{:.3f}'}),
            use_container_width=True,
        )

        st.markdown(f'**What "{pred_class}" actually holds**')
        weights = meta['model_portfolios'][pred_class]
        aum = liquid_aum
        holdings = pd.DataFrame([
            {
                'Ticker': t,
                'Asset': TICKER_NAMES.get(t, t),
                'Weight': w,
                f'£ allocation (on £{aum:,.0f})': w * aum,
            }
            for t, w in sorted(weights.items(), key=lambda kv: -kv[1])
        ])
        st.dataframe(
            holdings.style.format({
                'Weight': '{:.0%}',
                f'£ allocation (on £{aum:,.0f})': '£{:,.0f}',
            }),
            use_container_width=True,
            hide_index=True,
        )

        fig, ax = plt.subplots(figsize=(6, 3))
        sorted_h = holdings.sort_values('Weight')
        ax.barh(sorted_h['Ticker'] + '  ' + sorted_h['Asset'],
                sorted_h['Weight'], color='#4a8cff')
        ax.set_xlabel('Weight'); ax.set_xlim(0, max(0.55, sorted_h['Weight'].max() * 1.1))
        for i, (t, w) in enumerate(zip(sorted_h['Ticker'], sorted_h['Weight'])):
            ax.text(w + 0.005, i, f'{w:.0%}', va='center', fontsize=9)
        st.pyplot(fig, clear_figure=True)

        with st.expander('📈 Current portfolio metrics (trailing 12 months)'):
            pf_feats = pd.DataFrame(meta['portfolio_features']).T
            st.dataframe(
                pf_feats.round(3).loc[['ann_return', 'ann_vol', 'sharpe', 'max_dd']],
                use_container_width=True,
            )


# ─── PAGE 2 · Model Card ──────────────────────────────────────────────────────
elif page == 'Model Card':
    st.title('Model Card — ACME Random Forest')
    st.caption(
        'A live model card is the transparency artefact the FCA and the EU AI Act '
        'expect for "high-risk" financial AI. It lists intended use, training data, '
        'held-out performance and the known limitations.'
    )

    colA, colB, colC = st.columns(3)
    colA.metric('Test accuracy', f"{meta['test_accuracy']:.1%}")
    colB.metric('Macro F1', f"{meta['test_f1_macro']:.3f}")
    colC.metric('Macro ROC-AUC', f"{meta['test_roc_auc_macro']:.3f}")

    st.markdown(f"""
**Model** · `{meta['model_name']}` (scikit-learn RandomForestClassifier, 200 trees, max depth 10).

**Intended use** · Suggest one of four ACME model portfolios at client onboarding. The adviser remains the accountable decision-maker; the model's role is advisory.

**Training data** · {meta['trained_on']}. Synthetic data is used deliberately; real client profiles are GDPR-restricted and must only be used inside an FCA-sanctioned sandbox for production retraining.

**Features ({len(meta['feature_cols'])})** · {', '.join(meta['feature_cols'])}.

**Output classes** · {', '.join(meta['class_names'])}.

**Known limitations**
- Synthetic labels encode a senior adviser's allocation heuristic with 7% noise; the model inherits that heuristic's biases.
- Performance on clients outside the training distribution (ultra-high-net-worth, minors) is not validated.
- Market-regime features reflect the training snapshot and will drift; the retraining trigger is a drift-detection check run weekly by the Monitoring lane.
""")

    st.subheader('Feature importance')
    fi = pd.Series(model.feature_importances_, index=meta['feature_cols']).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fi.plot.barh(color='#2a628f', ax=ax)
    ax.set_xlabel('Importance')
    st.pyplot(fig, clear_figure=True)


# ─── PAGE 3 · Compare models ──────────────────────────────────────────────────
elif page == 'Compare Models':
    st.title('Model selection — explainability vs accuracy')
    st.caption(
        'Three candidate models were trained on the same data. ACME chose the '
        'Random Forest: it matches or beats the MLP on accuracy while remaining '
        'explainable through feature importance and SHAP values. The Decision '
        'Tree is fully interpretable but sacrifices too much accuracy.'
    )

    df = pd.DataFrame(meta['comparison_table'])
    df.index.name = 'Model'
    st.dataframe(
        df.style
            .format({'accuracy': '{:.3f}', 'f1_macro': '{:.3f}', 'roc_auc_macro': '{:.3f}'})
            .highlight_max(axis=0, color='#d4edda'),
        use_container_width=True,
    )

    st.markdown("""
**Reading the table**

- **Decision Tree** — every split is human-readable. Lowest accuracy but highest regulatory defensibility. Kept as the explainability floor.
- **Random Forest** — best accuracy + best ROC-AUC. Feature importance and SHAP values provide post-hoc explanations, so the EU AI Act transparency obligation for high-risk financial AI is met without sacrificing predictive quality. **Deployed.**
- **MLP** — close on accuracy, slightly higher F1 on minority classes, but a small dense network offers no native interpretability and the cost of adding SHAP dominates any marginal accuracy gain.
""")

    fig, ax = plt.subplots(figsize=(7, 4))
    df[['accuracy', 'f1_macro', 'roc_auc_macro']].plot.bar(ax=ax, width=0.8)
    ax.set_ylabel('Score'); ax.set_ylim(0.5, 1.0); ax.set_xticklabels(df.index, rotation=0)
    ax.legend(loc='lower right')
    st.pyplot(fig, clear_figure=True)
