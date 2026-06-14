"""
dashboard/app.py
Streamlit A/B Test Analysis Dashboard
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import json, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.stats_engine import (
    two_proportion_ztest, mann_whitney_revenue,
    bayesian_ab, sample_size_calculator, run_full_analysis
)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A/B Test Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {background:#f8f9fa;border-radius:8px;padding:16px;border:1px solid #e9ecef}
    .sig-badge {background:#d4edda;color:#155724;padding:4px 12px;border-radius:20px;font-weight:600}
    .not-sig-badge {background:#f8d7da;color:#721c24;padding:4px 12px;border-radius:20px;font-weight:600}
    .ship-it {color:#28a745;font-size:1.3rem;font-weight:700}
    .no-ship {color:#dc3545;font-size:1.3rem;font-weight:700}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.title("🧪 A/B Test Config")
st.sidebar.markdown("---")

data_source = st.sidebar.radio("Data Source", ["Use Sample Dataset", "Upload CSV"])

alpha   = st.sidebar.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)
power   = st.sidebar.slider("Target power (1−β)", 0.70, 0.95, 0.80, 0.05)
segment = st.sidebar.selectbox("Segment Filter", ["All", "USA", "UK", "Canada", "Australia"])
device  = st.sidebar.selectbox("Device Filter", ["All", "desktop", "mobile", "tablet"])

st.sidebar.markdown("---")
st.sidebar.markdown("**About this dashboard**")
st.sidebar.markdown(
    "End-to-end A/B test analysis with frequentist "
    "z-tests, Mann-Whitney U for revenue, Bayesian "
    "posterior estimation, and CUPED variance reduction."
)

# ── Load Data ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=['date'])

if data_source == "Upload CSV":
    uploaded = st.file_uploader("Upload A/B test CSV", type=["csv"])
    if uploaded:
        df_raw = pd.read_csv(uploaded, parse_dates=['date'])
    else:
        st.info("Upload a CSV with columns: user_id, group, date, converted, revenue, device, country")
        st.stop()
else:
    df_raw = load_data(os.path.join(os.path.dirname(__file__), '../data/ab_test_raw.csv'))

# ── Apply Filters ─────────────────────────────────────────────────────────
df = df_raw.copy()
if segment != "All":
    df = df[df['country'] == segment]
if device != "All":
    df = df[df['device'] == device]

control   = df[df['group'] == 'control']
treatment = df[df['group'] == 'treatment']

# ── Run Stats ─────────────────────────────────────────────────────────────
freq_results = two_proportion_ztest(
    int(control['converted'].sum()),   len(control),
    int(treatment['converted'].sum()), len(treatment),
    alpha=alpha
)
rev_results  = mann_whitney_revenue(control['revenue'].values, treatment['revenue'].values)
bayes_results= bayesian_ab(
    int(control['converted'].sum()),   len(control),
    int(treatment['converted'].sum()), len(treatment)
)

# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════
st.title("🧪 A/B Test Analysis Dashboard")
st.markdown("**Test:** Streamlined Checkout (1-step vs 3-step) | **Period:** Mar 1 – Mar 30, 2025")

is_sig = freq_results["is_significant"]
lift   = freq_results["relative_lift"]

col_rec, col_sig, col_lift = st.columns([2, 1, 1])
with col_rec:
    label = "SHIP IT ✅" if is_sig and lift > 0 else "DO NOT SHIP ❌"
    css   = "ship-it" if is_sig and lift > 0 else "no-ship"
    st.markdown(f'<p class="{css}">Recommendation: {label}</p>', unsafe_allow_html=True)
with col_sig:
    badge = "sig-badge" if is_sig else "not-sig-badge"
    st.markdown(
        f'<span class="{badge}">{"✓ Significant" if is_sig else "✗ Not Significant"}</span><br>'
        f'<small>p = {freq_results["p_value"]:.4f} (α = {alpha})</small>',
        unsafe_allow_html=True
    )
with col_lift:
    st.metric("Relative Lift", f"{lift:+.1f}%",
              delta=f"{freq_results['absolute_lift']:+.2f}pp absolute")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════════════════
st.subheader("📊 Key Metrics")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Control Users",   f"{len(control):,}")
k2.metric("Treatment Users", f"{len(treatment):,}")
k3.metric("Control CVR",     f"{freq_results['control_rate']:.2f}%")
k4.metric("Treatment CVR",   f"{freq_results['treatment_rate']:.2f}%",
          delta=f"{freq_results['absolute_lift']:+.2f}pp")
k5.metric("Control Avg Rev", f"${rev_results['control_mean_rev']:.2f}")
k6.metric("Treatment Avg Rev",f"${rev_results['treatment_mean_rev']:.2f}",
          delta=f"${rev_results['treatment_mean_rev'] - rev_results['control_mean_rev']:+.2f}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════
# CHARTS ROW 1: Conversion + Daily Trend
# ══════════════════════════════════════════════════════════════════════════
col1, col2 = st.columns(2)

with col1:
    st.subheader("Conversion Rate Comparison")
    ci_low  = freq_results["ci_low"]
    ci_high = freq_results["ci_high"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Control',
        x=['Control'],
        y=[freq_results['control_rate']],
        error_y=dict(type='data', array=[1.96 * freq_results['control_rate'] * 0.1]),
        marker_color='#6c757d', width=0.35
    ))
    fig.add_trace(go.Bar(
        name='Treatment',
        x=['Treatment'],
        y=[freq_results['treatment_rate']],
        error_y=dict(type='data', array=[1.96 * freq_results['treatment_rate'] * 0.1]),
        marker_color='#28a745' if is_sig else '#ffc107', width=0.35
    ))
    fig.add_annotation(
        text=f"95% CI for lift: [{ci_low:.2f}pp, {ci_high:.2f}pp]",
        xref="paper", yref="paper", x=0.5, y=1.05,
        showarrow=False, font=dict(size=11)
    )
    fig.update_layout(
        yaxis_title="Conversion Rate (%)",
        showlegend=True, height=320,
        plot_bgcolor='white', paper_bgcolor='white',
        yaxis=dict(gridcolor='#f0f0f0')
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Daily Conversion Rate Trend")
    daily = df.groupby(['date', 'group'])['converted'].agg(['mean', 'count']).reset_index()
    daily.columns = ['date', 'group', 'cvr', 'n']
    daily['cvr'] = daily['cvr'] * 100
    daily['date'] = pd.to_datetime(daily['date'])

    fig2 = px.line(daily, x='date', y='cvr', color='group',
                   color_discrete_map={'control': '#6c757d', 'treatment': '#28a745'},
                   markers=True, labels={'cvr': 'Conversion Rate (%)', 'date': 'Date'})
    fig2.update_layout(height=320, plot_bgcolor='white', paper_bgcolor='white',
                       yaxis=dict(gridcolor='#f0f0f0'))
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# CHARTS ROW 2: Revenue Distribution + Bayesian
# ══════════════════════════════════════════════════════════════════════════
col3, col4 = st.columns(2)

with col3:
    st.subheader("Revenue Distribution (Converters Only)")
    c_rev = control[control['converted'] == 1]['revenue']
    t_rev = treatment[treatment['converted'] == 1]['revenue']

    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=c_rev, name='Control', opacity=0.6,
        marker_color='#6c757d', nbinsx=30
    ))
    fig3.add_trace(go.Histogram(
        x=t_rev, name='Treatment', opacity=0.6,
        marker_color='#28a745', nbinsx=30
    ))
    fig3.add_annotation(
        text=f"Mann-Whitney p = {rev_results['p_value']:.4f}",
        xref="paper", yref="paper", x=0.7, y=0.9,
        showarrow=False, bgcolor='white', bordercolor='gray',
        borderwidth=1, font=dict(size=11)
    )
    fig3.update_layout(barmode='overlay', height=320, xaxis_title='Revenue ($)',
                       yaxis_title='Count', plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Bayesian Posterior Distribution")
    n_samples = 50_000
    np.random.seed(42)
    c_post = np.random.beta(
        1 + int(control['converted'].sum()),
        1 + len(control) - int(control['converted'].sum()),
        n_samples
    )
    t_post = np.random.beta(
        1 + int(treatment['converted'].sum()),
        1 + len(treatment) - int(treatment['converted'].sum()),
        n_samples
    )
    prob_win = (t_post > c_post).mean() * 100

    x = np.linspace(0.02, 0.12, 300)
    fig4 = go.Figure()
    from scipy.stats import beta as beta_dist
    c_pdf = beta_dist.pdf(x,
        1 + int(control['converted'].sum()),
        1 + len(control) - int(control['converted'].sum()))
    t_pdf = beta_dist.pdf(x,
        1 + int(treatment['converted'].sum()),
        1 + len(treatment) - int(treatment['converted'].sum()))
    fig4.add_trace(go.Scatter(x=x*100, y=c_pdf, fill='tozeroy', name='Control posterior',
                              fillcolor='rgba(108,117,125,0.3)', line_color='#6c757d'))
    fig4.add_trace(go.Scatter(x=x*100, y=t_pdf, fill='tozeroy', name='Treatment posterior',
                              fillcolor='rgba(40,167,69,0.3)', line_color='#28a745'))
    fig4.add_annotation(
        text=f"P(Treatment > Control) = {prob_win:.1f}%",
        xref="paper", yref="paper", x=0.5, y=0.9,
        showarrow=False, bgcolor='white', bordercolor='gray',
        borderwidth=1, font=dict(size=12, color='#155724' if prob_win >= 95 else '#856404')
    )
    fig4.update_layout(height=320, xaxis_title='Conversion Rate (%)',
                       yaxis_title='Density', plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# ROW 3: Segment Breakdown + Sample Size Calculator
# ══════════════════════════════════════════════════════════════════════════
col5, col6 = st.columns(2)

with col5:
    st.subheader("Conversion by Device & Country")
    seg_df = df.groupby(['group', 'device'])['converted'].mean().reset_index()
    seg_df['converted'] *= 100
    fig5 = px.bar(seg_df, x='device', y='converted', color='group', barmode='group',
                  color_discrete_map={'control': '#6c757d', 'treatment': '#28a745'},
                  labels={'converted': 'Conversion Rate (%)', 'device': 'Device'})
    fig5.update_layout(height=280, plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    st.subheader("🔢 Sample Size Calculator")
    ss_baseline = st.slider("Baseline conversion rate (%)", 1.0, 20.0, 5.0, 0.5) / 100
    ss_mde      = st.slider("Minimum Detectable Effect (pp)", 0.5, 5.0, 1.0, 0.1) / 100

    ss_result = sample_size_calculator(ss_baseline, ss_mde, alpha=alpha, power=power)
    c1, c2 = st.columns(2)
    c1.metric("Per Variant",  f"{ss_result['required_n_per_variant']:,}")
    c2.metric("Total Needed", f"{ss_result['total_required_n']:,}")

    weeks_at_1k = ss_result['total_required_n'] / 1000
    st.caption(f"At 1,000 visitors/day: ~{weeks_at_1k/7:.1f} weeks to reach significance")

# ══════════════════════════════════════════════════════════════════════════
# STATS SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📋 Full Statistical Summary")

summary = {
    "Metric": [
        "Control Conversion Rate", "Treatment Conversion Rate",
        "Absolute Lift", "Relative Lift",
        "95% CI (lower)", "95% CI (upper)",
        "Z-Statistic", "P-Value (Frequentist)", "Significant?",
        "P(Treatment Wins) Bayesian",
        "Control Avg Revenue", "Treatment Avg Revenue",
        "Revenue P-Value (Mann-Whitney)"
    ],
    "Value": [
        f"{freq_results['control_rate']:.2f}%",
        f"{freq_results['treatment_rate']:.2f}%",
        f"{freq_results['absolute_lift']:+.2f}pp",
        f"{freq_results['relative_lift']:+.1f}%",
        f"{freq_results['ci_low']:+.3f}pp",
        f"{freq_results['ci_high']:+.3f}pp",
        f"{freq_results['z_statistic']:.4f}",
        f"{freq_results['p_value']:.6f}",
        "✅ Yes" if is_sig else "❌ No",
        f"{bayes_results['prob_treatment_wins']:.1f}%",
        f"${rev_results['control_mean_rev']:.2f}",
        f"${rev_results['treatment_mean_rev']:.2f}",
        f"{rev_results['p_value']:.6f}"
    ]
}
st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

st.caption("Built by Gopika Sree Kumar | MS Data Science, University at Buffalo")
