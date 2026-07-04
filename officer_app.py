"""
officer_app.py — AI-Driven Government Decision Intelligence Dashboard

Run:  streamlit run officer_app.py
"""
import datetime
import os
import re
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from modules.analytics_engine import AnalyticsEngine
from modules.copilot import AICopilot
from modules.data_processor import DataProcessor
from modules.decision_engine import DecisionEngine
from modules.nlp_engine import NLPEngine
from modules.optimizer import ResourceOptimizer
from modules.simulator import PolicySimulator

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(page_title="GovIntelligence — Officer Dashboard",
                   page_icon="🛡️", layout="wide")

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container{padding-top:1.5rem}
    .metric-card{background:#1b263b;border-radius:12px;padding:1.2rem;
        text-align:center;border:1px solid #415a77}
    .metric-val{font-size:2rem;font-weight:800;color:#00b4d8}
    .metric-lbl{font-size:.85rem;color:#778da9;margin-top:.2rem}
    .kpi-row{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1.5rem}
    .kpi-col{flex:1;min-width:150px}
</style>""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_data():
    dp = DataProcessor()
    df = dp.get_combined()
    return df, dp

df, dp = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.image("NYC-City-image.jpg", width=220)
st.sidebar.title("🛡️ GovIntelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "📊 Dashboard",
    "🧠 NLP Analysis",
    "📈 Analytics & Prediction",
    "⚖️ Decision Engine",
    "🔧 Resource Optimizer",
    "📜 Policy Simulator",
    "🤖 AI Copilot",
    "🗺️ Geo Map"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Data Refreshed:** "
                    f"{datetime.datetime.now().strftime('%H:%M:%S')}")
if st.sidebar.button("🔄 Reload Data"):
    st.cache_data.clear()
    st.rerun()

# ── PDF report button ──────────────────────────────────────────────────
st.sidebar.markdown("---")
if st.sidebar.button("📄 Generate Weekly PDF Report"):
    try:
        with st.spinner("Generating report…"):
            from generate_report import generate_weekly_report
            path = generate_weekly_report(df)
            if path:
                st.session_state.report_path = path
            else:
                st.sidebar.error("❌ Failed to generate report.")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")

# Keep download button alive using session state
if 'report_path' in st.session_state and st.session_state.report_path:
    path = st.session_state.report_path
    if os.path.exists(path):
        with open(path, 'rb') as f:
            st.sidebar.download_button(
                "⬇️ Download PDF Report", f,
                file_name=os.path.basename(path),
                mime='application/pdf'
            )
        st.sidebar.success("✅ Report ready for download!")

# ══════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Dashboard Overview")

    if df.empty:
        st.warning("No data loaded. Check dataset files.")
        st.stop()

    # ── KPIs ────────────────────────────────────────────────────────
    total = len(df)
    open_cnt = len(df[df['status'].isin(['Open', 'In Progress'])]) \
        if 'status' in df.columns else 0
    closed_cnt = len(df[df['status'] == 'Closed']) \
        if 'status' in df.columns else 0
    boroughs = df['borough'].nunique() if 'borough' in df.columns else 0
    agencies = df['agency'].nunique() if 'agency' in df.columns else 0
    live_cnt = len(df[df['source'] == 'live']) if 'source' in df.columns else 0

    cols = st.columns(6)
    kpis = [
        ("Total Complaints", f"{total:,}", "📋"),
        ("Open / In Progress", f"{open_cnt:,}", "🟡"),
        ("Closed", f"{closed_cnt:,}", "🟢"),
        ("Boroughs", f"{boroughs}", "🗽"),
        ("Agencies", f"{agencies}", "🏢"),
        ("Live Submissions", f"{live_cnt}", "👤"),
    ]
    for col, (label, val, icon) in zip(cols, kpis):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-val'>{icon} {val}</div>
            <div class='metric-lbl'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Borough histogram ───────────────────────────────────────────
    st.subheader("Complaints by Borough")
    if 'borough' in df.columns:
        bdata = df['borough'].value_counts().reset_index()
        bdata.columns = ['Borough', 'Count']
        fig = px.bar(bdata, x='Borough', y='Count', color='Borough',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    # ── Agency histogram ────────────────────────────────────────────
    st.subheader("Complaints by Agency")
    if 'agency' in df.columns:
        adata = df['agency'].value_counts().reset_index()
        adata.columns = ['Agency', 'Count']
        fig2 = px.bar(adata, x='Agency', y='Count', color='Agency',
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Status pie ──────────────────────────────────────────────────
    st.subheader("Status Breakdown")
    if 'status' in df.columns:
        sdata = df['status'].value_counts().reset_index()
        sdata.columns = ['Status', 'Count']
        fig3 = px.pie(sdata, names='Status', values='Count',
                      hole=0.45,
                      color_discrete_sequence=px.colors.qualitative.Set3)
        fig3.update_layout(height=380)
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE: NLP Analysis
# ══════════════════════════════════════════════════════════════════════
elif page == "🧠 NLP Analysis":
    st.title("🧠 NLP Sentiment & Urgency Analysis")
    nlp = NLPEngine()

    # ── Single complaint analysis ───────────────────────────────────
    st.subheader("Analyze a Complaint")
    sample_text = st.text_area("Enter complaint description:",
                               height=120,
                               placeholder="Loud music coming from the "
                                           "bar next door at 2 AM…")
    sample_type = st.selectbox("Complaint Type",
                               options=[''] + dp.get_complaint_types())

    if st.button("🔍 Analyze"):
        if sample_text.strip():
            result = nlp.analyze_complaint(sample_text, sample_type)
            c1, c2, c3 = st.columns(3)
            c1.metric("Sentiment", result['sentiment']['label'],
                      f"Polarity: {result['sentiment']['polarity']}")
            c2.metric("Urgency", result['urgency']['level'],
                      f"Score: {result['urgency']['score']}/10")
            c3.metric("Keywords", len(result['keywords']),
                      ', '.join(result['keywords'][:5]))
        else:
            st.warning("Please enter some text.")

    st.markdown("---")

    # ── Batch analysis ──────────────────────────────────────────────
    st.subheader("Batch Analysis (Top 50)")
    if not df.empty and 'complaint_type' in df.columns:
        sample_df = df.head(50).copy()
        sentiments, urgencies = [], []
        for _, row in sample_df.iterrows():
            txt = str(row.get('descriptor', ''))
            ct = row.get('complaint_type', '')
            s = nlp.analyze_sentiment(txt)
            u = nlp.detect_urgency(txt, ct)
            sentiments.append(s['label'])
            urgencies.append(u['level'])
        sample_df['Sentiment'] = sentiments
        sample_df['Urgency'] = urgencies

        c1, c2 = st.columns(2)
        with c1:
            fig_s = px.histogram(sample_df, x='Sentiment',
                                 title='Sentiment Distribution',
                                 color='Sentiment')
            st.plotly_chart(fig_s, use_container_width=True)
        with c2:
            fig_u = px.histogram(sample_df, x='Urgency',
                                 title='Urgency Distribution',
                                 color='Urgency')
            st.plotly_chart(fig_u, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE: Analytics & Prediction
# ══════════════════════════════════════════════════════════════════════
elif page == "📈 Analytics & Prediction":
    st.title("📈 Analytics & Prediction")
    analytics = AnalyticsEngine(df)

    # ── Trend chart ─────────────────────────────────────────────────
    st.subheader("Complaint Trend Over Time")
    trend = analytics.get_trend_data()
    if not trend.empty:
        fig_t = px.line(trend, x='date', y='count',
                        title='Daily Complaint Volume',
                        labels={'date': 'Date', 'count': 'Complaints'})
        fig_t.update_traces(line_color='#00b4d8')
        fig_t.add_scatter(x=trend['date'],
                          y=trend['count'].rolling(3, min_periods=1).mean(),
                          mode='lines', name='3-day MA',
                          line=dict(dash='dash', color='orange'))
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("No date data available for trends.")

    # ── Crisis early warning ────────────────────────────────────────
    st.subheader("🚨 Crisis Early Warning")
    alerts = analytics.detect_crisis()
    if alerts:
        for a in alerts:
            st.error(f"⚠️ **{a['date']}** — {a['count']} complaints "
                     f"(expected ~{a['expected']}, "
                     f"deviation: +{a['deviation']}σ)")
    else:
        st.success("✅ No crisis-level spikes detected in the current data.")

    # ── Prediction ──────────────────────────────────────────────────
    st.subheader("🔮 7-Day Prediction")
    pred = analytics.predict_next_week()
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Daily Avg", f"{pred['predicted_daily']}")
    c2.metric("Predicted Weekly Total", f"{pred['predicted_weekly']}")
    c3.metric("Confidence", pred['confidence'].capitalize())

    # ── Complaint type breakdown ────────────────────────────────────
    st.subheader("Top Complaint Types")
    ct_dist = analytics.get_complaint_type_distribution()
    if not ct_dist.empty:
        fig_ct = px.bar(ct_dist, x='complaint_type', y='count',
                        color='complaint_type',
                        title='Top 15 Complaint Types')
        fig_ct.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig_ct, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE: Decision Engine
# ══════════════════════════════════════════════════════════════════════
elif page == "⚖️ Decision Engine":
    st.title("⚖️ Decision Recommendation Engine")
    engine = DecisionEngine(df)

    st.markdown("""
    <div class='info-box'>
    The engine scores each complaint on Severity, Urgency, Sentiment,
    Borough Vulnerability, and Flag status to produce a composite
    priority score (0-100).
    </div>""", unsafe_allow_html=True)

    if df.empty:
        st.warning("No data available.")
        st.stop()

    with st.spinner("Scoring all complaints…"):
        scored = engine.score_all()

    # ── Priority distribution ───────────────────────────────────────
    st.subheader("Priority Distribution")
    if 'priority_label' in scored.columns:
        pcounts = scored['priority_label'].value_counts().reset_index()
        pcounts.columns = ['Priority', 'Count']

        PRIORITY_COLORS = {
            '🔴 Critical': '#d62828',
            '🟠 High':     '#e76f51',
            '🟡 Medium':   '#f4a261',
            '🟢 Low':      '#2a9d8f',
        }

        fig_p = px.pie(pcounts, names='Priority', values='Count',
                       hole=0.4, color='Priority',
                       color_discrete_map=PRIORITY_COLORS)
        st.plotly_chart(fig_p, use_container_width=True)

    # ── Top recommendations ─────────────────────────────────────────
    st.subheader("🔍 Top 10 Priority Recommendations")
    recs = engine.generate_recommendations(top_n=10)
    if recs:
        rec_df = pd.DataFrame(recs)
        st.dataframe(rec_df[['unique_key', 'complaint_type', 'borough',
                             'agency', 'priority_score', 'priority_label',
                             'action']],
                     use_container_width=True, height=400)

# ══════════════════════════════════════════════════════════════════════
# PAGE: Resource Optimizer
# ══════════════════════════════════════════════════════════════════════
elif page == "🔧 Resource Optimizer":
    st.title("🔧 Resource Allocation Optimizer")
    opt = ResourceOptimizer(df)

    total_res = st.slider("Total Available Resources (officers/inspectors)",
                          100, 2000, 500, step=50)

    alloc = opt.optimize_allocation(total_resources=total_res)

    if alloc:
        st.subheader("📊 Optimized Allocation")
        alloc_df = pd.DataFrame(alloc).T.reset_index()
        alloc_df.columns = ['Agency', 'Resources', 'Open Complaints',
                            'Complaints/Resource', 'Share %']
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)

        fig_r = px.bar(alloc_df, x='Agency', y='Resources',
                       color='Agency',
                       title=f'Resource Allocation (Total: {total_res})',
                       text='Resources')
        fig_r.update_traces(textposition='outside')
        st.plotly_chart(fig_r, use_container_width=True)

        st.subheader("💡 Reallocation Suggestions")
        suggestions = opt.suggest_reallocation()
        if suggestions:
            for s in suggestions:
                st.markdown(f"• {s}")
        else:
            st.success("✅ Resources are well-balanced across agencies.")

    else:
        st.info("Not enough data for optimization.")

# ══════════════════════════════════════════════════════════════════════
# PAGE: Policy Simulator
# ══════════════════════════════════════════════════════════════════════
elif page == "📜 Policy Simulator":
    st.title("📜 Policy Impact Simulator")
    sim = PolicySimulator(df)

    st.markdown("Adjust the budget allocation sliders to see projected "
                "impact on resolution rates and response times.")

    col1, col2 = st.columns(2)
    with col1:
        hpd_pct = st.slider("HPD Budget %", 5, 60, 35, 1)
        dsny_pct = st.slider("DSNY Budget %", 5, 60, 25, 1)
    with col2:
        nypd_pct = st.slider("NYPD Budget %", 5, 60, 30, 1)
        other_pct = max(0, 100 - hpd_pct - dsny_pct - nypd_pct)
        st.metric("Other Agencies Budget %", f"{other_pct}")

    if hpd_pct + dsny_pct + nypd_pct > 100:
        st.error("❌ Total exceeds 100%. Adjust sliders.")
    else:
        budget = {'HPD': hpd_pct, 'DSNY': dsny_pct,
                  'NYPD': nypd_pct, 'OTHER': other_pct}
        results = sim.simulate_budget(budget)

        st.subheader("📊 Projected Impact")
        rows = []
        for agency, info in results.items():
            rows.append({
                'Agency': agency,
                'Budget %': info['budget_pct'],
                'Current Resol. Rate': f"{info['current_resolution_rate']}%",
                'Projected Resol. Rate': f"{info['projected_resolution_rate']}%",
                'Current Avg Hours': info['current_avg_hours'],
                'Projected Avg Hours': info['projected_avg_hours'],
                'Budget Δ': f"{info['change_pct']:+.1f}%"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

        compare_df = pd.DataFrame([
            {'Agency': k, 'Current Rate': v['current_resolution_rate'],
             'Projected Rate': v['projected_resolution_rate']}
            for k, v in results.items()
        ])
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Bar(name='Current', x=compare_df['Agency'],
                                 y=compare_df['Current Rate'],
                                 marker_color='#415a77'))
        fig_sim.add_trace(go.Bar(name='Projected', x=compare_df['Agency'],
                                 y=compare_df['Projected Rate'],
                                 marker_color='#00b4d8'))
        fig_sim.update_layout(barmode='group',
                              title='Resolution Rate: Current vs Projected',
                              yaxis_title='Resolution Rate (%)')
        st.plotly_chart(fig_sim, use_container_width=True)

    st.markdown("---")
    st.subheader("🏛️ Policy Presets")
    policy = st.selectbox("Select a policy to simulate:",
                          ['proactive_patrols', 'housing_inspection',
                           'sanitation_blitz'])
    intensity = st.slider("Intensity (1-3)", 1.0, 3.0, 1.5, 0.5)
    impact = sim.simulate_policy_change(policy, intensity)
    if impact:
        for agency, vals in impact.items():
            st.info(f"🏢 **{agency}**: Resolution rate "
                    f"{vals['resolution_rate_change']:+.1f}%, "
                    f"Avg hours {vals['avg_hours_change']:+.1f}h")

# ══════════════════════════════════════════════════════════════════════
# PAGE: AI Copilot
# ══════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Copilot":
    st.title("🤖 AI Copilot")
    st.markdown("Ask questions about the complaint data in plain English.")

    copilot = AICopilot(df)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.markdown("**Try asking:**")
    qp_cols = st.columns(5)
    quick_prompts = [
        "Top 5 complaint types",
        "Any recommendation?",
        "Complaints by Borough",
        "Show trend"
    ]
    for col, qp in zip(qp_cols, quick_prompts):
        if col.button(qp):
            st.session_state.chat_history.append(
                {"role": "user", "content": qp})
            answer = copilot.process_query(qp)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer})

    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    user_q = st.chat_input("Ask about complaint data…")
    if user_q:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_q})
        answer = copilot.process_query(user_q)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer})
        st.rerun()

# ══════════════════════════════════════════════════════════════════════
# PAGE: Geo Map
# ══════════════════════════════════════════════════════════════════════
elif page == "🗺️ Geo Map":
    st.title("🗺️NYC 311 Complaints — Geographic Distribution")

    if df.empty:
        st.warning("No data loaded.")
        st.stop()

    map_df = df.copy()

    if 'latitude' in map_df.columns:
        map_df['latitude'] = pd.to_numeric(map_df['latitude'], errors='coerce')
    else:
        map_df['latitude'] = np.nan

    if 'longitude' in map_df.columns:
        map_df['longitude'] = pd.to_numeric(map_df['longitude'], errors='coerce')
    else:
        map_df['longitude'] = np.nan

    # Fallback: parse the 'location' WKT column  "POINT (-73.99 40.74)"
    if 'location' in map_df.columns and map_df['latitude'].isna().any():
        def _parse_wkt(val):
            try:
                m = re.search(r'POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)', str(val))
                if m:
                    return pd.Series([float(m.group(2)), float(m.group(1))])
            except Exception:
                pass
            return pd.Series([np.nan, np.nan])

        missing = map_df['latitude'].isna()
        parsed = map_df.loc[missing, 'location'].apply(_parse_wkt)
        parsed.columns = ['latitude', 'longitude']
        map_df.loc[missing, 'latitude'] = parsed['latitude']
        map_df.loc[missing, 'longitude'] = parsed['longitude']

    map_df = map_df.dropna(subset=['latitude', 'longitude'])

    map_df = map_df[
        (map_df['latitude'].between(40.4, 41.0)) &
        (map_df['longitude'].between(-74.3, -73.5))
    ]

    if map_df.empty:
        st.error("❌ No valid coordinates found in the dataset.")
        st.stop()

    st.metric("📍 Mappable Complaints", f"{len(map_df):,}")

    col1, col2, col3 = st.columns(3)
    with col1:
        borough_opts = sorted(map_df['borough'].dropna().unique().tolist()) if 'borough' in map_df.columns else []
        sel_borough = st.multiselect("Filter by Borough", options=borough_opts)
    with col2:
        agency_opts = sorted(map_df['agency'].dropna().unique().tolist()) if 'agency' in map_df.columns else []
        sel_agency = st.multiselect("Filter by Agency", options=agency_opts)
    with col3:
        type_opts = sorted(map_df['complaint_type'].dropna().unique().tolist()) if 'complaint_type' in map_df.columns else []
        sel_type = st.multiselect("Filter by Complaint Type", options=type_opts)

    filtered = map_df.copy()
    if sel_borough:
        filtered = filtered[filtered['borough'].isin(sel_borough)]
    if sel_agency:
        filtered = filtered[filtered['agency'].isin(sel_agency)]
    if sel_type:
        filtered = filtered[filtered['complaint_type'].isin(sel_type)]

    st.metric("📊 Showing", f"{len(filtered):,} complaints")

    if filtered.empty:
        st.warning("No complaints match the selected filters.")
        st.stop()

    agency_color_map = {
        'NYPD': '#d62828', 'HPD': '#457b9d', 'DSNY': '#2a9d8f',
        'DOT': '#6a4c93', 'DHS': '#e76f51', 'DOB': '#264653',
        'DOHMH': '#1d3557', 'TLC': '#e9c46a', 'DEP': '#606c38'
    }

    hover_cols = []
    for col_name in ['complaint_type', 'agency', 'borough', 'status', 'incident_address']:
        if col_name in filtered.columns:
            hover_cols.append(col_name)

    fig_map = px.scatter_mapbox(
        filtered,
        lat='latitude',
        lon='longitude',
        color='agency' if 'agency' in filtered.columns else None,
        color_discrete_map=agency_color_map,
        hover_name='complaint_type' if 'complaint_type' in filtered.columns else None,
        hover_data=hover_cols,
        zoom=9,
        height=600,
        
        mapbox_style='open-street-map',
    )

    fig_map.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig_map.update_traces(marker=dict(size=7, opacity=0.7))

    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("📋 Complaint Details")
    display_cols = [c for c in ['unique_key', 'complaint_type', 'agency', 'borough', 'status', 'incident_address', 'created_date'] if c in filtered.columns]
    st.dataframe(filtered[display_cols].head(200), use_container_width=True, height=350)
