"""
citizen_app.py — NYC 311 Citizen Complaint Submission Portal

Run:  streamlit run citizen_app.py
"""
import os
import sys

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, os.path.dirname(__file__))
from modules.data_processor import DataProcessor

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(page_title="NYC 311 — File a Complaint",
                   page_icon="🏙️", layout="wide")

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container{padding-top:2rem}
    .stApp{background:linear-gradient(135deg,#0d1b2a 0%,#1b263b 100%)}
    h1,h2,h3,h4,label,.stMarkdown{color:#e0e1dd !important}
    .stButton>button{background:#415a77;border:none;color:#e0e1dd;
        font-weight:bold;border-radius:8px;padding:.6rem 2rem}
    .stButton>button:hover{background:#778da9}
    .success-box{background:#2d6a4f;padding:1.5rem;border-radius:12px;
        color:#d8f3dc;margin-top:1rem}
    .warning-box{background:#9b2226;padding:1rem;border-radius:8px;
        color:#fec89a;margin-top:.5rem}
    .info-box{background:#1b4965;padding:1rem;border-radius:8px;
        color:#bee1e6;margin-bottom:1rem}
</style>""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding-bottom:.5rem'>
    <h1>🏙️ NYC 311 Citizen Portal</h1>
    <p style='color:#778da9;font-size:1.1rem'>
       File a complaint and help make New York City better.</p>
</div>""", unsafe_allow_html=True)

# ── Data processor ─────────────────────────────────────────────────────
dp = DataProcessor()

# ── Form ───────────────────────────────────────────────────────────────
with st.form("complaint_form", clear_on_submit=False):
    st.markdown("### 📝 Complaint Details")

    col1, col2 = st.columns(2)
    with col1:
        full_name = st.text_input("Full Name *", placeholder="John Doe")
        mobile = st.text_input("Mobile Number *",
                               placeholder="212-555-0123")
    with col2:
        borough = st.selectbox("Borough *",
                               options=dp.get_boroughs())
        complaint_type = st.selectbox("Complaint Type *",
                                     options=dp.get_complaint_types())

    location_type = st.selectbox("Location Type",
                                 options=[""] + dp.get_location_types())
    description = st.text_area("Detailed Description *", height=130,
                               placeholder="Describe the issue in detail…")
    street_address = st.text_input("Street Address (optional)",
                                   placeholder="123 Main Street")

    # ── Interactive map picker ──────────────────────────────────────
    st.markdown("### 🗺️ Pick Your Location on the Map")
    st.markdown("<div class='info-box'>Click anywhere on the map to set "
                "the exact latitude & longitude of the complaint.</div>",
                unsafe_allow_html=True)

    # Session-state defaults
    if 'picked_lat' not in st.session_state:
        st.session_state.picked_lat = 40.7128
    if 'picked_lon' not in st.session_state:
        st.session_state.picked_lon = -74.0060

    m = folium.Map(location=[st.session_state.picked_lat,
                             st.session_state.picked_lon],
                   zoom_start=11, tiles="OpenStreetMap")
    folium.Marker(
        [st.session_state.picked_lat, st.session_state.picked_lon],
        popup="Complaint Location", icon=folium.Icon(color='red')
    ).add_to(m)

    map_data = st_folium(m, height=350, width=700)

    if map_data and map_data.get('last_clicked'):
        st.session_state.picked_lat = map_data['last_clicked']['lat']
        st.session_state.picked_lon = map_data['last_clicked']['lng']

    col_a, col_b = st.columns(2)
    with col_a:
        lat = st.number_input("Latitude",
                              value=st.session_state.picked_lat,
                              format="%.6f")
    with col_b:
        lon = st.number_input("Longitude",
                              value=st.session_state.picked_lon,
                              format="%.6f")

    # ── Submit ──────────────────────────────────────────────────────
    submitted = st.form_submit_button("🚀  Submit Complaint", type="primary")

# ── Processing on submit ───────────────────────────────────────────────
if submitted:
    errors = []
    if not full_name.strip():
        errors.append("Full Name is required.")
    if not mobile.strip():
        errors.append("Mobile Number is required.")
    if not description.strip():
        errors.append("Detailed Description is required.")

    if errors:
        for e in errors:
            st.error(f"❌ {e}")
        st.stop()

    # ── Fake complaint detection ────────────────────────────────────
    fake_result = dp.detect_fake_complaint(borough, complaint_type)

    # ── Build complaint record ──────────────────────────────────────
    complaint_data = {
        'borough': borough,
        'complaint_type': complaint_type,
        'descriptor': description[:200],
        'descriptor_2': '',
        'location_type': location_type,
        'incident_address': street_address,
        'street_name': street_address.split()[0] if street_address else '',
        'latitude': lat,
        'longitude': lon,
        'incident_zip': '',
        'citizen_name': full_name,
        'citizen_mobile': mobile,
        'is_flagged': fake_result['is_flagged'],
        'flag_reason': fake_result.get('flag_reason', ''),
        'source': 'live'
    }

    complaint_id = dp.save_complaint(complaint_data)

    # ── Success message ─────────────────────────────────────────────
    st.markdown(f"""
    <div class='success-box'>
        <h3>✅ Complaint Submitted Successfully!</h3>
        <p><strong>Complaint ID:</strong> <code>{complaint_id}</code></p>
        <p><strong>Agency:</strong> {dp.get_agency_for_type(complaint_type)}
           — {dp.get_agency_name(dp.get_agency_for_type(complaint_type))}</p>
        <p><strong>Borough:</strong> {borough}</p>
        <p><strong>Status:</strong> Open</p>
    </div>""", unsafe_allow_html=True)

    if fake_result['is_flagged']:
        st.markdown(f"""
        <div class='warning-box'>
            ⚠️ <strong>Flagged for Review:</strong>
            {fake_result['flag_reason']}<br>
            Risk Score: {fake_result['risk_score']}/100
        </div>""", unsafe_allow_html=True)