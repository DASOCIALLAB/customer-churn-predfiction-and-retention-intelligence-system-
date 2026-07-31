"""Custom CSS injection and shared UI helpers."""

import streamlit as st


def inject_css():
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg,#0b0e17 0%, #10121c 100%); }
        .kpi-card {
            background: linear-gradient(135deg, #6d28d9 0%, #2563eb 100%);
            border-radius: 16px; padding: 20px 24px; box-shadow: 0 8px 24px rgba(76,29,149,0.35);
            color: white; margin-bottom: 8px; height: 100%;
        }
        .kpi-label { font-size:13px; opacity:0.85; text-transform:uppercase; letter-spacing:0.5px; }
        .kpi-value { font-size:30px; font-weight:700; margin-top:4px; }
        .kpi-sub { font-size:12px; opacity:0.75; margin-top:4px; }
        .risk-badge-high { background:#ef4444; padding:8px 18px; border-radius:20px; color:white; font-weight:700; font-size:18px; }
        .risk-badge-medium { background:#f59e0b; padding:8px 18px; border-radius:20px; color:white; font-weight:700; font-size:18px; }
        .risk-badge-low { background:#10b981; padding:8px 18px; border-radius:20px; color:white; font-weight:700; font-size:18px; }
        .rec-card {
            background: linear-gradient(135deg,#1e1b3a,#1a2740); border-radius: 12px; padding: 14px 18px;
            margin-bottom: 10px; border: 1px solid rgba(139,92,246,0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label, value, sub=""):
    return (
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>'
    )
