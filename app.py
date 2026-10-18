"""Main Streamlit Application Entry Point."""
import streamlit as st

st.set_page_config(
    page_title="EpiWatch",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("EpiWatch — AI-Powered Epidemiological Intelligence")

st.markdown("""
Welcome to **EpiWatch**, an integrated epidemiological intelligence platform.

Use the sidebar to navigate through the stakeholder dashboards and analytical views.

### Available Dashboards:
- **Overview**: Regional summaries, disease trends, and active alerts.
- **Alerts**: Intelligent alerts triggered by data anomalies and threshold breaches.
- **Recommendations**: Evidence-based public-health recommendations.
- **AI Assistant**: Natural-language interface to epidemiological insights.

*Data is sourced from validated analytics pipelines. Do not use experimental outputs for critical medical decisions.*
""")

# Note: Streamlit natively supports multipage apps by placing files in a `pages/` directory next to `app.py`.
# We will create a symlink or physically place them in `pages/` for Streamlit to discover.
