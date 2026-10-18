import streamlit as st
from src.dashboard.data_service import get_alerts

st.set_page_config(page_title="Alerts - EpiWatch", layout="wide")
st.title("Active Alerts")

alerts_df = get_alerts()

if alerts_df.empty:
    st.info("No alerts generated yet. The system is operating normally.")
    st.stop()

# Filters
col1, col2 = st.columns(2)
with col1:
    regions = ["All"] + list(alerts_df['region'].dropna().unique()) if 'region' in alerts_df.columns else ["All"]
    selected_region = st.selectbox("Filter by Region", regions)

with col2:
    severities = ["All"] + list(alerts_df['severity'].dropna().unique()) if 'severity' in alerts_df.columns else ["All"]
    selected_severity = st.selectbox("Filter by Severity", severities)

filtered_df = alerts_df.copy()
if selected_region != "All":
    filtered_df = filtered_df[filtered_df['region'] == selected_region]
if selected_severity != "All":
    filtered_df = filtered_df[filtered_df['severity'] == selected_severity]

if filtered_df.empty:
    st.info("No alerts match the selected filters.")
else:
    for idx, row in filtered_df.iterrows():
        severity_color = "red" if str(row.get('severity', '')).lower() in ['high', 'critical review required'] else "orange"
        with st.expander(f"{row.get('alert_id', 'Unknown')} - {row.get('region', 'Unknown')} ({row.get('severity', 'Unknown')})"):
            st.markdown(f"**Category:** {row.get('category', 'N/A')}")
            st.markdown(f"**Explanation:** {row.get('explanation', 'N/A')}")
            st.markdown(f"**Status:** {row.get('status', 'N/A')}  |  **Time:** {row.get('timestamp', 'N/A')}")
            st.markdown(f"**Trigger Metric:** {row.get('triggering_metric', 'N/A')} = {row.get('observed_value', 'N/A')} (Threshold: {row.get('threshold', 'N/A')})")
