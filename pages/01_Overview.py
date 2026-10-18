import streamlit as st
from src.dashboard.data_service import get_alerts, get_risk_assessments, get_forecasts

st.set_page_config(page_title="Overview - EpiWatch", layout="wide")

st.title("Regional Overview")

# Load data
risk_df = get_risk_assessments()
forecast_df = get_forecasts()
alerts_df = get_alerts()

if risk_df.empty and forecast_df.empty:
    st.warning("No data available. Please ensure analytical pipelines have run.")
    st.stop()

# Get available regions
regions = []
if not risk_df.empty and 'region' in risk_df.columns:
    regions.extend(risk_df['region'].unique())
if not forecast_df.empty and 'region' in forecast_df.columns:
    regions.extend(forecast_df['region'].unique())
    
regions = list(set(regions))
if not regions:
    st.warning("No region data found.")
    st.stop()

selected_region = st.selectbox("Select Region", regions)

st.subheader(f"Metrics for {selected_region}")

col1, col2, col3 = st.columns(3)

# Risk Metric
if not risk_df.empty and 'region' in risk_df.columns:
    reg_risk = risk_df[risk_df['region'] == selected_region]
    if not reg_risk.empty:
        latest_risk = reg_risk.iloc[-1]
        score = latest_risk.get('integrated_regional_risk_score', 'N/A')
        cat = latest_risk.get('integrated_risk_category', 'N/A')
        col1.metric("Integrated Risk Score", f"{score:.1f}" if isinstance(score, float) else score, cat)
else:
    col1.metric("Integrated Risk Score", "No Data")

# Forecast Metric
if not forecast_df.empty and 'region' in forecast_df.columns:
    reg_fc = forecast_df[forecast_df['region'] == selected_region]
    if not reg_fc.empty:
        if 'date' in reg_fc.columns: reg_fc = reg_fc.sort_values('date')
        latest_fc = reg_fc.iloc[-1]
        rate = latest_fc.get('forecast_growth_rate', 'N/A')
        col2.metric("Forecasted Growth Rate", f"{rate:.2f}" if isinstance(rate, float) else rate)
else:
    col2.metric("Forecasted Growth Rate", "No Data")

# Alerts Metric
if not alerts_df.empty and 'region' in alerts_df.columns:
    reg_alerts = alerts_df[(alerts_df['region'] == selected_region) & (alerts_df['status'] != 'resolved')]
    col3.metric("Active Alerts", len(reg_alerts))
else:
    col3.metric("Active Alerts", 0)

st.divider()
st.subheader("Disease Trend Forecast")
if not forecast_df.empty and 'region' in forecast_df.columns:
    reg_fc = forecast_df[forecast_df['region'] == selected_region]
    if not reg_fc.empty and 'date' in reg_fc.columns and 'forecast_growth_rate' in reg_fc.columns:
        st.line_chart(reg_fc.set_index('date')['forecast_growth_rate'])
    else:
        st.info("Insufficient columns for trend chart.")
else:
    st.info("No forecast data available for chart.")
