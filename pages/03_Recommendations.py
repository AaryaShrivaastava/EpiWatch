import streamlit as st
from src.dashboard.data_service import get_recommendations

st.set_page_config(page_title="Recommendations - EpiWatch", layout="wide")
st.title("Public-Health Recommendations")

rec_df = get_recommendations()

if rec_df.empty:
    st.info("No recommendations available.")
    st.stop()

regions = ["All"] + list(rec_df['region'].dropna().unique()) if 'region' in rec_df.columns else ["All"]
selected_region = st.selectbox("Filter by Region", regions)

filtered_df = rec_df.copy()
if selected_region != "All":
    filtered_df = filtered_df[filtered_df['region'] == selected_region]

if filtered_df.empty:
    st.info("No recommendations for the selected region.")
else:
    for idx, row in filtered_df.iterrows():
        with st.container(border=True):
            st.markdown(f"### {row.get('recommendation_title', 'Recommendation')}")
            st.markdown(f"**Region:** {row.get('region', 'N/A')}  |  **Target Stakeholder:** {row.get('target_stakeholder', 'N/A')}  |  **Priority:** {row.get('priority', 'N/A')}")
            st.markdown(f"**Rationale:** {row.get('rationale', 'N/A')}")
            if 'evidence_summary' in row:
                st.markdown(f"**Evidence:** {row['evidence_summary']}")
            if 'limitation' in row:
                st.caption(f"*Limitation:* {row['limitation']}")
