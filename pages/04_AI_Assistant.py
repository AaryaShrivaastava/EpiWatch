import streamlit as st
import sys
from pathlib import Path

# Need to ensure src can be imported
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

st.set_page_config(page_title="AI Assistant - EpiWatch", layout="wide")
st.title("Epidemiology AI Assistant")

st.markdown("""
Ask questions about the current epidemiological situation. 
*Example: "Please generate recommendations for Bhopal."*
""")

try:
    from src.agent.agent_pipeline import AgentPipeline
    agent_available = True
except ImportError:
    agent_available = False

if not agent_available:
    st.error("The AI Agent module (Phase 12) is not fully implemented or available in the path.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask the EpiWatch Agent..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Analyzing data and generating response..."):
            try:
                pipeline = AgentPipeline()
                response = pipeline.process_request(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error communicating with agent: {e}")
