import os
   os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
"""
Ice Age Trail Guidebook Q&A — Streamlit chat interface.

Run locally:
    streamlit run app.py

Reads keys from environment variables VOYAGE_API_KEY and XAI_API_KEY,
or from Streamlit secrets when deployed to Streamlit Cloud.
"""
import os
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (mobile-friendly, hiking-themed)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ice Age Trail Companion",
    page_icon="🥾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Load API keys (prefers Streamlit secrets in production, env vars locally)
# ---------------------------------------------------------------------------
def _load_keys():
    # Streamlit Cloud secrets (production)
    try:
        if 'VOYAGE_API_KEY' in st.secrets:
            os.environ['VOYAGE_API_KEY'] = st.secrets['VOYAGE_API_KEY']
            os.environ['XAI_API_KEY']    = st.secrets['XAI_API_KEY']
            return True
    except Exception:
        pass
    # Local development: read from environment (or a local secrets/.env file)
    if os.environ.get('VOYAGE_API_KEY') and os.environ.get('XAI_API_KEY'):
        return True
    env_file = Path(__file__).parent / 'secrets' / '.env'
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if '=' in line:
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip()
        return True
    return False

if not _load_keys():
    st.error("API keys not found. Set VOYAGE_API_KEY and XAI_API_KEY in Streamlit secrets or in secrets/.env")
    st.stop()

# Import the pipeline AFTER env is loaded
from rag_pipeline import ask

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🥾 Ice Age Trail Companion")
st.caption(
    "Ask anything about the Ice Age National Scenic Trail. "
    "Answers come only from the official 2023 IATA guidebook, with citations."
)

# Disclaimer (visible on first render)
with st.expander("⚠️ Important — read this once before relying on the app", expanded=False):
    st.markdown("""
**This app gives best-effort answers from the official 2023 Ice Age Trail Guidebook only.**

- It may say "the guidebook doesn't address this" — that's *correct* behavior, not a failure.
- It does not have current trail conditions, weather, hunting season schedules, or trail closures. Always check **IceAgeTrail.org/alerts** before a hike for those.
- For anything safety-critical (water reliability, road crossings, weather, hazards), confirm with the Ice Age Trail Alliance or the relevant agency.
- Answers cite which segment or section they come from in [brackets]. Use those to verify against the printed guidebook if anything seems off.
""")

# ---------------------------------------------------------------------------
# Session state for chat history
# ---------------------------------------------------------------------------
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        if msg['role'] == 'assistant' and 'sources' in msg:
            with st.expander("Sources used", expanded=False):
                for s in msg['sources']:
                    seg = s.get('segment_name') or s.get('region') or 'Document section'
                    st.markdown(f"- **{seg}** ({s.get('token_count', '?')} tokens)")

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
prompt = st.chat_input("Ask about the trail — e.g., 'Where can I camp on the Bear Lake Segment?'")

if prompt:
    # Echo user message
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    # Generate answer with progress indicator
    with st.chat_message('assistant'):
        placeholder = st.empty()
        placeholder.markdown("🥾 Searching guidebook...")
        t0 = time.time()
        try:
            result = ask(prompt)
            elapsed = time.time() - t0
            answer_md = result['answer']
            placeholder.markdown(answer_md)
            with st.expander(f"Sources used ({elapsed:.1f}s)", expanded=False):
                for s in result['used_parents']:
                    seg = s.get('segment_name') or s.get('region') or 'Document section'
                    st.markdown(f"- **{seg}** ({s.get('token_count', '?')} tokens)")

            st.session_state.messages.append({
                'role': 'assistant',
                'content': answer_md,
                'sources': result['used_parents'],
            })
        except Exception as e:
            placeholder.error(f"Something went wrong: {e}")
            st.session_state.messages.append({
                'role': 'assistant',
                'content': f"⚠️ Error: {e}",
            })

# Footer
st.divider()
st.caption(
    "Built for hikers. Source: *Ice Age National Scenic Trail Official Guidebook* "
    "(Ice Age Trail Alliance, 2023). Always verify safety-critical info with **IceAgeTrail.org**."
)
