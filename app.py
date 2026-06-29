import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from dotenv import load_dotenv
from rag_chain import build_chain

load_dotenv()

SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "My calls keep dropping — what should I do?",
    "How do I activate international roaming?",
    "Why is my bill higher than usual this month?",
    "My phone shows SIM not detected after a restart",
    "How do I enable Wi-Fi calling?",
    "I was charged for roaming but had a bundle active",
    "How do I unlock my phone for another network?",
]

st.set_page_config(
    page_title="Telecom Customer Support",
    page_icon="📡",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Page background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e3a5f;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: #1e293b;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 4px 8px;
    margin-bottom: 8px;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: #1e293b !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* Sample question buttons */
div[data-testid="stButton"] > button {
    background: #1e3a5f;
    color: #93c5fd;
    border: 1px solid #2563eb;
    border-radius: 8px;
    font-size: 0.78rem;
    text-align: left;
    transition: all 0.2s ease;
}
div[data-testid="stButton"] > button:hover {
    background: #2563eb;
    color: #ffffff;
    border-color: #60a5fa;
}

/* Clear button */
div[data-testid="stButton"]:last-child > button {
    background: #3b0d0d;
    color: #fca5a5;
    border-color: #7f1d1d;
}
div[data-testid="stButton"]:last-child > button:hover {
    background: #7f1d1d;
    color: #ffffff;
}

/* Title */
h1 { color: #60a5fa !important; }
p, li { color: #cbd5e1; }

/* Status badge */
.status-badge {
    display: inline-block;
    background: #14532d;
    color: #86efac;
    border: 1px solid #166534;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_chain():
    return build_chain()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Telecom Support")
    st.markdown('<span class="status-badge">● Online</span>', unsafe_allow_html=True)
    st.caption("Powered by RAG · Qwen3-32B on Groq")
    st.divider()

    st.markdown("#### 💬 Sample Questions")
    st.caption("Click one to send instantly.")

    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True, key=f"sq_{q[:20]}"):
            st.session_state.pending_question = q

    st.divider()

    st.markdown("#### ℹ️ Knowledge Sources")
    col1, col2, col3 = st.columns(3)
    col1.metric("FAQ", "✓", delta=None)
    col2.metric("Tickets", "✓", delta=None)
    col3.metric("Guide", "✓", delta=None)

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
col_main, col_pad = st.columns([3, 1])

with col_main:
    st.markdown("# 🛠️ Customer Care Assistant")
    st.caption(
        "Ask me anything about your mobile service — "
        "connectivity, billing, SIM, roaming, and more. "
        "All answers include source citations."
    )
    st.divider()

    # Chat history
    if not st.session_state.messages:
        st.info(
            "👋 Welcome! Type your question below or pick one from the sidebar.",
            icon="📡",
        )
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Input
    question = st.chat_input("Describe your issue…")
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base…"):
                chain = get_chain()
            response = st.write_stream(chain(question))

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()