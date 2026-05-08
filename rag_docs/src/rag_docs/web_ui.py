"""Streamlit UI for the Encompass RAG assistant.

Talks to the FastAPI backend at http://localhost:8000/api/query and renders the
answer plus the new response shape (`sources` list + `relevant_endpoints` list).
"""
import time
from typing import Any, Dict, List

import requests
import streamlit as st

st.set_page_config(
    page_title="Encompass Docs RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.markdown("""
<style>
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
    }

    h1, h2, h3 {
        color: #FFFFFF;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
    }

    .user-message {
        background-color: #1A1E2E;
        color: #FFFFFF;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }

    .assistant-message {
        background-color: #1E2A38;
        color: #FFFFFF;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 4px solid #10B981;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }

    .stButton button {
        background-color: #2D3748;
        color: #FFFFFF;
        font-weight: bold;
        border: none;
        transition: background-color 0.3s;
    }

    .stButton button:hover {
        background-color: #3B4A63;
    }

    .stTextArea textarea {
        background-color: #1A202C;
        color: #FFFFFF;
        border: 1px solid #2D3748;
    }

    .title {
        text-align: center;
        color: #FFFFFF;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }

    .stForm {
        background-color: #1A202C;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2D3748;
    }

    [data-testid="column"]:first-child {
        background-color: #171923;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #2D3748;
    }

    [data-testid="column"]:nth-child(2) {
        background-color: #1A1A2E;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #2D3748;
    }

    .source-preview {
        background-color: #1A202C;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 3px solid #4A5568;
        font-family: monospace;
        font-size: 0.9em;
    }

    .endpoint-info {
        background-color: #1E293B;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #334155;
    }

    .method-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85em;
        margin-right: 10px;
        color: #FFFFFF;
    }

    .method-get { background-color: #10B981; }
    .method-post { background-color: #3B82F6; }
    .method-put { background-color: #F59E0B; }
    .method-delete { background-color: #EF4444; }
    .method-patch { background-color: #8B5CF6; }

    .kind-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75em;
        margin-right: 8px;
        background-color: #374151;
        color: #E5E7EB;
    }
    .kind-guide     { background-color: #1E40AF; }
    .kind-reference { background-color: #065F46; }
    .kind-changelog { background-color: #7C2D12; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🚀 Encompass API Assistant</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #E0E0E0;'>Ask questions about Encompass API — answers cite documentation and surface relevant endpoints.</p>",
    unsafe_allow_html=True,
)

# --- Session state ----------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_sources" not in st.session_state:
    st.session_state.show_sources = True


# --- API client -------------------------------------------------------------

def query_rag_api(question: str) -> Dict[str, Any]:
    url = "http://localhost:8000/api/query"
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"query": question},
            timeout=3600,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to the API. Is the FastAPI server running on :8000?"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out."}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


# --- Render helpers ---------------------------------------------------------

def _method_badge_html(method: str) -> str:
    method_norm = (method or "").upper().strip()
    css = f"method-{method_norm.lower()}" if method_norm in ("GET", "POST", "PUT", "DELETE", "PATCH") else ""
    return f"<span class='method-badge {css}'>{method_norm or '—'}</span>"


def _kind_badge_html(kind: str) -> str:
    kind_norm = (kind or "").lower()
    css = f"kind-{kind_norm}" if kind_norm in ("guide", "reference", "changelog") else ""
    label = kind_norm or "doc"
    return f"<span class='kind-badge {css}'>{label}</span>"


def display_sources_and_endpoints(response_data: Dict[str, Any]) -> None:
    sources: List[Dict[str, Any]] = response_data.get("sources") or []
    endpoints: List[Dict[str, Any]] = response_data.get("relevant_endpoints") or []

    if not sources and not endpoints:
        st.info("No source documents or endpoints were retrieved for this query.")
        return

    tab_docs, tab_endpoints, tab_summary = st.tabs([
        f"📚 Documentation ({len(sources)})",
        f"🔧 API Endpoints ({len(endpoints)})",
        "📊 Summary",
    ])

    with tab_docs:
        if not sources:
            st.info("No documentation chunks were retrieved.")
        else:
            st.subheader("Documentation chunks (RRF-fused, FAISS + BM25)")
            for i, src in enumerate(sources, 1):
                title = src.get("title", "(untitled)")
                kind = src.get("kind", "")
                breadcrumb = " > ".join(src.get("breadcrumb") or []) or "—"
                with st.expander(f"📄 [{i}] {title}", expanded=False):
                    st.markdown(
                        f"{_kind_badge_html(kind)} **Section:** {breadcrumb}  "
                        f"&nbsp;•&nbsp; **Score:** `{src.get('score', 0):.4f}`",
                        unsafe_allow_html=True,
                    )
                    if src.get("url"):
                        st.markdown(f"**🔗 URL:** [{src['url']}]({src['url']})")
                    st.markdown("**Preview:**")
                    st.code(src.get("preview", ""), language="markdown")
                    if st.checkbox("Show full chunk", key=f"src_full_{i}"):
                        st.text_area(
                            label="Full chunk",
                            value=src.get("full_content", src.get("preview", "")),
                            height=300,
                            key=f"src_textarea_{i}",
                        )

    with tab_endpoints:
        if not endpoints:
            st.info("No API endpoints surfaced for this query (gate didn't fire).")
        else:
            st.subheader("Postman endpoints (gated by token-overlap + score-ratio)")
            for i, ep in enumerate(endpoints, 1):
                method = ep.get("method", "")
                name = ep.get("name", "Endpoint")
                with st.expander(f"{method} — {name}", expanded=(i == 1)):
                    st.markdown(_method_badge_html(method), unsafe_allow_html=True)
                    st.code(ep.get("path", ""), language="text")
                    folder = ep.get("folder_path") or []
                    if folder:
                        st.markdown(f"**Folder:** {' > '.join(folder)}")
                    desc = ep.get("description") or ""
                    if desc.strip():
                        st.markdown("**Description:**")
                        st.code(desc, language="text")

    with tab_summary:
        st.subheader("Retrieval summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total chunks", len(sources))
        with col2:
            st.metric("API endpoints", len(endpoints))
        with col3:
            kinds = [s.get("kind", "") for s in sources]
            st.metric("Distinct doc kinds", len({k for k in kinds if k}))

        if sources:
            kind_counts: Dict[str, int] = {}
            for s in sources:
                k = s.get("kind") or "other"
                kind_counts[k] = kind_counts.get(k, 0) + 1
            st.markdown("### By kind")
            st.bar_chart(kind_counts)


# --- Layout -----------------------------------------------------------------

col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("<h3 style='color: #3B82F6;'>Ask a Question</h3>", unsafe_allow_html=True)

    with st.form(key="query_form"):
        user_question = st.text_area(
            "Enter your question about the Encompass API:",
            height=150,
            placeholder="e.g., How do I create a borrower pair? What's the parameter for loan templates?",
        )
        col_submit, col_toggle = st.columns(2)
        with col_submit:
            submit_button = st.form_submit_button("🔍 Submit Question", use_container_width=True)
        with col_toggle:
            st.form_submit_button("📚 Toggle Sources", use_container_width=True)

    with st.expander("⚙️ Settings", expanded=False):
        st.session_state.show_sources = st.checkbox(
            "Show sources & endpoints", value=st.session_state.show_sources,
        )
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # API status indicator
    try:
        health = requests.get("http://localhost:8000/api/health", timeout=2)
        if health.status_code == 200 and health.json().get("rag_system_loaded"):
            st.success("✅ RAG System Ready")
        else:
            st.warning("⚠️ RAG System Loading…")
    except Exception:
        st.error("❌ Cannot connect to API")

# Submit handling
if submit_button and user_question:
    st.session_state.chat_history.append({
        "role": "user", "content": user_question, "timestamp": time.time(),
    })
    with st.spinner("🔍 Searching documentation and generating answer..."):
        response_data = query_rag_api(user_question)
        if "error" in response_data:
            st.session_state.chat_history.append({
                "role": "error", "content": response_data["error"], "timestamp": time.time(),
            })
        else:
            st.session_state.chat_history.append({
                "role": "assistant", "content": response_data, "timestamp": time.time(),
            })

# Conversation column
with col2:
    st.markdown("<h3 style='color: #10B981;'>Conversation</h3>", unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.info(
            "💡 Ask a question to start. Try: "
            "“How do I create a borrower pair?” or "
            "“Which endpoint returns webhook subscriptions?”"
        )

    for message in reversed(st.session_state.chat_history):
        if message["role"] == "user":
            st.markdown(
                f"<div class='user-message'><strong>🧑 You:</strong><br/>{message['content']}</div>",
                unsafe_allow_html=True,
            )
        elif message["role"] == "assistant":
            response_content = message["content"]
            answer = response_content.get("answer", "No answer generated")
            st.markdown(
                f"<div class='assistant-message'><strong>🤖 Assistant:</strong><br/>{answer}</div>",
                unsafe_allow_html=True,
            )
            if st.session_state.show_sources:
                st.markdown("---")
                display_sources_and_endpoints(response_content)
        elif message["role"] == "error":
            st.error(message["content"])

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #9CA3AF;'>"
    "FAISS-IP + dual BM25 + Postman gate · Jina v3 embeddings · "
    "served via FastAPI &amp; LangChain"
    "</p>",
    unsafe_allow_html=True,
)
