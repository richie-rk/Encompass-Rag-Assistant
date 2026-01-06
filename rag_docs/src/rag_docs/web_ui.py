import streamlit as st
import requests
import json
import time
from typing import Dict, List, Any

st.set_page_config(
    page_title="Encompass Docs RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
    /* Keep all your existing styles... */
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

    .stAlert {
        background-color: #2D3748;
        color: #FFFFFF;
    }

    .stException, .stError {
        background-color: #7F1D1D;
        color: #FFFFFF;
        padding: 10px;
        border-radius: 5px;
    }

    .stSpinner > div > div {
        border-color: #3B82F6 transparent transparent !important;
    }

    .css-1d391kg, .css-1lcbmhc {
        background-color: #171923;
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
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85em;
        margin-right: 10px;
    }

    .method-get { background-color: #10B981; }
    .method-post { background-color: #3B82F6; }
    .method-put { background-color: #F59E0B; }
    .method-delete { background-color: #EF4444; }
    .method-patch { background-color: #8B5CF6; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🚀 Encompass API Assistant</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #E0E0E0;'>Ask questions about Encompass API - Get answers with source references</p>",
    unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_sources" not in st.session_state:
    st.session_state.show_sources = True


def query_rag_api(question: str) -> Dict[str, Any]:
    """Query the RAG API with enhanced response handling"""
    url = "http://localhost:8000/api/query"
    headers = {"Content-Type": "application/json"}
    data = {"query": question}

    try:
        response = requests.post(url, headers=headers, json=data, timeout=3600)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to the API. Please make sure the FastAPI server is running."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The server might be processing a complex query."}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error occurred: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


def display_sources(response_data: Dict[str, Any]):
    """Display sources in organized tabs with enhanced formatting"""

    if not response_data.get('context7_sources') and not response_data.get('postman_sources') and not response_data.get(
            'csv_sources'):
        st.info("No source documents were retrieved for this query.")
        return

    # Create tabs for different source types
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Context7 Docs", "🔧 Postman Endpoints", "📄 CSV Documentation", "📊 Summary"])

    with tab1:
        context7_sources = response_data.get('context7_sources', [])
        if context7_sources:
            st.subheader("Context7 Documentation")
            for i, source in enumerate(context7_sources, 1):
                with st.expander(f"📄 Source {i}: {source.get('type', 'Documentation').title()}", expanded=False):
                    if source.get('url'):
                        st.markdown(f"**🔗 URL:** [{source['url']}]({source['url']})")

                    st.markdown("**Preview:**")
                    st.code(source.get('content', 'No preview available'), language='text')

                    if st.checkbox(f"Show full content for source {i}", key=f"context7_{i}"):
                        st.markdown("**Full Content:**")
                        st.text_area(
                            "Full documentation content",
                            value=source.get('full_content', source.get('content', '')),
                            height=300,
                            key=f"full_context7_{i}"
                        )
        else:
            st.info("No Context7 documentation sources were used for this query.")

    with tab2:
        postman_sources = response_data.get('postman_sources', [])
        if postman_sources:
            st.subheader("Postman API Endpoints")
            for i, source in enumerate(postman_sources, 1):
                endpoint = source
                method = endpoint.get('method', 'UNKNOWN').upper()

                # Create method badge HTML
                method_class = f"method-{method.lower()}" if method in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'] else ""

                with st.expander(f"{method} {endpoint.get('endpoint_name', 'Endpoint')}", expanded=False):
                    col1, col2 = st.columns([1, 3])

                    with col1:
                        st.markdown(f"<span class='method-badge {method_class}'>{method}</span>",
                                    unsafe_allow_html=True)

                    with col2:
                        st.code(endpoint.get('path', 'N/A'), language='text')

                    st.markdown("**Preview:**")
                    st.code(endpoint.get('content', 'No preview available'), language='text')

                    # Show raw endpoint data if available
                    if endpoint.get('raw_data'):
                        if st.checkbox(f"Show complete endpoint details", key=f"postman_{i}"):
                            st.json(endpoint['raw_data'])

                    if st.checkbox(f"Show full formatted content", key=f"postman_full_{i}"):
                        st.text_area(
                            "Full endpoint documentation",
                            value=endpoint.get('full_content', endpoint.get('content', '')),
                            height=300,
                            key=f"full_postman_{i}"
                        )
        else:
            st.info("No Postman endpoints were referenced for this query.")

    with tab3:
        csv_sources = response_data.get('csv_sources', [])
        if csv_sources:
            st.subheader("CSV Documentation Sources")
            for i, source in enumerate(csv_sources, 1):
                title = source.get('title', 'Document')
                doc_type = source.get('type', 'Documentation').title()

                with st.expander(f"📄 {title} - {doc_type}", expanded=False):
                    if source.get('url'):
                        st.markdown(f"**🔗 URL:** [{source['url']}]({source['url']})")

                    if source.get('title'):
                        st.markdown(f"**📑 Title:** {source['title']}")

                    st.markdown("**Preview:**")
                    st.code(source.get('content', 'No preview available'), language='text')

                    if st.checkbox(f"Show full content", key=f"csv_{i}"):
                        st.markdown("**Full Content:**")
                        st.text_area(
                            "Full documentation content",
                            value=source.get('full_content', source.get('content', '')),
                            height=300,
                            key=f"full_csv_{i}"
                        )
        else:
            st.info("No CSV documentation sources were used for this query.")

    with tab4:
        st.subheader("Source Summary")

        total_sources = response_data.get('total_sources_used', 0)
        context7_count = len(response_data.get('context7_sources', []))
        postman_count = len(response_data.get('postman_sources', []))
        csv_count = len(response_data.get('csv_sources', []))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sources", total_sources)
        with col2:
            st.metric("Context7", context7_count)
        with col3:
            st.metric("Postman", postman_count)
        with col4:
            st.metric("CSV Docs", csv_count)

        # Show source types breakdown
        if total_sources > 0:
            st.markdown("### Source Breakdown")
            source_data = {
                "Context7": context7_count,
                "Postman Endpoints": postman_count,
                "CSV Documentation": csv_count
            }
            st.bar_chart(source_data)


# Create layout
col1, col2 = st.columns([2, 3])

# Input area in the left column
with col1:
    st.markdown("<h3 style='color: #3B82F6;'>Ask a Question</h3>", unsafe_allow_html=True)

    with st.form(key="query_form"):
        user_question = st.text_area(
            "Enter your question about Encompass API:",
            height=150,
            placeholder="e.g., How do I authenticate with the loan pipeline endpoint? What are the required parameters for creating a loan?"
        )

        col_submit, col_sources = st.columns(2)
        with col_submit:
            submit_button = st.form_submit_button("🔍 Submit Question", use_container_width=True)
        with col_sources:
            show_sources = st.form_submit_button("📚 Toggle Sources", use_container_width=True)

    # Settings
    with st.expander("⚙️ Settings", expanded=False):
        st.session_state.show_sources = st.checkbox("Show source documents", value=st.session_state.show_sources)

        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # API Status
    try:
        health_response = requests.get("http://localhost:8000/api/health", timeout=2)
        if health_response.status_code == 200:
            health_data = health_response.json()
            if health_data.get('rag_system_loaded'):
                st.success("✅ RAG System Ready")
            else:
                st.warning("⚠️ RAG System Loading...")
        else:
            st.error("❌ API Unavailable")
    except:
        st.error("❌ Cannot connect to API")

# Process the query when submitted
if submit_button and user_question:
    # Add user question to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_question,
        "timestamp": time.time()
    })

    # Query the API
    with st.spinner("🔍 Searching documentation and generating answer..."):
        response_data = query_rag_api(user_question)

        if "error" in response_data:
            st.error(response_data["error"])
            st.session_state.chat_history.append({
                "role": "error",
                "content": response_data["error"],
                "timestamp": time.time()
            })
        else:
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_data,
                "timestamp": time.time()
            })

# Display chat history in the right column
with col2:
    st.markdown("<h3 style='color: #10B981;'>Conversation</h3>", unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.info(
            "💡 Ask a question to start! Try: 'How do I authenticate API requests?' or 'What endpoints are available for loan data?'")

    for message in reversed(st.session_state.chat_history):  # Show newest first
        if message["role"] == "user":
            st.markdown(
                f"<div class='user-message'><strong>🧑 You:</strong><br/>{message['content']}</div>",
                unsafe_allow_html=True
            )

        elif message["role"] == "assistant":
            response_content = message['content']

            # Display the answer
            answer = response_content.get('answer', 'No answer generated')
            st.markdown(
                f"<div class='assistant-message'><strong>🤖 Assistant:</strong><br/>{answer}</div>",
                unsafe_allow_html=True
            )

            # Display sources if enabled
            if st.session_state.show_sources:
                with st.container():
                    st.markdown("---")
                    display_sources(response_content)

        elif message["role"] == "error":
            st.error(message["content"])

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #9CA3AF;'>Powered by FastAPI, LangChain, BGE Embeddings & Hybrid Search</p>",
    unsafe_allow_html=True
)

