import streamlit as st
import requests
import json
import time

st.set_page_config(
    page_title="Encompass Docs RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
    /* Overall page styling */
    .stApp {
        background-color: #121212;  /* Very dark gray for main background */
        color: #E0E0E0;
    }

    /* Header styling */
    h1, h2, h3 {
        color: #FFFFFF;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
    }

    /* User message styling */
    .user-message {
        background-color: #1A1E2E;  /* Dark navy blue */
        color: #FFFFFF;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }

    /* Assistant message styling */
    .assistant-message {
        background-color: #1E2A38;  /* Dark slate blue */
        color: #FFFFFF;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 4px solid #10B981;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }

    /* Button styling */
    .stButton button {
        background-color: #2D3748;  /* Dark slate gray */
        color: #FFFFFF;
        font-weight: bold;
        border: none;
        transition: background-color 0.3s;
    }

    .stButton button:hover {
        background-color: #3B4A63;  /* Slightly lighter on hover */
    }

    /* Input area styling */
    .stTextArea textarea {
        background-color: #1A202C;  /* Dark blue-gray */
        color: #FFFFFF;
        border: 1px solid #2D3748;
    }

    /* Title styling */
    .title {
        text-align: center;
        color: #FFFFFF;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }

    /* Info box styling */
    .stAlert {
        background-color: #2D3748;  /* Dark slate gray */
        color: #FFFFFF;
    }

    /* Error message styling */
    .stException, .stError {
        background-color: #7F1D1D;  /* Dark red */
        color: #FFFFFF;
        padding: 10px;
        border-radius: 5px;
    }

    /* Spinner styling */
    .stSpinner > div > div {
        border-color: #3B82F6 transparent transparent !important;
    }

    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #171923;  /* Very dark blue-gray */
    }

    /* Form styling */
    .stForm {
        background-color: #1A202C;  /* Dark blue-gray */
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2D3748;
    }

    /* Column styling for left column */
    [data-testid="column"]:first-child {
        background-color: #171923;  /* Very dark blue-gray */
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #2D3748;
    }

    /* Column styling for right column */
    [data-testid="column"]:nth-child(2) {
        background-color: #1A1A2E;  /* Very dark blue */
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #2D3748;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("<h1 class='title'>Encompass Docs Assistant</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #E0E0E0;'>Ask questions about Encompass documentation and get answers powered by RAG.</p>",
    unsafe_allow_html=True)

# Initialize chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



def query_rag_api(question):
    url = "http://localhost:8000/api/query"
    headers = {"Content-Type": "application/json"}
    data = {"query": question}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to the API. Please make sure the FastAPI server is running."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The server might be overloaded."}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error occurred: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


# Create a two-column layout
col1, col2 = st.columns([2, 3])

# Input area in the left column
with col1:
    st.markdown("<h3 style='color: #3B82F6;'>Ask a Question</h3>", unsafe_allow_html=True)
    with st.form(key="query_form"):
        user_question = st.text_area("Enter your question about Encompass:", height=150,
                                     placeholder="e.g., How to use loanpipeline endpoint?")
        submit_button = st.form_submit_button("Submit Question")

    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.experimental_rerun()

# Process the query when the form is submitted
if submit_button and user_question:
    # Add user question to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_question, "timestamp": time.time()})

    # Query the API with a spinner
    with st.spinner("Generating answer..."):
        response_data = query_rag_api(user_question)

        if "error" in response_data:
            st.error(response_data["error"])
        else:
            # Add assistant response to chat history
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response_data, "timestamp": time.time()})

# Display chat history in the right column
with col2:
    st.markdown("<h3 style='color: #10B981;'>Conversation</h3>", unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.info("Ask a question to start the conversation!")

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"<div class='user-message'><strong>You:</strong> {message['content']}</div>",
                        unsafe_allow_html=True)
        else:
            response_content = message['content']

            if isinstance(response_content, dict):

                formatted_response = response_content.get('answer', str(response_content))
            else:

                formatted_response = str(response_content)

            st.markdown(f"<div class='assistant-message'><strong>Assistant:</strong> {formatted_response}</div>",
                        unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #9CA3AF;'>Powered by FastAPI, LangChain, and Ollama</p>",
            unsafe_allow_html=True)