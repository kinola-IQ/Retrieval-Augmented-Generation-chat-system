import uvicorn
import os
from dotenv import load_dotenv
# import json
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from scripts import backend_startup
load_dotenv()

# start backend
backend_startup.start_backend()

st.set_page_config(
    page_title="Morocura Chat Assistant",
    page_icon="🤖",
    layout="wide",
)


# api configuration
host = os.environ.get("HOST","0.0.0.0")
port = os.environ.get("PORT",8000)
DEFAULT_BACKEND_URL = f"http://{host}:{port}/v1"


@st.cache_data(ttl=30)
def check_backend_health(api_url: str) -> dict:
    status = {
        "healthy": False,
        "message": "Backend not available yet.",
    }
    try:
        response = requests.get(f"{api_url}/services/health", timeout=90)
        if response.ok:
            payload = response.json()
            status = {
                "healthy": True,
                "message": "Backend is healthy and ready to answer questions.",
                "details": payload,
            }
        else:
            status["message"] = f"Backend responded with status {response.status_code}."
    except requests.exceptions.RequestException as exc:
        status["message"] = f"Unable to connect to backend: {exc}"
    return status


def post_question(api_url: str, prompt: str) -> dict:
    payload = {"prompt": prompt}
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{api_url}/chat", json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def render_sidebar() -> dict:
    st.sidebar.title("Settings")
    st.sidebar.write(
        "Configure the RAG backend endpoint and understand how the app connects to the PDF knowledge base."
    )
    api_url = st.sidebar.text_input("API base URL", DEFAULT_BACKEND_URL)
    st.sidebar.markdown(
        "Use the FastAPI backend route mounted at `/v1/chat`. Start the backend with:"
    )
    st.sidebar.code("uvicorn api.server:server --reload --host 127.0.0.1 --port 8080")
    st.sidebar.markdown(
        "If your backend runs on a different host or port, update the API base URL above."
    )
    return {"api_url": api_url}


def render_app() -> None:
    st.title("RAG PDF Assistant")
    st.markdown(
        "Ask questions about the ingested PDF knowledge base and receive grounded answers "
        "with source citations. This interface connects to the existing FastAPI backend."
    )

    settings = render_sidebar()
    api_url = settings["api_url"].rstrip("/")

    health = check_backend_health(api_url)
    if health["healthy"]:
        st.success(health["message"])
        if "details" in health:
            with st.expander("Health details"):
                st.json(health["details"])
    else:
        st.warning(health["message"])

    if "history" not in st.session_state:
        st.session_state.history = []

    if backend_startup.BACKEND is True:
        prompt = st.text_area(
            "Ask a question",
            placeholder="What can you tell me about the documents in the knowledge base?",
            height=180,
        )

        submit_button = st.button("Send question")

        if submit_button and prompt:
            with st.spinner("Querying backend..."):
                try:
                    response = post_question(api_url, prompt)
                    answer = response.get("response", "No answer returned.")
                    sources = response.get("sources", [])
                    entry = {
                        "prompt": prompt,
                        "response": answer,
                        "sources": sources,
                    }
                    st.markdown(f"**Response:**\n{answer}")
                    st.session_state.history.insert(0, entry)
                except requests.exceptions.RequestException as exc:
                    st.error(f"Request failed: {exc}")
                except ValueError as exc:
                    st.error(f"Unexpected response: {exc}")

        if st.session_state.history:
            st.markdown("---")
            st.header("Conversation history")
            for idx, entry in enumerate(st.session_state.history, 1):
                st.subheader(f"Question {idx}")
                st.markdown(f"**You asked:** {entry['prompt']}")
                st.markdown(f"**Answer:** {entry['response']}")
                if entry["sources"]:
                    with st.expander("View retrieved sources"):
                        for source in entry["sources"]:
                            source_text = source.get('text', '')
                            truncated_text = source_text[:300] + ('...' if len(source_text) > 300 else '')
                            st.markdown(
                                f"- **Source:** {source.get('source', 'unknown')}  \n"
                                f"**Score:** {source.get('score', 'N/A')}  \n"
                                f"**Text:** {truncated_text}"
                            )

        st.markdown("---")
        st.info(
            "This UI is designed to support the project’s PDF-based Retrieval-Augmented Generation flow: "
            "users submit a natural-language query, the backend retrieves relevant PDF context, and a Hugging Face model generates an answer." 
        )
    else:
        st_autorefresh(interval=1000, limit=2)

if __name__ == "__main__":
    render_app()