import streamlit as st
import json
from bot import QABot

st.set_page_config(
    page_title="Local RAG Web Scraper Bot",
    page_icon="[RAG]",
    layout="wide"
)

# Initialize bot once per session
if "bot" not in st.session_state:
    with st.spinner("Initializing Local RAG Pipeline (Loading Transformers)..."):
        st.session_state.bot = QABot()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar information
with st.sidebar:
    st.title("Architecture & Models")
    st.markdown("""
    **100% Free & Local Architecture**
    - **Offline Memory:** Dictionary Knowledge Base
    - **Search Engine:** DuckDuckGo Search API
    - **Defense Layer:** Anti-Bot Header Spoofing & Paywall Filter
    - **Chunking:** Sliding Window with Overlap
    - **Reranker:** `all-MiniLM-L6-v2` (Bi-Encoder)
    - **Generator:** `facebook/bart-large-cnn` (Seq2Seq LM)
    - **API Keys Required:** None (0)
    """)
    st.markdown("---")
    if st.button("Clear Conversation History"):
        st.session_state.messages = []
        st.rerun()

# Main Header
st.title("Local RAG Q&A Assistant")
st.caption("A multi-stage local RAG system featuring anti-bot web scraping, overlapping chunking, dense vector reranking, and Seq2Seq generation.")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "details" in msg and msg["details"]:
            with st.expander("Pipeline Inspection (RAG / Transformer Stages)"):
                meta = msg["details"].get("pipeline_meta", {})
                
                st.subheader("1. Web Documents Ingestion & Anti-Bot Defense")
                docs = meta.get("documents", [])
                if docs:
                    for i, d in enumerate(docs):
                        st.markdown(f"**Doc {i+1}:** [{d.get('title', 'Link')}]({d.get('url', '#')})  \n*Status:* `{d.get('status', 'N/A')}` | *Type:* `{d.get('source_type', 'N/A')}`")
                else:
                    st.write("No external documents scraped.")

                st.subheader("2. Overlapping Chunking")
                total_chunks = meta.get("total_chunks", 0)
                st.write(f"Total overlapping chunks generated: `{total_chunks}`")

                st.subheader("3. Semantic Reranking (all-MiniLM-L6-v2)")
                top_chunks = meta.get("top_chunks", [])
                if top_chunks:
                    for i, tc in enumerate(top_chunks):
                        score = tc.get("similarity_score", 0.0)
                        st.markdown(f"**Rank {i+1}** (Cosine Similarity: `{score}`):")
                        st.info(tc.get("text", ""))

                st.subheader("4. Structured JSON Context Fed to Generator")
                json_context = msg["details"].get("json_context", "{}")
                try:
                    parsed_json = json.loads(json_context)
                    st.json(parsed_json)
                except Exception:
                    st.code(json_context, language="json")

# User Input
query = st.chat_input("Ask a question...")

if query:
    # Display user query
    st.chat_message("user").markdown(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Processing pipeline (Scraping -> Chunking -> Reranking -> Generating)..."):
            result = st.session_state.bot.ask(query, return_details=True)

        if isinstance(result, dict):
            answer_text = result.get("answer", "")
            details = result
        else:
            answer_text = str(result)
            details = None

        st.markdown(answer_text)

        if details and details.get("pipeline_meta"):
            with st.expander("Pipeline Inspection (RAG / Transformer Stages)"):
                meta = details.get("pipeline_meta", {})
                
                st.subheader("1. Web Documents Ingestion & Anti-Bot Defense")
                docs = meta.get("documents", [])
                if docs:
                    for i, d in enumerate(docs):
                        st.markdown(f"**Doc {i+1}:** [{d.get('title', 'Link')}]({d.get('url', '#')})  \n*Status:* `{d.get('status', 'N/A')}` | *Type:* `{d.get('source_type', 'N/A')}`")
                else:
                    st.write("No external documents scraped.")

                st.subheader("2. Overlapping Chunking")
                total_chunks = meta.get("total_chunks", 0)
                st.write(f"Total overlapping chunks generated: `{total_chunks}`")

                st.subheader("3. Semantic Reranking (all-MiniLM-L6-v2)")
                top_chunks = meta.get("top_chunks", [])
                if top_chunks:
                    for i, tc in enumerate(top_chunks):
                        score = tc.get("similarity_score", 0.0)
                        st.markdown(f"**Rank {i+1}** (Cosine Similarity: `{score}`):")
                        st.info(tc.get("text", ""))

                st.subheader("4. Structured JSON Context Fed to Generator")
                json_context = details.get("json_context", "{}")
                try:
                    parsed_json = json.loads(json_context)
                    st.json(parsed_json)
                except Exception:
                    st.code(json_context, language="json")

    # Store in history
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "details": details
    })
