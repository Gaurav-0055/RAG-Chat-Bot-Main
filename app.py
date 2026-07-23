import os
from pathlib import Path
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
DB_PATH = ".chroma_db"

st.set_page_config(page_title="The Reading Room", page_icon="🗂️", layout="centered")

# ------------------------------------------------------------------
# Visual identity: "The Reading Room" — an archive/index-card aesthetic
# built around the idea of a document being catalogued and retrieved,
# not a generic chatbot skin.
# ------------------------------------------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,500;0,600;1,500&family=Inter:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap');

        :root {
            --ink: #232019;
            --paper: #EDE7D9;
            --paper-deep: #E1D9C6;
            --brass: #9C6F32;
            --forest: #3C5B4E;
            --card-border: #C9BE9F;
        }

        .stApp {
            background: var(--paper);
        }

        .block-container {
            padding-top: 2.2rem;
            max-width: 760px;
        }

        footer, #MainMenu { visibility: hidden; }

        /* ---------- Header / hero ---------- */
        .archive-eyebrow {
            font-family: 'Space Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--brass);
            border-top: 1px solid var(--card-border);
            border-bottom: 1px solid var(--card-border);
            padding: 0.35rem 0;
            margin-bottom: 0.9rem;
            display: flex;
            justify-content: space-between;
        }

        .archive-title {
            font-family: 'Lora', serif;
            font-weight: 600;
            font-size: 2.3rem;
            color: var(--ink);
            margin-bottom: 0.1rem;
            letter-spacing: -0.01em;
        }

        .archive-subtitle {
            font-family: 'Inter', sans-serif;
            font-size: 0.95rem;
            color: #5b5648;
            margin-bottom: 1.6rem;
        }

        /* ---------- Chat bubbles ---------- */
        [data-testid="stChatMessage"] {
            background: #F7F3E8;
            border: 1px solid var(--card-border);
            border-radius: 4px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.8rem;
            box-shadow: 2px 2px 0 rgba(35,32,25,0.05);
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stChatMessage"] p {
            color: var(--ink);
            line-height: 1.55;
        }

        /* User messages get a brass rule on the left, assistant a forest rule */
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
            border-left: 3px solid var(--brass);
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
            border-left: 3px solid var(--forest);
        }

        /* ---------- Chat input ---------- */
        [data-testid="stChatInput"] textarea {
            font-family: 'Inter', sans-serif;
            background: #F7F3E8 !important;
            border: 1px solid var(--card-border) !important;
        }

        /* ---------- Source / citation cards ---------- */
        .catalog-card {
            font-family: 'Space Mono', monospace;
            font-size: 0.78rem;
            background: #F7F3E8;
            border: 1px dashed var(--card-border);
            border-radius: 3px;
            padding: 0.55rem 0.75rem;
            margin-bottom: 0.4rem;
            color: #4a4638;
        }
        .catalog-card b { color: var(--forest); }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: var(--paper-deep);
            border-right: 1px solid var(--card-border);
        }
        section[data-testid="stSidebar"] * {
            font-family: 'Inter', sans-serif;
        }
        .sidebar-stamp {
            font-family: 'Space Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--brass);
            border: 1px solid var(--brass);
            border-radius: 3px;
            padding: 0.2rem 0.5rem;
            display: inline-block;
            margin-bottom: 0.4rem;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_backend():
    """Establishes global application instances for vector lookup and the model."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Missing GEMINI_API_KEY environment credentials.")
        return None, None

    os.environ["GOOGLE_API_KEY"] = api_key

    if not os.path.exists(DB_PATH):
        st.error("Persistent vector directory not found. Please run ingest.py first.")
        return None, None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # Context window width expanded to k=6 to handle cross-sentence context logic better
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    return retriever, llm


retriever, llm = initialize_rag_backend()

# ---------------- Sidebar: index status card ----------------
with st.sidebar:
    st.markdown('<span class="sidebar-stamp">Index Status</span>', unsafe_allow_html=True)
    if retriever and llm:
        st.markdown("🟢 **Connected**")
        st.caption(f"Store: `{DB_PATH}`")
        st.caption("Model: gemini-2.5-flash")
        st.caption("Retrieval depth: k = 6")
    else:
        st.markdown("🔴 **Not connected**")
        st.caption("Run `ingest.py` to build the index.")
    st.divider()
    st.caption("Ask a question and the assistant will cite the passages it drew on, like index cards pulled from a catalog.")

# Application state instantiation
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I've loaded your document index. What would you like to know?", "sources": []}
    ]

# ---------------- Header ----------------
st.markdown(
    '<div class="archive-eyebrow"><span>Archive No. 001</span><span>Gemini · RAG Index</span></div>',
    unsafe_allow_html=True
)
st.markdown('<div class="archive-title">The Reading Room</div>', unsafe_allow_html=True)
st.markdown('<div class="archive-subtitle">Ask your documents anything — answers are drawn strictly from what\'s on file.</div>', unsafe_allow_html=True)

# Conversation rendering
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📇 {len(msg['sources'])} passage(s) referenced"):
                for i, src in enumerate(msg["sources"], start=1):
                    st.markdown(
                        f'<div class="catalog-card"><b>CARD {i:02d}</b> — {src}</div>',
                        unsafe_allow_html=True
                    )

# Main execution frame loop
if user_query := st.chat_input("Ask a question about your documents..."):

    st.session_state.messages.append({"role": "user", "content": user_query, "sources": []})
    with st.chat_message("user"):
        st.write(user_query)

    if retriever and llm:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Pulling passages from the index..."):
                try:
                    # Isolate text payloads from matching chunk documents
                    matched_docs = retriever.invoke(user_query)
                    context_text = "\n\n".join([doc.page_content for doc in matched_docs])
                    source_snippets = [
                        (doc.page_content[:140] + "…") if len(doc.page_content) > 140 else doc.page_content
                        for doc in matched_docs
                    ]

                    # RAG Guardrail Prompt Architecture
                    prompt_template = (
                        f"You are a precise document analysis assistant.\n"
                        f"Answer the question based strictly on the provided context. If the answer isn't present, "
                        f"say you don't know.\n\n"
                        f"--- CONTEXT ---\n{context_text}\n---------------\n\n"
                        f"Question: {user_query}\n"
                        f"Answer:"
                    )

                    execution_result = llm.invoke(prompt_template)
                    output_text = execution_result.content

                    response_placeholder.write(output_text)
                    if source_snippets:
                        with st.expander(f"📇 {len(source_snippets)} passage(s) referenced"):
                            for i, src in enumerate(source_snippets, start=1):
                                st.markdown(
                                    f'<div class="catalog-card"><b>CARD {i:02d}</b> — {src}</div>',
                                    unsafe_allow_html=True
                                )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": output_text, "sources": source_snippets}
                    )

                except Exception as e:
                    error_msg = f"An execution error occurred: {str(e)}"
                    response_placeholder.write(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
    else:
        st.warning("Backend pipeline initialization failed. Check your environment keys.")
