import os
from pathlib import Path
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
DB_PATH = ".chroma_db"

st.set_page_config(page_title="RAG Document Assistant", page_icon="🤖", layout="centered")

# Visual layout adjustments
st.markdown("""
    <style>
        .block-container { padding-top: 3rem; }
        footer { visibility: hidden; }
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

# Application state instantiation
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I've loaded your document index. What would you like to know?"}
    ]

st.title("📄 Document Knowledge Base")
st.caption("Retrieval-Augmented Generation Chatbot powered by Gemini")

# Conversation rendering
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Main execution frame loop
if user_query := st.chat_input("Ask a question about your documents..."):
    
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
        
    if retriever and llm:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Searching document context..."):
                try:
                    # Isolate text payloads from matching chunk documents
                    matched_docs = retriever.invoke(user_query)
                    context_text = "\n\n".join([doc.page_content for doc in matched_docs])
                    
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
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                    
                except Exception as e:
                    error_msg = f"An execution error occurred: {str(e)}"
                    response_placeholder.write(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        st.warning("Backend pipeline initialization failed. Check your environment keys.")