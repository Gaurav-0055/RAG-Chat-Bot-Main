import os
import time
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Resolve environment keys relative to current script path
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

DATA_PATH = "data"
DB_PATH = ".chroma_db"

def build_vector_store():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[-] Error: Gemini API key missing from .env file.")
        return
    
    os.environ["GOOGLE_API_KEY"] = api_key

    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"[-] Missing target documents inside '{DATA_PATH}' directory.")
        return

    print("[+] Reading raw PDF documents...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_documents = loader.load()
    
    if not raw_documents:
        print("[-] Document parsing returned empty results.")
        return

    print(f"[+] Successfully extracted {len(raw_documents)} pages.")

    # Chunk layout tuning to balance performance and granular retrieval context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_documents(raw_documents)
    total_chunks = len(chunks)
    print(f"[+] Generated {total_chunks} chunk segments.")

    print("[+] Initializing database storage target...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    
    # Safe batching limits to safeguard against free-tier 100 RPM ceilings
    BATCH_SIZE = 40
    print("[+] Executing paced document ingestion pipeline...")
    
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        current_batch = (i // BATCH_SIZE) + 1
        max_batches = -(-total_chunks // BATCH_SIZE)
        
        print(f"    -> Committing batch {current_batch}/{max_batches}...")
        
        for attempt in range(3):
            try:
                vector_store.add_documents(batch)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print("    [!] Rate limit triggered. Sleeping for 35 seconds...")
                    time.sleep(35)
                else:
                    raise e
                    
        if i + BATCH_SIZE < total_chunks:
            time.sleep(5)
            
    print(f"[+] Ingestion complete. Data compiled locally at: {DB_PATH}")

if __name__ == "__main__":
    build_vector_store()