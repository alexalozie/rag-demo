# --- SQLite3 Shim for ChromaDB on Streamlit Community Cloud (Linux) ---
# Must run BEFORE importing streamlit, chromadb, or langchain modules.
try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import shutil
from pathlib import Path
import streamlit as st

# LangChain Core & Chains
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Loaders, Retrievers, Re-rankers & Vector Stores
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

# Stacks: Cloud (OpenAI) & Local (Ollama)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama

# =====================================================================
# ⚙️ CONFIGURATION BLOCK
# =====================================================================
DB_DIR = "./chroma_db_storage"
DATA_DIR = "./source_documents"
COLLECTION_NAME = "rag_demo_docs"
MAX_HISTORY_TURNS = 6

CANDIDATE_K = 10
RERANK_TOP_N = 3
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
# =====================================================================

st.set_page_config(page_title="Hybrid RAG Assistant", page_icon="🤖", layout="wide")


def get_cloud_api_key() -> str:
    """Safely checks Streamlit Cloud secrets first, then environment variables."""
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")


def is_running_on_streamlit_cloud() -> bool:
    """Detects Streamlit Community Cloud container environment or configured cloud secret."""
    if os.path.exists("/mount/src"):
        return True
    try:
        return "OPENAI_API_KEY" in st.secrets
    except Exception:
        return False


def ensure_sample_data(allow_seed: bool = True):
    """Creates the data directory and seeds sample data only on initial startup."""
    data_path = Path(DATA_DIR)
    data_path.mkdir(exist_ok=True)
    if allow_seed and not any(f.is_file() for f in data_path.iterdir()):
        sample_txt = data_path / "sample_policy.txt"
        sample_txt.write_text(
            "Company remote work strategy updated in 2026: "
            "Employees can work from any location globally for up to 90 days per calendar year. "
            "The travel allowance program is managed by the Global Mobility Team led by Marcus Vance.",
            encoding="utf-8",
        )


def trigger_reindex():
    """Clears persisted ChromaDB on disk and increments index_version to bust cache."""
    db_path = Path(DB_DIR)
    if db_path.exists():
        shutil.rmtree(db_path)
    st.session_state.index_version += 1


def load_and_chunk_documents(directory: str):
    """Loads TXT and PDF files and splits them into overlapping chunks."""
    documents = []
    data_path = Path(directory)
    if not data_path.exists():
        return []

    for file_path in sorted(data_path.iterdir()):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            documents.extend(TextLoader(str(file_path), encoding="utf-8").load())
        elif suffix == ".pdf":
            documents.extend(PyPDFLoader(str(file_path)).load())

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    return splitter.split_documents(documents)


@st.cache_resource(show_spinner="Initializing models & FlashRank re-ranker...")
def get_models(stack_mode: str, api_key: str = ""):
    """Caches embeddings, LLM, and FlashRank compressor across Streamlit reruns."""
    compressor = FlashrankRerank(model=RERANK_MODEL, top_n=RERANK_TOP_N)
    if stack_mode == "CLOUD":
        os.environ["OPENAI_API_KEY"] = api_key
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        llm = ChatOllama(model="llama3.2", temperature=0, keep_alive="10m")
    return embeddings, llm, compressor


@st.cache_resource(show_spinner="Building hybrid BM25 + Chroma index...")
def get_rag_chain(stack_mode: str, api_key: str, index_version: int):
    """Builds and caches the conversational hybrid RAG chain."""
    embeddings, llm, compressor = get_models(stack_mode, api_key)

    chunks = load_and_chunk_documents(DATA_DIR)
    if not chunks:
        return None

    db_path = Path(DB_DIR)
    if db_path.exists() and any(db_path.iterdir()):
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=DB_DIR,
        )
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=DB_DIR,
        )

    dense_retriever = vector_store.as_retriever(search_kwargs={"k": CANDIDATE_K})
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = CANDIDATE_K

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.4, 0.6],
    )
    reranked_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given a chat history and the latest user question which might reference "
            "context in the chat history, formulate a standalone question that can be "
            "understood without the chat history. Do NOT answer the question—just "
            "reformulate it if needed, or return it as-is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, reranked_retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert system. Answer the user query using strictly "
            "the provided background context below. If the text does not contain "
            "the answer, say explicitly 'Information not found in database.'\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, qa_chain)


def extract_sources(context_docs):
    """Formats retrieved chunks with filenames, page numbers, and FlashRank scores."""
    formatted = []
    for doc in context_docs:
        filename = Path(doc.metadata.get("source", "Unknown")).name
        page = doc.metadata.get("page")
        score = doc.metadata.get("relevance_score")
        title = f"{filename} (p. {page + 1})" if page is not None else filename
        formatted.append({
            "title": title,
            "score": f"{score:.2f}" if score is not None else "N/A",
            "snippet": doc.page_content.strip(),
        })
    return formatted


def stream_rag_response(rag_chain, user_query: str, lc_history: list, state_holder: dict):
    """Yields answer tokens for st.write_stream() while capturing retrieved context docs."""
    for chunk in rag_chain.stream({"input": user_query, "chat_history": lc_history}):
        if "context" in chunk:
            state_holder["context"] = chunk["context"]
        if "answer" in chunk and chunk["answer"]:
            yield chunk["answer"]


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_version" not in st.session_state:
    st.session_state.index_version = 0
if "confirm_delete_all" not in st.session_state:
    st.session_state.confirm_delete_all = False
if "initial_seed_done" not in st.session_state:
    ensure_sample_data(allow_seed=True)
    st.session_state.initial_seed_done = True

# --- Sidebar: Stack Settings & File Manager ---
with st.sidebar:
    st.header("⚙️ RAG Configuration")

    cloud_env = is_running_on_streamlit_cloud()
    default_index = 1 if cloud_env else 0  # Index 1 = "CLOUD", Index 0 = "LOCAL"

    stack_mode = st.selectbox(
        "Model Stack",
        ["LOCAL", "CLOUD"],
        index=default_index,
        help="Auto-defaults to CLOUD when deployed on Streamlit Cloud.",
    )

    api_key = ""
    if stack_mode == "CLOUD":
        detected_key = get_cloud_api_key()
        api_key = st.text_input(
            "OpenAI API Key",
            value=detected_key,
            type="password",
            help="Loaded automatically from Streamlit Secrets if configured.",
        )
        if detected_key and api_key == detected_key:
            st.caption("🔐 Loaded `OPENAI_API_KEY` from Secrets/Environment.")

    st.divider()
    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Add PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Save & Re-index Uploads", use_container_width=True):
        Path(DATA_DIR).mkdir(exist_ok=True)
        for uf in uploaded_files:
            dest = Path(DATA_DIR) / uf.name
            dest.write_bytes(uf.getbuffer())
        st.session_state.confirm_delete_all = False
        trigger_reindex()
        st.rerun()

    st.divider()
    st.subheader("🗂️ Indexed File Manager")
    data_path = Path(DATA_DIR)
    data_path.mkdir(exist_ok=True)
    active_files = sorted(
        [f for f in data_path.iterdir() if f.is_file() and f.suffix.lower() in {".pdf", ".txt"}]
    )

    if not active_files:
        st.caption("No documents indexed. Upload a PDF or TXT file above.")
    else:
        for file_path in active_files:
            size_kb = file_path.stat().st_size / 1024
            col_name, col_del = st.columns([0.8, 0.2])
            with col_name:
                st.markdown(f"📄 **{file_path.name}**  \n`{size_kb:.1f} KB`")
            with col_del:
                if st.button(
                    "🗑️",
                    key=f"del_{file_path.name}",
                    help=f"Delete {file_path.name} and re-index",
                ):
                    file_path.unlink(missing_ok=True)
                    st.session_state.confirm_delete_all = False
                    trigger_reindex()
                    st.rerun()

        # Two-Step Delete All Confirmation
        if not st.session_state.confirm_delete_all:
            if st.button(
                "🗑️ Delete All Documents",
                use_container_width=True,
                help="Remove all files from the knowledge base",
            ):
                st.session_state.confirm_delete_all = True
                st.rerun()
        else:
            st.warning(f"Delete all {len(active_files)} document(s) and wipe the vector index?")
            c_yes, c_no = st.columns(2)
            with c_yes:
                if st.button("✅ Confirm", type="primary", use_container_width=True):
                    for f in active_files:
                        f.unlink(missing_ok=True)
                    st.session_state.confirm_delete_all = False
                    st.session_state.messages = []
                    trigger_reindex()
                    st.rerun()
            with c_no:
                if st.button("✖️ Cancel", use_container_width=True):
                    st.session_state.confirm_delete_all = False
                    st.rerun()

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Rebuild DB", use_container_width=True):
            trigger_reindex()
            st.rerun()

# --- Main Chat Area ---
st.title("🤖 Conversational Hybrid RAG")
st.caption("BM25 + ChromaDB + FlashRank Cross-Encoder Re-Ranking")

if stack_mode == "CLOUD" and not api_key:
    st.warning("Enter your OpenAI API Key in the sidebar (or add `OPENAI_API_KEY` to Streamlit Cloud Secrets).")
    st.stop()

rag_chain = get_rag_chain(stack_mode, api_key, st.session_state.index_version)
if rag_chain is None:
    st.info("Your knowledge base is empty. Upload a PDF or TXT file in the sidebar to begin.")
    st.stop()

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Retrieved Sources & FlashRank Scores"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['title']}** — Relevance Score: `{src['score']}`")
                    st.caption(src["snippet"])

# Handle new user input with real-time token streaming
if user_query := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    lc_history = []
    for m in st.session_state.messages[:-1][-(MAX_HISTORY_TURNS * 2) :]:
        if m["role"] == "user":
            lc_history.append(HumanMessage(content=m["content"]))
        else:
            lc_history.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant"):
        stream_state = {"context": []}
        answer = st.write_stream(
            stream_rag_response(rag_chain, user_query, lc_history, stream_state)
        )
        sources = extract_sources(stream_state["context"])

        if sources:
            with st.expander("📚 Retrieved Sources & FlashRank Scores"):
                for src in sources:
                    st.markdown(f"**{src['title']}** — Relevance Score: `{src['score']}`")
                    st.caption(src["snippet"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })