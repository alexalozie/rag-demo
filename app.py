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


def ensure_sample_data():
    """Creates the data directory and sample policy file if empty."""
    Path(DATA_DIR).mkdir(exist_ok=True)
    sample_txt = Path(DATA_DIR) / "sample_policy.txt"
    if not any(Path(DATA_DIR).iterdir()):
        sample_txt.write_text(
            "Company remote work strategy updated in 2026: "
            "Employees can work from any location globally for up to 90 days per calendar year. "
            "The travel allowance program is managed by the Global Mobility Team led by Marcus Vance.",
            encoding="utf-8",
        )


def load_and_chunk_documents(directory: str):
    """Loads TXT and PDF files and splits them into overlapping chunks."""
    documents = []
    for file_path in Path(directory).iterdir():
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
    ensure_sample_data()
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


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_version" not in st.session_state:
    st.session_state.index_version = 0

# --- Sidebar: Stack Settings & Document Upload ---
with st.sidebar:
    st.header("⚙️ RAG Configuration")
    stack_mode = st.selectbox("Model Stack", ["LOCAL", "CLOUD"], index=0)
    api_key = ""
    if stack_mode == "CLOUD":
        api_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
        )

    st.divider()
    st.subheader("📂 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Save & Re-index Documents", use_container_width=True):
        Path(DATA_DIR).mkdir(exist_ok=True)
        for uf in uploaded_files:
            dest = Path(DATA_DIR) / uf.name
            dest.write_bytes(uf.getbuffer())
        if Path(DB_DIR).exists():
            shutil.rmtree(DB_DIR)
        st.session_state.index_version += 1
        st.success(f"Indexed {len(uploaded_files)} file(s)!")

    ensure_sample_data()
    indexed_files = [f.name for f in Path(DATA_DIR).iterdir() if f.is_file()]
    st.caption(f"**Active files ({len(indexed_files)}):** {', '.join(indexed_files)}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Rebuild DB", use_container_width=True):
            if Path(DB_DIR).exists():
                shutil.rmtree(DB_DIR)
            st.session_state.index_version += 1
            st.rerun()

# --- Main Chat Area ---
st.title("🤖 Conversational Hybrid RAG")
st.caption("BM25 + ChromaDB + FlashRank Cross-Encoder Re-Ranking")

if stack_mode == "CLOUD" and not api_key:
    st.warning("Enter your OpenAI API Key in the sidebar to run in CLOUD mode.")
    st.stop()

rag_chain = get_rag_chain(stack_mode, api_key, st.session_state.index_version)
if rag_chain is None:
    st.info("Upload a PDF or TXT file in the sidebar to get started.")
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

# Handle new user input
if user_query := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Convert UI messages into LangChain message history
    lc_history = []
    for m in st.session_state.messages[:-1][-(MAX_HISTORY_TURNS * 2) :]:
        if m["role"] == "user":
            lc_history.append(HumanMessage(content=m["content"]))
        else:
            lc_history.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Contextualizing, retrieving & re-ranking..."):
            result = rag_chain.invoke({
                "input": user_query,
                "chat_history": lc_history,
            })
            answer = result["answer"]
            sources = extract_sources(result.get("context", []))

        st.markdown(answer)
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