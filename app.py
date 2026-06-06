import os
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Study Buddy",
    page_icon="📚",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1a1d27; border-right: 1px solid #2e3148; }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background-color: #1e2130;
        border: 1.5px dashed #3b3f5c;
        border-radius: 10px;
        padding: 10px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6c63ff, #4facfe);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Input box */
    .stTextInput > div > div > input {
        background-color: #1e2130;
        color: white;
        border: 1px solid #3b3f5c;
        border-radius: 8px;
    }

    /* Chat messages */
    .user-msg {
        background: linear-gradient(135deg, #6c63ff22, #4facfe22);
        border: 1px solid #6c63ff44;
        border-radius: 12px 12px 4px 12px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e0e0ff;
    }
    .ai-msg {
        background-color: #1e2130;
        border: 1px solid #2e3148;
        border-radius: 12px 12px 12px 4px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #d0d0e8;
    }
    .source-badge {
        display: inline-block;
        background: #2e3148;
        border: 1px solid #3b3f5c;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 11px;
        color: #7c83d0;
        margin: 4px 3px 0 0;
    }

    /* Header */
    h1 { color: #a29bfe !important; }
    h2, h3 { color: #6c63ff !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1e2130;
        border: 1px solid #2e3148;
        border-radius: 10px;
        padding: 12px;
    }

    /* Success/info boxes */
    .stSuccess { background-color: #1a2e1a !important; border-color: #2d6a2d !important; }
    .stInfo    { background-color: #1a1e2e !important; border-color: #2d4a6a !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────
def init_state():
    defaults = {
        "chat_history": [],
        "indexed": False,
        "num_chunks": 0,
        "indexed_files": [],
        "collection": None,
        "embedding_model": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Cached loaders ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

@st.cache_resource(show_spinner="Setting up vector database...")
def get_chroma_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name="pdf_documents")


# ── Core pipeline functions ───────────────────────────────────────────
def extract_text_from_pdfs(uploaded_files):
    """Extract text from all uploaded PDFs with metadata."""
    all_documents = []
    for file in uploaded_files:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        all_documents.append({"source": file.name, "text": text})
    return all_documents


def chunk_documents(all_documents):
    """Split documents into overlapping chunks with source metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_chunks = []
    for doc in all_documents:
        chunks = splitter.split_text(doc["text"])
        for chunk_text in chunks:
            all_chunks.append({"source": doc["source"], "text": chunk_text})
    return all_chunks


def build_vector_index(all_chunks, embedding_model, collection):
    """Embed all chunks and store them in ChromaDB."""
    # Clear existing data
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    texts = [c["text"] for c in all_chunks]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    collection.add(
        ids=[str(i) for i in range(len(all_chunks))],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": c["source"]} for c in all_chunks],
    )
    return len(all_chunks)


def retrieve_context(question, embedding_model, collection, n_results=3):
    """Embed the question and retrieve the top-n matching chunks."""
    query_emb = embedding_model.encode(question, convert_to_numpy=True)
    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=n_results,
    )
    return results


def generate_answer(question, results, groq_api_key):
    """Send retrieved context + question to Groq LLaMA 3."""
    client = Groq(api_key=groq_api_key)
    context = "\n\n".join(results["documents"][0])
    sources = list({m["source"] for m in results["metadatas"][0]})

    prompt = f"""Use ONLY the provided context to answer the question.
If the answer is not in the context, say "I couldn't find this in the uploaded documents."

Context:
{context}

Question:
{question}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content, sources


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 Smart Study Buddy")
    st.markdown("*Ask questions across multiple PDFs*")
    st.divider()

    # API Key input
    st.markdown("### 🔑 Groq API Key")
    groq_key = st.text_input(
        "Enter your Groq API key",
        type="password",
        placeholder="gsk_...",
        help="Get your free key at console.groq.com",
    )

    st.divider()

    # File upload
    st.markdown("### 📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drop your PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Index button
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected**")
        for f in uploaded_files:
            st.markdown(f"• `{f.name}`")
        st.markdown("")

        if st.button("⚡ Process & Index Documents"):
            if not groq_key:
                st.error("Please enter your Groq API key first.")
            else:
                with st.spinner("Extracting text from PDFs..."):
                    all_documents = extract_text_from_pdfs(uploaded_files)

                with st.spinner("Chunking documents..."):
                    all_chunks = chunk_documents(all_documents)

                with st.spinner("Loading embedding model..."):
                    model = load_embedding_model()

                with st.spinner(f"Embedding {len(all_chunks)} chunks..."):
                    collection = get_chroma_collection()
                    n = build_vector_index(all_chunks, model, collection)

                st.session_state.indexed = True
                st.session_state.num_chunks = n
                st.session_state.indexed_files = [f.name for f in uploaded_files]
                st.session_state.embedding_model = model
                st.session_state.collection = collection
                st.success(f"✅ Indexed {n} chunks from {len(all_documents)} file(s)!")

    st.divider()

    # Index stats
    if st.session_state.indexed:
        st.markdown("### 📊 Index Stats")
        col1, col2 = st.columns(2)
        col1.metric("Chunks", st.session_state.num_chunks)
        col2.metric("Files", len(st.session_state.indexed_files))
        st.markdown("**Indexed files:**")
        for fn in st.session_state.indexed_files:
            st.markdown(f"✅ `{fn}`")

    # Clear button
    if st.session_state.chat_history:
        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ── Main area ─────────────────────────────────────────────────────────
st.title("📚 Smart Study Buddy")
st.markdown("*Multi-PDF RAG Q&A — Powered by nomic-embed + ChromaDB + LLaMA 3*")
st.divider()

# Welcome state
if not st.session_state.indexed:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\n\nEnter your Groq API key in the sidebar")
    with col2:
        st.info("**Step 2**\n\nUpload one or more PDF files")
    with col3:
        st.info("**Step 3**\n\nClick **Process & Index** then start asking questions!")
    st.stop()

# Chat display
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">🧑‍💻 <strong>You</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        sources_html = "".join(f'<span class="source-badge">📄 {s}</span>' for s in msg.get("sources", []))
        st.markdown(
            f'<div class="ai-msg">🤖 <strong>Assistant</strong><br><br>{msg["content"]}'
            f'<br><br>{sources_html}</div>',
            unsafe_allow_html=True,
        )

# Question input
st.markdown("---")
with st.form("question_form", clear_on_submit=True):
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        question = st.text_input(
            "Ask a question",
            placeholder="e.g. What is backpropagation? Summarise chapter 2.",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Ask ➤")

if submitted and question.strip():
    if not groq_key:
        st.error("Please enter your Groq API key in the sidebar.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("Searching documents and generating answer..."):
            results = retrieve_context(
                question,
                st.session_state.embedding_model,
                st.session_state.collection,
            )
            answer, sources = generate_answer(question, results, groq_key)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
        st.rerun()
