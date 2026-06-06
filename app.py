import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
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
    .stApp { background-color: #0f1117; }
    [data-testid="stSidebar"] { background-color: #1a1d27; border-right: 1px solid #2e3148; }
    [data-testid="stFileUploader"] {
        background-color: #1e2130;
        border: 1.5px dashed #3b3f5c;
        border-radius: 10px;
        padding: 10px;
    }
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
    .stTextInput > div > div > input {
        background-color: #1e2130;
        color: white;
        border: 1px solid #3b3f5c;
        border-radius: 8px;
    }
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
    h1 { color: #a29bfe !important; }
    h2, h3 { color: #6c63ff !important; }
    [data-testid="stMetric"] {
        background: #1e2130;
        border: 1px solid #2e3148;
        border-radius: 10px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────
def init_state():
    defaults = {
        "chat_history": [],
        "indexed": False,
        "num_chunks": 0,
        "indexed_files": [],
        # FAISS index + parallel chunk store
        "faiss_index": None,
        "chunk_texts": [],       # list[str]  — raw text of each chunk
        "chunk_sources": [],     # list[str]  — filename for each chunk
        "embedding_model": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Cached model loader ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    return SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
    )


# ── Pipeline helpers ──────────────────────────────────────────────────
def extract_text_from_pdfs(uploaded_files):
    docs = []
    for f in uploaded_files:
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        docs.append({"source": f.name, "text": text})
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = []
    for doc in docs:
        for piece in splitter.split_text(doc["text"]):
            chunks.append({"source": doc["source"], "text": piece})
    return chunks


def build_faiss_index(chunks, model):
    """Embed all chunks and build an in-memory FAISS flat-L2 index."""
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,   # cosine via inner-product on unit vectors
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product == cosine for normalised vecs
    index.add(embeddings)

    return index, texts, [c["source"] for c in chunks]


def retrieve(question, model, index, texts, sources, k=3):
    q_emb = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    _, indices = index.search(q_emb, k)
    top_texts   = [texts[i]   for i in indices[0]]
    top_sources = [sources[i] for i in indices[0]]
    return top_texts, top_sources


def generate_answer(question, top_texts, top_sources, groq_api_key):
    client  = Groq(api_key=groq_api_key)
    context = "\n\n".join(top_texts)
    unique_sources = list(dict.fromkeys(top_sources))   # preserve order, dedupe

    prompt = f"""Use ONLY the provided context to answer the question.
If the answer is not in the context, say "I couldn't find this in the uploaded documents."

Context:
{context}

Question:
{question}
"""
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content, unique_sources


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 Smart Study Buddy")
    st.markdown("*Ask questions across multiple PDFs*")
    st.divider()

    st.markdown("### 🔑 Groq API Key")
    groq_key = st.text_input(
        "Groq API key",
        type="password",
        placeholder="gsk_…",
        help="Free key at console.groq.com",
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### 📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected**")
        for f in uploaded_files:
            st.markdown(f"• `{f.name}`")
        st.markdown("")

        if st.button("⚡ Process & Index Documents"):
            if not groq_key:
                st.error("Please enter your Groq API key first.")
            else:
                with st.spinner("Extracting text…"):
                    docs = extract_text_from_pdfs(uploaded_files)

                with st.spinner("Chunking…"):
                    chunks = chunk_documents(docs)

                with st.spinner("Loading embedding model…"):
                    model = load_embedding_model()

                with st.spinner(f"Embedding {len(chunks)} chunks…"):
                    idx, texts, srcs = build_faiss_index(chunks, model)

                # Store everything in session state
                st.session_state.faiss_index      = idx
                st.session_state.chunk_texts      = texts
                st.session_state.chunk_sources    = srcs
                st.session_state.embedding_model  = model
                st.session_state.indexed          = True
                st.session_state.num_chunks       = len(chunks)
                st.session_state.indexed_files    = [f.name for f in uploaded_files]

                st.success(f"✅ Indexed {len(chunks)} chunks from {len(docs)} file(s)!")

    st.divider()

    if st.session_state.indexed:
        st.markdown("### 📊 Index Stats")
        c1, c2 = st.columns(2)
        c1.metric("Chunks", st.session_state.num_chunks)
        c2.metric("Files",  len(st.session_state.indexed_files))
        for fn in st.session_state.indexed_files:
            st.markdown(f"✅ `{fn}`")

    if st.session_state.chat_history:
        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ── Main ──────────────────────────────────────────────────────────────
st.title("📚 Smart Study Buddy")
st.markdown("*Multi-PDF RAG Q&A — Powered by nomic-embed + **FAISS** + LLaMA 3*")
st.divider()

if not st.session_state.indexed:
    c1, c2, c3 = st.columns(3)
    c1.info("**Step 1**\n\nEnter your Groq API key in the sidebar")
    c2.info("**Step 2**\n\nUpload one or more PDF files")
    c3.info("**Step 3**\n\nClick **Process & Index** then ask away!")
    st.stop()

# Chat history display
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-msg">🧑‍💻 <strong>You</strong><br>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        badges = "".join(
            f'<span class="source-badge">📄 {s}</span>'
            for s in msg.get("sources", [])
        )
        st.markdown(
            f'<div class="ai-msg">🤖 <strong>Assistant</strong><br><br>'
            f'{msg["content"]}<br><br>{badges}</div>',
            unsafe_allow_html=True,
        )

# Question form
st.markdown("---")
with st.form("q_form", clear_on_submit=True):
    cq, cb = st.columns([5, 1])
    with cq:
        question = st.text_input(
            "question",
            placeholder="e.g. What is backpropagation? Summarise chapter 2.",
            label_visibility="collapsed",
        )
    with cb:
        submitted = st.form_submit_button("Ask ➤")

if submitted and question.strip():
    if not groq_key:
        st.error("Please enter your Groq API key in the sidebar.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("Searching & generating answer…"):
            top_texts, top_sources = retrieve(
                question,
                st.session_state.embedding_model,
                st.session_state.faiss_index,
                st.session_state.chunk_texts,
                st.session_state.chunk_sources,
            )
            answer, sources = generate_answer(question, top_texts, top_sources, groq_key)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
        st.rerun()
