import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq

# ── YOUR GROQ API KEY (hidden from users) ─────────────────────────────
GROQ_API_KEY = "GROQ_API_KEY"   # ← paste your key here
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Study Buddy",
    page_icon="📚",
    layout="centered",
)

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0f1117; color: #e0e0e0; }

[data-testid="stSidebar"] { display: none; }

h1 { color: #a29bfe !important; text-align: center; font-size: 2rem !important; }
.subtitle { text-align: center; color: #6b7280; font-size: 0.95rem; margin-top: -12px; margin-bottom: 28px; }

/* Upload box */
[data-testid="stFileUploader"] {
    background: #1e2130;
    border: 2px dashed #3b3f5c;
    border-radius: 14px;
    padding: 8px;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #6c63ff; }

/* File pills */
.file-pill {
    display: inline-block;
    background: #2a2d45;
    border: 1px solid #3b3f5c;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: #a29bfe;
    margin: 3px;
}

/* Ask button */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #4facfe) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 10px 0 !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stButton > button:disabled { opacity: 0.35 !important; cursor: not-allowed !important; }

/* Text input */
.stTextInput > label { color: #9ca3af !important; font-size: 13px !important; }
.stTextInput > div > div > input {
    background: #1e2130 !important;
    color: #e0e0e0 !important;
    border: 1.5px solid #3b3f5c !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus { border-color: #6c63ff !important; box-shadow: 0 0 0 3px #6c63ff22 !important; }

/* Chat bubbles */
.bubble-user {
    background: linear-gradient(135deg, #6c63ff33, #4facfe22);
    border: 1px solid #6c63ff55;
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px;
    margin: 10px 0 10px 40px;
    color: #ddd6fe;
    line-height: 1.6;
}
.bubble-ai {
    background: #1e2130;
    border: 1px solid #2e3148;
    border-radius: 16px 16px 16px 4px;
    padding: 14px 18px;
    margin: 10px 40px 10px 0;
    color: #d0d0e8;
    line-height: 1.7;
}
.label { font-size: 11px; font-weight: 600; letter-spacing: 0.06em; margin-bottom: 4px; }
.label-user { color: #818cf8; text-align: right; }
.label-ai   { color: #4facfe; }

/* Source tags */
.src-row { margin-top: 8px; }
.src-tag {
    display: inline-block;
    background: #16213e;
    border: 1px solid #2e3f6e;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 11px;
    color: #60a5fa;
    margin: 2px 3px 0 0;
}

/* Divider */
hr { border-color: #2e3148 !important; margin: 24px 0 !important; }

/* Status badge */
.status-ok {
    background: #14532d22;
    border: 1px solid #16a34a55;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 13px;
    color: #4ade80;
    margin-bottom: 16px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────
for key, val in {
    "chat_history": [],
    "indexed": False,
    "faiss_index": None,
    "chunk_texts": [],
    "chunk_sources": [],
    "embedding_model": None,
    "indexed_files": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Cached model ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model (first time only)…")
def load_model():
    return SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
    )


# ── Pipeline functions (exact logic from your notebook) ───────────────
def extract_text(uploaded_files):
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    tokenized_chunks = []
    for doc in all_documents:
        chunks = text_splitter.split_text(doc["text"])
        for chunk_text in chunks:
            tokenized_chunks.append({
                "source": doc["source"],
                "text": chunk_text,
            })
    return tokenized_chunks


def build_faiss_index(tokenized_chunks, embedding_model):
    texts = [chunk["text"] for chunk in tokenized_chunks]
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    sources = [chunk["source"] for chunk in tokenized_chunks]
    return index, texts, sources


def retrieve(question, embedding_model, index, texts, sources, n_results=3):
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    _, indices = index.search(query_embedding, n_results)
    top_texts   = [texts[i]   for i in indices[0]]
    top_sources = [sources[i] for i in indices[0]]
    return top_texts, top_sources


def generate_answer(question, top_texts, top_sources):
    client = Groq(api_key=GROQ_API_KEY)

    context = "\n\n".join(top_texts)

    prompt = f"""
    Use the provided context to answer the question.

    Context:
    {context}

    Question:
    {question}
    """

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content
    unique_sources = list(dict.fromkeys(top_sources))
    return answer, unique_sources


# ── UI ────────────────────────────────────────────────────────────────
st.title("📚 Smart Study Buddy")
st.markdown('<p class="subtitle">Upload your PDFs and ask any question</p>', unsafe_allow_html=True)

# ── SECTION 1 : Upload ────────────────────────────────────────────────
st.markdown("#### 📄 Upload PDFs")
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    pills = "".join(f'<span class="file-pill">📄 {f.name}</span>' for f in uploaded_files)
    st.markdown(pills, unsafe_allow_html=True)
    st.markdown("")

    if st.button("⚡ Process & Index Documents"):
        with st.spinner("Reading PDFs…"):
            docs = extract_text(uploaded_files)

        with st.spinner("Splitting into chunks…"):
            tokenized_chunks = chunk_documents(docs)

        with st.spinner("Loading embedding model…"):
            model = load_model()

        with st.spinner(f"Embedding {len(tokenized_chunks)} chunks into FAISS…"):
            index, texts, sources = build_faiss_index(tokenized_chunks, model)

        st.session_state.faiss_index     = index
        st.session_state.chunk_texts     = texts
        st.session_state.chunk_sources   = sources
        st.session_state.embedding_model = model
        st.session_state.indexed         = True
        st.session_state.indexed_files   = [f.name for f in uploaded_files]
        st.session_state.chat_history    = []   # reset chat on new upload

        st.success(f"✅ Indexed {len(tokenized_chunks)} chunks from {len(docs)} file(s)!")

st.markdown("<hr>", unsafe_allow_html=True)

# ── SECTION 2 : Ask ───────────────────────────────────────────────────
if not st.session_state.indexed:
    st.info("👆 Upload and process your PDFs first, then ask questions here.")
else:
    st.markdown(
        f'<div class="status-ok">✅ {len(st.session_state.indexed_files)} file(s) ready — '
        + ", ".join(f"<b>{f}</b>" for f in st.session_state.indexed_files)
        + "</div>",
        unsafe_allow_html=True,
    )

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="label label-user">YOU</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="label label-ai">ASSISTANT</div>', unsafe_allow_html=True)
            srcs = "".join(f'<span class="src-tag">📄 {s}</span>' for s in msg.get("sources", []))
            st.markdown(
                f'<div class="bubble-ai">{msg["content"]}'
                + (f'<div class="src-row">{srcs}</div>' if srcs else "")
                + "</div>",
                unsafe_allow_html=True,
            )

    # Question input
    st.markdown("#### 💬 Ask a Question")
    question = st.text_input(
        "question",
        placeholder="e.g. What is backpropagation? Explain gradient descent.",
        label_visibility="collapsed",
    )

    if st.button("Ask ➤"):
        if question.strip():
            st.session_state.chat_history.append({"role": "user", "content": question})

            with st.spinner("Searching documents and generating answer…"):
                top_texts, top_sources = retrieve(
                    question,
                    st.session_state.embedding_model,
                    st.session_state.faiss_index,
                    st.session_state.chunk_texts,
                    st.session_state.chunk_sources,
                )
                answer, sources = generate_answer(question, top_texts, top_sources)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
            st.rerun()
        else:
            st.warning("Please type a question first.")

    # Clear chat
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()