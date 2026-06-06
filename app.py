import streamlit as st
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq

# ─────────────────────────────────────────────────────────────────────
#  GROQ API KEY  — hidden from users entirely
#  Streamlit Cloud: Settings → Secrets → add  GROQ_API_KEY = "gsk_..."
#  Local testing:  replace st.secrets["GROQ_API_KEY"] with your key string
# ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Study Buddy",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }

.stApp { background: #07090f; color: #c8d0e0; }
.block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── top bar ── */
.topbar {
  display:flex; align-items:center; gap:14px;
  padding:13px 32px;
  background:#0b0e1c; border-bottom:1px solid #181f38;
  position:sticky; top:0; z-index:200;
}
.tb-dot {
  width:9px; height:9px; border-radius:50%;
  background:#7c6cfc; box-shadow:0 0 10px #7c6cfc99;
  animation:tb-pulse 2.2s ease-in-out infinite;
}
@keyframes tb-pulse{0%,100%{opacity:1}50%{opacity:.3}}
.tb-title { font-size:16px; font-weight:700; color:#e0dbff; letter-spacing:-.01em; }
.tb-stack { font-family:'JetBrains Mono',monospace; font-size:11px; color:#313a5c; margin-left:auto; }
.tb-tag {
  background:#11152a; border:1px solid #1d2545;
  border-radius:20px; padding:4px 14px;
  font-size:11px; color:#515c85;
  font-family:'JetBrains Mono',monospace;
}

/* ── two columns ── */
.cols-wrap { display:grid; grid-template-columns:310px 1fr; height:calc(100vh - 50px); }

/* ── left panel ── */
.left {
  background:#0b0e1c; border-right:1px solid #181f38;
  display:flex; flex-direction:column; overflow-y:auto;
  padding:20px 16px; gap:0;
}
.sec-label {
  font-size:10px; font-weight:700; letter-spacing:.13em;
  text-transform:uppercase; color:#313a5c;
  display:flex; align-items:center; gap:8px; margin-bottom:12px;
}
.sec-label::after{content:'';flex:1;height:1px;background:#181f38;}

[data-testid="stFileUploader"] {
  background:#07090f !important; border:1.5px dashed #1d2545 !important;
  border-radius:12px !important; transition:border-color .2s !important;
}
[data-testid="stFileUploader"]:hover { border-color:#7c6cfc !important; }

.chip {
  display:flex; align-items:center; gap:7px;
  background:#0f1222; border:1px solid #1d2545;
  border-radius:8px; padding:7px 11px; margin-bottom:5px;
  font-size:12px; color:#7a84a8;
  font-family:'JetBrains Mono',monospace; word-break:break-all;
}
.cdot { width:6px; height:6px; border-radius:50%; flex-shrink:0; }
.cdot-v { background:#7c6cfc; box-shadow:0 0 5px #7c6cfc88; }
.cdot-g { background:#4ade80; box-shadow:0 0 5px #4ade8088; }

.stat-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:14px 0; }
.stat-box  {
  background:#07090f; border:1px solid #181f38;
  border-radius:10px; padding:12px; text-align:center;
}
.sv { font-size:22px; font-weight:700; color:#7c6cfc; line-height:1; }
.sl { font-size:10px; color:#313a5c; margin-top:4px; text-transform:uppercase; letter-spacing:.06em; }

/* buttons */
.stButton > button {
  background:linear-gradient(135deg,#7c6cfc,#4facfe) !important;
  color:#fff !important; border:none !important;
  border-radius:10px !important; font-weight:600 !important;
  font-size:14px !important; padding:11px 0 !important;
  width:100% !important; letter-spacing:.02em !important;
  font-family:'Outfit',sans-serif !important;
  transition:opacity .2s, transform .15s !important;
}
.stButton > button:hover  { opacity:.88 !important; transform:translateY(-1px) !important; }
.stButton > button:active { transform:translateY(0) !important; }
.stButton > button:disabled { opacity:.25 !important; cursor:not-allowed !important; }

/* ── right: chat ── */
.right { display:flex; flex-direction:column; overflow:hidden; background:#07090f; }

.chat-top {
  padding:13px 28px; background:#0b0e1c;
  border-bottom:1px solid #181f38;
  display:flex; align-items:center; gap:10px; flex-shrink:0;
}
.ct-title { font-size:15px; font-weight:600; color:#c8d0e0; }
.ct-sub   { font-size:11px; color:#313a5c; margin-left:auto;
            font-family:'JetBrains Mono',monospace; }

.chat-body { flex:1; overflow-y:auto; padding:24px 32px; display:flex; flex-direction:column; gap:16px; }

::-webkit-scrollbar        { width:4px; }
::-webkit-scrollbar-track  { background:transparent; }
::-webkit-scrollbar-thumb  { background:#1d2545; border-radius:10px; }

/* empty */
.empty { margin:auto; text-align:center; padding:40px 20px; }
.e-icon { font-size:42px; margin-bottom:14px; opacity:.35; }
.e-txt  { font-size:14px; line-height:1.85; color:#252f50; }
.e-txt b { color:#313a5c; }

/* bubbles */
.lbl-you { font-size:10px; font-weight:700; letter-spacing:.1em;
           color:#7c6cfc; text-align:right; margin-bottom:5px;
           font-family:'JetBrains Mono',monospace; }
.lbl-ai  { font-size:10px; font-weight:700; letter-spacing:.1em;
           color:#4facfe; margin-bottom:5px;
           font-family:'JetBrains Mono',monospace; }

.bbl-you {
  background:linear-gradient(135deg,#7c6cfc16,#4facfe0d);
  border:1px solid #7c6cfc28; border-radius:16px 16px 4px 16px;
  padding:13px 17px; color:#ddd6fe;
  margin-left:60px; font-size:14px; line-height:1.7;
}
.bbl-ai {
  background:#0f1222; border:1px solid #181f38;
  border-radius:16px 16px 16px 4px;
  padding:14px 17px; color:#c8d0e0;
  margin-right:60px; font-size:14px; line-height:1.8;
}
.src-row { margin-top:10px; display:flex; flex-wrap:wrap; gap:5px; }
.src-tag {
  background:#07090f; border:1px solid #1d2545;
  border-radius:6px; padding:3px 10px; font-size:11px; color:#4facfe;
  font-family:'JetBrains Mono',monospace;
}

/* input bar */
.inp-bar {
  padding:14px 28px; background:#0b0e1c;
  border-top:1px solid #181f38; flex-shrink:0;
}
.ready-strip {
  background:#091510; border:1px solid #183525;
  border-radius:8px; padding:7px 14px; margin-bottom:10px;
  font-size:11px; color:#4ade80;
  font-family:'JetBrains Mono',monospace;
  display:flex; align-items:center; gap:8px; flex-wrap:wrap;
}
.rdot { width:6px;height:6px;border-radius:50%;background:#4ade80;flex-shrink:0; }

.stTextInput > label { display:none !important; }
.stTextInput > div > div > input {
  background:#0f1222 !important; color:#e0e4f4 !important;
  border:1.5px solid #1d2545 !important; border-radius:10px !important;
  font-size:14px !important; padding:12px 16px !important;
  font-family:'Outfit',sans-serif !important;
  transition:border-color .2s, box-shadow .2s !important;
}
.stTextInput > div > div > input:focus {
  border-color:#7c6cfc !important;
  box-shadow:0 0 0 3px #7c6cfc16 !important;
}
.stTextInput > div > div > input::placeholder { color:#252f50 !important; }

/* progress bar */
.stProgress > div > div > div > div {
  background:linear-gradient(90deg,#7c6cfc,#4facfe) !important;
  border-radius:10px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────
_defaults = {
    "chat_history":   [],
    "indexed":        False,
    "indexed_files":  [],
    "num_chunks":     0,
    "collection":     None,
    "embed_model":    None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────
#  CACHED RESOURCES
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    # all-MiniLM-L6-v2 — no trust_remote_code, works on Python 3.14
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Starting ChromaDB…")
def get_chroma_collection():
    # chromadb 1.x API — EphemeralClient for Streamlit Cloud (no disk needed)
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(name="pdf_documents")


# ─────────────────────────────────────────────────────────────────────
#  PIPELINE  (mirrors your notebook logic exactly)
# ─────────────────────────────────────────────────────────────────────
def extract_text_from_pdfs(uploaded_files):
    all_documents = []
    for file in uploaded_files:
        reader = PdfReader(file)
        text   = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        all_documents.append({"source": file.name, "text": text})
    return all_documents


def chunk_documents(all_documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )
    all_chunks = []
    for doc in all_documents:
        for chunk_text in text_splitter.split_text(doc["text"]):
            all_chunks.append({"source": doc["source"], "text": chunk_text})
    return all_chunks


def create_chunk_embeddings(all_chunks, embedding_model):
    texts = [chunk["text"] for chunk in all_chunks]
    return embedding_model.encode(texts, convert_to_numpy=True)


def store_in_vectordb(all_chunks, chunk_embeddings, collection):
    # clear old data so re-indexing works cleanly
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    collection.add(
        ids        = [str(i) for i in range(len(all_chunks))],
        documents  = [c["text"]   for c in all_chunks],
        embeddings = chunk_embeddings.tolist(),
        metadatas  = [{"source": c["source"]} for c in all_chunks],
    )
    return len(all_chunks)


def retrieve(question, embedding_model, collection, n_results=3):
    query_embedding = embedding_model.encode(question, convert_to_numpy=True)
    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
    )


def generate_answer(question, results):
    client  = Groq(api_key=GROQ_API_KEY)
    context = "\n\n".join(results["documents"][0])
    prompt  = f"""
    Use the provided context to answer the question.

    Context:
    {context}

    Question:
    {question}
    """
    response = client.chat.completions.create(
        model    = GROQ_MODEL,
        messages = [{"role": "user", "content": prompt}],
    )
    answer  = response.choices[0].message.content
    sources = list(dict.fromkeys(m["source"] for m in results["metadatas"][0]))
    return answer, sources


# ─────────────────────────────────────────────────────────────────────
#  TOP BAR
# ─────────────────────────────────────────────────────────────────────
n_files  = len(st.session_state.indexed_files)
tag_text = f"{n_files} doc{'s' if n_files!=1 else ''} indexed" if st.session_state.indexed else "no index"

st.markdown(f"""
<div class="topbar">
  <div class="tb-dot"></div>
  <span class="tb-title">📚 Smart Study Buddy</span>
  <span class="tb-stack">chromadb · all-MiniLM · LLaMA-3.3-70B</span>
  <span class="tb-tag">{tag_text}</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([0.27, 0.73], gap="small")


# ═══════════════════════════════════
#  LEFT — Upload + Index
# ═══════════════════════════════════
with col_left:

    st.markdown('<div class="sec-label">📄 Upload PDFs</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "pdfs", type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for f in uploaded_files:
            st.markdown(
                f'<div class="chip"><span class="cdot cdot-v"></span>{f.name}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.indexed:
        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-box"><div class="sv">{len(st.session_state.indexed_files)}</div><div class="sl">PDFs</div></div>
          <div class="stat-box"><div class="sv">{st.session_state.num_chunks}</div><div class="sl">Chunks</div></div>
        </div>
        """, unsafe_allow_html=True)

    btn_label = "⚡ Process & Index" if not st.session_state.indexed else "🔄 Re-index"
    do_index  = st.button(btn_label, disabled=not uploaded_files)

    if st.session_state.indexed:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">✅ Indexed files</div>', unsafe_allow_html=True)
        for fn in st.session_state.indexed_files:
            st.markdown(
                f'<div class="chip"><span class="cdot cdot-g"></span>{fn}</div>',
                unsafe_allow_html=True,
            )

    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # ── run pipeline ──
    if do_index and uploaded_files:
        prog = st.progress(0, text="Reading PDFs…")
        docs   = extract_text_from_pdfs(uploaded_files)
        prog.progress(20, text="Chunking text…")
        chunks = chunk_documents(docs)
        prog.progress(40, text="Loading embedding model…")
        model  = load_embedding_model()
        prog.progress(60, text=f"Embedding {len(chunks)} chunks…")
        embs   = create_chunk_embeddings(chunks, model)
        prog.progress(80, text="Storing in ChromaDB…")
        col    = get_chroma_collection()
        n      = store_in_vectordb(chunks, embs, col)
        prog.progress(100, text="Done!")

        st.session_state.collection    = col
        st.session_state.embed_model   = model
        st.session_state.indexed       = True
        st.session_state.num_chunks    = n
        st.session_state.indexed_files = [f.name for f in uploaded_files]
        st.session_state.chat_history  = []

        prog.empty()
        st.success(f"✅ {n} chunks from {len(docs)} file(s) stored in ChromaDB")
        st.rerun()


# ═══════════════════════════════════
#  RIGHT — Chat
# ═══════════════════════════════════
with col_right:

    n_ex = len(st.session_state.chat_history) // 2
    sub  = f"{n_ex} exchange{'s' if n_ex!=1 else ''}" if n_ex else "waiting…"

    st.markdown(f"""
    <div class="chat-top">
      <span style="font-size:18px">💬</span>
      <span class="ct-title">Ask your documents</span>
      <span class="ct-sub">{sub}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── messages ──
    with st.container():
        if not st.session_state.indexed:
            st.markdown("""
            <div class="empty">
              <div class="e-icon">📂</div>
              <div class="e-txt">
                Upload your PDFs on the left<br>
                and click <b>Process &amp; Index</b><br>
                to start asking questions.
              </div>
            </div>
            """, unsafe_allow_html=True)

        elif not st.session_state.chat_history:
            files_bold = ", ".join(f"<b>{f}</b>" for f in st.session_state.indexed_files)
            st.markdown(f"""
            <div class="empty">
              <div class="e-icon">✨</div>
              <div class="e-txt">Ready! Ask anything about<br>{files_bold}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="lbl-you">YOU</div>
                    <div class="bbl-you">{msg["content"]}</div>
                    """, unsafe_allow_html=True)
                else:
                    srcs     = "".join(f'<span class="src-tag">📄 {s}</span>' for s in msg.get("sources", []))
                    src_html = f'<div class="src-row">{srcs}</div>' if srcs else ""
                    st.markdown(f"""
                    <div class="lbl-ai">ASSISTANT</div>
                    <div class="bbl-ai">{msg["content"]}{src_html}</div>
                    """, unsafe_allow_html=True)

    # ── input bar ──
    if st.session_state.indexed:
        files_str = "  ·  ".join(st.session_state.indexed_files)
        st.markdown(f"""
        <div class="ready-strip">
          <div class="rdot"></div>{files_str}
        </div>
        """, unsafe_allow_html=True)

    ic, bc = st.columns([6, 1])
    with ic:
        question = st.text_input(
            "q",
            placeholder="Ask anything about your documents…",
            label_visibility="collapsed",
            disabled=not st.session_state.indexed,
        )
    with bc:
        ask = st.button("Ask ➤", disabled=not st.session_state.indexed)


# ─────────────────────────────────────────────────────────────────────
#  HANDLE ASK
# ─────────────────────────────────────────────────────────────────────
if ask and question.strip():
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.spinner("Searching documents and generating answer…"):
        results         = retrieve(question, st.session_state.embed_model, st.session_state.collection)
        answer, sources = generate_answer(question, results)
    st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
    st.rerun()
