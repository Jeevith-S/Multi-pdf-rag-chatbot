import streamlit as st
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq

# ─────────────────────────────────────────────
#  YOUR GROQ API KEY  (hidden from users)
#  For Streamlit Cloud: use st.secrets["GROQ_API_KEY"]
#  For local:          paste key directly as string below
# ─────────────────────────────────────────────
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Study Buddy",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }

/* ── app shell ── */
.stApp { background: #07090f; color: #c8d0e0; }
.block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── top bar ── */
.topbar {
  display: flex; align-items: center; gap: 14px;
  padding: 14px 32px;
  background: #0c0f1c;
  border-bottom: 1px solid #1a2035;
  position: sticky; top: 0; z-index: 200;
}
.tb-pulse {
  width: 10px; height: 10px; border-radius: 50%;
  background: #7c6cfc;
  box-shadow: 0 0 10px #7c6cfc99;
  animation: tb-blink 2.2s ease-in-out infinite;
}
@keyframes tb-blink { 0%,100%{opacity:1} 50%{opacity:.3} }
.tb-title { font-size: 17px; font-weight: 700; color: #e2dcff; letter-spacing: -.01em; }
.tb-mono  { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #3d4a70; margin-left: auto; }
.tb-tag {
  background: #131829; border: 1px solid #1e2a48;
  border-radius: 20px; padding: 4px 14px;
  font-size: 11px; color: #5a6a99;
  font-family: 'JetBrains Mono', monospace;
}

/* ── two-col wrapper ── */
.app-grid {
  display: grid;
  grid-template-columns: 320px 1fr;
  height: calc(100vh - 53px);
  overflow: hidden;
}

/* ── left panel ── */
.left-panel {
  background: #0c0f1c;
  border-right: 1px solid #1a2035;
  display: flex; flex-direction: column;
  overflow-y: auto; overflow-x: hidden;
  padding: 20px 18px;
}
.section-label {
  font-size: 10px; font-weight: 700;
  letter-spacing: .14em; text-transform: uppercase;
  color: #3d4a70; margin-bottom: 12px;
  display: flex; align-items: center; gap: 8px;
}
.section-label::after { content:''; flex:1; height:1px; background:#1a2035; }

/* upload zone */
[data-testid="stFileUploader"] {
  background: #08090f !important;
  border: 1.5px dashed #1e2a48 !important;
  border-radius: 12px !important;
  padding: 6px !important;
  transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: #7c6cfc !important;
}

/* file chips */
.chip {
  display: flex; align-items: center; gap: 7px;
  background: #11152a; border: 1px solid #1e2a48;
  border-radius: 8px; padding: 7px 11px;
  font-size: 12px; color: #8892b0;
  font-family: 'JetBrains Mono', monospace;
  margin-bottom: 5px;
  word-break: break-all;
}
.chip-dot { width:6px;height:6px;border-radius:50%;flex-shrink:0; }
.chip-dot-blue  { background:#7c6cfc; box-shadow:0 0 6px #7c6cfc88; }
.chip-dot-green { background:#4ade80; box-shadow:0 0 6px #4ade8088; }

/* stats */
.stat-row { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:14px 0; }
.stat-box {
  background:#08090f; border:1px solid #1a2035;
  border-radius:10px; padding:12px; text-align:center;
}
.stat-val { font-size:22px; font-weight:700; color:#7c6cfc; line-height:1; }
.stat-lbl { font-size:10px; color:#3d4a70; margin-top:4px;
            text-transform:uppercase; letter-spacing:.06em; }

/* process button */
.stButton > button {
  background: linear-gradient(135deg,#7c6cfc,#4facfe) !important;
  color:#fff !important; border:none !important;
  border-radius:10px !important; font-weight:600 !important;
  font-size:14px !important; padding:11px 0 !important;
  width:100% !important; letter-spacing:.02em !important;
  font-family:'Outfit',sans-serif !important;
  transition: opacity .2s, transform .15s !important;
}
.stButton > button:hover  { opacity:.88 !important; transform:translateY(-1px) !important; }
.stButton > button:active { transform:translateY(0) !important; }
.stButton > button:disabled { opacity:.28 !important; cursor:not-allowed !important; }

/* ── right panel ── */
.right-panel {
  display:flex; flex-direction:column;
  overflow:hidden;
}

/* chat header */
.chat-header {
  padding:14px 28px; background:#0a0c18;
  border-bottom:1px solid #1a2035;
  display:flex; align-items:center; gap:10px;
}
.ch-title { font-size:15px; font-weight:600; color:#c8d0e0; }
.ch-sub   { font-size:12px; color:#3d4a70; margin-left:auto;
            font-family:'JetBrains Mono',monospace; }

/* chat scroll area */
.chat-scroll {
  flex:1; overflow-y:auto; overflow-x:hidden;
  padding:24px 28px;
  display:flex; flex-direction:column; gap:18px;
}
::-webkit-scrollbar        { width:4px; }
::-webkit-scrollbar-track  { background:transparent; }
::-webkit-scrollbar-thumb  { background:#1e2a48; border-radius:10px; }

/* empty state */
.empty-state {
  margin:auto; text-align:center;
  padding:40px 20px; color:#1e2a48;
}
.empty-icon { font-size:44px; margin-bottom:14px; opacity:.5; }
.empty-txt  { font-size:14px; line-height:1.8; color:#2e3a5a; }
.empty-txt b { color:#3d4a70; }

/* bubble: user */
.msg-you-label { font-size:10px; font-weight:700; letter-spacing:.1em;
                 color:#7c6cfc; text-align:right; margin-bottom:5px;
                 font-family:'JetBrains Mono',monospace; }
.bubble-you {
  background: linear-gradient(135deg,#7c6cfc18,#4facfe0e);
  border:1px solid #7c6cfc30;
  border-radius:16px 16px 4px 16px;
  padding:13px 17px; color:#ddd6fe;
  margin-left:60px; font-size:14px; line-height:1.65;
}

/* bubble: ai */
.msg-ai-label  { font-size:10px; font-weight:700; letter-spacing:.1em;
                 color:#4facfe; margin-bottom:5px;
                 font-family:'JetBrains Mono',monospace; }
.bubble-ai {
  background:#0f1322; border:1px solid #1a2035;
  border-radius:16px 16px 16px 4px;
  padding:14px 17px; color:#c8d0e0;
  margin-right:60px; font-size:14px; line-height:1.8;
}
.src-row { margin-top:10px; display:flex; flex-wrap:wrap; gap:5px; }
.src-tag {
  background:#07090f; border:1px solid #1e2a48;
  border-radius:6px; padding:3px 10px;
  font-size:11px; color:#4facfe;
  font-family:'JetBrains Mono',monospace;
}

/* typing dots */
.typing { display:flex; gap:5px; align-items:center; padding:4px 0; }
.typing span {
  width:7px;height:7px;border-radius:50%;background:#3d4a70;
  animation:dot-blink 1.3s ease-in-out infinite;
}
.typing span:nth-child(2){animation-delay:.2s}
.typing span:nth-child(3){animation-delay:.4s}
@keyframes dot-blink{0%,80%,100%{opacity:.2}40%{opacity:1}}

/* input bar */
.input-bar {
  padding:14px 28px;
  background:#0a0c18; border-top:1px solid #1a2035;
}
.ready-bar {
  background:#0a1810; border:1px solid #1a3a28;
  border-radius:8px; padding:7px 14px;
  font-size:11px; color:#4ade80;
  font-family:'JetBrains Mono',monospace;
  display:flex; align-items:center; gap:8px;
  margin-bottom:10px; flex-wrap:wrap;
}
.ready-dot { width:6px;height:6px;border-radius:50%;background:#4ade80;flex-shrink:0; }

.stTextInput > label { display:none !important; }
.stTextInput > div > div > input {
  background:#0f1322 !important; color:#e0e0f0 !important;
  border:1.5px solid #1e2a48 !important;
  border-radius:10px !important;
  font-size:14px !important; padding:12px 16px !important;
  font-family:'Outfit',sans-serif !important;
  transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput > div > div > input:focus {
  border-color:#7c6cfc !important;
  box-shadow:0 0 0 3px #7c6cfc18 !important;
}
.stTextInput > div > div > input::placeholder { color:#2e3a5a !important; }

/* progress */
.stProgress > div > div { background: linear-gradient(90deg,#7c6cfc,#4facfe) !important;
                          border-radius:10px !important; }

/* alerts */
.stSuccess,.stInfo,.stWarning,.stError { border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for k, v in {
    "chat_history":   [],
    "indexed":        False,
    "indexed_files":  [],
    "num_chunks":     0,
    "collection":     None,
    "embed_model":    None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
#  CACHED RESOURCES
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    # nomic-embed-text-v1.5 — same as your notebook
    return SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
    )

@st.cache_resource(show_spinner="Starting ChromaDB…")
def get_chroma_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name="pdf_documents")


# ─────────────────────────────────────────────
#  PIPELINE  (exact logic from your notebook)
# ─────────────────────────────────────────────
def extract_text_from_pdfs(uploaded_files):
    """Step 1 — pypdf extraction, same as notebook."""
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
    """Step 2 — RecursiveCharacterTextSplitter, same as notebook (bug fixed: all docs)."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size    = 500,
        chunk_overlap = 100,
    )
    all_chunks = []
    for doc in all_documents:
        for chunk_text in text_splitter.split_text(doc["text"]):
            all_chunks.append({"source": doc["source"], "text": chunk_text})
    return all_chunks


def create_chunk_embeddings(all_chunks, embedding_model):
    """Step 3 — sentence-transformers encode, same as notebook."""
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    return embeddings


def store_in_vectordb(all_chunks, chunk_embeddings, collection):
    """Step 4 — ChromaDB store, same as notebook."""
    # clear existing entries first so re-index works cleanly
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    collection.add(
        ids       = [str(i) for i in range(len(all_chunks))],
        documents = [chunk["text"]   for chunk in all_chunks],
        embeddings= chunk_embeddings.tolist(),
        metadatas = [{"source": chunk["source"]} for chunk in all_chunks],
    )
    return len(all_chunks)


def retrieve(question, embedding_model, collection, n_results=3):
    """Step 5 — query embedding + ChromaDB search, same as notebook."""
    query_embedding = embedding_model.encode(question, convert_to_numpy=True)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
    )
    return results


def generate_answer(question, results):
    """Step 6 — Groq LLaMA 3, same as notebook."""
    client  = Groq(api_key=GROQ_API_KEY)
    context = "\n\n".join(results["documents"][0])

    prompt = f"""
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


# ─────────────────────────────────────────────
#  TOP BAR
# ─────────────────────────────────────────────
n_files = len(st.session_state.indexed_files)
tag_txt = f"{n_files} doc{'s' if n_files!=1 else ''} indexed" if st.session_state.indexed else "no index"

st.markdown(f"""
<div class="topbar">
  <div class="tb-pulse"></div>
  <span class="tb-title">📚 Smart Study Buddy</span>
  <span class="tb-mono">ChromaDB · nomic-embed · LLaMA-3.3-70B</span>
  <span class="tb-tag">{tag_txt}</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TWO-COLUMN LAYOUT  (left = upload, right = chat)
# ─────────────────────────────────────────────
col_left, col_right = st.columns([0.28, 0.72], gap="small")


# ══════════════════════════════════════════════
#  LEFT  —  Upload & Index
# ══════════════════════════════════════════════
with col_left:
    st.markdown('<div class="section-label">📄 Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "upload",
        type             = ["pdf"],
        accept_multiple_files = True,
        label_visibility = "collapsed",
    )

    # show chips for each selected file
    if uploaded_files:
        for f in uploaded_files:
            st.markdown(
                f'<div class="chip"><span class="chip-dot chip-dot-blue"></span>{f.name}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # stats cards (shown after indexing)
    if st.session_state.indexed:
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-box">
            <div class="stat-val">{len(st.session_state.indexed_files)}</div>
            <div class="stat-lbl">PDFs</div>
          </div>
          <div class="stat-box">
            <div class="stat-val">{st.session_state.num_chunks}</div>
            <div class="stat-lbl">Chunks</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # process button
    btn_lbl = "⚡ Process & Index" if not st.session_state.indexed else "🔄 Re-index Docs"
    do_index = st.button(btn_lbl, disabled=not uploaded_files)

    # indexed files list
    if st.session_state.indexed:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">✅ Indexed</div>', unsafe_allow_html=True)
        for fn in st.session_state.indexed_files:
            st.markdown(
                f'<div class="chip"><span class="chip-dot chip-dot-green"></span>{fn}</div>',
                unsafe_allow_html=True,
            )

    # clear chat button at bottom
    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # run the pipeline when button clicked
    if do_index and uploaded_files:
        prog = st.progress(0, text="Reading PDFs…")

        docs   = extract_text_from_pdfs(uploaded_files);  prog.progress(20, text="Chunking…")
        chunks = chunk_documents(docs);                    prog.progress(40, text="Loading model…")
        model  = load_embedding_model();                   prog.progress(60, text="Embedding chunks…")
        embs   = create_chunk_embeddings(chunks, model);   prog.progress(80, text="Storing in ChromaDB…")
        col    = get_chroma_collection()
        n      = store_in_vectordb(chunks, embs, col);     prog.progress(100, text="Done!")

        st.session_state.collection    = col
        st.session_state.embed_model   = model
        st.session_state.indexed       = True
        st.session_state.num_chunks    = n
        st.session_state.indexed_files = [f.name for f in uploaded_files]
        st.session_state.chat_history  = []

        prog.empty()
        st.success(f"✅ {n} chunks indexed from {len(docs)} file(s)")
        st.rerun()


# ══════════════════════════════════════════════
#  RIGHT  —  Chat
# ══════════════════════════════════════════════
with col_right:

    n_exchanges = len(st.session_state.chat_history) // 2
    sub_txt     = f"{n_exchanges} exchange{'s' if n_exchanges!=1 else ''}" if n_exchanges else "ready"

    st.markdown(f"""
    <div class="chat-header">
      <span style="font-size:18px">💬</span>
      <span class="ch-title">Ask your documents</span>
      <span class="ch-sub">{sub_txt}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── chat messages ──
    chat_placeholder = st.container()
    with chat_placeholder:
        if not st.session_state.indexed:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">📂</div>
              <div class="empty-txt">
                Upload your PDFs on the left<br>
                and click <b>Process &amp; Index</b><br>
                to start asking questions.
              </div>
            </div>
            """, unsafe_allow_html=True)

        elif not st.session_state.chat_history:
            files_bold = ", ".join(f"<b>{f}</b>" for f in st.session_state.indexed_files)
            st.markdown(f"""
            <div class="empty-state">
              <div class="empty-icon">✨</div>
              <div class="empty-txt">
                Ready! Ask anything about<br>{files_bold}
              </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="msg-you-label">YOU</div>
                    <div class="bubble-you">{msg["content"]}</div>
                    """, unsafe_allow_html=True)
                else:
                    srcs     = "".join(f'<span class="src-tag">📄 {s}</span>' for s in msg.get("sources", []))
                    src_html = f'<div class="src-row">{srcs}</div>' if srcs else ""
                    st.markdown(f"""
                    <div class="msg-ai-label">ASSISTANT</div>
                    <div class="bubble-ai">{msg["content"]}{src_html}</div>
                    """, unsafe_allow_html=True)

    # ── input bar ──
    if st.session_state.indexed:
        files_str = "  ·  ".join(st.session_state.indexed_files)
        st.markdown(f"""
        <div class="ready-bar">
          <div class="ready-dot"></div>{files_str}
        </div>
        """, unsafe_allow_html=True)

    inp_col, btn_col = st.columns([6, 1])
    with inp_col:
        question = st.text_input(
            "q",
            placeholder      = "Ask anything about your documents…",
            label_visibility = "collapsed",
            disabled         = not st.session_state.indexed,
        )
    with btn_col:
        ask = st.button("Ask ➤", disabled=not st.session_state.indexed)


# ─────────────────────────────────────────────
#  HANDLE ASK
# ─────────────────────────────────────────────
if ask and question.strip():
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("Searching documents and generating answer…"):
        results        = retrieve(question, st.session_state.embed_model, st.session_state.collection)
        answer, sources = generate_answer(question, results)

    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })
    st.rerun()
