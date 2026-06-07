import streamlit as st
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq

# ── API KEY ───────────────────────────────────────────────────────────
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("⚠️ Add GROQ_API_KEY in App Settings → Secrets")
    st.stop()

GROQ_MODEL = "llama-3.3-70b-versatile"

st.set_page_config(
    page_title="Study Buddy AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── pure black shell ── */
.stApp { background: #000 !important; color: #ececec; }
.block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── full viewport grid: sidebar | main ── */
.root-grid {
  display: grid;
  grid-template-columns: 240px 1fr;
  height: 100vh;
  overflow: hidden;
  background: #000;
}

/* ════════════════════════
   LEFT SIDEBAR
════════════════════════ */
.sidebar {
  background: #0a0a0a;
  border-right: 1px solid #1c1c1c;
  display: flex;
  flex-direction: column;
  padding: 20px 14px;
  overflow-y: auto;
  gap: 6px;
}

.logo-row {
  display: flex; align-items: center; gap: 10px;
  padding: 4px 6px 16px;
  border-bottom: 1px solid #1c1c1c;
  margin-bottom: 6px;
}
.logo-box {
  width: 30px; height: 30px; border-radius: 7px;
  background: linear-gradient(135deg,#19c37d,#0f7a4e);
  display: flex; align-items: center; justify-content: center;
  font-size: 15px; flex-shrink: 0;
}
.logo-text { font-size: 14px; font-weight: 600; color: #ececec; }
.logo-sub  { font-size: 11px; color: #555; }

.sidebar-label {
  font-size: 10px; font-weight: 600; letter-spacing: .1em;
  text-transform: uppercase; color: #444;
  padding: 8px 6px 4px;
}

/* upload */
[data-testid="stFileUploader"] {
  background: #111 !important;
  border: 1px dashed #2a2a2a !important;
  border-radius: 10px !important;
  transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: #19c37d !important; }

/* file rows */
.frow {
  display: flex; align-items: center; gap: 8px;
  padding: 6px 8px; border-radius: 7px;
  transition: background .15s; cursor: default;
}
.frow:hover { background: #161616; }
.frow-icon {
  width: 26px; height: 26px; border-radius: 5px;
  background: #1c1c1c; display: flex; align-items: center;
  justify-content: center; font-size: 11px; font-weight: 600;
  color: #19c37d; flex-shrink: 0; font-family: 'JetBrains Mono', monospace;
}
.frow-name { font-size: 12px; color: #aaa; flex: 1; min-width: 0;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.frow-ok   { color: #19c37d; font-size: 12px; flex-shrink: 0; }

/* stat pills */
.stats-row { display: flex; gap: 6px; padding: 4px 0; }
.spill {
  flex: 1; background: #111; border: 1px solid #1c1c1c;
  border-radius: 8px; padding: 9px 6px; text-align: center;
}
.spill-v { font-size: 17px; font-weight: 700; color: #19c37d; line-height: 1; }
.spill-l { font-size: 9px; color: #444; margin-top: 3px;
           text-transform: uppercase; letter-spacing: .05em; }

/* process button */
.stButton > button {
  background: #19c37d !important;
  color: #000 !important; border: none !important;
  border-radius: 8px !important; font-weight: 600 !important;
  font-size: 13px !important; padding: 9px 0 !important;
  width: 100% !important; font-family: 'Inter', sans-serif !important;
  transition: background .15s, transform .1s !important;
  letter-spacing: .01em !important;
}
.stButton > button:hover  { background: #15a86a !important; }
.stButton > button:active { transform: scale(.98) !important; }
.stButton > button:disabled {
  background: #0d3324 !important;
  color: #0d5c3a !important; cursor: not-allowed !important;
}

.clr-btn > button {
  background: transparent !important;
  color: #555 !important;
  border: 1px solid #222 !important;
  font-weight: 400 !important;
}
.clr-btn > button:hover { background: #111 !important; color: #aaa !important; }

.divider { height: 1px; background: #1c1c1c; margin: 8px 0; }

.stack-info {
  font-size: 10px; color: #333; line-height: 1.8;
  padding: 4px 6px; margin-top: auto;
}
.stack-info span { color: #444; }

/* ════════════════════════
   MAIN PANEL
════════════════════════ */
.main-panel {
  display: flex; flex-direction: column;
  overflow: hidden; background: #000;
}

/* top bar */
.topbar {
  display: flex; align-items: center; gap: 10px;
  padding: 12px 28px;
  border-bottom: 1px solid #1c1c1c;
  background: #000; flex-shrink: 0;
}
.tb-title { font-size: 14px; font-weight: 500; color: #888; }
.tb-badge {
  display: flex; align-items: center; gap: 5px;
  background: #111; border: 1px solid #222;
  border-radius: 6px; padding: 3px 10px;
  font-size: 11px; color: #555;
  font-family: 'JetBrains Mono', monospace;
}
.tb-dot { width:5px;height:5px;border-radius:50%;background:#19c37d;
          box-shadow:0 0 5px #19c37d; animation:glow 2s ease-in-out infinite; }
@keyframes glow{0%,100%{opacity:1}50%{opacity:.3}}
.tb-right { margin-left: auto; font-size: 11px; color: #333;
            font-family: 'JetBrains Mono', monospace; }

/* scroll area */
.scroll-area {
  flex: 1; overflow-y: auto; overflow-x: hidden;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #222; border-radius: 10px; }

/* ── WELCOME (centered, minimal like image 2) ── */
.welcome {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  min-height: calc(100vh - 160px);
  padding: 0 20px; text-align: center;
}
.welcome-title {
  font-size: 28px; font-weight: 500; color: #ececec;
  margin-bottom: 32px; letter-spacing: -.02em;
}
.welcome-title span { color: #19c37d; }

/* suggestion pills — exactly like image 2 */
.pill-row {
  display: flex; flex-wrap: wrap; gap: 10px;
  justify-content: center; max-width: 600px;
}
.pill {
  display: inline-flex; align-items: center; gap: 7px;
  background: #111; border: 1px solid #2a2a2a;
  border-radius: 100px; padding: 8px 18px;
  font-size: 13px; color: #aaa;
  cursor: pointer; transition: all .2s;
  white-space: nowrap;
}
.pill:hover { background: #1a1a1a; border-color: #19c37d; color: #ececec; }
.pill-icon { font-size: 14px; }

/* ready state */
.ready-state {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  min-height: calc(100vh - 160px);
  padding: 20px; text-align: center;
}
.ready-title { font-size: 22px; font-weight: 500; color: #ececec; margin-bottom: 12px; }
.ready-sub   { font-size: 14px; color: #555; margin-bottom: 24px; }
.doc-tags { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
.doc-tag {
  display: inline-flex; align-items: center; gap: 6px;
  background: #0d1f18; border: 1px solid #0d3324;
  border-radius: 100px; padding: 5px 14px;
  font-size: 12px; color: #19c37d;
  font-family: 'JetBrains Mono', monospace;
}

/* ── MESSAGES ── */
.msg-row {
  padding: 22px 0;
  border-bottom: 1px solid #0f0f0f;
}
.msg-row.user { background: #000; }
.msg-row.ai   { background: #0a0a0a; }

.msg-wrap {
  max-width: 680px; margin: 0 auto;
  padding: 0 28px;
  display: flex; gap: 14px; align-items: flex-start;
}

.av {
  width: 32px; height: 32px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 600; flex-shrink: 0; margin-top: 2px;
}
.av-user { background: #1c1c1c; color: #ececec; font-size: 11px; }
.av-ai   { background: linear-gradient(135deg,#19c37d,#0f7a4e); color: #000; }

.msg-body { flex: 1; min-width: 0; }
.msg-name { font-size: 12px; font-weight: 600; color: #555; margin-bottom: 6px; }
.msg-name.ai-name { color: #19c37d; }
.msg-text {
  font-size: 14px; color: #d1d5db; line-height: 1.8;
  word-wrap: break-word;
}
.msg-text p  { margin-bottom: 10px; }
.msg-text p:last-child { margin-bottom: 0; }
.msg-text strong { color: #ececec; font-weight: 600; }
.msg-text code {
  background: #111; border: 1px solid #2a2a2a;
  border-radius: 4px; padding: 1px 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px; color: #7effc4;
}
.msg-text ul, .msg-text ol { padding-left: 18px; margin: 8px 0; }
.msg-text li { margin-bottom: 4px; }

.src-wrap { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; }
.src-chip {
  display: inline-flex; align-items: center; gap: 5px;
  background: #0d1f18; border: 1px solid #0d3324;
  border-radius: 100px; padding: 3px 12px;
  font-size: 11px; color: #19c37d;
  font-family: 'JetBrains Mono', monospace;
}

/* ── INPUT BAR ── */
.input-zone {
  padding: 14px 28px 18px;
  background: #000;
  border-top: 1px solid #1c1c1c;
  flex-shrink: 0;
}
.input-inner { max-width: 680px; margin: 0 auto; }

.stTextInput > label { display: none !important; }
.stTextInput > div > div > input {
  background: #111 !important;
  color: #ececec !important;
  border: 1px solid #2a2a2a !important;
  border-radius: 12px !important;
  font-size: 14px !important;
  padding: 14px 18px !important;
  font-family: 'Inter', sans-serif !important;
  transition: border-color .2s, box-shadow .2s !important;
  box-shadow: 0 0 0 0 transparent !important;
}
.stTextInput > div > div > input:focus {
  border-color: #19c37d !important;
  box-shadow: 0 0 0 3px #19c37d14 !important;
}
.stTextInput > div > div > input::placeholder { color: #333 !important; }

.input-hint {
  text-align: center; margin-top: 8px;
  font-size: 11px; color: #2a2a2a;
}

/* progress */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg,#19c37d,#7effc4) !important;
  border-radius: 10px !important;
}

/* alerts */
.stSuccess { background: #081410 !important; border-left-color: #19c37d !important;
             color: #19c37d !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────
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


# ── CACHED RESOURCES ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Starting ChromaDB…")
def get_chroma_collection():
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(name="pdf_documents")


# ── PIPELINE ──────────────────────────────────────────────────────────
def extract_text_from_pdfs(files):
    docs = []
    for f in files:
        reader = PdfReader(f)
        text   = ""
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + "\n"
        docs.append({"source": f.name, "text": text})
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = []
    for doc in docs:
        for piece in splitter.split_text(doc["text"]):
            chunks.append({"source": doc["source"], "text": piece})
    return chunks

def create_chunk_embeddings(chunks, model):
    return model.encode([c["text"] for c in chunks], convert_to_numpy=True)

def store_in_vectordb(chunks, embeddings, collection):
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    collection.add(
        ids        = [str(i) for i in range(len(chunks))],
        documents  = [c["text"]   for c in chunks],
        embeddings = embeddings.tolist(),
        metadatas  = [{"source": c["source"]} for c in chunks],
    )
    return len(chunks)

def retrieve(question, model, collection, n=3):
    q_emb = model.encode(question, convert_to_numpy=True)
    return collection.query(query_embeddings=[q_emb.tolist()], n_results=n)

def generate_answer(question, results):
    client  = Groq(api_key=GROQ_API_KEY)
    context = "\n\n".join(results["documents"][0])
    prompt  = f"""You are a helpful study assistant. Answer the question using ONLY the provided context.
Be clear, concise and well structured. If the answer isn't in the context, say so.

Context:
{context}

Question: {question}"""
    resp    = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    answer  = resp.choices[0].message.content
    sources = list(dict.fromkeys(m["source"] for m in results["metadatas"][0]))
    return answer, sources

def fmt_size(b):
    return f"{b/1024:.0f} KB" if b < 1048576 else f"{b/1048576:.1f} MB"


# ══════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════
left, right = st.columns([0.19, 0.81], gap="small")


# ════════════════════════
#  SIDEBAR
# ════════════════════════
with left:
    # Logo
    st.markdown("""
    <div class="logo-row">
      <div class="logo-box">📚</div>
      <div>
        <div class="logo-text">Study Buddy</div>
        <div class="logo-sub">RAG Chat</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "pdfs", type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # File list
    if uploaded_files:
        for f in uploaded_files:
            ok = "✓" if (st.session_state.indexed and f.name in st.session_state.indexed_files) else ""
            st.markdown(f"""
            <div class="frow">
              <div class="frow-icon">PDF</div>
              <div class="frow-name" title="{f.name}">{f.name}</div>
              <span class="frow-ok">{ok}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Stats
    if st.session_state.indexed:
        st.markdown(f"""
        <div class="stats-row">
          <div class="spill"><div class="spill-v">{len(st.session_state.indexed_files)}</div><div class="spill-l">PDFs</div></div>
          <div class="spill"><div class="spill-v">{st.session_state.num_chunks}</div><div class="spill-l">Chunks</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Index button
    lbl      = "⚡  Process & Index" if not st.session_state.indexed else "🔄  Re-index"
    do_index = st.button(lbl, disabled=not uploaded_files)

    # Clear chat
    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="clr-btn">', unsafe_allow_html=True)
        if st.button("🗑  Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Stack info
    st.markdown("""
    <div class="stack-info">
      <span>Model</span><br>all-MiniLM-L6-v2<br>
      <span>DB</span><br>ChromaDB<br>
      <span>LLM</span><br>LLaMA-3.3-70B
    </div>
    """, unsafe_allow_html=True)

    # Run pipeline
    if do_index and uploaded_files:
        prog = st.progress(0, text="Reading PDFs…")
        docs   = extract_text_from_pdfs(uploaded_files);   prog.progress(20, text="Chunking…")
        chunks = chunk_documents(docs);                     prog.progress(40, text="Loading model…")
        model  = load_embedding_model();                    prog.progress(60, text="Embedding…")
        embs   = create_chunk_embeddings(chunks, model);    prog.progress(80, text="Storing…")
        col_db = get_chroma_collection()
        n      = store_in_vectordb(chunks, embs, col_db);  prog.progress(100, text="Done!")

        st.session_state.collection    = col_db
        st.session_state.embed_model   = model
        st.session_state.indexed       = True
        st.session_state.num_chunks    = n
        st.session_state.indexed_files = [f.name for f in uploaded_files]
        st.session_state.chat_history  = []
        prog.empty()
        st.success(f"✅ {n} chunks ready")
        st.rerun()


# ════════════════════════
#  MAIN PANEL
# ════════════════════════
with right:

    # Top bar
    n_ex = len(st.session_state.chat_history) // 2
    st.markdown(f"""
    <div class="topbar">
      <span class="tb-title">Chat</span>
      <div class="tb-badge">
        <div class="tb-dot"></div>
        llama-3.3-70b-versatile
      </div>
      <span class="tb-right">{n_ex} message{'s' if n_ex!=1 else ''}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Messages / welcome ──
    with st.container():

        if not st.session_state.indexed:
            # WELCOME — minimal centered exactly like image 2
            st.markdown("""
            <div class="welcome">
              <div class="welcome-title">Where should we <span>begin?</span></div>
              <div class="pill-row">
                <div class="pill"><span class="pill-icon">📄</span> Upload a PDF</div>
                <div class="pill"><span class="pill-icon">⚡</span> Process &amp; Index</div>
                <div class="pill"><span class="pill-icon">💬</span> Ask a question</div>
                <div class="pill"><span class="pill-icon">📌</span> Get cited answers</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        elif not st.session_state.chat_history:
            # READY — show indexed docs as pills
            tags = "".join(
                f'<div class="doc-tag">📄 {f}</div>'
                for f in st.session_state.indexed_files
            )
            st.markdown(f"""
            <div class="ready-state">
              <div class="ready-title">Ready to answer</div>
              <div class="ready-sub">Ask anything about your documents</div>
              <div class="doc-tags">{tags}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # CHAT MESSAGES
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="msg-row user">
                      <div class="msg-wrap">
                        <div class="av av-user">J</div>
                        <div class="msg-body">
                          <div class="msg-name">You</div>
                          <div class="msg-text">{msg["content"]}</div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    body = msg["content"].replace("\n\n","</p><p>").replace("\n","<br>")
                    srcs = "".join(f'<span class="src-chip">📄 {s}</span>' for s in msg.get("sources",[]))
                    src_html = f'<div class="src-wrap">{srcs}</div>' if srcs else ""
                    st.markdown(f"""
                    <div class="msg-row ai">
                      <div class="msg-wrap">
                        <div class="av av-ai">AI</div>
                        <div class="msg-body">
                          <div class="msg-name ai-name">Study Buddy AI</div>
                          <div class="msg-text"><p>{body}</p></div>
                          {src_html}
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Input bar ──
    st.markdown('<div class="input-inner">', unsafe_allow_html=True)

    if st.session_state.indexed:
        files_str = "  ·  ".join(st.session_state.indexed_files)
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;
             font-size:11px;color:#19c37d;font-family:'JetBrains Mono',monospace">
          <span style="width:5px;height:5px;border-radius:50%;background:#19c37d;
                       box-shadow:0 0 5px #19c37d;display:inline-block"></span>
          {files_str}
        </div>
        """, unsafe_allow_html=True)

    q_col, b_col = st.columns([11, 1])
    with q_col:
        question = st.text_input(
            "q",
            placeholder="Ask anything…" if st.session_state.indexed else "Upload and index PDFs first…",
            label_visibility="collapsed",
            disabled=not st.session_state.indexed,
        )
    with b_col:
        ask = st.button("➤", disabled=not st.session_state.indexed)

    st.markdown('<div class="input-hint">Study Buddy AI · answers based on your documents only</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── HANDLE ASK ────────────────────────────────────────────────────────
if ask and question.strip():
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.spinner(""):
        results         = retrieve(question, st.session_state.embed_model, st.session_state.collection)
        answer, sources = generate_answer(question, results)
    st.session_state.chat_history.append({
        "role": "assistant", "content": answer, "sources": sources
    })
    st.rerun()
