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
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── App shell ── */
.stApp { background: #000 !important; }
.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* ════════════════════════════════════
   REAL STREAMLIT SIDEBAR
════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: #0a0a0a !important;
  border-right: 1px solid #1c1c1c !important;
  width: 280px !important;
  min-width: 280px !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 24px 20px 20px !important;
}
[data-testid="collapsedControl"] {
  color: #555 !important;
  background: #111 !important;
}

/* sidebar logo */
.sb-logo {
  display: flex; align-items: center; gap: 11px;
  padding-bottom: 20px;
  border-bottom: 1px solid #1c1c1c;
  margin-bottom: 20px;
}
.sb-logo-icon {
  width: 36px; height: 36px; border-radius: 9px;
  background: linear-gradient(135deg,#19c37d,#0d6644);
  display: flex; align-items: center; justify-content: center;
  font-size: 18px; flex-shrink: 0;
  box-shadow: 0 2px 12px #19c37d33;
}
.sb-logo-name { font-size: 15px; font-weight: 600; color: #ececec; }
.sb-logo-sub  { font-size: 11px; color: #444; margin-top: 1px; }

/* section label */
.sb-label {
  font-size: 10px; font-weight: 700; letter-spacing: .12em;
  text-transform: uppercase; color: #3a3a3a;
  margin-bottom: 10px; padding: 0 2px;
}

/* file upload zone */
[data-testid="stFileUploader"] {
  background: #111 !important;
  border: 1.5px dashed #222 !important;
  border-radius: 10px !important;
  transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: #19c37d !important; }
[data-testid="stFileUploader"] section { padding: 12px !important; }
[data-testid="stFileUploader"] label   { color: #555 !important; font-size: 13px !important; }

/* file item rows */
.frow {
  display: flex; align-items: center; gap: 9px;
  padding: 7px 8px; border-radius: 8px;
  margin-bottom: 3px; background: transparent;
  transition: background .15s;
}
.frow:hover { background: #141414; }
.frow-badge {
  width: 30px; height: 30px; border-radius: 6px;
  background: #1a1a1a; border: 1px solid #252525;
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 700;
  color: #19c37d; flex-shrink: 0;
  font-family: 'JetBrains Mono', monospace;
}
.frow-info  { flex: 1; min-width: 0; }
.frow-name  { font-size: 12px; font-weight: 500; color: #bbb;
              white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.frow-size  { font-size: 10px; color: #3a3a3a; margin-top: 1px; }
.frow-ok    { color: #19c37d; font-size: 13px; flex-shrink: 0; }

/* stat cards */
.stat-pair { display: flex; gap: 8px; margin: 14px 0; }
.stat-card {
  flex: 1; background: #0d0d0d; border: 1px solid #1c1c1c;
  border-radius: 10px; padding: 12px 8px; text-align: center;
}
.sc-val { font-size: 20px; font-weight: 700; color: #19c37d; line-height: 1; }
.sc-lbl { font-size: 10px; color: #3a3a3a; margin-top: 4px;
          text-transform: uppercase; letter-spacing: .05em; }

/* ── SIDEBAR BUTTONS ── */
[data-testid="stSidebar"] .stButton > button {
  background: #19c37d !important;
  color: #000 !important; border: none !important;
  border-radius: 9px !important; font-weight: 600 !important;
  font-size: 13px !important; padding: 10px 0 !important;
  width: 100% !important;
  font-family: 'Inter', sans-serif !important;
  transition: background .15s, transform .1s !important;
}
[data-testid="stSidebar"] .stButton > button:hover  { background: #14a86a !important; }
[data-testid="stSidebar"] .stButton > button:active { transform: scale(.98) !important; }
[data-testid="stSidebar"] .stButton > button:disabled {
  background: #0d2e1e !important; color: #174d31 !important; cursor: not-allowed !important;
}

.clr-wrap [data-testid="stSidebar"] .stButton > button,
.clr-wrap .stButton > button {
  background: transparent !important;
  border: 1px solid #222 !important;
  color: #555 !important;
  font-weight: 400 !important;
}
.clr-wrap .stButton > button:hover { background: #141414 !important; color: #aaa !important; }

[data-testid="stSidebar"] .stProgress > div > div > div > div {
  background: linear-gradient(90deg,#19c37d,#7effc4) !important;
  border-radius: 10px !important;
}

.sb-divider { height: 1px; background: #1c1c1c; margin: 14px 0; }

.sb-stack {
  padding: 6px 2px; font-size: 10px; color: #2e2e2e; line-height: 2;
}
.sb-stack b { color: #3a3a3a; font-weight: 600; }

/* ════════════════════════════════════
   MAIN CONTENT AREA
════════════════════════════════════ */
.main-wrap {
  display: flex; flex-direction: column;
  height: 100vh; background: #000;
  overflow: hidden;
}

/* top bar */
.topbar {
  display: flex; align-items: center; gap: 12px;
  padding: 13px 32px;
  border-bottom: 1px solid #1a1a1a;
  background: #000; flex-shrink: 0;
}
.tb-title { font-size: 14px; font-weight: 500; color: #666; }
.tb-model {
  display: flex; align-items: center; gap: 6px;
  background: #0d0d0d; border: 1px solid #1c1c1c;
  border-radius: 6px; padding: 4px 12px;
  font-size: 11px; color: #444;
  font-family: 'JetBrains Mono', monospace;
}
.tb-mdot {
  width: 5px; height: 5px; border-radius: 50%;
  background: #19c37d; box-shadow: 0 0 5px #19c37d;
  animation: glow 2s ease-in-out infinite;
}
@keyframes glow { 0%,100%{opacity:1} 50%{opacity:.25} }
.tb-count { margin-left: auto; font-size: 11px; color: #2a2a2a;
            font-family: 'JetBrains Mono', monospace; }

/* messages scroll zone */
.msg-scroll {
  flex: 1; overflow-y: auto; overflow-x: hidden;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1c1c1c; border-radius: 10px; }

/* ── Welcome ── */
.welcome {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  min-height: calc(100vh - 130px);
  text-align: center; padding: 20px;
}
.welcome-h {
  font-size: 30px; font-weight: 500; color: #ececec;
  letter-spacing: -.02em; margin-bottom: 30px;
}
.welcome-h em { color: #19c37d; font-style: normal; }

.pills {
  display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;
  max-width: 640px;
}
.pill {
  display: inline-flex; align-items: center; gap: 8px;
  background: #0d0d0d; border: 1px solid #252525;
  border-radius: 100px; padding: 10px 20px;
  font-size: 13px; color: #888; transition: all .18s;
  cursor: default; white-space: nowrap;
}
.pill:hover { background: #141414; border-color: #19c37d33; color: #ccc; }

/* ── Ready state ── */
.ready {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  min-height: calc(100vh - 130px);
  text-align: center; padding: 20px;
}
.ready-h   { font-size: 22px; font-weight: 500; color: #ececec; margin-bottom: 8px; }
.ready-sub { font-size: 14px; color: #444; margin-bottom: 22px; }
.doc-tags  { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
.doc-tag {
  display: inline-flex; align-items: center; gap: 6px;
  background: #091a12; border: 1px solid #112e1e;
  border-radius: 100px; padding: 5px 16px;
  font-size: 12px; color: #19c37d;
  font-family: 'JetBrains Mono', monospace;
}

/* ── Chat messages ── */
.msg-row { width: 100%; padding: 20px 0; border-bottom: 1px solid #0d0d0d; }
.msg-row.user { background: #000; }
.msg-row.ai   { background: #080808; }

.msg-inner {
  max-width: 720px; margin: 0 auto;
  padding: 0 32px;
  display: flex; gap: 16px; align-items: flex-start;
}

.av {
  width: 34px; height: 34px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700; flex-shrink: 0; margin-top: 1px;
}
.av-u { background: #1c1c1c; color: #aaa; }
.av-a { background: linear-gradient(135deg,#19c37d,#0d6644); color: #000; font-size: 11px; }

.mc { flex: 1; min-width: 0; }
.mc-name  { font-size: 12px; font-weight: 600; color: #3a3a3a; margin-bottom: 7px; }
.mc-name.ai { color: #19c37d; }
.mc-text {
  font-size: 15px; color: #ccc; line-height: 1.8; word-wrap: break-word;
}
.mc-text p           { margin-bottom: 10px; }
.mc-text p:last-child{ margin-bottom: 0; }
.mc-text strong      { color: #ececec; }
.mc-text code {
  background: #0d0d0d; border: 1px solid #1c1c1c;
  border-radius: 4px; padding: 2px 6px;
  font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #7effc4;
}
.mc-text ul, .mc-text ol { padding-left: 20px; margin: 8px 0; }
.mc-text li { margin-bottom: 4px; }

.src-row  { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; }
.src-chip {
  display: inline-flex; align-items: center; gap: 5px;
  background: #091a12; border: 1px solid #112e1e;
  border-radius: 100px; padding: 3px 12px;
  font-size: 11px; color: #19c37d;
  font-family: 'JetBrains Mono', monospace;
}

/* ── Input bar (ChatGPT style) ── */
.input-bar {
  padding: 14px 32px 20px;
  background: #000; border-top: 1px solid #1a1a1a;
  flex-shrink: 0;
}
.input-center { max-width: 720px; margin: 0 auto; }

.active-docs {
  display: flex; align-items: center; gap: 7px;
  font-size: 11px; color: #19c37d; margin-bottom: 9px;
  font-family: 'JetBrains Mono', monospace;
}
.ad-dot {
  width: 5px; height: 5px; border-radius: 50%;
  background: #19c37d; box-shadow: 0 0 5px #19c37d; flex-shrink: 0;
}

/* main area input styling */
.block-container .stTextInput > label { display: none !important; }
.block-container .stTextInput > div > div > input {
  background: #0d0d0d !important;
  color: #ececec !important;
  border: 1px solid #252525 !important;
  border-radius: 14px !important;
  font-size: 15px !important;
  padding: 15px 20px !important;
  font-family: 'Inter', sans-serif !important;
  transition: border-color .2s, box-shadow .2s !important;
  box-shadow: 0 2px 12px #00000088 !important;
}
.block-container .stTextInput > div > div > input:focus {
  border-color: #19c37d !important;
  box-shadow: 0 0 0 3px #19c37d12, 0 2px 12px #00000088 !important;
}
.block-container .stTextInput > div > div > input::placeholder { color: #2e2e2e !important; }

/* send button inside main area */
.block-container .stButton > button {
  background: #19c37d !important;
  color: #000 !important; border: none !important;
  border-radius: 10px !important; font-weight: 700 !important;
  font-size: 18px !important; padding: 11px 0 !important;
  width: 100% !important; transition: background .15s !important;
}
.block-container .stButton > button:hover    { background: #14a86a !important; }
.block-container .stButton > button:disabled { background: #0d2e1e !important; color: #163d25 !important; }

.input-hint {
  text-align: center; font-size: 11px; color: #1e1e1e; margin-top: 8px;
}

/* ── inline uploader: hide everything, show only clean button ── */
.block-container [data-testid="stFileUploader"] {
  background: transparent !important; border: none !important;
}
.block-container [data-testid="stFileUploader"] section {
  padding: 0 !important; min-height: unset !important; background: transparent !important; border: none !important;
}
.block-container [data-testid="stFileUploader"] section > div { padding: 0 !important; }
.block-container [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] { display:none !important; }
.block-container [data-testid="stFileUploader"] small { display:none !important; }
/* the Browse button becomes the + icon */
.block-container [data-testid="stFileUploader"] button {
  background: #1a1a1a !important;
  border: 1px solid #2a2a2a !important;
  border-radius: 10px !important;
  color: #888 !important;
  width: 48px !important; height: 48px !important;
  min-height: unset !important; padding: 0 !important;
  font-size: 20px !important;
  transition: background .15s, border-color .15s, color .15s !important;
}
.block-container [data-testid="stFileUploader"] button:hover {
  background: #222 !important; border-color: #19c37d !important; color: #19c37d !important;
}
.block-container [data-testid="stFileUploader"] label { display:none !important; }

/* streamlit success */
.stSuccess { background: #061410 !important; border-left-color: #19c37d !important;
             color: #19c37d !important; border-radius: 8px !important; font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE ─────────────────────────────────────────────────────
for k, v in {
    "chat_history": [], "indexed": False,
    "indexed_files": [], "num_chunks": 0,
    "collection": None, "embed_model": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── CACHED RESOURCES ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Starting ChromaDB…")
def get_chroma_collection():
    return chromadb.EphemeralClient().get_or_create_collection("pdf_docs")


# ── PIPELINE ──────────────────────────────────────────────────────────
def extract_text_from_pdfs(files):
    docs = []
    for f in files:
        reader = PdfReader(f)
        text = "".join(p.extract_text() or "" for p in reader.pages)
        docs.append({"source": f.name, "text": text})
    return docs

def chunk_documents(docs):
    sp = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return [{"source": d["source"], "text": t}
            for d in docs for t in sp.split_text(d["text"])]

def embed(chunks, model):
    return model.encode([c["text"] for c in chunks], convert_to_numpy=True)

def store(chunks, embeddings, col):
    ex = col.get()
    if ex["ids"]: col.delete(ids=ex["ids"])
    col.add(ids=[str(i) for i in range(len(chunks))],
            documents=[c["text"] for c in chunks],
            embeddings=embeddings.tolist(),
            metadatas=[{"source": c["source"]} for c in chunks])
    return len(chunks)

def retrieve(q, model, col, n=3):
    return col.query(query_embeddings=[model.encode(q, convert_to_numpy=True).tolist()], n_results=n)

def answer(q, res):
    ctx = "\n\n".join(res["documents"][0])
    r   = Groq(api_key=GROQ_API_KEY).chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role":"user","content":
            f"Answer using ONLY this context. Be clear and structured.\n\nContext:\n{ctx}\n\nQuestion: {q}"}])
    return r.choices[0].message.content, list(dict.fromkeys(m["source"] for m in res["metadatas"][0]))

def fsize(b): return f"{b/1024:.0f} KB" if b < 1048576 else f"{b/1048576:.1f} MB"


# ════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════
with st.sidebar:

    st.markdown("""
    <div class="sb-logo">
      <div class="sb-logo-icon">📚</div>
      <div>
        <div class="sb-logo-name">Study Buddy AI</div>
        <div class="sb-logo-sub">RAG Document Chat</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Documents</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("pdfs", type=["pdf"],
                                accept_multiple_files=True,
                                label_visibility="collapsed")

    if uploaded:
        for f in uploaded:
            ok = "✓" if (st.session_state.indexed and f.name in st.session_state.indexed_files) else ""
            st.markdown(f"""
            <div class="frow">
              <div class="frow-badge">PDF</div>
              <div class="frow-info">
                <div class="frow-name" title="{f.name}">{f.name}</div>
                <div class="frow-size">{fsize(f.size)}</div>
              </div>
              <span class="frow-ok">{ok}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.indexed:
        st.markdown(f"""
        <div class="stat-pair">
          <div class="stat-card"><div class="sc-val">{len(st.session_state.indexed_files)}</div><div class="sc-lbl">PDFs</div></div>
          <div class="stat-card"><div class="sc-val">{st.session_state.num_chunks}</div><div class="sc-lbl">Chunks</div></div>
        </div>""", unsafe_allow_html=True)

    lbl      = "⚡  Process & Index" if not st.session_state.indexed else "🔄  Re-index"
    do_index = st.button(lbl, disabled=not uploaded)

    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="clr-wrap">', unsafe_allow_html=True)
        if st.button("🗑  Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-stack">
      <b>Embed</b><br>all-MiniLM-L6-v2<br>
      <b>DB</b><br>ChromaDB<br>
      <b>LLM</b><br>LLaMA-3.3-70B · Groq
    </div>""", unsafe_allow_html=True)

    # Pipeline
    if do_index and uploaded:
        p = st.progress(0, text="Reading PDFs…")
        docs   = extract_text_from_pdfs(uploaded);  p.progress(20, text="Chunking…")
        chunks = chunk_documents(docs);              p.progress(40, text="Loading model…")
        m      = load_embedding_model();             p.progress(60, text="Embedding…")
        embs   = embed(chunks, m);                   p.progress(80, text="Storing…")
        col_db = get_chroma_collection()
        n      = store(chunks, embs, col_db);        p.progress(100, text="Done!")

        st.session_state.update({
            "collection": col_db, "embed_model": m,
            "indexed": True, "num_chunks": n,
            "indexed_files": [f.name for f in uploaded],
            "chat_history": [],
        })
        p.empty()
        st.success(f"✅ {n} chunks indexed")
        st.rerun()


# ════════════════════════════════════════
#  MAIN CONTENT
# ════════════════════════════════════════
n_ex = len(st.session_state.chat_history) // 2

# Top bar
st.markdown(f"""
<div class="topbar">
  <span class="tb-title">Chat</span>
  <div class="tb-model">
    <div class="tb-mdot"></div>
    llama-3.3-70b-versatile
  </div>
  <span class="tb-count">{n_ex} message{'s' if n_ex!=1 else ''}</span>
</div>
""", unsafe_allow_html=True)

# Messages area
if not st.session_state.indexed:
    st.markdown("""
    <div class="welcome">
      <div class="welcome-h">Where should we <em>begin?</em></div>
      <div class="pills">
        <div class="pill">📄&nbsp; Upload a PDF</div>
        <div class="pill">⚡&nbsp; Process &amp; Index</div>
        <div class="pill">💬&nbsp; Ask a question</div>
        <div class="pill">📌&nbsp; Get cited answers</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

elif not st.session_state.chat_history:
    tags = "".join(f'<div class="doc-tag">📄 {f}</div>' for f in st.session_state.indexed_files)
    st.markdown(f"""
    <div class="ready">
      <div class="ready-h">Ready to answer</div>
      <div class="ready-sub">Ask anything about your documents below</div>
      <div class="doc-tags">{tags}</div>
    </div>
    """, unsafe_allow_html=True)

else:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-row user">
              <div class="msg-inner">
                <div class="av av-u">J</div>
                <div class="mc">
                  <div class="mc-name">You</div>
                  <div class="mc-text">{msg["content"]}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            body = msg["content"].replace("\n\n","</p><p>").replace("\n","<br>")
            srcs = "".join(f'<span class="src-chip">📄 {s}</span>' for s in msg.get("sources",[]))
            sh   = f'<div class="src-row">{srcs}</div>' if srcs else ""
            st.markdown(f"""
            <div class="msg-row ai">
              <div class="msg-inner">
                <div class="av av-a">AI</div>
                <div class="mc">
                  <div class="mc-name ai">Study Buddy AI</div>
                  <div class="mc-text"><p>{body}</p></div>
                  {sh}
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

# ── INPUT BAR ─────────────────────────────────────────────────────────
st.markdown('<div class="input-bar"><div class="input-center">', unsafe_allow_html=True)

# Active docs strip
if st.session_state.indexed:
    docs_str = "  ·  ".join(st.session_state.indexed_files)
    st.markdown(f"""
    <div class="active-docs">
      <span class="ad-dot"></span>{docs_str}
    </div>""", unsafe_allow_html=True)

# Three columns: upload | text input | send
uc, qc, bc = st.columns([1.2, 11, 1.2])

with uc:
    inline_upload = st.file_uploader(
        "add",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="inline_uploader",
    )

with qc:
    question = st.text_input(
        "q",
        placeholder="Upload a PDF and ask anything…",
        label_visibility="collapsed",
    )

with bc:
    ask = st.button("➤", disabled=(not question.strip()))

st.markdown('<div class="input-hint">Study Buddy AI · answers sourced from your documents only</div>', unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)

# ── Auto-process when files uploaded via inline uploader ──────────────
if inline_upload:
    prog = st.progress(0, text="Reading PDFs…")
    docs   = extract_text_from_pdfs(inline_upload);  prog.progress(20, text="Chunking…")
    chunks = chunk_documents(docs);                   prog.progress(40, text="Loading model…")
    m      = load_embedding_model();                  prog.progress(60, text="Embedding…")
    embs   = embed(chunks, m);                        prog.progress(80, text="Storing…")
    col_db = get_chroma_collection()
    n      = store(chunks, embs, col_db);             prog.progress(100, text="Done!")
    st.session_state.update({
        "collection": col_db, "embed_model": m,
        "indexed": True, "num_chunks": n,
        "indexed_files": [f.name for f in inline_upload],
        "chat_history": [],
    })
    prog.empty()
    st.rerun()

# ── Handle ask ────────────────────────────────────────────────────────
if ask and question.strip():
    if not st.session_state.indexed:
        st.warning("Upload and process a PDF first using the 📎 button on the left of the input.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner(""):
            res       = retrieve(question, st.session_state.embed_model, st.session_state.collection)
            ans, srcs = answer(question, res)
        st.session_state.chat_history.append({"role": "assistant", "content": ans, "sources": srcs})
        st.rerun()
