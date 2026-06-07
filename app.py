import streamlit as st
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq
import datetime

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

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp { background: #0f0f0f !important; color: #ececec; }
.block-container { padding: 24px 32px 80px 32px !important; max-width: 860px !important; margin: 0 auto !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { color: #555 !important; }

/* ════════════════════════════
   SIDEBAR
════════════════════════════ */
[data-testid="stSidebar"] {
  background: #161616 !important;
  border-right: 1px solid #252525 !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 20px 16px 24px !important;
}

/* Logo */
.sb-brand {
  display: flex; align-items: center; gap: 11px;
  padding: 4px 6px 20px;
  border-bottom: 1px solid #252525;
  margin-bottom: 8px;
}
.sb-brand-icon {
  width: 34px; height: 34px; border-radius: 8px;
  background: linear-gradient(135deg,#19c37d,#0d6644);
  display: flex; align-items: center; justify-content: center;
  font-size: 17px; flex-shrink: 0;
  box-shadow: 0 2px 10px #19c37d33;
}
.sb-brand-name { font-size: 14px; font-weight: 700; color: #ececec; letter-spacing:-.01em; }
.sb-brand-sub  { font-size: 10px; color: #444; margin-top: 1px; }

/* Section headers */
.sb-section {
  font-size: 10px; font-weight: 700; letter-spacing: .12em;
  text-transform: uppercase; color: #3a3a3a;
  padding: 16px 6px 8px;
}

/* Chat history items */
.hist-item {
  display: flex; align-items: center; gap: 9px;
  padding: 8px 10px; border-radius: 8px;
  margin-bottom: 2px; cursor: pointer;
  transition: background .15s;
  color: #999; font-size: 13px;
}
.hist-item:hover { background: #202020; color: #ececec; }
.hist-item.active { background: #1f1f1f; color: #ececec; }
.hist-icon { font-size: 13px; flex-shrink: 0; opacity: .6; }
.hist-label { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; }
.hist-time  { font-size: 10px; color: #333; flex-shrink: 0; }

.hist-date-group {
  font-size: 10px; font-weight: 600; color: #2e2e2e;
  letter-spacing: .06em; text-transform: uppercase;
  padding: 10px 10px 4px;
}

/* PDF library items */
.pdf-item {
  display: flex; align-items: center; gap: 9px;
  padding: 7px 10px; border-radius: 8px;
  margin-bottom: 2px; transition: background .15s;
}
.pdf-item:hover { background: #1f1f1f; }
.pdf-badge {
  width: 28px; height: 28px; border-radius: 6px;
  background: #1a1a1a; border: 1px solid #252525;
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 700; color: #19c37d;
  font-family: 'JetBrains Mono', monospace; flex-shrink: 0;
}
.pdf-info { flex: 1; min-width: 0; }
.pdf-name { font-size: 12px; font-weight: 500; color: #bbb;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pdf-meta { font-size: 10px; color: #3a3a3a; margin-top: 1px; }
.pdf-ok   { color: #19c37d; font-size: 12px; flex-shrink: 0; }

/* Stats grid */
.stats-grid {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 7px; margin: 10px 0;
}
.stat-box {
  background: #111; border: 1px solid #1f1f1f;
  border-radius: 9px; padding: 11px 10px; text-align: center;
}
.stat-val { font-size: 18px; font-weight: 700; color: #19c37d; line-height: 1; }
.stat-lbl { font-size: 9px; color: #333; margin-top: 4px;
            text-transform: uppercase; letter-spacing: .06em; }

.model-row {
  display: flex; flex-direction: column; gap: 5px; margin-top: 8px;
}
.model-pill {
  display: flex; align-items: center; justify-content: space-between;
  background: #111; border: 1px solid #1f1f1f;
  border-radius: 7px; padding: 7px 11px;
  font-size: 11px;
}
.mp-label { color: #3a3a3a; font-weight: 600; text-transform: uppercase;
            letter-spacing: .06em; font-size: 10px; }
.mp-value { color: #666; font-family: 'JetBrains Mono', monospace; font-size: 11px; }

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
  background: #1a1a1a !important;
  color: #ececec !important;
  border: 1px solid #2a2a2a !important;
  border-radius: 9px !important;
  font-weight: 500 !important;
  font-size: 13px !important;
  padding: 9px 14px !important;
  width: 100% !important;
  text-align: left !important;
  transition: background .15s, border-color .15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: #222 !important; border-color: #333 !important;
}

.sb-process-btn [data-testid="stSidebar"] .stButton > button,
.sb-process-btn .stButton > button {
  background: #19c37d !important;
  color: #000 !important; border: none !important;
  font-weight: 600 !important;
}
.sb-process-btn .stButton > button:hover { background: #14a86a !important; }
.sb-process-btn .stButton > button:disabled {
  background: #0d2e1e !important; color: #0d3d22 !important;
}

.sb-danger .stButton > button {
  background: transparent !important;
  color: #555 !important; border: 1px solid #1f1f1f !important;
  font-size: 12px !important;
}
.sb-danger .stButton > button:hover {
  background: #1a0e0e !important; border-color: #3a1a1a !important; color: #ff6b6b !important;
}

[data-testid="stSidebar"] .stProgress > div > div > div > div {
  background: linear-gradient(90deg,#19c37d,#7effc4) !important;
  border-radius: 10px !important;
}

[data-testid="stSidebar"] .stFileUploader {
  background: #111 !important;
  border: 1.5px dashed #252525 !important;
  border-radius: 10px !important;
}
[data-testid="stSidebar"] .stFileUploader:hover { border-color: #19c37d !important; }

/* ════════════════════════════
   MAIN AREA
════════════════════════════ */

/* Top header */
.main-header {
  display: flex; align-items: center; gap: 12px;
  padding: 0 0 20px 0;
  border-bottom: 1px solid #1f1f1f;
  margin-bottom: 24px;
}
.mh-title { font-size: 15px; font-weight: 600; color: #ececec; }
.mh-badge {
  display: flex; align-items: center; gap: 6px;
  background: #111; border: 1px solid #1f1f1f;
  border-radius: 6px; padding: 4px 12px;
  font-size: 11px; color: #444;
  font-family: 'JetBrains Mono', monospace;
}
.mh-dot {
  width: 5px; height: 5px; border-radius: 50%;
  background: #19c37d; box-shadow: 0 0 5px #19c37d;
  animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.2} }
.mh-right { margin-left: auto; font-size: 11px; color: #2e2e2e;
            font-family: 'JetBrains Mono', monospace; }

/* ── Welcome ── */
.welcome {
  display: flex; flex-direction: column;
  align-items: center;
  padding: 60px 20px 40px;
  text-align: center;
}
.wlc-icon { font-size: 44px; margin-bottom: 18px; }
.wlc-title { font-size: 26px; font-weight: 600; color: #ececec;
             letter-spacing: -.02em; margin-bottom: 10px; }
.wlc-title em { color: #19c37d; font-style: normal; }
.wlc-sub   { font-size: 15px; color: #444; margin-bottom: 36px; line-height: 1.7; }

.action-grid {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 12px; width: 100%; max-width: 500px;
}
.action-card {
  background: #161616; border: 1px solid #222;
  border-radius: 12px; padding: 18px 16px;
  text-align: left; cursor: default;
  transition: border-color .2s, background .2s;
}
.action-card:hover { background: #1a1a1a; border-color: #2e2e2e; }
.ac-icon  { font-size: 20px; margin-bottom: 10px; }
.ac-title { font-size: 13px; font-weight: 600; color: #ccc; margin-bottom: 4px; }
.ac-desc  { font-size: 12px; color: #3a3a3a; line-height: 1.5; }

/* ── Chat messages ── */
.msg-row {
  display: flex; gap: 14px; align-items: flex-start;
  padding: 18px 0; border-bottom: 1px solid #161616;
}
.msg-row:last-child { border-bottom: none; }

.av {
  width: 34px; height: 34px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700; flex-shrink: 0; margin-top: 2px;
}
.av-u { background: #222; color: #aaa; border: 1px solid #2e2e2e; }
.av-a { background: linear-gradient(135deg,#19c37d,#0d6644); color: #000; font-size: 11px; font-weight: 700; }

.mc { flex: 1; min-width: 0; }
.mc-name   { font-size: 12px; font-weight: 600; margin-bottom: 7px; }
.mc-name.you { color: #555; }
.mc-name.ai  { color: #19c37d; }

.mc-text {
  font-size: 14.5px; color: #c8c8c8; line-height: 1.8;
  word-wrap: break-word;
}
.mc-text p            { margin-bottom: 10px; }
.mc-text p:last-child { margin-bottom: 0; }
.mc-text strong       { color: #ececec; font-weight: 600; }
.mc-text code {
  background: #111; border: 1px solid #222;
  border-radius: 4px; padding: 2px 7px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12.5px; color: #7effc4;
}
.mc-text ul, .mc-text ol { padding-left: 20px; margin: 8px 0; }
.mc-text li { margin-bottom: 4px; color: #b0b0b0; }
.mc-text h3 { color: #ececec; font-size: 15px; margin: 12px 0 6px; }

/* Sources */
.sources-wrap { margin-top: 14px; }
.sources-title { font-size: 10px; font-weight: 700; letter-spacing: .1em;
                 text-transform: uppercase; color: #2e2e2e; margin-bottom: 8px; }
.src-chips { display: flex; flex-wrap: wrap; gap: 6px; }
.src-chip {
  display: inline-flex; align-items: center; gap: 6px;
  background: #0d1a11; border: 1px solid #112a18;
  border-radius: 8px; padding: 5px 12px;
  font-size: 11px; color: #19c37d;
  font-family: 'JetBrains Mono', monospace;
  transition: background .15s;
}
.src-chip:hover { background: #112016; }

/* ── INPUT BAR ── */
.input-wrap {
  position: fixed; bottom: 0; left: 0; right: 0;
  background: #0f0f0f;
  border-top: 1px solid #1f1f1f;
  padding: 14px 32px 18px;
  z-index: 999;
}
.input-inner {
  max-width: 860px; margin: 0 auto;
}

/* active docs bar */
.active-docs {
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 10px; flex-wrap: wrap;
}
.ad-label { font-size: 10px; color: #2e2e2e; font-weight: 600;
            text-transform: uppercase; letter-spacing: .08em; margin-right: 2px; }
.ad-tag {
  display: inline-flex; align-items: center; gap: 5px;
  background: #0d1a11; border: 1px solid #112a18;
  border-radius: 100px; padding: 2px 10px;
  font-size: 11px; color: #19c37d;
  font-family: 'JetBrains Mono', monospace;
}
.ad-dot {
  width: 4px; height: 4px; border-radius: 50%;
  background: #19c37d; flex-shrink: 0;
}

/* input row */
.block-container .stTextInput > label { display: none !important; }
.block-container .stTextInput > div > div > input {
  background: #161616 !important;
  color: #ececec !important;
  border: 1px solid #2a2a2a !important;
  border-radius: 14px !important;
  font-size: 14px !important;
  padding: 14px 20px !important;
  font-family: 'Inter', sans-serif !important;
  transition: border-color .2s, box-shadow .2s !important;
  box-shadow: 0 4px 20px #00000066 !important;
}
.block-container .stTextInput > div > div > input:focus {
  border-color: #19c37d !important;
  box-shadow: 0 0 0 3px #19c37d14, 0 4px 20px #00000066 !important;
}
.block-container .stTextInput > div > div > input::placeholder { color: #2e2e2e !important; }

/* send button */
.block-container .stButton > button {
  background: #19c37d !important;
  color: #000 !important; border: none !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
  font-size: 17px !important;
  padding: 13px 0 !important;
  width: 100% !important;
  transition: background .15s !important;
}
.block-container .stButton > button:hover { background: #14a86a !important; }
.block-container .stButton > button:disabled {
  background: #0d2e1e !important; color: #0d3d22 !important; cursor: not-allowed !important;
}

.input-hint {
  text-align: center; font-size: 11px; color: #1e1e1e; margin-top: 8px;
}

/* inline uploader — clip icon style */
.block-container [data-testid="stFileUploader"] {
  background: transparent !important; border: none !important;
}
.block-container [data-testid="stFileUploader"] section {
  padding: 0 !important; min-height: unset !important;
  background: transparent !important; border: none !important;
}
.block-container [data-testid="stFileUploader"] section > div { padding: 0 !important; }
.block-container [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] { display:none !important; }
.block-container [data-testid="stFileUploader"] small { display:none !important; }
.block-container [data-testid="stFileUploader"] button {
  background: #161616 !important;
  border: 1px solid #2a2a2a !important;
  border-radius: 10px !important;
  color: #555 !important;
  width: 48px !important; height: 48px !important;
  min-height: unset !important; padding: 0 !important;
  font-size: 20px !important;
  transition: background .15s, border-color .15s, color .15s !important;
}
.block-container [data-testid="stFileUploader"] button:hover {
  background: #1a1a1a !important; border-color: #19c37d !important; color: #19c37d !important;
}
.block-container [data-testid="stFileUploader"] label { display:none !important; }

/* misc */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #222; border-radius: 10px; }

.divider { height: 1px; background: #1f1f1f; margin: 10px 0; }

.stSuccess { background: #061410 !important; border-left-color: #19c37d !important;
             color: #19c37d !important; border-radius: 8px !important; font-size: 13px !important; }
.stWarning { border-radius: 8px !important; font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
def _init():
    defaults = {
        "chats":          [],          # list of {title, history, timestamp}
        "active_chat":    0,           # index into chats
        "pdf_library":    [],          # list of {name, size, chunks, indexed}
        "collection":     None,
        "embed_model":    None,
        "total_chunks":   0,
        "page":           "chat",      # chat | docs
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

def current_chat():
    if not st.session_state.chats:
        return None
    return st.session_state.chats[st.session_state.active_chat]

def current_history():
    c = current_chat()
    return c["history"] if c else []


# ─────────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Starting ChromaDB…")
def get_collection():
    return chromadb.EphemeralClient().get_or_create_collection("pdf_docs")

def extract_text(files):
    docs = []
    for f in files:
        reader = PdfReader(f)
        text = "".join(p.extract_text() or "" for p in reader.pages)
        docs.append({"source": f.name, "text": text, "pages": len(reader.pages)})
    return docs

def chunk_docs(docs):
    sp = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return [{"source": d["source"], "text": t} for d in docs for t in sp.split_text(d["text"])]

def store(chunks, col):
    ex = col.get()
    if ex["ids"]: col.delete(ids=ex["ids"])
    model = load_model()
    embs  = model.encode([c["text"] for c in chunks], convert_to_numpy=True)
    col.add(ids=[str(i) for i in range(len(chunks))],
            documents=[c["text"] for c in chunks],
            embeddings=embs.tolist(),
            metadatas=[{"source": c["source"]} for c in chunks])
    return model, len(chunks)

def retrieve(q, model, col, n=3):
    return col.query(query_embeddings=[model.encode(q, convert_to_numpy=True).tolist()], n_results=n)

def generate(q, res):
    ctx = "\n\n".join(res["documents"][0])
    r   = Groq(api_key=GROQ_API_KEY).chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role":"user","content":
            f"Answer clearly and concisely using ONLY this context. Use formatting where helpful.\n\nContext:\n{ctx}\n\nQuestion: {q}"}])
    return r.choices[0].message.content, list(dict.fromkeys(m["source"] for m in res["metadatas"][0]))

def fsize(b): return f"{b/1024:.0f} KB" if b < 1048576 else f"{b/1048576:.1f} MB"

def now_label():
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

def process_and_index(files):
    prog = st.progress(0, text="Reading PDFs…")
    docs   = extract_text(files);       prog.progress(25, text="Chunking…")
    chunks = chunk_docs(docs);          prog.progress(50, text="Embedding…")
    col    = get_collection()
    model, n = store(chunks, col);      prog.progress(100, text="Done!")
    prog.empty()

    st.session_state.collection   = col
    st.session_state.embed_model  = model
    st.session_state.total_chunks = n
    st.session_state.pdf_library  = [
        {"name": f.name, "size": f.size, "pages": d["pages"], "chunks": 0, "indexed": True}
        for f, d in zip(files, docs)
    ]
    # distribute chunks roughly
    per = n // len(files) if files else 0
    for p in st.session_state.pdf_library:
        p["chunks"] = per
    return n


# ════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════
with st.sidebar:

    # Brand
    st.markdown("""
    <div class="sb-brand">
      <div class="sb-brand-icon">📚</div>
      <div>
        <div class="sb-brand-name">Study Buddy AI</div>
        <div class="sb-brand-sub">Multi-PDF RAG Chat</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── New Chat ──
    if st.button("＋  New Chat"):
        st.session_state.chats.append({
            "title":     "New Chat",
            "history":   [],
            "timestamp": now_label(),
        })
        st.session_state.active_chat = len(st.session_state.chats) - 1
        st.rerun()

    # ── Chat History ──
    if st.session_state.chats:
        st.markdown('<div class="sb-section">Chat History</div>', unsafe_allow_html=True)
        for i, chat in enumerate(reversed(st.session_state.chats)):
            idx = len(st.session_state.chats) - 1 - i
            is_active = idx == st.session_state.active_chat
            cls = "hist-item active" if is_active else "hist-item"
            label = chat["title"][:28] + ("…" if len(chat["title"]) > 28 else "")
            if st.button(f"💬  {label}", key=f"chat_{idx}"):
                st.session_state.active_chat = idx
                st.rerun()

    # ── PDF Library ──
    st.markdown('<div class="sb-section">PDF Library</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "upload", type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        for f in uploaded:
            in_lib = any(p["name"] == f.name for p in st.session_state.pdf_library)
            ok = "✓" if in_lib else ""
            st.markdown(f"""
            <div class="pdf-item">
              <div class="pdf-badge">PDF</div>
              <div class="pdf-info">
                <div class="pdf-name" title="{f.name}">{f.name}</div>
                <div class="pdf-meta">{fsize(f.size)}</div>
              </div>
              <span class="pdf-ok">{ok}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sb-process-btn">', unsafe_allow_html=True)
        if st.button("⚡  Process & Index All"):
            n = process_and_index(uploaded)
            if not st.session_state.chats:
                st.session_state.chats.append({"title":"New Chat","history":[],"timestamp":now_label()})
                st.session_state.active_chat = 0
            st.success(f"✅ {n} chunks indexed from {len(uploaded)} PDF(s)")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    elif st.session_state.pdf_library:
        for p in st.session_state.pdf_library:
            st.markdown(f"""
            <div class="pdf-item">
              <div class="pdf-badge">PDF</div>
              <div class="pdf-info">
                <div class="pdf-name" title="{p['name']}">{p['name']}</div>
                <div class="pdf-meta">{p['pages']} pages · {p['chunks']} chunks</div>
              </div>
              <span class="pdf-ok">✓</span>
            </div>""", unsafe_allow_html=True)

    # ── Statistics ──
    st.markdown('<div class="sb-section">Statistics</div>', unsafe_allow_html=True)
    n_pdfs = len(st.session_state.pdf_library)
    n_chunks = st.session_state.total_chunks
    n_chats  = len(st.session_state.chats)
    st.markdown(f"""
    <div class="stats-grid">
      <div class="stat-box"><div class="stat-val">{n_pdfs}</div><div class="stat-lbl">PDFs</div></div>
      <div class="stat-box"><div class="stat-val">{n_chunks}</div><div class="stat-lbl">Chunks</div></div>
      <div class="stat-box"><div class="stat-val">{n_chats}</div><div class="stat-lbl">Chats</div></div>
      <div class="stat-box"><div class="stat-val">{"✓" if st.session_state.collection else "–"}</div><div class="stat-lbl">Indexed</div></div>
    </div>
    <div class="model-row">
      <div class="model-pill"><span class="mp-label">Embed</span><span class="mp-value">MiniLM-L6</span></div>
      <div class="model-pill"><span class="mp-label">DB</span><span class="mp-value">ChromaDB</span></div>
      <div class="model-pill"><span class="mp-label">LLM</span><span class="mp-value">LLaMA-3.3-70B</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Clear ──
    if st.session_state.chats:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sb-danger">', unsafe_allow_html=True)
        if st.button("🗑  Clear All Chats"):
            st.session_state.chats = []
            st.session_state.active_chat = 0
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════
#  MAIN AREA
# ════════════════════════════════════════
indexed    = bool(st.session_state.collection)
history    = current_history()
n_msgs     = len(history) // 2

# ── Header ──
st.markdown(f"""
<div class="main-header">
  <span class="mh-title">💬 Chat</span>
  <div class="mh-badge">
    <div class="mh-dot"></div>
    llama-3.3-70b-versatile
  </div>
  <span class="mh-right">{n_msgs} message{'s' if n_msgs!=1 else ''}</span>
</div>
""", unsafe_allow_html=True)

# ── Welcome or messages ──
if not indexed and not history:
    st.markdown("""
    <div class="welcome">
      <div class="wlc-icon">📚</div>
      <div class="wlc-title">What would you like to <em>know?</em></div>
      <div class="wlc-sub">Upload PDFs from the sidebar, click Process & Index,<br>then ask anything across all your documents.</div>
      <div class="action-grid">
        <div class="action-card">
          <div class="ac-icon">📋</div>
          <div class="ac-title">Summarize PDFs</div>
          <div class="ac-desc">Get a concise summary of any uploaded document</div>
        </div>
        <div class="action-card">
          <div class="ac-icon">⚖️</div>
          <div class="ac-title">Compare Documents</div>
          <div class="ac-desc">Find similarities and differences across PDFs</div>
        </div>
        <div class="action-card">
          <div class="ac-icon">❓</div>
          <div class="ac-title">Generate Quiz</div>
          <div class="ac-desc">Create study questions from your materials</div>
        </div>
        <div class="action-card">
          <div class="ac-icon">📝</div>
          <div class="ac-title">Create Notes</div>
          <div class="ac-desc">Extract key points and structured notes</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

elif indexed and not history:
    tags = "".join(f'<div class="ad-tag"><span class="ad-dot"></span>{p["name"]}</div>'
                   for p in st.session_state.pdf_library)
    st.markdown(f"""
    <div class="welcome">
      <div class="wlc-icon">✨</div>
      <div class="wlc-title">Ready to answer</div>
      <div class="wlc-sub">Your documents are indexed. Ask anything below.</div>
      <div class="active-docs" style="justify-content:center;margin-top:0">{tags}</div>
    </div>
    """, unsafe_allow_html=True)

else:
    for msg in history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-row">
              <div class="av av-u">J</div>
              <div class="mc">
                <div class="mc-name you">You</div>
                <div class="mc-text">{msg["content"]}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            body = msg["content"].replace("\n\n","</p><p>").replace("\n","<br>")
            srcs = "".join(f'<span class="src-chip">📄 {s}</span>' for s in msg.get("sources",[]))
            sh   = f'<div class="sources-wrap"><div class="sources-title">Sources</div><div class="src-chips">{srcs}</div></div>' if srcs else ""
            st.markdown(f"""
            <div class="msg-row">
              <div class="av av-a">AI</div>
              <div class="mc">
                <div class="mc-name ai">Study Buddy AI</div>
                <div class="mc-text"><p>{body}</p></div>
                {sh}
              </div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════
#  FIXED INPUT BAR
# ════════════════════════════════════════
st.markdown('<div class="input-wrap"><div class="input-inner">', unsafe_allow_html=True)

if st.session_state.pdf_library:
    tags_str = "".join(
        f'<div class="ad-tag"><span class="ad-dot"></span>{p["name"]}</div>'
        for p in st.session_state.pdf_library
    )
    st.markdown(f"""
    <div class="active-docs">
      <span class="ad-label">Loaded</span>{tags_str}
    </div>""", unsafe_allow_html=True)

uc, qc, bc = st.columns([1, 11, 1])

with uc:
    inline_file = st.file_uploader(
        "clip", type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="clip_uploader",
    )

with qc:
    question = st.text_input(
        "q",
        placeholder="Ask anything about your documents…" if indexed else "Upload PDFs from the sidebar first…",
        label_visibility="collapsed",
    )

with bc:
    ask = st.button("➤", disabled=not (indexed and question.strip()))

st.markdown("""
<div class="input-hint">
  Study Buddy AI · answers sourced from your indexed documents only
</div>""", unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)


# ── Handle inline upload ──────────────────────────────────────────────
if inline_file:
    n = process_and_index(inline_file)
    if not st.session_state.chats:
        st.session_state.chats.append({"title":"New Chat","history":[],"timestamp":now_label()})
        st.session_state.active_chat = 0
    st.rerun()


# ── Handle ask ────────────────────────────────────────────────────────
if ask and question.strip() and indexed:
    # ensure there's an active chat
    if not st.session_state.chats:
        st.session_state.chats.append({"title": question[:30], "history": [], "timestamp": now_label()})
        st.session_state.active_chat = 0

    chat = st.session_state.chats[st.session_state.active_chat]

    # auto-title chat from first question
    if chat["title"] in ("New Chat", "") and not chat["history"]:
        chat["title"] = question[:32] + ("…" if len(question) > 32 else "")

    chat["history"].append({"role": "user", "content": question})

    with st.spinner(""):
        res       = retrieve(question, st.session_state.embed_model, st.session_state.collection)
        ans, srcs = generate(question, res)

    chat["history"].append({"role": "assistant", "content": ans, "sources": srcs})
    st.rerun()
