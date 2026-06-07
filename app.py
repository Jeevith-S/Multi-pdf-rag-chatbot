import streamlit as st
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("⚠️ Add GROQ_API_KEY in App Settings → Secrets")
    st.stop()

GROQ_MODEL = "llama-3.3-70b-versatile"

st.set_page_config(
    page_title="Study Buddy AI",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp { background: #111 !important; }
.block-container {
  max-width: 760px !important;
  padding: 48px 32px 120px !important;
  margin: 0 auto !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }

/* ── Title ── */
.app-title {
  text-align: center;
  margin-bottom: 40px;
}
.app-title h1 {
  font-size: 28px;
  font-weight: 600;
  color: #f0f0f0;
  letter-spacing: -0.02em;
  margin-bottom: 6px;
}
.app-title p {
  font-size: 15px;
  color: #555;
}

/* ── Upload card ── */
.upload-card {
  background: #1a1a1a;
  border: 1px solid #2a2a2a;
  border-radius: 16px;
  padding: 28px 28px 20px;
  margin-bottom: 28px;
}
.card-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: #444;
  margin-bottom: 14px;
}

[data-testid="stFileUploader"] {
  background: #111 !important;
  border: 1.5px dashed #2a2a2a !important;
  border-radius: 12px !important;
  transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: #19c37d !important;
}

/* uploaded file pills */
.file-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 9px 12px;
  background: #111;
  border: 1px solid #222;
  border-radius: 10px;
  margin-top: 8px;
}
.file-icon {
  width: 32px; height: 32px;
  background: #1a2e22;
  border: 1px solid #1a3a28;
  border-radius: 7px;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700;
  color: #19c37d; flex-shrink: 0;
}
.file-name {
  font-size: 13px; color: #bbb; font-weight: 500;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1;
}
.file-check { color: #19c37d; font-size: 14px; flex-shrink: 0; }

/* ── Process button ── */
.stButton > button {
  background: #19c37d !important;
  color: #000 !important;
  border: none !important;
  border-radius: 10px !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  padding: 12px 0 !important;
  width: 100% !important;
  margin-top: 14px !important;
  transition: background .15s, transform .1s !important;
  font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover  { background: #14a86a !important; }
.stButton > button:active { transform: scale(.98) !important; }
.stButton > button:disabled {
  background: #0d2e1e !important;
  color: #1a4a30 !important;
  cursor: not-allowed !important;
}

/* progress */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, #19c37d, #7effc4) !important;
  border-radius: 10px !important;
}

/* ── Ready badge ── */
.ready-badge {
  display: flex;
  align-items: center;
  gap: 8px;
  background: #0d1f16;
  border: 1px solid #163524;
  border-radius: 10px;
  padding: 11px 16px;
  margin-bottom: 28px;
  font-size: 13px;
  color: #19c37d;
}
.rb-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: #19c37d;
  box-shadow: 0 0 6px #19c37d;
  flex-shrink: 0;
  animation: glow 2s ease-in-out infinite;
}
@keyframes glow { 0%,100%{opacity:1} 50%{opacity:.3} }
.rb-files { color: #555; font-size: 12px; margin-left: auto; }

/* ── Chat messages ── */
.chat-wrap { display: flex; flex-direction: column; gap: 0; }

.msg {
  display: flex;
  gap: 14px;
  padding: 20px 0;
  border-bottom: 1px solid #191919;
  align-items: flex-start;
}
.msg:last-child { border-bottom: none; }

.av {
  width: 34px; height: 34px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 700;
  flex-shrink: 0; margin-top: 1px;
}
.av-u { background: #222; color: #aaa; border: 1px solid #2a2a2a; }
.av-a { background: linear-gradient(135deg, #19c37d, #0d6644); color: #000; font-size: 11px; }

.msg-body { flex: 1; min-width: 0; }
.msg-name {
  font-size: 12px; font-weight: 600;
  margin-bottom: 8px;
}
.name-u { color: #444; }
.name-a { color: #19c37d; }

.msg-text {
  font-size: 15px;
  line-height: 1.8;
  color: #c8c8c8;
  word-wrap: break-word;
}
.msg-text p             { margin-bottom: 10px; }
.msg-text p:last-child  { margin-bottom: 0; }
.msg-text strong        { color: #ececec; }
.msg-text ul, .msg-text ol { padding-left: 22px; margin: 8px 0; }
.msg-text li            { margin-bottom: 5px; color: #b0b0b0; }
.msg-text code {
  background: #1a1a1a;
  border: 1px solid #2a2a2a;
  border-radius: 4px;
  padding: 2px 7px;
  font-size: 13px;
  color: #7effc4;
}

.src-row { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; }
.src-tag {
  display: inline-flex; align-items: center; gap: 5px;
  background: #0d1f16;
  border: 1px solid #163524;
  border-radius: 100px;
  padding: 4px 12px;
  font-size: 11px;
  color: #19c37d;
}

/* ── Fixed input bar ── */
.fixed-bar {
  position: fixed;
  bottom: 0; left: 0; right: 0;
  background: #111;
  border-top: 1px solid #1f1f1f;
  padding: 14px 24px 20px;
  z-index: 999;
}
.bar-inner {
  max-width: 760px;
  margin: 0 auto;
  display: flex;
  gap: 10px;
  align-items: center;
}
.bar-hint {
  text-align: center;
  font-size: 11px;
  color: #222;
  margin-top: 8px;
  max-width: 760px;
  margin-left: auto;
  margin-right: auto;
}

/* input inside bar */
.stTextInput > label { display: none !important; }
.stTextInput > div > div > input {
  background: #1a1a1a !important;
  color: #ececec !important;
  border: 1px solid #2a2a2a !important;
  border-radius: 12px !important;
  font-size: 15px !important;
  padding: 14px 18px !important;
  font-family: 'Inter', sans-serif !important;
  box-shadow: 0 4px 24px #00000055 !important;
  transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput > div > div > input:focus {
  border-color: #19c37d !important;
  box-shadow: 0 0 0 3px #19c37d14, 0 4px 24px #00000055 !important;
}
.stTextInput > div > div > input::placeholder { color: #2e2e2e !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #222; border-radius: 10px; }

.stSuccess { background: #061410 !important; border-left-color: #19c37d !important;
             color: #19c37d !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────
for k, v in {
    "history": [], "indexed": False,
    "pdf_names": [], "collection": None, "embed_model": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Pipeline (all internal) ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def _collection():
    return chromadb.EphemeralClient().get_or_create_collection("docs")

def run_pipeline(files):
    # 1 extract
    docs = []
    for f in files:
        reader = PdfReader(f)
        text = "".join(p.extract_text() or "" for p in reader.pages)
        docs.append({"source": f.name, "text": text})
    # 2 chunk
    sp = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = [{"source": d["source"], "text": t}
              for d in docs for t in sp.split_text(d["text"])]
    # 3 embed + store
    model = _model()
    col   = _collection()
    ex    = col.get()
    if ex["ids"]: col.delete(ids=ex["ids"])
    embs  = model.encode([c["text"] for c in chunks], convert_to_numpy=True)
    col.add(ids=[str(i) for i in range(len(chunks))],
            documents=[c["text"] for c in chunks],
            embeddings=embs.tolist(),
            metadatas=[{"source": c["source"]} for c in chunks])
    return model, col

def ask_question(q, model, col):
    res = col.query(
        query_embeddings=[model.encode(q, convert_to_numpy=True).tolist()],
        n_results=3,
    )
    ctx = "\n\n".join(res["documents"][0])
    r   = Groq(api_key=GROQ_API_KEY).chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content":
            f"Answer clearly using ONLY the context below. Format well.\n\nContext:\n{ctx}\n\nQuestion: {q}"}],
    )
    return (r.choices[0].message.content,
            list(dict.fromkeys(m["source"] for m in res["metadatas"][0])))


# ── Page ──────────────────────────────────────────────────────────────

# Title
st.markdown("""
<div class="app-title">
  <h1>📚 Study Buddy AI</h1>
  <p>Upload your PDFs and ask questions — that's it.</p>
</div>
""", unsafe_allow_html=True)

# Upload card
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
st.markdown('<div class="card-label">Upload PDFs</div>', unsafe_allow_html=True)

files = st.file_uploader(
    "upload", type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if files:
    for f in files:
        ok = "✓" if (st.session_state.indexed and f.name in st.session_state.pdf_names) else ""
        st.markdown(f"""
        <div class="file-row">
          <div class="file-icon">PDF</div>
          <span class="file-name">{f.name}</span>
          <span class="file-check">{ok}</span>
        </div>""", unsafe_allow_html=True)

    if st.button("⚡  Process & Index", disabled=not files):
        with st.spinner("Processing your PDFs…"):
            p = st.progress(0)
            p.progress(30, text="Reading and chunking…")
            model, col = run_pipeline(files)
            p.progress(100, text="Done!")
            p.empty()
        st.session_state.indexed    = True
        st.session_state.pdf_names  = [f.name for f in files]
        st.session_state.embed_model = model
        st.session_state.collection  = col
        st.session_state.history     = []
        st.success(f"✅ {len(files)} PDF(s) ready — start asking questions below!")
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Ready badge
if st.session_state.indexed:
    names = "  ·  ".join(st.session_state.pdf_names)
    st.markdown(f"""
    <div class="ready-badge">
      <span class="rb-dot"></span>
      <span>Documents ready</span>
      <span class="rb-files">{names}</span>
    </div>
    """, unsafe_allow_html=True)

# Chat history
if st.session_state.history:
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg">
              <div class="av av-u">J</div>
              <div class="msg-body">
                <div class="msg-name name-u">You</div>
                <div class="msg-text">{msg["content"]}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            body = msg["content"].replace("\n\n","</p><p>").replace("\n","<br>")
            srcs = "".join(f'<span class="src-tag">📄 {s}</span>' for s in msg.get("sources",[]))
            sh   = f'<div class="src-row">{srcs}</div>' if srcs else ""
            st.markdown(f"""
            <div class="msg">
              <div class="av av-a">AI</div>
              <div class="msg-body">
                <div class="msg-name name-a">Study Buddy AI</div>
                <div class="msg-text"><p>{body}</p></div>
                {sh}
              </div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🗑  Clear Chat"):
        st.session_state.history = []
        st.rerun()

# ── Fixed bottom input bar ────────────────────────────────────────────
st.markdown('<div class="fixed-bar"><div class="bar-inner">', unsafe_allow_html=True)

qc, bc = st.columns([11, 1])
with qc:
    question = st.text_input(
        "q",
        placeholder="Ask anything about your documents…" if st.session_state.indexed else "Process your PDFs above first…",
        label_visibility="collapsed",
        disabled=not st.session_state.indexed,
    )
with bc:
    send = st.button("➤", disabled=not (st.session_state.indexed and question.strip()))

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="bar-hint">Answers are based only on your uploaded documents.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Handle send
if send and question.strip() and st.session_state.indexed:
    st.session_state.history.append({"role": "user", "content": question})
    with st.spinner(""):
        ans, srcs = ask_question(question, st.session_state.embed_model, st.session_state.collection)
    st.session_state.history.append({"role": "assistant", "content": ans, "sources": srcs})
    st.rerun()
