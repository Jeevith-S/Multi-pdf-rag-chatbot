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
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp { background: #111 !important; }
.block-container {
  max-width: 780px !important;
  padding: 48px 32px 120px !important;
  margin: 0 auto !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { color: #666 !important; }

/* ══════════════════════════
   SIDEBAR
══════════════════════════ */
[data-testid="stSidebar"] {
  background: #161616 !important;
  border-right: 1px solid #242424 !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 24px 16px !important;
}

/* brand */
.sb-brand {
  display: flex; align-items: center; gap: 10px;
  padding: 0 4px 20px;
  border-bottom: 1px solid #242424;
  margin-bottom: 20px;
}
.sb-icon {
  width: 34px; height: 34px; border-radius: 8px;
  background: linear-gradient(135deg,#19c37d,#0d6644);
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; flex-shrink: 0;
}
.sb-name { font-size: 14px; font-weight: 600; color: #ececec; }
.sb-sub  { font-size: 11px; color: #555; }

/* section label */
.sb-label {
  font-size: 10px; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase;
  color: #444; padding: 0 4px; margin-bottom: 10px;
}

/* new chat button */
[data-testid="stSidebar"] .stButton > button {
  background: #1e1e1e !important;
  color: #d0d0d0 !important;
  border: 1px solid #2a2a2a !important;
  border-radius: 9px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 9px 14px !important;
  width: 100% !important;
  text-align: left !important;
  transition: background .15s, border-color .15s !important;
  font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: #242424 !important;
  border-color: #333 !important;
  color: #fff !important;
}

/* chat history items */
.chat-hist-item {
  display: flex; align-items: center; gap: 9px;
  padding: 9px 10px; border-radius: 8px;
  margin-bottom: 2px; cursor: pointer;
  transition: background .15s;
}
.chat-hist-item:hover { background: #1e1e1e; }
.chat-hist-item.active-chat { background: #1e1e1e; border: 1px solid #2a2a2a; }
.chi-icon { font-size: 13px; opacity: .5; flex-shrink: 0; }
.chi-text {
  font-size: 13px; color: #aaa; font-weight: 400;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1;
}
.chi-text.active-text { color: #ececec; font-weight: 500; }
.chi-time { font-size: 10px; color: #3a3a3a; flex-shrink: 0; }

.sb-divider { height: 1px; background: #1f1f1f; margin: 14px 0; }

/* clear button in sidebar */
.sb-clear .stButton > button {
  background: transparent !important;
  color: #444 !important;
  border: 1px solid #1e1e1e !important;
  font-size: 12px !important;
}
.sb-clear .stButton > button:hover {
  background: #1a0e0e !important;
  border-color: #3a1a1a !important;
  color: #ff6b6b !important;
}

/* ══════════════════════════
   MAIN AREA
══════════════════════════ */

.app-title {
  text-align: center;
  margin-bottom: 36px;
}
.app-title h1 {
  font-size: 26px; font-weight: 600;
  color: #f0f0f0; letter-spacing: -.02em;
  margin-bottom: 6px;
}
.app-title p { font-size: 15px; color: #666; }

/* upload card */
.upload-card {
  background: #1a1a1a;
  border: 1px solid #262626;
  border-radius: 16px;
  padding: 26px 26px 20px;
  margin-bottom: 24px;
}
.card-label {
  font-size: 11px; font-weight: 600;
  letter-spacing: .1em; text-transform: uppercase;
  color: #555; margin-bottom: 14px;
}

[data-testid="stFileUploader"] {
  background: #111 !important;
  border: 1.5px dashed #2a2a2a !important;
  border-radius: 12px !important;
  transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: #19c37d !important; }

/* file pills */
.file-row {
  display: flex; align-items: center; gap: 10px;
  padding: 9px 12px;
  background: #111; border: 1px solid #222;
  border-radius: 10px; margin-top: 8px;
}
.file-icon {
  width: 30px; height: 30px; border-radius: 6px;
  background: #0d1f16; border: 1px solid #163524;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 700; color: #19c37d; flex-shrink: 0;
}
.file-name { font-size: 13px; color: #ccc; font-weight: 500; flex: 1;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.file-check { color: #19c37d; font-size: 14px; }

/* process button */
.stButton > button {
  background: #19c37d !important;
  color: #000 !important; border: none !important;
  border-radius: 10px !important;
  font-size: 14px !important; font-weight: 600 !important;
  padding: 12px 0 !important; width: 100% !important;
  margin-top: 14px !important;
  transition: background .15s, transform .1s !important;
  font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover  { background: #14a86a !important; }
.stButton > button:active { transform: scale(.98) !important; }
.stButton > button:disabled {
  background: #0d2e1e !important; color: #1a4a30 !important; cursor: not-allowed !important;
}

/* progress */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg,#19c37d,#7effc4) !important;
  border-radius: 10px !important;
}

/* ready badge */
.ready-badge {
  display: flex; align-items: center; gap: 8px;
  background: #0d1f16; border: 1px solid #163524;
  border-radius: 10px; padding: 11px 16px;
  margin-bottom: 24px; font-size: 13px; color: #19c37d;
}
.rb-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: #19c37d; box-shadow: 0 0 6px #19c37d;
  flex-shrink: 0; animation: glow 2s ease-in-out infinite;
}
@keyframes glow { 0%,100%{opacity:1} 50%{opacity:.3} }
.rb-files { color: #888; font-size: 12px; margin-left: auto; }

/* messages */
.msg {
  display: flex; gap: 14px;
  padding: 20px 0;
  border-bottom: 1px solid #1a1a1a;
  align-items: flex-start;
}
.msg:last-child { border-bottom: none; }

.av {
  width: 34px; height: 34px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700;
  flex-shrink: 0; margin-top: 2px;
}
.av-u { background: #222; color: #bbb; border: 1px solid #2a2a2a; }
.av-a { background: linear-gradient(135deg,#19c37d,#0d6644); color: #000; font-size: 11px; }

.msg-body { flex: 1; min-width: 0; }
.msg-name { font-size: 12px; font-weight: 600; margin-bottom: 8px; }
.name-u { color: #666; }
.name-a { color: #19c37d; }

.msg-text {
  font-size: 15px; line-height: 1.85; color: #d0d0d0; word-wrap: break-word;
}
.msg-text p             { margin-bottom: 10px; }
.msg-text p:last-child  { margin-bottom: 0; }
.msg-text strong        { color: #f0f0f0; }
.msg-text ul, .msg-text ol { padding-left: 22px; margin: 8px 0; }
.msg-text li            { margin-bottom: 5px; color: #bbb; }
.msg-text code {
  background: #1a1a1a; border: 1px solid #2a2a2a;
  border-radius: 4px; padding: 2px 7px;
  font-size: 13px; color: #7effc4;
}

.src-row { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; }
.src-tag {
  display: inline-flex; align-items: center; gap: 5px;
  background: #0d1f16; border: 1px solid #163524;
  border-radius: 100px; padding: 4px 12px;
  font-size: 11px; color: #19c37d;
}

/* ── fixed bottom input bar ── */
.fixed-bar {
  position: fixed; bottom: 0; left: 0; right: 0;
  background: #111; border-top: 1px solid #1f1f1f;
  padding: 14px 24px 18px; z-index: 999;
}
.bar-inner {
  max-width: 780px; margin: 0 auto;
}
.bar-hint {
  text-align: center; font-size: 11px;
  color: #3a3a3a; margin-top: 8px;
}

/* text input */
.stTextInput > label { display: none !important; }
.stTextInput > div > div > input {
  background: #1a1a1a !important;
  color: #ececec !important;
  border: 1px solid #2e2e2e !important;
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
/* ✅ FIXED: placeholder now clearly visible */
.stTextInput > div > div > input::placeholder { color: #777 !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #222; border-radius: 10px; }

.stSuccess { background: #061410 !important; border-left-color: #19c37d !important;
             color: #19c37d !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────
for k, v in {
    "sessions":    [],       # list of {title, history}
    "active":      0,
    "indexed":     False,
    "pdf_names":   [],
    "collection":  None,
    "embed_model": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def cur_history():
    if not st.session_state.sessions:
        return []
    return st.session_state.sessions[st.session_state.active]["history"]


# ── Pipeline ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def _col():
    return chromadb.EphemeralClient().get_or_create_collection("docs")

def run_pipeline(files):
    docs = []
    for f in files:
        reader = PdfReader(f)
        text = "".join(p.extract_text() or "" for p in reader.pages)
        docs.append({"source": f.name, "text": text})
    sp = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = [{"source": d["source"], "text": t}
              for d in docs for t in sp.split_text(d["text"])]
    model = _model()
    col   = _col()
    ex    = col.get()
    if ex["ids"]: col.delete(ids=ex["ids"])
    embs  = model.encode([c["text"] for c in chunks], convert_to_numpy=True)
    col.add(ids=[str(i) for i in range(len(chunks))],
            documents=[c["text"] for c in chunks],
            embeddings=embs.tolist(),
            metadatas=[{"source": c["source"]} for c in chunks])
    return model, col

def ask_llm(q, model, col):
    res = col.query(
        query_embeddings=[model.encode(q, convert_to_numpy=True).tolist()],
        n_results=3,
    )
    ctx = "\n\n".join(res["documents"][0])
    r   = Groq(api_key=GROQ_API_KEY).chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content":
            f"Answer clearly using ONLY the context. Format well with paragraphs.\n\nContext:\n{ctx}\n\nQuestion: {q}"}],
    )
    return (r.choices[0].message.content,
            list(dict.fromkeys(m["source"] for m in res["metadatas"][0])))


# ════════════════════════════════════
#  SIDEBAR — chat history only
# ════════════════════════════════════
with st.sidebar:

    st.markdown("""
    <div class="sb-brand">
      <div class="sb-icon">📚</div>
      <div>
        <div class="sb-name">Study Buddy AI</div>
        <div class="sb-sub">PDF Chat</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # New chat
    if st.button("＋  New Chat"):
        st.session_state.sessions.append({"title": "New Chat", "history": []})
        st.session_state.active = len(st.session_state.sessions) - 1
        st.rerun()

    # History list
    if st.session_state.sessions:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sb-label">Chats</div>', unsafe_allow_html=True)

        for i, s in enumerate(reversed(st.session_state.sessions)):
            idx     = len(st.session_state.sessions) - 1 - i
            active  = idx == st.session_state.active
            label   = s["title"][:30] + ("…" if len(s["title"]) > 30 else "")
            txt_cls = "chi-text active-text" if active else "chi-text"
            bg_cls  = "chat-hist-item active-chat" if active else "chat-hist-item"
            msgs    = len(s["history"]) // 2
            time_lbl = f"{msgs} msg{'s' if msgs!=1 else ''}"

            st.markdown(f"""
            <div class="{bg_cls}" onclick="">
              <span class="chi-icon">💬</span>
              <span class="{txt_cls}">{label}</span>
              <span class="chi-time">{time_lbl}</span>
            </div>""", unsafe_allow_html=True)

            # invisible button to make it clickable
            if st.button("  ", key=f"sel_{idx}", help=label):
                st.session_state.active = idx
                st.rerun()

    # Clear all
    if st.session_state.sessions:
        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-clear">', unsafe_allow_html=True)
        if st.button("🗑  Clear All Chats"):
            st.session_state.sessions = []
            st.session_state.active   = 0
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════
#  MAIN AREA
# ════════════════════════════════════

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
            p.progress(40, text="Chunking and embedding…")
            model, col = run_pipeline(files)
            p.progress(100, text="Done!")
            p.empty()

        st.session_state.indexed     = True
        st.session_state.pdf_names   = [f.name for f in files]
        st.session_state.embed_model = model
        st.session_state.collection  = col

        # auto-create a new chat session for this batch
        title = files[0].name.replace(".pdf", "") if len(files) == 1 else f"{len(files)} PDFs"
        st.session_state.sessions.append({"title": title, "history": []})
        st.session_state.active = len(st.session_state.sessions) - 1

        st.success(f"✅ {len(files)} PDF(s) ready — ask anything below!")
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
    </div>""", unsafe_allow_html=True)

# Chat messages
history = cur_history()
if history:
    for msg in history:
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


# ════════════════════════════════════
#  FIXED INPUT BAR
# ════════════════════════════════════
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

st.markdown('<div class="bar-hint">Answers are based only on your uploaded documents.</div>', unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)

# Handle send
if send and question.strip() and st.session_state.indexed:
    if not st.session_state.sessions:
        st.session_state.sessions.append({"title": question[:30], "history": []})
        st.session_state.active = 0

    sess = st.session_state.sessions[st.session_state.active]
    if not sess["history"]:
        sess["title"] = question[:30] + ("…" if len(question) > 30 else "")

    sess["history"].append({"role": "user", "content": question})
    with st.spinner(""):
        ans, srcs = ask_llm(question, st.session_state.embed_model, st.session_state.collection)
    sess["history"].append({"role": "assistant", "content": ans, "sources": srcs})
    st.rerun()
