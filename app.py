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

# ── PAGE CONFIG ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Study Buddy AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;1,14..32,400&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin:0; padding:0; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── Shell ── */
.stApp { background: #212121; color: #ececec; }
.block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── Root layout: sidebar + main ── */
.app-shell {
  display: grid;
  grid-template-columns: 260px 1fr;
  height: 100vh;
  overflow: hidden;
}

/* ══════════════════════════════════
   SIDEBAR
══════════════════════════════════ */
.sidebar {
  background: #171717;
  border-right: 1px solid #2f2f2f;
  display: flex;
  flex-direction: column;
  padding: 0;
  overflow: hidden;
}

.sidebar-header {
  padding: 18px 16px 14px;
  border-bottom: 1px solid #2f2f2f;
}

.brand {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 16px;
}
.brand-icon {
  width: 32px; height: 32px; border-radius: 8px;
  background: linear-gradient(135deg, #10a37f, #1a7f64);
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; flex-shrink: 0;
  box-shadow: 0 2px 8px #10a37f44;
}
.brand-name { font-size: 15px; font-weight: 600; color: #ececec; }
.brand-sub  { font-size: 11px; color: #8e8ea0; margin-top: 1px; }

.new-chat-btn {
  display: flex; align-items: center; justify-content: center; gap: 8px;
  width: 100%; padding: 9px 14px;
  background: transparent;
  border: 1px solid #3f3f3f;
  border-radius: 8px;
  color: #ececec; font-size: 13px; font-weight: 500;
  cursor: pointer; transition: background .15s;
  font-family: 'Inter', sans-serif;
}
.new-chat-btn:hover { background: #2a2a2a; }

/* sidebar sections */
.sidebar-section { padding: 16px 12px 8px; }
.sidebar-section-title {
  font-size: 10px; font-weight: 600; letter-spacing: .1em;
  text-transform: uppercase; color: #4a4a5a;
  padding: 0 4px; margin-bottom: 10px;
}

/* upload zone */
[data-testid="stFileUploader"] {
  background: #1e1e1e !important;
  border: 1.5px dashed #3a3a4a !important;
  border-radius: 10px !important;
  transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: #10a37f !important; }
[data-testid="stFileUploader"] label  { color: #8e8ea0 !important; font-size:13px !important; }
[data-testid="stFileUploader"] section { padding: 12px !important; }

/* file pills in sidebar */
.file-pill {
  display: flex; align-items: center; gap: 8px;
  padding: 7px 10px; border-radius: 8px;
  margin-bottom: 3px; cursor: default;
  transition: background .15s;
}
.file-pill:hover { background: #2a2a2a; }
.fp-icon {
  width: 28px; height: 28px; border-radius: 6px;
  background: #2a2a3a; border: 1px solid #3a3a4a;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; flex-shrink: 0;
}
.fp-info { min-width: 0; flex: 1; }
.fp-name { font-size: 12px; font-weight: 500; color: #c5c5d2;
           white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.fp-size { font-size: 10px; color: #4a4a5a; margin-top: 1px; }
.fp-check { color: #10a37f; font-size: 13px; flex-shrink: 0; }

/* stat pills */
.stat-row-h { display: flex; gap: 6px; margin: 10px 0 4px; }
.stat-pill {
  flex: 1; background: #1e1e1e; border: 1px solid #2f2f2f;
  border-radius: 8px; padding: 10px 8px; text-align: center;
}
.sp-val { font-size: 18px; font-weight: 700; color: #10a37f; line-height: 1; }
.sp-lbl { font-size: 10px; color: #4a4a5a; margin-top: 3px; text-transform: uppercase; letter-spacing: .05em; }

/* sidebar buttons */
.stButton > button {
  background: #10a37f !important;
  color: #fff !important; border: none !important;
  border-radius: 8px !important; font-weight: 500 !important;
  font-size: 13px !important; padding: 9px 0 !important;
  width: 100% !important; font-family: 'Inter', sans-serif !important;
  transition: background .15s, transform .1s !important;
  letter-spacing: .01em !important;
}
.stButton > button:hover   { background: #0d8f6e !important; }
.stButton > button:active  { transform: scale(.98) !important; }
.stButton > button:disabled{ background: #1e3a32 !important; color: #2a6a54 !important; cursor: not-allowed !important; }

.btn-secondary > button {
  background: transparent !important;
  border: 1px solid #3f3f3f !important;
  color: #8e8ea0 !important;
}
.btn-secondary > button:hover { background: #2a2a2a !important; color: #ececec !important; }

/* progress */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, #10a37f, #1de9b6) !important;
  border-radius: 10px !important;
}

/* ══════════════════════════════════
   MAIN CHAT AREA
══════════════════════════════════ */
.main-area {
  display: flex; flex-direction: column;
  overflow: hidden; background: #212121;
}

/* top header bar */
.chat-header {
  padding: 13px 24px;
  border-bottom: 1px solid #2f2f2f;
  display: flex; align-items: center; gap: 12px;
  background: #212121;
  flex-shrink: 0;
}
.ch-model-badge {
  display: flex; align-items: center; gap: 6px;
  background: #2a2a2a; border: 1px solid #3a3a3a;
  border-radius: 6px; padding: 4px 10px;
  font-size: 12px; color: #8e8ea0;
  font-family: 'JetBrains Mono', monospace;
}
.ch-model-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: #10a37f; box-shadow: 0 0 6px #10a37f;
  animation: mdot 2s ease-in-out infinite;
}
@keyframes mdot { 0%,100%{opacity:1} 50%{opacity:.3} }
.ch-title { font-size: 15px; font-weight: 600; color: #ececec; }
.ch-right  { margin-left: auto; display: flex; align-items: center; gap: 8px; }
.ch-counter {
  font-size: 11px; color: #4a4a5a;
  font-family: 'JetBrains Mono', monospace;
}

/* messages scroll */
.messages-wrap {
  flex: 1; overflow-y: auto; overflow-x: hidden;
  padding: 0; scroll-behavior: smooth;
}
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 10px; }

/* ── Welcome / empty state ── */
.welcome-wrap {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  height: 100%; padding: 40px 20px; text-align: center;
  min-height: 400px;
}
.welcome-logo {
  width: 56px; height: 56px; border-radius: 16px;
  background: linear-gradient(135deg, #10a37f, #1a7f64);
  display: flex; align-items: center; justify-content: center;
  font-size: 28px; margin-bottom: 20px;
  box-shadow: 0 4px 20px #10a37f33;
}
.welcome-title { font-size: 24px; font-weight: 600; color: #ececec; margin-bottom: 8px; }
.welcome-sub   { font-size: 15px; color: #8e8ea0; line-height: 1.7; max-width: 380px; }
.welcome-cards { display: flex; gap: 10px; margin-top: 28px; flex-wrap: wrap; justify-content: center; }
.wcard {
  background: #2a2a2a; border: 1px solid #3a3a3a;
  border-radius: 10px; padding: 14px 16px;
  font-size: 13px; color: #c5c5d2; max-width: 180px;
  text-align: left; line-height: 1.5;
}
.wcard-icon { font-size: 18px; margin-bottom: 7px; }

/* ── Message rows (ChatGPT style) ── */
.msg-row {
  width: 100%;
  padding: 20px 0;
  border-bottom: 1px solid #2a2a2a;
  display: flex; gap: 0;
}
.msg-row.user-row { background: #212121; }
.msg-row.ai-row   { background: #2a2a2a; }

.msg-inner {
  max-width: 720px; margin: 0 auto;
  width: 100%; padding: 0 24px;
  display: flex; gap: 16px; align-items: flex-start;
}

.msg-avatar {
  width: 36px; height: 36px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 15px; flex-shrink: 0; font-weight: 600;
  margin-top: 2px;
}
.avatar-user {
  background: linear-gradient(135deg, #7c6cfc, #4facfe);
  color: #fff; font-size: 13px;
}
.avatar-ai {
  background: linear-gradient(135deg, #10a37f, #1a7f64);
  color: #fff;
}

.msg-content { flex: 1; min-width: 0; }
.msg-author  { font-size: 13px; font-weight: 600; color: #ececec; margin-bottom: 6px; }
.msg-text    { font-size: 15px; color: #d1d5db; line-height: 1.75; word-wrap: break-word; }
.msg-text p  { margin-bottom: 10px; }
.msg-text p:last-child { margin-bottom: 0; }
.msg-text strong { color: #ececec; }
.msg-text code {
  background: #1e1e2e; border: 1px solid #3a3a4a;
  border-radius: 4px; padding: 2px 6px;
  font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #a5f3c8;
}
.msg-text pre {
  background: #1e1e2e; border: 1px solid #3a3a4a;
  border-radius: 8px; padding: 14px 16px; margin: 10px 0;
  overflow-x: auto;
}
.msg-text pre code { background: none; border: none; padding: 0; }
.msg-text ul, .msg-text ol { padding-left: 20px; margin: 8px 0; }
.msg-text li { margin-bottom: 4px; }

/* sources under AI message */
.sources-wrap { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; }
.src-chip {
  display: inline-flex; align-items: center; gap: 5px;
  background: #1e1e2e; border: 1px solid #3a3a4a;
  border-radius: 20px; padding: 3px 12px;
  font-size: 11px; color: #10a37f;
  font-family: 'JetBrains Mono', monospace;
}
.src-chip::before { content: '📄'; font-size: 10px; }

/* typing indicator */
.typing-row { padding: 20px 0; background: #2a2a2a; border-bottom: 1px solid #2a2a2a; }
.typing-dots { display: flex; gap: 5px; align-items: center; padding: 4px 0; }
.typing-dots span {
  width: 8px; height: 8px; border-radius: 50%; background: #4a4a5a;
  animation: tdot 1.4s ease-in-out infinite;
}
.typing-dots span:nth-child(2) { animation-delay: .2s; }
.typing-dots span:nth-child(3) { animation-delay: .4s; }
@keyframes tdot { 0%,80%,100%{opacity:.2;transform:scale(.8)} 40%{opacity:1;transform:scale(1)} }

/* ══════════════════════════════════
   INPUT BAR (ChatGPT style)
══════════════════════════════════ */
.input-zone {
  padding: 16px 24px 20px;
  background: #212121;
  border-top: 1px solid #2f2f2f;
  flex-shrink: 0;
}
.input-inner { max-width: 720px; margin: 0 auto; }

.stTextInput > label { display: none !important; }
.stTextInput > div > div > input {
  background: #2a2a2a !important;
  color: #ececec !important;
  border: 1px solid #3f3f3f !important;
  border-radius: 12px !important;
  font-size: 15px !important;
  padding: 14px 18px !important;
  font-family: 'Inter', sans-serif !important;
  transition: border-color .2s, box-shadow .2s !important;
  box-shadow: 0 2px 10px #0004 !important;
}
.stTextInput > div > div > input:focus {
  border-color: #10a37f !important;
  box-shadow: 0 0 0 3px #10a37f18, 0 2px 10px #0004 !important;
}
.stTextInput > div > div > input::placeholder { color: #4a4a5a !important; }

.input-footer {
  text-align: center; margin-top: 8px;
  font-size: 11px; color: #3a3a3a;
}

/* send + clear buttons inside input zone */
.input-btn-row { display: flex; gap: 8px; margin-top: 8px; }

/* alerts */
.stSuccess { background: #0d2016 !important; border-color: #10a37f44 !important; color: #10a37f !important; border-radius: 10px !important; }
.stInfo    { border-radius: 10px !important; }
.stError   { border-radius: 10px !important; }
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
    "is_thinking":    False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── CACHED RESOURCES ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Starting ChromaDB…")
def get_chroma_collection():
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(name="pdf_documents")


# ── PIPELINE ─────────────────────────────────────────────────────────
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
    prompt  = f"""You are a helpful study assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say so clearly. Be clear, concise, and well-structured.

Context:
{context}

Question:
{question}"""
    resp    = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    answer  = resp.choices[0].message.content
    sources = list(dict.fromkeys(m["source"] for m in results["metadatas"][0]))
    return answer, sources

def fmt_size(b):
    return f"{b/1024:.0f} KB" if b < 1048576 else f"{b/1048576:.1f} MB"


# ══════════════════════════════════════════════════════════
#  LAYOUT: two st.columns act as sidebar + main
# ══════════════════════════════════════════════════════════
sidebar_col, main_col = st.columns([0.22, 0.78], gap="small")


# ══════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════
with sidebar_col:

    # Brand
    st.markdown("""
    <div class="brand" style="padding:18px 4px 6px">
      <div class="brand-icon">📚</div>
      <div>
        <div class="brand-name">Study Buddy AI</div>
        <div class="brand-sub">RAG Document Chat</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Upload section
    st.markdown('<div class="sidebar-section-title">📄 Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Show selected file pills
    if uploaded_files:
        for f in uploaded_files:
            is_indexed = st.session_state.indexed and f.name in st.session_state.indexed_files
            check = "✓" if is_indexed else ""
            st.markdown(f"""
            <div class="file-pill">
              <div class="fp-icon">PDF</div>
              <div class="fp-info">
                <div class="fp-name">{f.name}</div>
                <div class="fp-size">{fmt_size(f.size)}</div>
              </div>
              <span class="fp-check">{check}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Stats (after indexing)
    if st.session_state.indexed:
        st.markdown(f"""
        <div class="stat-row-h">
          <div class="stat-pill">
            <div class="sp-val">{len(st.session_state.indexed_files)}</div>
            <div class="sp-lbl">PDFs</div>
          </div>
          <div class="stat-pill">
            <div class="sp-val">{st.session_state.num_chunks}</div>
            <div class="sp-lbl">Chunks</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Index button
    btn_lbl  = "⚡  Process & Index" if not st.session_state.indexed else "🔄  Re-index Files"
    do_index = st.button(btn_lbl, disabled=not uploaded_files)

    # Clear chat button
    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
            if st.button("🗑️  Clear Conversation"):
                st.session_state.chat_history = []
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # Model info at bottom
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:10px;color:#3a3a3a;line-height:1.7;padding:0 2px">
      <div style="margin-bottom:4px;color:#4a4a5a;font-weight:600;letter-spacing:.08em;text-transform:uppercase">Stack</div>
      ChromaDB · all-MiniLM-L6-v2<br>Groq · LLaMA-3.3-70B
    </div>
    """, unsafe_allow_html=True)

    # ── Run pipeline ──
    if do_index and uploaded_files:
        prog = st.progress(0, text="Reading PDFs…")
        docs   = extract_text_from_pdfs(uploaded_files);    prog.progress(20, text="Chunking…")
        chunks = chunk_documents(docs);                      prog.progress(40, text="Loading model…")
        model  = load_embedding_model();                     prog.progress(60, text="Embedding…")
        embs   = create_chunk_embeddings(chunks, model);     prog.progress(80, text="Storing in ChromaDB…")
        col_db = get_chroma_collection()
        n      = store_in_vectordb(chunks, embs, col_db);   prog.progress(100, text="Done!")

        st.session_state.collection    = col_db
        st.session_state.embed_model   = model
        st.session_state.indexed       = True
        st.session_state.num_chunks    = n
        st.session_state.indexed_files = [f.name for f in uploaded_files]
        st.session_state.chat_history  = []
        prog.empty()
        st.success(f"✅ {n} chunks indexed")
        st.rerun()


# ══════════════════════════════════
#  MAIN CHAT AREA
# ══════════════════════════════════
with main_col:

    # ── Header bar ──
    n_ex = len(st.session_state.chat_history) // 2
    model_label = "llama-3.3-70b-versatile"
    st.markdown(f"""
    <div class="chat-header">
      <span class="ch-title">💬 Chat</span>
      <div class="ch-model-badge">
        <div class="ch-model-dot"></div>
        {model_label}
      </div>
      <div class="ch-right">
        <span class="ch-counter">{n_ex} message{'s' if n_ex!=1 else ''}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Message area ──
    msgs_container = st.container()

    with msgs_container:
        if not st.session_state.indexed:
            # Welcome state
            st.markdown("""
            <div class="welcome-wrap">
              <div class="welcome-logo">📚</div>
              <div class="welcome-title">Study Buddy AI</div>
              <div class="welcome-sub">
                Upload your PDFs and ask questions across all your documents.<br>
                Powered by ChromaDB and LLaMA 3.
              </div>
              <div class="welcome-cards">
                <div class="wcard">
                  <div class="wcard-icon">📄</div>
                  Upload multiple PDFs from the left panel
                </div>
                <div class="wcard">
                  <div class="wcard-icon">⚡</div>
                  Click Process &amp; Index to embed your documents
                </div>
                <div class="wcard">
                  <div class="wcard-icon">💬</div>
                  Ask any question and get cited answers
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        elif not st.session_state.chat_history:
            # Ready state
            files_list = "".join(
                f'<div style="display:inline-flex;align-items:center;gap:6px;'
                f'background:#1e1e2e;border:1px solid #3a3a4a;border-radius:6px;'
                f'padding:3px 12px;font-size:12px;color:#10a37f;margin:3px;'
                f'font-family:JetBrains Mono,monospace">📄 {f}</div>'
                for f in st.session_state.indexed_files
            )
            st.markdown(f"""
            <div class="welcome-wrap">
              <div class="welcome-logo">✨</div>
              <div class="welcome-title">Ready to chat!</div>
              <div class="welcome-sub">Your documents are indexed and ready.<br>Ask anything below.</div>
              <div style="margin-top:20px;display:flex;flex-wrap:wrap;justify-content:center;gap:4px">
                {files_list}
              </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # Chat messages — ChatGPT style full-width rows
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="msg-row user-row">
                      <div class="msg-inner">
                        <div class="msg-avatar avatar-user">J</div>
                        <div class="msg-content">
                          <div class="msg-author">You</div>
                          <div class="msg-text">{msg["content"]}</div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    srcs = "".join(f'<span class="src-chip">{s}</span>' for s in msg.get("sources", []))
                    src_html = f'<div class="sources-wrap">{srcs}</div>' if srcs else ""
                    # render newlines as paragraph breaks
                    body = msg["content"].replace("\n\n", "</p><p>").replace("\n", "<br>")
                    st.markdown(f"""
                    <div class="msg-row ai-row">
                      <div class="msg-inner">
                        <div class="msg-avatar avatar-ai">🤖</div>
                        <div class="msg-content">
                          <div class="msg-author">Study Buddy AI</div>
                          <div class="msg-text"><p>{body}</p></div>
                          {src_html}
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Input bar ──
    st.markdown('<div class="input-inner">', unsafe_allow_html=True)

    if st.session_state.indexed:
        # active files strip
        files_str = "  ·  ".join(st.session_state.indexed_files)
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:7px;margin-bottom:10px;
             font-size:11px;color:#10a37f;font-family:'JetBrains Mono',monospace">
          <span style="width:6px;height:6px;border-radius:50%;background:#10a37f;
                display:inline-block;box-shadow:0 0 6px #10a37f"></span>
          {files_str}
        </div>
        """, unsafe_allow_html=True)

    q_col, btn_col = st.columns([10, 1])
    with q_col:
        question = st.text_input(
            "message",
            placeholder="Ask anything about your documents…" if st.session_state.indexed else "Upload and index PDFs first…",
            label_visibility="collapsed",
            disabled=not st.session_state.indexed,
        )
    with btn_col:
        ask = st.button("➤", disabled=not st.session_state.indexed, help="Send message")

    st.markdown("""
    <div class="input-footer">
      Study Buddy AI may produce inaccurate information — always verify with source documents.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── HANDLE ASK ────────────────────────────────────────────────────────
if ask and question.strip():
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.spinner(""):
        results         = retrieve(question, st.session_state.embed_model, st.session_state.collection)
        answer, sources = generate_answer(question, results)
    st.session_state.chat_history.append({
        "role": "assistant", "content": answer, "sources": sources,
    })
    st.rerun()
