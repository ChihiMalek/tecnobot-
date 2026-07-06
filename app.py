"""
=============================================================
  TECNOBOT ELITE v3  —  APPLICATION PRINCIPALE
  Fichier : 2_app.py
  Moteur  : Qwen2.5:7b  +  RAG ChromaDB  +  nomic-embed-text
=============================================================
  USAGE :
    streamlit run 2_app.py
=============================================================
"""
 
import os
import time
import sqlite3
import hashlib
import base64
from pathlib import Path
 
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import ollama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
 
# ─── CONFIGURATION ───────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
PATH_DB     = BASE_DIR / "chroma_db"
DB_FILE     = BASE_DIR / "db_elite_v3.db"
BG_IMAGE    = BASE_DIR / "votre_image.jpg"  # ⚠️ REMPLACEZ PAR LE CHEMIN DE VOTRE IMAGE
MODEL_LLM   = "qwen2.5:7b"
MODEL_EMBED = "nomic-embed-text"
RAG_K       = 4          # Nombre de chunks récupérés par requête
RAG_SCORE   = 0.30       # Score minimal de similarité
 
 
# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TecnoBot Elite v3",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
 
# ─────────────────────────────────────────────────────────────
#  DESIGN — THEME ELITE AVEC IMAGE DE FOND ET ANIMATIONS
# ─────────────────────────────────────────────────────────────
def apply_design():
    b64 = ""
    if BG_IMAGE.exists():
        with open(BG_IMAGE, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
    
    # CSS pour l'image de fond avec effet de parallaxe
    bg_css = f"""
        background-image: url("data:image/jpeg;base64,{b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    """ if b64 else "background: linear-gradient(135deg, #0a0e1a 0%, #1a1f2e 100%);"
 
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Rajdhani:wght@500;700&display=swap');
 
    /* ── ANIMATIONS ─────────────────────────────────────────── */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    @keyframes glow {{
        0% {{ text-shadow: 0 0 5px rgba(0,229,160,0.2); }}
        50% {{ text-shadow: 0 0 20px rgba(0,229,160,0.6); }}
        100% {{ text-shadow: 0 0 5px rgba(0,229,160,0.2); }}
    }}
    
    @keyframes float {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-10px); }}
        100% {{ transform: translateY(0px); }}
    }}
    
    @keyframes pulse {{
        0% {{ transform: scale(1); opacity: 0.3; }}
        50% {{ transform: scale(1.05); opacity: 0.6; }}
        100% {{ transform: scale(1); opacity: 0.3; }}
    }}
    
    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    @keyframes slideInLeft {{
        from {{ opacity: 0; transform: translateX(-30px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}
    
    @keyframes slideInRight {{
        from {{ opacity: 0; transform: translateX(30px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}
    
    /* ── PARTICULES ANIMÉES ───────────────────────────────── */
    .particles {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }}
    
    .particle {{
        position: absolute;
        background: rgba(0, 229, 160, 0.15);
        border-radius: 50%;
        animation: floatParticle linear infinite;
    }}
    
    @keyframes floatParticle {{
        0% {{ transform: translateY(100vh) rotate(0deg); opacity: 0; }}
        10% {{ opacity: 0.5; }}
        90% {{ opacity: 0.5; }}
        100% {{ transform: translateY(-100vh) rotate(360deg); opacity: 0; }}
    }}
    
    /* ── ROOT ─────────────────────────────────────────── */
    .stApp {{
        {bg_css}
        position: relative;
    }}
    
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: linear-gradient(135deg, 
            rgba(4, 8, 14, 0.75) 0%,
            rgba(8, 12, 20, 0.85) 50%,
            rgba(4, 8, 14, 0.75) 100%);
        backdrop-filter: blur(12px);
        z-index: 0;
        animation: gradientShift 10s ease infinite;
        background-size: 200% 200%;
    }}
    
    /* Contenu principal avec animation d'apparition */
    .main-content {{
        position: relative;
        z-index: 1;
        animation: fadeIn 0.8s ease-out;
    }}
 
    /* ── TYPOGRAPHY ───────────────────────────────────── */
    html, body, [class*="css"] {{
        font-family: 'Rajdhani', sans-serif;
        color: #c9d1d9;
    }}
    
    h1, h2, h3 {{ 
        font-family: 'Rajdhani', sans-serif; 
        text-transform: uppercase; 
        letter-spacing: 2px;
        animation: slideInLeft 0.6s ease-out;
    }}
    
    /* Effet de lueur animée sur les titres */
    .glow-text {{
        animation: glow 3s ease-in-out infinite;
    }}
 
    /* ── SIDEBAR ──────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: rgba(8, 12, 16, 0.98) !important;
        border-right: 1px solid #1e2d3d;
        backdrop-filter: blur(10px);
        animation: slideInLeft 0.5s ease-out;
    }}
    
    [data-testid="stSidebar"]:hover {{
        border-right-color: #00e5a0;
        transition: border-color 0.3s ease;
    }}
    
    [data-testid="stSidebar"] * {{ color: #c9d1d9 !important; }}
 
    /* ── CARDS AVEC ANIMATIONS ────────────────────────── */
    .elite-card {{
        background: rgba(13, 17, 23, 0.85);
        border: 1px solid #1e2d3d;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        backdrop-filter: blur(4px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideInRight 0.5s ease-out;
    }}
    
    .elite-card:hover {{
        transform: translateY(-4px);
        border-color: #00e5a0;
        box-shadow: 0 8px 32px rgba(0, 229, 160, 0.15);
        transition: all 0.3s ease;
    }}
    
    .elite-card-title {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: #00e5a0;
        letter-spacing: 3px;
        text-transform: uppercase;
        border-bottom: 1px solid #1e2d3d;
        padding-bottom: 10px;
        margin-bottom: 14px;
        transition: letter-spacing 0.3s ease;
    }}
    
    .elite-card:hover .elite-card-title {{
        letter-spacing: 4px;
    }}
 
    /* ── NEON TEXT ────────────────────────────────────── */
    .neon  {{ 
        color: #00e5a0; 
        text-shadow: 0 0 8px rgba(0,229,160,0.4);
        animation: glow 2s ease-in-out infinite;
    }}
    .blue  {{ color: #00b8ff; }}
    .amber {{ color: #ffa940; }}
    .red   {{ color: #ff5f56; }}
 
    /* ── PAGE TITLE ───────────────────────────────────── */
    .page-title {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: #00e5a0;
        letter-spacing: 4px;
        text-transform: uppercase;
        padding: 0 0 16px;
        border-bottom: 1px solid #1e2d3d;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
        animation: slideInLeft 0.5s ease-out;
    }}
    
    .page-title::before {{
        content: '';
        display: inline-block;
        width: 4px; height: 16px;
        background: #00e5a0;
        border-radius: 2px;
        animation: pulse 2s ease-in-out infinite;
    }}
 
    /* ── METRIC CARDS ─────────────────────────────────── */
    .metric-row {{ display: flex; gap: 12px; margin-bottom: 16px; }}
    .metric-box {{
        flex: 1;
        background: rgba(13,17,23,0.9);
        border: 1px solid #1e2d3d;
        border-radius: 10px;
        padding: 14px 16px;
        transition: all 0.3s ease;
        animation: slideInRight 0.5s ease-out;
    }}
    
    .metric-box:hover {{
        transform: scale(1.02);
        border-color: #00e5a0;
        box-shadow: 0 4px 20px rgba(0, 229, 160, 0.1);
    }}
    
    .metric-label {{ font-size: 10px; color: #8b949e; letter-spacing: 1px; text-transform: uppercase; }}
    .metric-value {{ 
        font-family: 'JetBrains Mono', monospace; 
        font-size: 24px; 
        font-weight: 700; 
        margin-top: 4px;
        transition: all 0.3s ease;
    }}
    
    .metric-box:hover .metric-value {{
        transform: scale(1.05);
        display: inline-block;
    }}
 
    /* ── BADGES ───────────────────────────────────────── */
    .badge {{
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.5px;
        font-weight: 700;
        transition: all 0.3s ease;
    }}
    
    .badge:hover {{
        transform: translateY(-2px);
        box-shadow: 0 2px 8px rgba(0, 229, 160, 0.2);
    }}
    
    .badge-green {{ background: rgba(0,229,160,0.12); color: #00e5a0; border: 1px solid rgba(0,229,160,0.3); }}
    .badge-blue  {{ background: rgba(0,184,255,0.12); color: #00b8ff; border: 1px solid rgba(0,184,255,0.3); }}
    .badge-amber {{ background: rgba(255,169,64,0.12); color: #ffa940; border: 1px solid rgba(255,169,64,0.3); }}
    .badge-red   {{ background: rgba(255,95,86,0.12);  color: #ff5f56; border: 1px solid rgba(255,95,86,0.3); }}
 
    /* ── CHAT MESSAGES AVEC ANIMATIONS ─────────────────── */
    .chat-user {{
        background: rgba(0,229,160,0.08);
        border: 1px solid rgba(0,229,160,0.25);
        border-radius: 10px 10px 2px 10px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 14px;
        animation: slideInRight 0.4s ease-out;
        transition: all 0.3s ease;
    }}
    
    .chat-user:hover {{
        transform: translateX(-4px);
        border-color: #00e5a0;
    }}
    
    .chat-bot {{
        background: rgba(13,17,23,0.9);
        border: 1px solid #1e2d3d;
        border-radius: 10px 10px 10px 2px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 14px;
        animation: slideInLeft 0.4s ease-out;
        transition: all 0.3s ease;
    }}
    
    .chat-bot:hover {{
        transform: translateX(4px);
        border-color: #00e5a0;
    }}
    
    .chat-source {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: #00e5a0;
        margin-top: 8px;
        padding: 6px 10px;
        background: rgba(0,229,160,0.05);
        border-left: 2px solid #00e5a0;
        border-radius: 0 4px 4px 0;
        transition: border-left-width 0.3s ease;
    }}
    
    .chat-source:hover {{
        border-left-width: 4px;
    }}
 
    /* ── BUTTONS AVEC ANIMATIONS ───────────────────────── */
    .stButton > button {{
        background: linear-gradient(135deg, #00e5a0 0%, #00b8ff 100%) !important;
        color: #000 !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        border: none !important;
        border-radius: 8px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative;
        overflow: hidden;
    }}
    
    .stButton > button::before {{
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }}
    
    .stButton > button:hover::before {{
        width: 300px;
        height: 300px;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 229, 160, 0.3);
        letter-spacing: 2px !important;
    }}
    
    .stButton > button:active {{
        transform: translateY(1px);
    }}
 
    /* ── INPUTS AVEC ANIMATIONS ────────────────────────── */
    .stTextInput > div > div > input,
    .stTextArea textarea,
    .stSelectbox > div > div {{
        background: rgba(13,17,23,0.9) !important;
        border: 1px solid #1e2d3d !important;
        color: #c9d1d9 !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus {{
        border-color: #00e5a0 !important;
        box-shadow: 0 0 12px rgba(0, 229, 160, 0.2) !important;
        transform: scale(1.01);
    }}
 
    /* ── SLIDER ───────────────────────────────────────── */
    .stSlider > div > div > div > div {{ 
        background: linear-gradient(90deg, #00e5a0, #00b8ff) !important;
        transition: all 0.3s ease;
    }}
    
    .stSlider > div > div > div > div:hover {{
        transform: scale(1.1);
    }}
 
    /* ── TABS AVEC ANIMATIONS ──────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        background: rgba(8,12,16,0.8) !important;
        border-radius: 8px 8px 0 0;
        border-bottom: 1px solid #1e2d3d;
        gap: 0;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        color: #8b949e !important;
        background: transparent !important;
        border: none !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        color: #00e5a0 !important;
        transform: translateY(-2px);
    }}
    
    .stTabs [aria-selected="true"] {{
        color: #00e5a0 !important;
        border-bottom: 2px solid #00e5a0 !important;
        animation: slideInLeft 0.3s ease-out;
    }}
 
    /* ── RAG PIPELINE BAR AVEC ANIMATIONS ───────────────── */
    .rag-bar {{
        display: flex;
        align-items: center;
        gap: 0;
        flex-wrap: wrap;
        margin: 12px 0;
    }}
    
    .rag-step {{
        flex: 1; min-width: 90px;
        background: rgba(13,17,23,0.9);
        border: 1px solid #1e2d3d;
        border-radius: 8px;
        padding: 10px 8px;
        text-align: center;
        transition: all 0.3s ease;
        animation: fadeIn 0.6s ease-out;
    }}
    
    .rag-step:hover {{
        transform: translateY(-5px);
        border-color: #00e5a0;
        box-shadow: 0 4px 15px rgba(0, 229, 160, 0.15);
    }}
    
    .rag-step-title {{ font-size: 10px; color: #00e5a0; font-weight: 700; letter-spacing: 0.5px; }}
    .rag-step-sub   {{ font-size: 9px;  color: #8b949e; margin-top: 3px; }}
    .rag-arrow {{ 
        color: #1e2d3d; 
        font-size: 20px; 
        padding: 0 4px;
        animation: pulse 2s ease-in-out infinite;
    }}
 
    /* ── CODE BLOCK ───────────────────────────────────── */
    .code-block {{
        background: #0d1117;
        border: 1px solid #1e2d3d;
        border-radius: 8px;
        padding: 14px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #00e5a0;
        line-height: 1.7;
        overflow-x: auto;
        margin: 8px 0;
        transition: all 0.3s ease;
    }}
    
    .code-block:hover {{
        border-color: #00e5a0;
        box-shadow: 0 4px 12px rgba(0, 229, 160, 0.1);
    }}
    
    /* ── SCROLLBAR PERSONNALISÉE ───────────────────────── */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: rgba(13, 17, 23, 0.9);
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: #00e5a0;
        border-radius: 4px;
        transition: background 0.3s ease;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: #00b8ff;
    }}
    
    /* ── LOADING SPINNER PERSONNALISÉ ──────────────────── */
    .custom-spinner {{
        width: 40px;
        height: 40px;
        border: 3px solid rgba(0, 229, 160, 0.2);
        border-top-color: #00e5a0;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }}
    
    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}
    </style>
    
    <!-- PARTICULES ANIMÉES -->
    <div class="particles" id="particles"></div>
    
    <script>
    // Création des particules animées
    function createParticles() {{
        const container = document.getElementById('particles');
        if (!container) return;
        
        const particleCount = 50;
        
        for (let i = 0; i < particleCount; i++) {{
            const particle = document.createElement('div');
            particle.className = 'particle';
            const size = Math.random() * 4 + 2;
            particle.style.width = size + 'px';
            particle.style.height = size + 'px';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.animationDuration = Math.random() * 10 + 8 + 's';
            particle.style.animationDelay = Math.random() * 5 + 's';
            container.appendChild(particle);
        }}
    }}
    
    document.addEventListener('DOMContentLoaded', createParticles);
    </script>
    """, unsafe_allow_html=True)
 
 
apply_design()
 
 
# ─────────────────────────────────────────────────────────────
#  SÉCURITÉ — AUTH SQLite + SHA-256
# ─────────────────────────────────────────────────────────────
def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()
 
 
def init_db():
    conn = sqlite3.connect(str(DB_FILE))
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role     TEXT NOT NULL,
            name     TEXT NOT NULL,
            created  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT,
            role      TEXT,
            content   TEXT,
            sources   TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Comptes par défaut
    defaults = [
        ("chihi",  hash_pw("1234"),  "Embedded ML Engineer", "Chihi Malek"),
        ("admin",  hash_pw("admin"), "Administrateur",       "Admin Système"),
    ]
    for u, p, r, n in defaults:
        c.execute("INSERT OR IGNORE INTO users VALUES (?,?,?,?,CURRENT_TIMESTAMP)", (u, p, r, n))
    conn.commit()
    conn.close()
 
 
def check_login(username: str, password: str):
    conn = sqlite3.connect(str(DB_FILE))
    c = conn.cursor()
    c.execute("SELECT password, role, name FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row and row[0] == hash_pw(password):
        return {"role": row[1], "name": row[2]}
    return None
 
 
def save_message(username: str, role: str, content: str, sources: str = ""):
    conn = sqlite3.connect(str(DB_FILE))
    c = conn.cursor()
    c.execute(
        "INSERT INTO chat_history (username,role,content,sources) VALUES (?,?,?,?)",
        (username, role, content, sources),
    )
    conn.commit()
    conn.close()
 
 
init_db()
 
 
# ─────────────────────────────────────────────────────────────
#  RAG — CHARGEMENT CHROMADB
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag_db():
    """Charge la base vectorielle ChromaDB une seule fois."""
    if not PATH_DB.exists():
        return None
    try:
        embeddings = OllamaEmbeddings(model=MODEL_EMBED)
        db = Chroma(
            persist_directory=str(PATH_DB),
            embedding_function=embeddings,
            collection_name="tecnobot_knowledge",
        )
        return db
    except Exception as e:
        st.warning(f"ChromaDB non chargée : {e}")
        return None
 
 
def retrieve_context(db, query: str):
    """Recherche les chunks pertinents et retourne contexte + sources."""
    if db is None:
        return "", []
    try:
        results = db.similarity_search_with_relevance_scores(query, k=RAG_K)
        filtered = [(doc, score) for doc, score in results if score >= RAG_SCORE]
        if not filtered:
            return "", []
        context = "\n\n---\n\n".join([doc.page_content for doc, _ in filtered])
        sources = [
            {
                "file":  doc.metadata.get("source", "inconnu"),
                "page":  doc.metadata.get("page", "?"),
                "score": round(score, 3),
            }
            for doc, score in filtered
        ]
        return context, sources
    except Exception:
        return "", []
 
 
# ─────────────────────────────────────────────────────────────
#  LLM — APPEL QWEN2.5 AVEC CONTEXTE RAG
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Tu es TecnoBot Elite, un assistant expert en :
- Électrotechnique et électronique de puissance
- Systèmes embarqués et Embedded Machine Learning
- Diagnostic de pannes électriques complexes
- Convertisseurs AC/DC, onduleurs, IGBT, variateurs de vitesse
- Normes IEC, CEI et bonnes pratiques industrielles
 
RÈGLES STRICTES :
1. Si un CONTEXTE est fourni, base ta réponse UNIQUEMENT sur ce contexte.
2. Cite toujours les sources utilisées à la fin de ta réponse.
3. Si la question dépasse le contexte, signale-le clairement.
4. Réponds en français sauf si la question est en anglais.
5. Sois précis, technique et structuré dans tes réponses.
6. Pour les diagnostics, liste les causes probables par ordre de probabilité."""
 
 
def ask_qwen(query: str, context: str, history: list) -> str:
    """Appel au modèle Qwen2.5 avec injection du contexte RAG."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
 
    # Ajout des derniers échanges (fenêtre glissante de 6 messages)
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
 
    # Construction du prompt final avec contexte RAG
    if context:
        user_content = f"""CONTEXTE TECHNIQUE (extrait de la base de connaissances) :
{context}
 
QUESTION DE L'INGÉNIEUR :
{query}"""
    else:
        user_content = f"""[Aucun document pertinent trouvé dans la base RAG]
 
QUESTION :
{query}
 
Note : Réponds en précisant que la réponse est basée sur tes connaissances générales."""
 
    messages.append({"role": "user", "content": user_content})
 
    response = ollama.chat(model=MODEL_LLM, messages=messages)
    return response["message"]["content"]
 
 
# ─────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────
if "auth" not in st.session_state:
    st.session_state.auth = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_db" not in st.session_state:
    st.session_state.rag_db = None
 
 
# ─────────────────────────────────────────────────────────────
#  PAGE LOGIN
# ─────────────────────────────────────────────────────────────
if not st.session_state.auth:
    st.markdown("""
    <div style="text-align:center; padding: 40px 0 20px;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:32px;
                    color:#00e5a0; letter-spacing:8px; font-weight:700; animation: glow 2s ease-in-out infinite;">
            TECNOBOT
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:11px;
                    color:#00b8ff; letter-spacing:10px; margin-top:6px;">ELITE  v3  —  QWEN2.5</div>
        <div style="font-size:12px; color:#8b949e; margin-top:12px; letter-spacing:1px;">
            INTELLIGENCE ARTIFICIELLE APPLIQUÉE — EMBEDDED ML & ÉLECTRONIQUE DE PUISSANCE
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown('<div class="elite-card">', unsafe_allow_html=True)
        st.markdown('<div class="elite-card-title">TERMINAL ACCESS — AUTHENTIFICATION</div>', unsafe_allow_html=True)
 
        username = st.text_input("IDENTIFIANT SYSTÈME", placeholder="username")
        password = st.text_input("CLÉ D'ACCÈS", type="password", placeholder="••••••••")
 
        if st.button("CONNEXION AU TERMINAL", use_container_width=True):
            result = check_login(username, password)
            if result:
                st.session_state.auth = True
                st.session_state.user = username
                st.session_state.role = result["role"]
                st.session_state.full_name = result["name"]
                st.session_state.rag_db = load_rag_db()
                st.rerun()
            else:
                st.error("⛔  ACCÈS REFUSÉ — Identifiants incorrects")
 
        st.markdown('<div style="text-align:center; font-size:11px; color:#8b949e; margin-top:12px;">Comptes : chihi/1234 · admin/admin</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()
 
 
# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    # Profil utilisateur
    st.markdown(f"""
    <div style="padding:16px 0 12px;">
        <div style="width:48px;height:48px;border-radius:10px;
                    background:linear-gradient(135deg,#00e5a0,#00b8ff);
                    display:flex;align-items:center;justify-content:center;
                    font-family:'JetBrains Mono',monospace;font-weight:700;
                    font-size:16px;color:#000;margin-bottom:10px;
                    animation: float 3s ease-in-out infinite;">
            {st.session_state.full_name[:2].upper()}
        </div>
        <div style="font-size:16px;font-weight:700;color:#e6edf3;">{st.session_state.full_name}</div>
        <div style="font-size:11px;color:#00e5a0;letter-spacing:1px;margin-top:3px;">{st.session_state.role.upper()}</div>
    </div>
    <hr style="border-color:#1e2d3d;margin:0 0 12px;">
    """, unsafe_allow_html=True)
 
    # Navigation
    menu = st.radio(
        "MODULES",
        ["📊  DASHBOARD", "🤖  QWEN2.5 AI + RAG", "📈  SIGNAL LAB", "🔬  RAG PIPELINE", "⚙️  ADMIN"],
        label_visibility="collapsed",
    )
 
    st.markdown("<hr style='border-color:#1e2d3d;'>", unsafe_allow_html=True)
 
    # Statut RAG
    rag_ok = st.session_state.rag_db is not None
    st.markdown(f"""
    <div style="font-size:11px; color:#8b949e; margin-bottom:6px;">STATUT SYSTÈME</div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
        <span style="width:8px;height:8px;border-radius:50%;background:{'#00e5a0' if rag_ok else '#ff5f56'};display:inline-block;animation: pulse 2s ease-in-out infinite;"></span>
        <span style="font-size:12px;">ChromaDB : {'CONNECTÉ' if rag_ok else 'HORS LIGNE'}</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
        <span style="width:8px;height:8px;border-radius:50%;background:#00e5a0;display:inline-block;animation: pulse 2s ease-in-out infinite;"></span>
        <span style="font-size:12px;">Qwen2.5 : EN LIGNE</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
        <span style="width:8px;height:8px;border-radius:50%;background:#00e5a0;display:inline-block;animation: pulse 2s ease-in-out infinite;"></span>
        <span style="font-size:12px;">Auth : SHA-256</span>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("<hr style='border-color:#1e2d3d;'>", unsafe_allow_html=True)
 
    if st.button("DÉCONNEXION", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
 
 
# ─────────────────────────────────────────────────────────────
#  PAGE — DASHBOARD
# ─────────────────────────────────────────────────────────────
if menu == "📊  DASHBOARD":
    st.markdown('<div class="page-title">SYSTEM STATUS — DASHBOARD</div>', unsafe_allow_html=True)
 
    # Métriques
    col1, col2, col3, col4 = st.columns(4)
    db_docs = 0
    if st.session_state.rag_db:
        try:
            db_docs = st.session_state.rag_db._collection.count()
        except Exception:
            db_docs = 0
 
    with col1:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Chunks vectorisés</div><div class="metric-value neon">{db_docs:,}</div></div>', unsafe_allow_html=True)
    with col2:
        rag_status = "ONLINE" if st.session_state.rag_db else "OFFLINE"
        color = "neon" if st.session_state.rag_db else "red"
        st.markdown(f'<div class="metric-box"><div class="metric-label">ChromaDB</div><div class="metric-value {color}" style="font-size:18px;margin-top:8px;">{rag_status}</div></div>', unsafe_allow_html=True)
    with col3:
        nb_msg = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.markdown(f'<div class="metric-box"><div class="metric-label">Requêtes session</div><div class="metric-value blue">{nb_msg}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Modèle LLM</div><div class="metric-value amber" style="font-size:14px;margin-top:10px;">Qwen2.5:7b</div></div>', unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="elite-card">
            <div class="elite-card-title">PROFIL INGÉNIEUR</div>
            <table style="width:100%;font-size:13px;border-collapse:collapse;">
                <tr><td style="color:#8b949e;padding:7px 0;border-bottom:1px solid #1e2d3d;">Nom</td>
                    <td style="text-align:right;padding:7px 0;border-bottom:1px solid #1e2d3d;">{st.session_state.full_name}</td>
                </tr>
                <tr><td style="color:#8b949e;padding:7px 0;border-bottom:1px solid #1e2d3d;">Rôle</td>
                    <td style="text-align:right;padding:7px 0;border-bottom:1px solid #1e2d3d;">{st.session_state.role}</td>
                </tr>
                <tr><td style="color:#8b949e;padding:7px 0;border-bottom:1px solid #1e2d3d;">Moteur IA</td>
                    <td style="text-align:right;padding:7px 0;border-bottom:1px solid #1e2d3d;">Qwen2.5:7b</td>
                </tr>
                <tr><td style="color:#8b949e;padding:7px 0;border-bottom:1px solid #1e2d3d;">Embeddings</td>
                    <td style="text-align:right;padding:7px 0;border-bottom:1px solid #1e2d3d;">nomic-embed-text</td>
                </tr>
                <tr><td style="color:#8b949e;padding:7px 0;">Domaine</td>
                    <td style="text-align:right;padding:7px 0;">Embedded ML / Power Electronics</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
 
    with col_b:
        st.markdown("""
        <div class="elite-card">
            <div class="elite-card-title">ARCHITECTURE TECHNIQUE</div>
            <table style="width:100%;font-size:13px;border-collapse:collapse;">
                <tr><td style="color:#8b949e;padding:7px 0;border-bottom:1px solid #1e2d3d;">Loader</td>
                    <td style="text-align:right;padding:7px 0;border-bottom:1px solid #1e2d3d;">PyPDFDirectoryLoader</td>
                </tr>
                <tr><td style="color:#8b949e;padding:7px 0;border-bottom:1px solid #1e2d3d;">Splitter</td>
                    <td style="text-align:right;padding:7px 0;border-bottom:1px solid #1e2d3d;">RecursiveCharacterTextSplitter</td>
                </tr>
                <tr><td style="color:#8b949e;padding:7px 0;border-bottom:1px solid #1e2d3d;">Vector Store</td>
                    <td style="text-align:right;padding:7px 0;border-bottom:1px solid #1e2d3d;">ChromaDB (persistant)</td>
                </tr>
                <tr><td style="color:#8b949e;padding:7px 0;border-bottom:1px solid #1e2d3d;">Auth</td>
                    <td style="text-align:right;padding:7px 0;border-bottom:1px solid #1e2d3d;">SQLite + SHA-256</td>
                </tr>
                <tr><td style="color:#8b949e;padding:7px 0;">Chunk size</td>
                    <td style="text-align:right;padding:7px 0;">1000 tokens / overlap 150</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
 
    # Historique session
    if st.session_state.messages:
        st.markdown('<div class="elite-card"><div class="elite-card-title">DERNIÈRES INTERACTIONS</div>', unsafe_allow_html=True)
        for msg in st.session_state.messages[-6:]:
            role_label = "INGÉNIEUR" if msg["role"] == "user" else "TECNOBOT"
            color = "#00e5a0" if msg["role"] == "user" else "#00b8ff"
            preview = msg["content"][:120] + "..." if len(msg["content"]) > 120 else msg["content"]
            st.markdown(f'<div style="padding:8px 0;border-bottom:1px solid #1e2d3d;font-size:12px;"><span style="color:{color};font-family:JetBrains Mono,monospace;font-size:10px;">{role_label}</span><br/>{preview}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────
#  PAGE — QWEN2.5 AI + RAG
# ─────────────────────────────────────────────────────────────
elif menu == "🤖  QWEN2.5 AI + RAG":
    st.markdown('<div class="page-title">QWEN2.5 ENGINE — MODE RAG ACTIVÉ</div>', unsafe_allow_html=True)
 
    rag_label = "🟢 RAG ACTIF" if st.session_state.rag_db else "🔴 RAG HORS LIGNE (mode général)"
    st.markdown(f'<span class="badge {"badge-green" if st.session_state.rag_db else "badge-red"}">{rag_label}</span>', unsafe_allow_html=True)
 
    # Affichage de l'historique
    st.markdown("<br>", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user"><strong style="color:#00e5a0;font-size:11px;">INGÉNIEUR</strong><br/>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            sources_html = ""
            if msg.get("sources"):
                srcs = msg["sources"]
                sources_html = '<div class="chat-source">📚 SOURCES RAG : ' + " | ".join(
                    [f'{s["file"]} p.{s["page"]} (score:{s["score"]})' for s in srcs]
                ) + "</div>"
            st.markdown(f'<div class="chat-bot"><strong style="color:#00b8ff;font-size:11px;">TECNOBOT ELITE</strong><br/>{msg["content"]}{sources_html}</div>', unsafe_allow_html=True)
 
    # Exemples de questions
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("💡 EXEMPLES DE REQUÊTES TECHNIQUES"):
        exemples = [
            "Diagnostiquer une panne IGBT en pont H",
            "Expliquer la loi d'Ohm en régime sinusoïdal",
            "Causes d'échauffement d'un moteur asynchrone",
            "Comment calculer le rapport cyclique d'un convertisseur boost ?",
            "Filtrer les harmoniques d'ordre 3 dans un réseau triphasé",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(exemples):
            with cols[i % 2]:
                if st.button(ex, key=f"ex_{i}"):
                    st.session_state._pending_question = ex
                    st.rerun()
 
    # Champ de saisie
    prompt = st.chat_input("Posez votre question technique... (électrotechnique, diagnostic, ML embarqué)")
 
    # Si question depuis les exemples
    if hasattr(st.session_state, "_pending_question"):
        prompt = st.session_state._pending_question
        del st.session_state._pending_question
 
    if prompt:
        # Enregistrement question utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message(st.session_state.user, "user", prompt)
        st.markdown(f'<div class="chat-user"><strong style="color:#00e5a0;font-size:11px;">INGÉNIEUR</strong><br/>{prompt}</div>', unsafe_allow_html=True)
 
        with st.spinner("⚙️ Recherche RAG + Génération Qwen2.5..."):
            # 1. Récupération du contexte RAG
            context, sources = retrieve_context(st.session_state.rag_db, prompt)
 
            # 2. Appel LLM
            try:
                answer = ask_qwen(
                    query=prompt,
                    context=context,
                    history=st.session_state.messages[:-1],
                )
            except Exception as e:
                answer = f"❌ Erreur Ollama : {e}\n\nVérifiez qu'Ollama est lancé avec `ollama serve`."
                sources = []
 
        # Enregistrement et affichage réponse
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
        save_message(st.session_state.user, "assistant", answer,
                     str(sources) if sources else "")
 
        sources_html = ""
        if sources:
            sources_html = '<div class="chat-source">📚 SOURCES RAG : ' + " | ".join(
                [f'{s["file"]} p.{s["page"]} (score:{s["score"]})' for s in sources]
            ) + "</div>"
 
        st.markdown(f'<div class="chat-bot"><strong style="color:#00b8ff;font-size:11px;">TECNOBOT ELITE</strong><br/>{answer}{sources_html}</div>', unsafe_allow_html=True)
        st.rerun()
 
    if st.button("🗑️ Effacer la conversation", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()
 
 
# ─────────────────────────────────────────────────────────────
#  PAGE — SIGNAL LAB
# ─────────────────────────────────────────────────────────────
elif menu == "📈  SIGNAL LAB":
    st.markdown('<div class="page-title">SIGNAL LAB — LABORATOIRE DE SIMULATION</div>', unsafe_allow_html=True)
 
    tab1, tab2, tab3, tab4 = st.tabs(["⚡ Signal AC", "🔋 Signal DC", "🔁 Triphasé", "📐 Loi d'Ohm"])
 
    # ── Tab 1 : AC ─────────────────────────────────────────────
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            voltage = st.slider("Tension efficace (V)", 0, 400, 230, key="v_ac")
        with col2:
            freq = st.slider("Fréquence (Hz)", 10, 200, 50, key="f_ac")
        with col3:
            harmonic = st.slider("Harmonique rang 3 (%)", 0, 50, 0, key="h_ac")
 
        t = np.linspace(0, 4 / freq, 1000)
        Vp = voltage * np.sqrt(2)
        y = Vp * np.sin(2 * np.pi * freq * t)
        if harmonic > 0:
            y += Vp * (harmonic / 100) * np.sin(2 * np.pi * 3 * freq * t) * 0.3
 
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t * 1000, y=y, name="V(t)", line=dict(color="#00e5a0", width=2)))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,17,23,0.9)",
            title=f"Signal AC — {voltage}V / {freq}Hz",
            xaxis_title="Temps (ms)",
            yaxis_title="Tension (V)",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
 
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Vp (crête)", f"{Vp:.1f} V")
        col_b.metric("Vrms", f"{voltage} V")
        col_c.metric("Période T", f"{1000/freq:.1f} ms")
        col_d.metric("Pulsation ω", f"{2*np.pi*freq:.1f} rad/s")
 
    # ── Tab 2 : DC ─────────────────────────────────────────────
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            v_dc = st.slider("Tension DC (V)", 0, 800, 400, key="v_dc")
        with col2:
            ripple = st.slider("Ondulation (%) ", 0, 30, 5, key="r_dc")
 
        t = np.linspace(0, 0.1, 1000)
        y_dc = np.full_like(t, v_dc)
        if ripple > 0:
            y_dc += v_dc * (ripple / 100) * np.sin(2 * np.pi * 120 * t) * 0.5
 
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=t * 1000, y=y_dc, name="V_DC", line=dict(color="#00b8ff", width=2)))
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,17,23,0.9)",
            title=f"Signal DC — {v_dc}V (ondulation {ripple}%)",
            xaxis_title="Temps (ms)",
            yaxis_title="Tension (V)",
            height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)
 
    # ── Tab 3 : Triphasé ───────────────────────────────────────
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            v3 = st.slider("Tension de ligne (V)", 0, 690, 400, key="v_3ph")
        with col2:
            f3 = st.slider("Fréquence (Hz)", 10, 200, 50, key="f_3ph")
 
        Vp3 = v3 * np.sqrt(2) / np.sqrt(3)
        t3 = np.linspace(0, 4 / f3, 1000)
        colors = ["#00e5a0", "#00b8ff", "#ffa940"]
        phases = ["Phase A", "Phase B", "Phase C"]
 
        fig3 = go.Figure()
        for i, (color, name) in enumerate(zip(colors, phases)):
            y_ph = Vp3 * np.sin(2 * np.pi * f3 * t3 - i * 2 * np.pi / 3)
            fig3.add_trace(go.Scatter(x=t3 * 1000, y=y_ph, name=name,
                                      line=dict(color=color, width=2)))
 
        fig3.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,17,23,0.9)",
            title=f"Système Triphasé — {v3}V / {f3}Hz",
            xaxis_title="Temps (ms)",
            yaxis_title="Tension (V)",
            height=380,
        )
        st.plotly_chart(fig3, use_container_width=True)
 
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Tension de phase", f"{Vp3/np.sqrt(2):.1f} V")
        col_b.metric("Tension de ligne", f"{v3} V")
        col_c.metric("Puissance (cos=1, R=10Ω)", f"{3*(v3/np.sqrt(3))**2/10:.0f} W")
 
    # ── Tab 4 : Loi d'Ohm ─────────────────────────────────────
    with tab4:
        col1, col2, col3 = st.columns(3)
        with col1:
            v_ohm = st.slider("Tension (V)", 1, 400, 230, key="v_ohm")
        with col2:
            r_ohm = st.slider("Résistance R (Ω)", 1, 1000, 100, key="r_ohm")
        with col3:
            f_ohm = st.slider("Fréquence (Hz)", 10, 200, 50, key="f_ohm")
 
        t_ohm = np.linspace(0, 4 / f_ohm, 1000)
        Vp_ohm = v_ohm * np.sqrt(2)
        v_signal = Vp_ohm * np.sin(2 * np.pi * f_ohm * t_ohm)
        i_signal = v_signal / r_ohm
 
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=t_ohm * 1000, y=v_signal, name="V(t) [V]",
                                  line=dict(color="#00e5a0", width=2)))
        fig4.add_trace(go.Scatter(x=t_ohm * 1000, y=i_signal * r_ohm / 10, name="I(t)×10 [A scalé]",
                                  line=dict(color="#00b8ff", width=2, dash="dash")))
        fig4.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,17,23,0.9)",
            title=f"Loi d'Ohm — V={v_ohm}V / R={r_ohm}Ω / f={f_ohm}Hz",
            xaxis_title="Temps (ms)",
            yaxis_title="Amplitude",
            height=350,
        )
        st.plotly_chart(fig4, use_container_width=True)
 
        I_rms = v_ohm / r_ohm
        P = v_ohm ** 2 / r_ohm
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("V (rms)", f"{v_ohm} V")
        col_b.metric("I (rms)", f"{I_rms:.3f} A")
        col_c.metric("R", f"{r_ohm} Ω")
        col_d.metric("Puissance", f"{P:.1f} W")
 
 
# ─────────────────────────────────────────────────────────────
#  PAGE — RAG PIPELINE
# ─────────────────────────────────────────────────────────────
elif menu == "🔬  RAG PIPELINE":
    st.markdown('<div class="page-title">RAG PIPELINE — ARCHITECTURE DE CONNAISSANCES</div>', unsafe_allow_html=True)
 
    st.markdown("""
    <div class="elite-card">
        <div class="elite-card-title">FLUX DE VECTORISATION</div>
        <div class="rag-bar">
            <div class="rag-step">
                <div style="font-size:22px;">📄</div>
                <div class="rag-step-title">INGESTION</div>
                <div class="rag-step-sub">PyPDFDirectoryLoader</div>
            </div>
            <div class="rag-arrow">→</div>
            <div class="rag-step">
                <div style="font-size:22px;">✂️</div>
                <div class="rag-step-title">CHUNKING</div>
                <div class="rag-step-sub">size=1000 / overlap=150</div>
            </div>
            <div class="rag-arrow">→</div>
            <div class="rag-step">
                <div style="font-size:22px;">🧠</div>
                <div class="rag-step-title">EMBEDDING</div>
                <div class="rag-step-sub">nomic-embed-text (Ollama)</div>
            </div>
            <div class="rag-arrow">→</div>
            <div class="rag-step">
                <div style="font-size:22px;">🗄️</div>
                <div class="rag-step-title">STOCKAGE</div>
                <div class="rag-step-sub">ChromaDB (persistant)</div>
            </div>
            <div class="rag-arrow">→</div>
            <div class="rag-step">
                <div style="font-size:22px;">🔍</div>
                <div class="rag-step-title">RETRIEVAL</div>
                <div class="rag-step-sub">Similarité cosinus (k=4)</div>
            </div>
            <div class="rag-arrow">→</div>
            <div class="rag-step">
                <div style="font-size:22px;">⚡</div>
                <div class="rag-step-title">GÉNÉRATION</div>
                <div class="rag-step-sub">Qwen2.5:7b + contexte</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    # Test de recherche RAG en direct
    st.markdown('<div class="elite-card"><div class="elite-card-title">TEST DE RECHERCHE SÉMANTIQUE</div>', unsafe_allow_html=True)
    test_q = st.text_input("Requête de test", placeholder="Ex: panne onduleur IGBT")
    if st.button("🔍 Lancer la recherche RAG") and test_q:
        if st.session_state.rag_db:
            with st.spinner("Recherche en cours..."):
                results = st.session_state.rag_db.similarity_search_with_relevance_scores(test_q, k=5)
            for doc, score in results:
                color = "#00e5a0" if score > 0.6 else "#ffa940" if score > 0.3 else "#ff5f56"
            st.markdown(f"""
                <div style="background:rgba(13,17,23,0.9);border:1px solid #1e2d3d;
                            border-radius:8px;padding:12px;margin:8px 0;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                        <span style="font-family:JetBrains Mono,monospace;font-size:11px;color:#8b949e;">
                            📄 {doc.metadata.get('source','?')} — Page {doc.metadata.get('page','?')}
                        </span>
                        <span style="font-family:JetBrains Mono,monospace;font-size:11px;color:{color};">
                            SCORE : {score:.3f}
                        </span>
                    </div>
                    <div style="font-size:12px;color:#c9d1d9;line-height:1.6;">
                        {doc.page_content[:300]}...
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("ChromaDB non connectée. Lance d'abord : python 1_vectorisation.py")
    st.markdown("</div>", unsafe_allow_html=True)
 
    # Paramètres RAG
    st.markdown(f"""
    <div class="elite-card">
        <div class="elite-card-title">PARAMÈTRES RAG ACTUELS</div>
        <table style="width:100%;font-size:13px;border-collapse:collapse;">
            <tr><td style="color:#8b949e;padding:7px 0;border-bottom:1px solid #1e2d3d;">Chunks retournés (k)</td>
                <td style="text-align:right;color:#00e5a0;font-family:JetBrains Mono,monospace;">{RAG_K}</td>
            </tr>
            <tr><td style="color:#8b949e;padding:7px 0;border-bottom:1px solid #1e2d3d;">Score minimal</td>
                <td style="text-align:right;color:#00e5a0;font-family:JetBrains Mono,monospace;">{RAG_SCORE}</td>
            </tr>
            <tr><td style="color:#8b949e;padding:7px 0;border-bottom:1px solid #1e2d3d;">Chunk size</td>
                <td style="text-align:right;color:#00e5a0;font-family:JetBrains Mono,monospace;">1000 tokens</td>
            </tr>
            <tr><td style="color:#8b949e;padding:7px 0;">Chevauchement</td>
                <td style="text-align:right;color:#00e5a0;font-family:JetBrains Mono,monospace;">150 tokens</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────
#  PAGE — ADMIN
# ─────────────────────────────────────────────────────────────
elif menu == "⚙️  ADMIN":
    st.markdown('<div class="page-title">ADMINISTRATION SYSTÈME</div>', unsafe_allow_html=True)
 
    if st.session_state.role != "Administrateur":
        st.error("⛔ Accès réservé aux administrateurs.")
        st.stop()
 
    tab_users, tab_history = st.tabs(["👥 Gestion Utilisateurs", "📜 Historique Conversations"])
 
    with tab_users:
        st.markdown("**Ajouter un utilisateur**")
        col1, col2, col3, col4 = st.columns(4)
        with col1: new_user = st.text_input("Username")
        with col2: new_pass = st.text_input("Mot de passe", type="password")
        with col3: new_role = st.selectbox("Rôle", ["Expert Technique", "Embedded ML Engineer", "Administrateur", "Étudiant"])
        with col4: new_name = st.text_input("Nom complet")
 
        if st.button("➕ Ajouter"):
            if new_user and new_pass and new_name:
                conn = sqlite3.connect(str(DB_FILE))
                c = conn.cursor()
                try:
                    c.execute("INSERT INTO users VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
                              (new_user, hash_pw(new_pass), new_role, new_name))
                    conn.commit()
                    st.success(f"✅ Utilisateur {new_user} créé.")
                except sqlite3.IntegrityError:
                    st.error("Utilisateur déjà existant.")
                finally:
                    conn.close()
            else:
                st.warning("Remplis tous les champs.")
 
        st.markdown("**Utilisateurs enregistrés**")
        conn = sqlite3.connect(str(DB_FILE))
        import pandas as pd
        df_users = pd.read_sql("SELECT username, role, name, created FROM users", conn)
        conn.close()
        st.dataframe(df_users, use_container_width=True)
 
    with tab_history:
        conn = sqlite3.connect(str(DB_FILE))
        df_hist = pd.read_sql(
            "SELECT username, role, substr(content,1,80)||'...' AS apercu, timestamp FROM chat_history ORDER BY id DESC LIMIT 50",
            conn,
        )
        conn.close()
        st.dataframe(df_hist, use_container_width=True)