import streamlit as st
import re
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import docx2txt
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ── 1. Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NewsAI | Advanced Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Custom CSS (Dark Tech / Glassmorphism) ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600;700&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    color: #e0eaff;
}

/* ── Animated Background ── */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 50%, #0d1b4b 0%, #050c1f 40%, #000510 100%);
    min-height: 100vh;
}

/* ── Animated floating dots ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(2px 2px at 10% 20%, rgba(0,200,255,0.4) 0%, transparent 100%),
        radial-gradient(2px 2px at 80% 10%, rgba(100,100,255,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 50% 80%, rgba(0,200,255,0.2) 0%, transparent 100%),
        radial-gradient(2px 2px at 90% 60%, rgba(150,0,255,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 30% 90%, rgba(0,150,255,0.3) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(5, 15, 50, 0.85) !important;
    border-right: 1px solid rgba(0, 200, 255, 0.2) !important;
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] * { color: #c8d8ff !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #8899cc !important; font-size: 0.85rem !important; }

/* ── Header ── */
.main-header {
    text-align: center;
    padding: 30px 20px 10px;
    position: relative;
}
.main-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00d4ff 0%, #7b6fff 50%, #ff6fff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 2px;
    margin-bottom: 8px;
    text-shadow: none;
}
.main-subtitle {
    color: #6a80b0;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 4px;
    text-transform: uppercase;
}

/* ── Glowing Divider ── */
.glow-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, #7b6fff, transparent);
    margin: 15px auto;
    max-width: 600px;
    border-radius: 2px;
}

/* ── Cards ── */
.glass-card {
    background: rgba(10, 25, 80, 0.6);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 200, 255, 0.15);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(0, 100, 200, 0.1), inset 0 1px 0 rgba(255,255,255,0.05);
}

/* ── Text Area ── */
.stTextArea textarea {
    background: rgba(5, 15, 50, 0.8) !important;
    border: 1px solid rgba(0, 200, 255, 0.2) !important;
    border-radius: 12px !important;
    color: #c8d8ff !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.95rem !important;
    resize: vertical;
}
.stTextArea textarea:focus {
    border-color: rgba(0, 200, 255, 0.6) !important;
    box-shadow: 0 0 20px rgba(0, 200, 255, 0.1) !important;
}
.stTextArea textarea::placeholder { color: #445577 !important; }

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(5, 15, 50, 0.5) !important;
    border: 2px dashed rgba(0, 200, 255, 0.25) !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

/* ── Radio Buttons ── */
.stRadio > div {
    display: flex;
    gap: 10px;
    flex-direction: row !important;
}
.stRadio label {
    background: rgba(10, 25, 80, 0.7) !important;
    border: 1px solid rgba(0, 200, 255, 0.2) !important;
    border-radius: 30px !important;
    padding: 6px 20px !important;
    cursor: pointer;
    transition: all 0.3s;
    font-size: 0.9rem !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(5, 15, 50, 0.8) !important;
    border: 1px solid rgba(0, 200, 255, 0.2) !important;
    border-radius: 10px !important;
    color: #c8d8ff !important;
}

/* ── Main Button ── */
.stButton > button {
    background: linear-gradient(135deg, #0062ff 0%, #00c8ff 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 40px !important;
    padding: 14px 30px !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    box-shadow: 0 0 30px rgba(0, 150, 255, 0.4), 0 4px 15px rgba(0, 100, 200, 0.3) !important;
    transition: all 0.3s !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 50px rgba(0, 200, 255, 0.6) !important;
}

/* ── Result Card ── */
.result-card {
    background: rgba(5, 20, 70, 0.8);
    border: 1px solid rgba(0, 200, 255, 0.3);
    border-left: 4px solid #00d4ff;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 0 30px rgba(0, 200, 255, 0.1);
}
.result-label {
    font-size: 0.75rem;
    color: #445577;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 8px;
}
.result-category {
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 12px;
}
.cat-world    { color: #60b0ff; text-shadow: 0 0 20px rgba(96,176,255,0.5); }
.cat-sports   { color: #4ecf8a; text-shadow: 0 0 20px rgba(78,207,138,0.5); }
.cat-business { color: #f5a623; text-shadow: 0 0 20px rgba(245,166,35,0.5); }
.cat-tech     { color: #e060f0; text-shadow: 0 0 20px rgba(224,96,240,0.5); }

.confidence-badge {
    display: inline-block;
    background: rgba(0, 200, 255, 0.15);
    border: 1px solid rgba(0, 200, 255, 0.3);
    border-radius: 20px;
    padding: 4px 16px;
    color: #00d4ff;
    font-weight: 700;
    font-size: 0.95rem;
}

/* ── Sidebar Stats Card ── */
.stat-card {
    background: rgba(0, 200, 255, 0.05);
    border: 1px solid rgba(0, 200, 255, 0.1);
    border-radius: 12px;
    padding: 14px;
    margin-top: 10px;
}
.stat-label { font-size: 0.75rem; color: #445577; text-transform: uppercase; letter-spacing: 2px; }
.stat-value { font-size: 1rem; font-weight: 600; color: #4ecf8a; margin-top: 4px; }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 30px;
    border-top: 1px solid rgba(0, 200, 255, 0.1);
    margin-top: 40px;
    color: #445577;
    font-size: 0.9rem;
    line-height: 1.8;
}
.footer b { color: #00d4ff; }

/* ── Section Headings ── */
h3 { font-family: 'Exo 2', sans-serif !important; font-weight: 600 !important; color: #8899cc !important; font-size: 0.9rem !important; text-transform: uppercase !important; letter-spacing: 2px !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #00d4ff !important; }

/* ── Success / Error ── */
.stAlert { border-radius: 10px !important; background: rgba(5, 50, 20, 0.6) !important; border: 1px solid rgba(0, 200, 100, 0.3) !important; }

/* ── Hide streamlit branding ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── 3. Load Models ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    try:
        tfidf  = joblib.load("tfidf_vectorizer.joblib")
        le     = joblib.load("label_encoder.joblib")
        models = {
            "Linear SVC":           joblib.load("linear_svc_model.joblib"),
            "Logistic Regression":  joblib.load("logistic_regression_model.joblib"),
            "Naive Bayes":          joblib.load("naive_bayes_model.joblib"),
        }
        return tfidf, le, models
    except Exception:
        return None, None, None

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

MODEL_INFO = {
    "Linear SVC":          {"desc": "Best for Accuracy 🎯",  "color": "#00d4ff"},
    "Logistic Regression": {"desc": "Balanced Speed 🔄",     "color": "#7b6fff"},
    "Naive Bayes":         {"desc": "Fastest Model ⚡",       "color": "#4ecf8a"},
}

CAT_CLASS = {
    "World":    "cat-world",
    "Sports":   "cat-sports",
    "Business": "cat-business",
    "Sci/Tech": "cat-tech",
}

BAR_COLORS = {
    "World":    "#60b0ff",
    "Sports":   "#4ecf8a",
    "Business": "#f5a623",
    "Sci/Tech": "#e060f0",
}

# ── 4. Main App ─────────────────────────────────────────────────────────────────
def main():
    tfidf, le, models = load_assets()

    # ── Sidebar ──────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 10px 0 20px;'>
            <div style='font-size:1.8rem; margin-bottom:6px;'>⚙️</div>
            <div style='font-family:Orbitron,monospace; font-size:1.2rem; font-weight:700;
                        background:linear-gradient(135deg,#00d4ff,#7b6fff);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
                Settings
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Choose AI Model**")
        model_options = list(MODEL_INFO.keys()) if models else ["Assets Missing"]
        selected_model = st.selectbox("", model_options, label_visibility="collapsed")

        if models and selected_model in MODEL_INFO:
            info = MODEL_INFO[selected_model]
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Model Stats</div>
                <div class="stat-value">{info['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Category legend
        st.markdown("<div style='font-size:0.75rem; color:#445577; text-transform:uppercase; letter-spacing:2px; margin-bottom:10px;'>Categories</div>", unsafe_allow_html=True)
        for cat, col in BAR_COLORS.items():
            st.markdown(f"<div style='color:{col}; font-size:0.9rem; margin:4px 0;'>● {cat}</div>", unsafe_allow_html=True)

    # ── Header ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <div class="main-title">NewsAI | Advanced Classifier 🤖</div>
        <div class="glow-divider"></div>
        <div class="main-subtitle">Enterprise NLP System for News Classification</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Content Columns ───────────────────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2], gap="large")

    # ────── LEFT : Input ──────────────────────────────────────────────────────────
    with left_col:
        input_mode = st.radio(
            "Input Method",
            ["✏️ Type/Paste Text", "📁 Upload Document"],
            horizontal=True,
            label_visibility="collapsed",
        )

        input_text = ""

        if input_mode == "✏️ Type/Paste Text":
            input_text = st.text_area(
                "Article",
                placeholder="Paste news article content here...",
                height=220,
                label_visibility="collapsed",
            )
        else:
            accept = ['pdf', 'docx', 'txt']
            file = st.file_uploader("Upload Document", type=accept, label_visibility="collapsed")
            if file:
                try:
                    if file.type == "application/pdf" and PDF_AVAILABLE:
                        reader = PdfReader(file)
                        input_text = " ".join(
                            p.extract_text() for p in reader.pages if p.extract_text()
                        )
                    elif file.type == "text/plain":
                        input_text = file.read().decode("utf-8")
                    elif DOCX_AVAILABLE:
                        input_text = docx2txt.process(file)
                    else:
                        input_text = file.read().decode("utf-8", errors="ignore")
                    st.success("✅ Document loaded successfully!")
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🤖  RUN AI ANALYSIS", use_container_width=True)

    # ────── RIGHT : Results ───────────────────────────────────────────────────────
    with right_col:
        if analyze_btn and input_text.strip():
            if tfidf and models:
                with st.spinner("Analyzing neural patterns..."):
                    time.sleep(0.6)
                    cleaned   = clean_text(input_text)
                    vec       = tfidf.transform([cleaned])
                    model     = models[selected_model]
                    pred_idx  = model.predict(vec)[0]
                    category  = le.inverse_transform([pred_idx])[0]

                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(vec)[0]
                    else:
                        d = model.decision_function(vec)[0]
                        exp_d = np.exp(d - np.max(d))
                        probs = exp_d / exp_d.sum()

                    conf_scores = {c: p for c, p in zip(le.classes_, probs)}
                    color_class = CAT_CLASS.get(category, "")
                    confidence  = conf_scores[category] * 100

                    # Result card
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label">Classification Result</div>
                        <div class="result-category {color_class}">{category}</div>
                        <div class="confidence-badge">Confidence: {confidence:.1f}%</div>
                        <p style="margin-top:14px; color:#6a80b0; font-size:0.85rem; line-height:1.6;">
                            The model classified this article as <strong style="color:#c8d8ff;">{category}</strong> news
                            with <strong style="color:#00d4ff;">{confidence:.1f}%</strong> confidence
                            using {selected_model}.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Probability chart
                    st.markdown("### Probability Distribution")
                    cats   = list(conf_scores.keys())
                    vals   = list(conf_scores.values())
                    colors = [BAR_COLORS.get(c, "#8899cc") for c in cats]

                    fig, ax = plt.subplots(figsize=(5, 3))
                    fig.patch.set_facecolor("none")
                    ax.set_facecolor("#050c1f")

                    bars = ax.barh(cats, vals, color=colors, height=0.55,
                                   edgecolor="none")

                    # Glow effect using wider, transparent bars behind
                    for bar, col in zip(bars, colors):
                        ax.barh(bar.get_y() + bar.get_height() / 2,
                                bar.get_width(), height=0.7,
                                color=col, alpha=0.15, left=0)

                    ax.set_xlim(0, 1.18)
                    ax.tick_params(axis='both', colors='#6a80b0', labelsize=10)
                    ax.xaxis.set_visible(False)
                    for spine in ax.spines.values():
                        spine.set_visible(False)

                    for bar, col in zip(bars, colors):
                        w = bar.get_width()
                        ax.text(w + 0.03, bar.get_y() + bar.get_height() / 2,
                                f"{w*100:.1f}%",
                                va='center', color=col,
                                fontweight='bold', fontsize=10)

                    ax.set_yticks(range(len(cats)))
                    ax.set_yticklabels(cats, color='#8899cc', fontsize=11)

                    plt.tight_layout(pad=0.5)
                    st.pyplot(fig, use_container_width=True)

            elif not tfidf:
                st.error("⚠️ Model assets not found. Please ensure .joblib files are in the app directory.")

        elif analyze_btn and not input_text.strip():
            st.warning("⚠️ Please enter or upload some article text first.")

        else:
            st.markdown("""
            <div style="
                text-align:center;
                padding: 60px 20px;
                border: 2px dashed rgba(0,200,255,0.15);
                border-radius: 16px;
                color: #2a3a60;
            ">
                <div style="font-size:3.5rem; margin-bottom:12px;">🧠</div>
                <div style="font-size:0.9rem; letter-spacing:2px; text-transform:uppercase;">
                    Awaiting article input...
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────────────────────────
    st.markdown("<br><hr style='border-color:rgba(0,200,255,0.1);'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
        Developed with ❤️ by <b>Section 1&3 Team</b><br>
        <span>Aya Ahmed | Toka Nasr | Toka Alaa | Hemmat Hamdi | Nourhan Medhat</span><br>
        <span style="font-size:0.8rem; color:#2a3a60;">© 2026 Academic Deployment Project</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
