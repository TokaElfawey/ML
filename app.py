import streamlit as st
import re
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── إعدادات الصفحة الفخمة ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NewsIQ | AI Intelligence",
    page_icon="🔮",
    layout="wide",
)

# ── التصميم الخارق (Advanced CSS) ────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&family=Playfair+Display:ital,wght@0,900;1,900&display=swap');
    
    /* الخلفية والتنسيق العام */
    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b, #0f172a, #020617);
        color: #f8fafc;
    }

    /* شريط التنقل */
    .nav-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 20px 0;
        margin-bottom: 40px;
    }
    .nav-logo {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: -1px;
        background: linear-gradient(to right, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* العنوان الرئيسي الهيرو */
    .hero-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        font-size: 4.5rem;
        text-align: center;
        margin-bottom: 10px;
        line-height: 1.1;
        background: linear-gradient(135deg, #fff 30%, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 60px;
        font-weight: 300;
    }

    /* بطاقات Glassmorphism */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 35px;
        transition: all 0.4s ease;
        box-shadow: 0 20px 50px rgba(0,0,0,0.3);
    }
    .glass-card:hover {
        border-color: rgba(129, 140, 248, 0.4);
        transform: translateY(-5px);
    }

    /* مدخلات النصوص */
    .stTextArea textarea {
        background: rgba(0, 0, 0, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        color: #f8fafc !important;
        font-size: 1.1rem !important;
    }

    /* الزر الفخم */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 18px 30px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3) !important;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 30px rgba(99, 102, 241, 0.5) !important;
    }

    /* نتائج التصنيف */
    .result-label {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #818cf8;
        margin-bottom: 10px;
    }
    .category-display {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 900;
        margin: 10px 0;
    }
    
    /* ألوان التصنيفات */
    .world { color: #38bdf8; text-shadow: 0 0 20px rgba(56,189,248,0.4); }
    .sports { color: #4ade80; text-shadow: 0 0 20px rgba(74,222,128,0.4); }
    .business { color: #fbbf24; text-shadow: 0 0 20px rgba(251,191,36,0.4); }
    .tech { color: #f472b6; text-shadow: 0 0 20px rgba(244,114,182,0.4); }

    /* إخفاء عناصر streamlit الافتراضية */
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── الدوال البرمجية (كما هي لضمان العمل) ────────────────────────────────────────

@st.cache_resource
def load_assets():
    try:
        tfidf = joblib.load("tfidf_vectorizer.joblib")
        le = joblib.load("label_encoder.joblib")
        models = {
            "⚡ Linear SVC (Extreme Accuracy)": joblib.load("linear_svc_model.joblib"),
            "🧠 Logistic Regression (Balanced)": joblib.load("logistic_regression_model.joblib"),
            "📊 Naive Bayes (Lightweight)": joblib.load("naive_bayes_model.joblib")
        }
        return tfidf, le, models
    except:
        return None, None, None

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

tfidf, le, models = load_assets()

# ── هيكل الصفحة الفخم ──────────────────────────────────────────────────────────────

# Navbar
st.markdown('<div class="nav-bar"><div class="nav-logo">NewsIQ.ai</div></div>', unsafe_allow_html=True)

# Hero
st.markdown('<h1 class="hero-title">Future of News<br>Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Experience the power of neural text classification with high-precision models.</p>', unsafe_allow_html=True)

container = st.container()
with container:
    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p style="color:#94a3b8; font-weight:600; margin-bottom:15px;">INPUT ARTICLE</p>', unsafe_allow_html=True)
        input_text = st.text_area("", placeholder="Paste your news headline or full article content here...", height=280, label_visibility="collapsed")
        
        st.markdown('<p style="color:#94a3b8; font-weight:600; margin-top:25px;">SELECT INTELLIGENCE ENGINE</p>', unsafe_allow_html=True)
        model_choice = st.selectbox("", list(models.keys()) if models else ["Loading Assets..."], label_visibility="collapsed")
        
        st.markdown('<div style="margin-top:30px;">', unsafe_allow_html=True)
        analyze_btn = st.button("RUN ANALYSIS", use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    with col2:
        if analyze_btn and input_text:
            if tfidf and models:
                with st.spinner("Decoding Neural Patterns..."):
                    time.sleep(0.8)
                    
                    # Logic
                    cleaned = clean_text(input_text)
                    vec = tfidf.transform([cleaned])
                    model = models[model_choice]
                    pred_idx = model.predict(vec)[0]
                    category = le.inverse_transform([pred_idx])[0]
                    
                    # Confidence Logic
                    if hasattr(model, "predict_proba"):
                        prob = model.predict_proba(vec)[0][pred_idx]
                    else:
                        d = model.decision_function(vec)[0]
                        prob = (np.exp(d) / np.sum(np.exp(d)))[pred_idx]

                    cat_class = category.lower().replace("/", "").replace("sci", "tech")
                    
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="result-label">Analysis Complete via {model_choice.split(' ')[1]}</div>
                        <div class="category-display {cat_class}">{category}</div>
                        <div style="display:flex; align-items:center; gap:10px; margin-top:10px;">
                            <div style="flex-grow:1; height:8px; background:rgba(255,255,255,0.1); border-radius:10px;">
                                <div style="width:{prob*100}%; height:100%; background:linear-gradient(to right, #818cf8, #c084fc); border-radius:10px;"></div>
                            </div>
                            <span style="font-family:monospace; color:#818cf8; font-weight:700;">{prob*100:.1f}%</span>
                        </div>
                        <p style="color:#64748b; font-size:0.9rem; margin-top:20px;">
                            Our AI has identified this content with high confidence. The linguistic patterns strongly align with <b>{category}</b> news markers.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mini Plot
                    fig, ax = plt.subplots(figsize=(6, 3))
                    plt.rcParams['text.color'] = '#94a3b8'
                    ax.barh(["Other", category], [1-prob, prob], color=['#1e293b', '#6366f1'], height=0.6)
                    ax.set_facecolor('none')
                    fig.patch.set_alpha(0)
                    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
                    ax.set_xticks([])
                    st.pyplot(fig)

            else:
                st.error("Missing Intelligence Files (Joblib).")
        else:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; padding:100px 40px; border-style:dashed; opacity:0.6;">
                <div style="font-size:4rem; margin-bottom:20px;">📡</div>
                <h3 style="font-family:'Plus Jakarta Sans'; font-weight:600;">System Ready</h3>
                <p style="color:#64748b;">Awaiting input data for neural processing.</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center; margin-top:80px; padding:40px; color:#475569; font-size:0.8rem; letter-spacing:1px;">
    NEWSIQ PRO · POWERED BY ADVANCED NLP MODELS · SECTION 1 TEAM
</div>
""", unsafe_allow_html=True)
