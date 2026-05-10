import streamlit as st
import re
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── إعدادات الصفحة (The Identity) ──────────────────────────────────────────────────
st.set_page_config(
    page_title="NewsIQ AI | Neural Classifier",
    page_icon="💠",
    layout="wide",
)

# ── الـ CSS السري (The Secret Sauce for High-End Design) ──────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@100;300;400;600;900&family=Instrument+Serif:ital@0;1&display=swap');

    /* تصفير الألوان الأساسية */
    .stApp {
        background-color: #080808;
        color: #FFFFFF;
    }

    /* تصميم الـ Sidebar الجانبي للملخص */
    [data-testid="stSidebar"] {
        background-color: #0d0d0d;
        border-right: 1px solid #1a1a1a;
        padding: 20px;
    }

    /* العناوين الرئيسية */
    .main-title {
        font-family: 'Instrument Serif', serif;
        font-size: 5rem;
        font-weight: 400;
        line-height: 1;
        margin-bottom: 20px;
        background: linear-gradient(180deg, #fff 0%, #444 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ملخص المشروع (The Project Brief) */
    .brief-title {
        font-family: 'Outfit', sans-serif;
        font-size: 0.7rem;
        letter-spacing: 3px;
        color: #6366f1;
        margin-bottom: 10px;
        text-transform: uppercase;
    }
    .brief-content {
        font-family: 'Outfit', sans-serif;
        font-size: 0.95rem;
        color: #888;
        line-height: 1.6;
        margin-bottom: 40px;
    }

    /* الكروت التقنية */
    .tech-card {
        background: #0f0f0f;
        border: 1px solid #1a1a1a;
        border-radius: 30px;
        padding: 40px;
        margin-bottom: 20px;
    }

    /* تخصيص زر التحليل */
    .stButton > button {
        background: #fff !important;
        color: #000 !important;
        border-radius: 100px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        padding: 1.5rem 3rem !important;
        border: none !important;
        width: 100% !important;
        transition: 0.4s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
    }
    .stButton > button:hover {
        transform: scale(0.98);
        background: #6366f1 !important;
        color: #fff !important;
    }

    /* نتيجة التصنيف */
    .result-box {
        text-align: center;
        padding: 60px 20px;
    }
    .result-category {
        font-family: 'Instrument Serif', serif;
        font-size: 6rem;
        font-style: italic;
        color: #6366f1;
        margin: 0;
    }
    
    /* إخفاء العلامات المزعجة */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# ── الدوال البرمجية (Core Logic) ──────────────────────────────────────────────────

@st.cache_resource
def load_models():
    try:
        return {
            "tfidf": joblib.load("tfidf_vectorizer.joblib"),
            "le": joblib.load("label_encoder.joblib"),
            "models": {
                "Neural SVC Engine": joblib.load("linear_svc_model.joblib"),
                "Logistic Core": joblib.load("logistic_regression_model.joblib"),
                "Probabilistic NB": joblib.load("naive_bayes_model.joblib")
            }
        }
    except: return None

assets = load_models()

# ── التصميم الخارجي (The Interface) ────────────────────────────────────────────────

# Sidebar - Project Brief
with st.sidebar:
    st.markdown('<p class="brief-title">Project Abstract</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="brief-content">
    مشروع <b>NewsIQ</b> هو نموذج متقدم لمعالجة اللغات الطبيعية (NLP) مصمم لتصنيف الأخبار العالمية بدقة استثنائية. 
    من خلال دمج خوارزميات التعلم الآلي مع تقنيات Vectorization المتطورة، يستطيع النظام تحليل السياق اللغوي للنصوص وتحديد تصنيفها (رياضة، اقتصاد، تقنية، سياسة) في أجزاء من الثانية.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="brief-title">Technical Stack</p>', unsafe_allow_html=True)
    st.markdown('<div class="brief-content">Python, Scikit-Learn, TF-IDF, Streamlit</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="margin-top:100px;"></div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.6rem; color:#444;">DEVELOPED BY</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:Outfit; font-size:0.8rem; font-weight:600;">TOKA NASR SAEED</p>', unsafe_allow_html=True)

# Main Screen
st.markdown('<h1 class="main-title">Intelligence<br>Beyond News.</h1>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="tech-card">', unsafe_allow_html=True)
    st.markdown('<p style="color:#444; font-size:0.7rem; letter-spacing:2px; margin-bottom:15px;">INPUT FEED</p>', unsafe_allow_html=True)
    input_text = st.text_area("Analysis input", placeholder="Drop your article text here to begin analysis...", height=250, label_visibility="collapsed")
    
    st.markdown('<p style="color:#444; font-size:0.7rem; letter-spacing:2px; margin-top:25px; margin-bottom:10px;">PROCESSING ENGINE</p>', unsafe_allow_html=True)
    engine = st.selectbox("", list(assets["models"].keys()) if assets else ["Loading..."], label_visibility="collapsed")
    
    st.markdown('<div style="margin-top:40px;"></div>', unsafe_allow_html=True)
    run_btn = st.button("RUN NEURAL SCAN")
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    if run_btn and input_text:
        if assets:
            with st.spinner("Decoding semantics..."):
                time.sleep(1)
                # Processing
                clean = input_text.lower()
                clean = re.sub(r'[^a-z\s]', '', clean)
                vec = assets["tfidf"].transform([clean])
                model = assets["models"][engine]
                pred_idx = model.predict(vec)[0]
                category = assets["le"].inverse_transform([pred_idx])[0]
                
                # Confidence Score
                if hasattr(model, "predict_proba"):
                    conf = model.predict_proba(vec)[0][pred_idx]
                else:
                    d = model.decision_function(vec)[0]
                    conf = (np.exp(d) / np.sum(np.exp(d)))[pred_idx]

                st.markdown(f"""
                <div class="result-box">
                    <p style="color:#444; font-size:0.8rem; letter-spacing:4px; text-transform:uppercase;">Classification result</p>
                    <h2 class="result-category">{category}</h2>
                    <p style="font-family:Outfit; font-size:1.2rem; color:#888; margin-top:20px;">
                        Reliability Index: <span style="color:#fff; font-weight:900;">{conf*100:.1f}%</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Engine failure: Resources not found.")
    else:
        st.markdown("""
        <div style="height:450px; display:flex; align-items:center; justify-content:center; border:1px dashed #222; border-radius:30px; color:#333;">
            <p style="font-family:Outfit; letter-spacing:2px; font-size:0.8rem;">SYSTEM IDLE / AWAITING INPUT</p>
        </div>
        """, unsafe_allow_html=True)

# Signature Footer
st.markdown('<div style="text-align:center; padding:50px; color:#222; font-family:Outfit; font-size:0.6rem; letter-spacing:5px;">SECTION 1 • COMPUTER SCIENCE • 2026</div>', unsafe_allow_html=True)
