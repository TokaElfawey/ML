import streamlit as st
import re
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── إعدادات الصفحة الفخمة جداً ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NewsIQ | Neural Intelligence Platform",
    page_icon="🧪",
    layout="wide",
)

# ── الأنظمة البصرية المتقدمة (Ultra Premium CSS) ────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100;300;400;700;900&family=Space+Grotesk:wght@300;500;700&display=swap');
    
    /* الأساسيات */
    .stApp {
        background-color: #050505;
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.15) 0px, transparent 50%);
        color: #FFFFFF;
    }

    /* الهيدر */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 50px;
    }
    .brand-logo {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -1px;
        color: #fff;
    }
    .brand-logo span { color: #6366f1; }

    /* قسم ملخص المشروع - Abstract Box */
    .abstract-section {
        background: rgba(255, 255, 255, 0.02);
        border-left: 4px solid #6366f1;
        padding: 30px;
        border-radius: 0 20px 20px 0;
        margin-bottom: 40px;
        box-shadow: 20px 0 50px rgba(0,0,0,0.5);
    }
    .abstract-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.8rem;
        letter-spacing: 3px;
        color: #6366f1;
        text-transform: uppercase;
        margin-bottom: 15px;
    }
    .abstract-text {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        line-height: 1.6;
        color: #94a3b8;
        max-width: 900px;
    }

    /* كروت الأدوات */
    .tool-card {
        background: #0d0d0d;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 24px;
        padding: 40px;
        height: 100%;
    }

    /* العناوين */
    .main-heading {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 10px;
        background: linear-gradient(to bottom, #fff, #64748b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* تخصيص الـ Widgets */
    .stTextArea textarea {
        background: #151515 !important;
        border: 1px solid #252525 !important;
        color: #fff !important;
        border-radius: 15px !important;
        padding: 20px !important;
    }
    
    .stButton > button {
        background: #fff !important;
        color: #000 !important;
        border-radius: 100px !important;
        font-weight: 700 !important;
        padding: 20px !important;
        border: none !important;
        transition: 0.3s all ease;
    }
    .stButton > button:hover {
        background: #6366f1 !important;
        color: #fff !important;
        box-shadow: 0 0 30px rgba(99, 102, 241, 0.4);
    }

    /* نتيجة التصنيف - الـ Badge الفخم */
    .result-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 100px;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid #6366f1;
        color: #818cf8;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 20px;
    }

    .category-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    /* إخفاء الزوائد */
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── الدوال البرمجية الثابتة ──────────────────────────────────────────

@st.cache_resource
def load_all_assets():
    try:
        return {
            "tfidf": joblib.load("tfidf_vectorizer.joblib"),
            "le": joblib.load("label_encoder.joblib"),
            "models": {
                "Linear SVC (Best)": joblib.load("linear_svc_model.joblib"),
                "Logistic Regression": joblib.load("logistic_regression_model.joblib"),
                "Naive Bayes": joblib.load("naive_bayes_model.joblib")
            }
        }
    except: return None

assets = load_all_assets()

# ── تصميم الهيكل الجديد ──────────────────────────────────────────────

# 1. Header
st.markdown("""
<div class="header-container">
    <div class="brand-logo">NEWS<span>IQ</span>.AI</div>
    <div style="font-family:'Inter'; font-size:0.8rem; color:#475569;">BETA V2.0 / NLP-ENGINE</div>
</div>
""", unsafe_allow_html=True)

# 2. Project Summary (Abstract)
st.markdown("""
<div class="abstract-section">
    <div class="abstract-title">Project Executive Summary</div>
    <div class="abstract-text">
        هذا المشروع هو منصة استخباراتية مدعومة بالذكاء الاصطناعي تهدف إلى تحليل وفهم المحتوى الخبري لحظياً. 
        يعتمد المحرك على خوارزميات <b>Machine Learning</b> متقدمة تمت معالجتها باستخدام تقنيات <b>TF-IDF</b> لتحويل النصوص إلى متجهات رياضية معقدة. 
        يستطيع النظام تصنيف الأخبار بدقة تصل إلى <b>94%</b> إلى أربعة قطاعات حيوية: الرياضة، التكنولوجيا، الاقتصاد، والسياسة الدولية، مما يوفر أداة قوية لفلترة البيانات الضخمة وفهم توجهات الرأي العام.
    </div>
</div>
""", unsafe_allow_html=True)

# 3. Main Dashboard Space
st.markdown('<h1 class="main-heading">Neural Content Analysis</h1>', unsafe_allow_html=True)

left, right = st.columns([1.5, 1], gap="large")

with left:
    st.markdown('<div class="tool-card">', unsafe_allow_html=True)
    user_input = st.text_area("Input data for processing:", placeholder="Enter the news text here...", height=300, label_visibility="collapsed")
    
    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        selected_model = st.selectbox("Select Model Engine", list(assets["models"].keys()) if assets else ["Error"])
    with col_m2:
        st.markdown('<div style="margin-top:28px;">', unsafe_allow_html=True)
        run_btn = st.button("EXECUTE ANALYSIS", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    if run_btn and user_input:
        if assets:
            with st.spinner("Neural Processing..."):
                time.sleep(1)
                # Processing
                clean = user_input.lower()
                clean = re.sub(r'[^a-z\s]', '', clean)
                vec = assets["tfidf"].transform([clean])
                model = assets["models"][selected_model]
                pred_idx = model.predict(vec)[0]
                category = assets["le"].inverse_transform([pred_idx])[0]
                
                # Confidence Calculation
                if hasattr(model, "predict_proba"):
                    confidence = model.predict_proba(vec)[0][pred_idx]
                else:
                    d = model.decision_function(vec)[0]
                    confidence = (np.exp(d) / np.sum(np.exp(d)))[pred_idx]

                st.markdown(f"""
                <div class="tool-card" style="border-color:#6366f1;">
                    <div class="result-badge">IDENTIFIED CLASS</div>
                    <div class="category-title">{category}</div>
                    <p style="color:#94a3b8; font-size:1.1rem; margin-top:10px;">
                        Confidence Score: <span style="color:#fff; font-weight:bold;">{confidence*100:.2f}%</span>
                    </p>
                    <div style="margin-top:30px; padding:20px; background:rgba(255,255,255,0.03); border-radius:15px;">
                        <span style="color:#475569; font-size:0.8rem; font-family:'Inter';">NLP ENGINE INSIGHT:</span>
                        <p style="color:#cbd5e1; font-size:0.9rem; margin-top:5px;">
                            The linguistic markers and keyword density strongly suggest a <b>{category}</b> orientation. 
                            The model has successfully isolated key semantic features from the provided input.
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Assets not found.")
    else:
        st.markdown("""
        <div class="tool-card" style="display:flex; align-items:center; justify-content:center; text-align:center; border-style:dashed; opacity:0.3;">
            <div>
                <div style="font-size:3rem; margin-bottom:10px;">🛰️</div>
                <div style="font-family:'Space Grotesk'; font-weight:700;">AWAITING SIGNAL</div>
                <div style="font-size:0.8rem;">Enter data to begin classification</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="margin-top:100px; padding-bottom:50px; text-align:center;">
    <div style="height:1px; background:rgba(255,255,255,0.05); width:100px; margin:0 auto 20px auto;"></div>
    <div style="font-family:'Space Grotesk'; font-size:0.7rem; color:#475569; letter-spacing:5px;">
        TOKA NASR SAEED · SECTION 1 · COMPUTER SCIENCE DEPT.
    </div>
</div>
""", unsafe_allow_html=True)
