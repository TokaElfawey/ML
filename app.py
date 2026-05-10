import streamlit as st
import re
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import docx2txt
from pypdf import PdfReader

# ── 1. إعدادات الصفحة الفخمة ──────────────────────────────────────────────────
st.set_page_config(
    page_title="NewsIQ | AI Intelligence Hub",
    page_icon="🧠",
    layout="wide",
)

# ── 2. دالة خلفية الفيديو (إضافة الروح للموقع) ──────────────────────────────────
def add_bg_video():
    video_url = "https://assets.mixkit.co/videos/preview/mixkit-digital-particles-in-blue-background-9121-large.mp4"
    st.markdown(f"""
        <style>
        [data-testid="stAppViewContainer"] {{ background: none; }}
        #myVideo {{
            position: fixed; right: 0; bottom: 0;
            min-width: 100%; min-height: 100%;
            z-index: -1; filter: brightness(0.4);
        }}
        </style>
        <video autoplay loop muted playsinline id="myVideo">
            <source src="{video_url}" type="video/mp4">
        </video>
    """, unsafe_allow_html=True)

# ── 3. Custom CSS (دمج الهوية البصرية مع الـ Glassmorphism) ─────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: #e8e6f0; }

    /* الهيدر اللوجو */
    .nav-logo { font-family: 'DM Serif Display', serif; font-size: 2.5rem; color: #fff; text-align: center; margin-bottom: 10px; }
    .nav-logo span { color: #00f5ff; }

    /* كروت النتائج الزجاجية */
    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 20px;
    }

    /* ألوان التصنيفات */
    .result-category { font-family: 'DM Serif Display', serif; font-size: 3rem; margin-bottom: 10px; }
    .cat-world { color: #60b0ff; text-shadow: 0 0 15px #60b0ff66; }
    .cat-sports { color: #4ecf8a; text-shadow: 0 0 15px #4ecf8a66; }
    .cat-biz { color: #f5a623; text-shadow: 0 0 15px #f5a62366; }
    .cat-tech { color: #e060f0; text-shadow: 0 0 15px #e060f066; }

    /* زر التحليل المضيء */
    .stButton button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 15px !important;
        font-weight: 800 !important;
        font-size: 1.2rem !important;
        box-shadow: 0 10px 20px rgba(0, 210, 255, 0.3) !important;
        transition: 0.3s;
    }
    .stButton button:hover { transform: translateY(-3px); box-shadow: 0 15px 25px rgba(0, 210, 255, 0.5) !important; }

    /* الفوتر */
    .footer { text-align: center; padding: 40px; background: rgba(0,0,0,0.4); border-top: 1px solid rgba(255,255,255,0.05); margin-top: 50px; }
</style>
""", unsafe_allow_html=True)

# ── 4. الدوال البرمجية (NLP) ──────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    try:
        tfidf = joblib.load("tfidf_vectorizer.joblib")
        le = joblib.load("label_encoder.joblib")
        models = {
            "Linear SVC": joblib.load("linear_svc_model.joblib"),
            "Logistic Regression": joblib.load("logistic_regression_model.joblib"),
            "Naive Bayes": joblib.load("naive_bayes_model.joblib")
        }
        return tfidf, le, models
    except:
        return None, None, None

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

# ── 5. بناء الواجهة الرئيسية ─────────────────────────────────────────────────
def main():
    add_bg_video()
    tfidf, le, models = load_assets()

    st.markdown('<div class="nav-logo">News<span>IQ</span></div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#94a3b8; font-size:1.2rem;'>Advanced Neural News Classification Hub</p>", unsafe_allow_html=True)

    # تقسيم الصفحة
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        input_method = st.tabs(["✍️ Paste Text", "📁 Upload Document"])
        
        input_text = ""
        with input_method[0]:
            input_text = st.text_area("Article Content", placeholder="Enter the news story here...", height=250, label_visibility="collapsed")
        
        with input_method[1]:
            file = st.file_uploader("Upload PDF/DOCX/TXT", type=['pdf', 'docx', 'txt'])
            if file:
                if file.type == "application/pdf":
                    reader = PdfReader(file)
                    input_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
                elif file.type == "text/plain":
                    input_text = file.read().decode("utf-8")
                else:
                    input_text = docx2txt.process(file)
                st.success("File context loaded successfully!")

        st.markdown("</div>", unsafe_allow_html=True)
        
        selected_model = st.selectbox("Intelligence Model Core", list(models.keys()) if models else ["Assets Missing"])
        analyze_btn = st.button("⚡ START NEURAL ANALYSIS", use_container_width=True)

    with right_col:
        if analyze_btn and input_text:
            if tfidf and models:
                with st.spinner("Decoding language patterns..."):
                    time.sleep(0.8) # لمحاكاة سرعة التحليل
                    cleaned = clean_text(input_text)
                    vec = tfidf.transform([cleaned])
                    
                    model = models[selected_model]
                    pred_idx = model.predict(vec)[0]
                    category = le.inverse_transform([pred_idx])[0]

                    # حساب الثقة (Confidence)
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(vec)[0]
                    else:
                        d_func = model.decision_function(vec)[0]
                        exp_scores = np.exp(d_func - np.max(d_func))
                        probs = exp_scores / exp_scores.sum()
                    
                    conf_scores = {c: p for c, p in zip(le.classes_, probs)}

                    # تحديد لون التصنيف بناءً على كودك
                    color_class = {"World": "cat-world", "Sports": "cat-sports", "Business": "cat-biz", "Sci/Tech": "cat-tech"}.get(category, "")

                    # عرض كارت النتيجة الفخم
                    st.markdown(f"""
                    <div class="card" style="border-left: 5px solid #00f5ff;">
                        <p style="font-size:0.8rem; color:#8884a0; text-transform:uppercase; letter-spacing:2px;">Classification Result</p>
                        <div class="result-category {color_class}">{category}</div>
                        <div style="background:rgba(0,245,255,0.1); padding:10px; border-radius:10px; display:inline-block;">
                            <span style="color:#00f5ff; font-weight:bold; font-size:1.2rem;">Confidence: {conf_scores[category]*100:.2f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # رسم بياني احترافي (Dark Theme)
                    st.write("### Probability Distribution")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_alpha(0)
                    ax.patch.set_alpha(0)
                    
                    cats = list(conf_scores.keys())
                    vals = list(conf_scores.values())
                    colors = ['#f5a623', '#e060f0', '#4ecf8a', '#60b0ff']

                    bars = ax.barh(cats, vals, color=colors, edgecolor='white', linewidth=0.5)
                    ax.set_xlim(0, 1)
                    ax.tick_params(axis='both', colors='#cbd5e1', labelsize=12)
                    for spine in ax.spines.values(): spine.set_visible(False)
                    
                    # إضافة النسب المئوية على البارات
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width*100:.1f}%', 
                                va='center', color='white', fontweight='bold')

                    st.pyplot(fig)
            else:
                st.error("Assets Error: Please check model files.")
        else:
            st.markdown("""
            <div class="card" style="text-align:center; opacity:0.3; padding:100px 20px; border-style: dashed;">
                <h1 style="font-size:4rem; margin:0;">🧠</h1>
                <p>Awaiting article input for analysis...</p>
            </div>
            """, unsafe_allow_html=True)

    # Footer المطور
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; color: #94a3b8; font-family: sans-serif; font-size: 18px;'>
            Developed with ❤️ by <b style='color: #00f5ff;'>Section 1 Team</b><br>
            <span style='font-size: 12px;'>Toka Nasr | Aya Ahmed | Toka Alaa | Hemmat Hamdi | Nourhan Medhat</span><br>
            © 2026 Academic Project
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
