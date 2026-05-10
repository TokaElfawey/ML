import streamlit as st
import re
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── إعدادات الصفحة ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NewsIQ — AI News Classifier",
    page_icon="🧠",
    layout="wide",
)

# ── Custom CSS (نفس التصميم الاحترافي الذي اخترته) ────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background: #0a0a0f;
        color: #e8e6f0;
    }
    .stApp { background: #0a0a0f; }
    .nav-bar { display: flex; align-items: center; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.07); margin-bottom: 20px; }
    .nav-logo { font-family: 'DM Serif Display', serif; font-size: 1.8rem; color: #fff; }
    .nav-logo span { color: #7c6af7; }
    .card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 16px; padding: 25px; margin-bottom: 15px; }
    .result-category { font-family: 'DM Serif Display', serif; font-size: 2.5rem; margin-bottom: 5px; }
    .cat-world { color: #60b0ff; } .cat-sports { color: #4ecf8a; } .cat-biz { color: #f5a623; } .cat-tech { color: #e060f0; }
</style>
""", unsafe_allow_html=True)

# ── الدوال الأساسية (NLP & Model Loading) ────────────────────────────────────────

@st.cache_resource
def load_assets():
    """تحميل الـ Vectorizer والـ Encoder والموديلات الحقيقية"""
    try:
        tfidf = joblib.load("tfidf_vectorizer.joblib")
        le = joblib.load("label_encoder.joblib")
        models = {
            "Linear SVC": joblib.load("linear_svc_model.joblib"),
            "Logistic Regression": joblib.load("logistic_regression_model.joblib"),
            "Naive Bayes": joblib.load("naive_bayes_model.joblib")
        }
        return tfidf, le, models
    except Exception as e:
        st.error(f"خطأ في تحميل الملفات: {e}")
        return None, None, None

def clean_text(text):
    """تنظيف النص بنفس الطريقة المستخدمة في التدريب"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── واجهة المستخدم ──────────────────────────────────────────────────────────────

tfidf, le, models = load_assets()

# Navbar
st.markdown('<div class="nav-bar"><div class="nav-logo">News<span>IQ</span></div></div>', unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div style="text-align: center; padding: 40px 0;">
    <h1 style="font-family: 'DM Serif Display'; font-size: 3.5rem;">AI News <em style="color:#7c6af7">Classifier</em></h1>
    <p style="color: #8884a0; font-size: 1.1rem;">صنف الأخبار العالمية بدقة باستخدام خوارزميات معالجة اللغات الطبيعية</p>
</div>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    st.markdown('<p style="color:#5a567a; font-family:monospace;">ARTICLE TEXT</p>', unsafe_allow_html=True)
    input_text = st.text_area("", placeholder="انسخ نص الخبر هنا...", height=250, label_visibility="collapsed")

    st.markdown('<p style="color:#5a567a; font-family:monospace; margin-top:20px;">CHOOSE AI MODEL</p>', unsafe_allow_html=True)
    selected_model_name = st.selectbox("", list(models.keys()) if models else ["Loading..."], label_visibility="collapsed")

    analyze_btn = st.button("⚡ Analyze News Article", type="primary", use_container_width=True)

with right_col:
    if analyze_btn and input_text:
        if tfidf and models:
            with st.spinner("جاري التحليل..."):
                time.sleep(0.5) # للمحاكاة فقط

                # 1. المعالجة والتحويل
                cleaned = clean_text(input_text)
                vec = tfidf.transform([cleaned])

                # 2. التنبؤ
                current_model = models[selected_model_name]
                pred_idx = current_model.predict(vec)[0]
                category = le.inverse_transform([pred_idx])[0]

                # 3. حساب الاحتمالات (Confidence)
                conf_scores = {}
                if hasattr(current_model, "predict_proba"):
                    probs = current_model.predict_proba(vec)[0]
                    conf_scores = {c: p for c, p in zip(le.classes_, probs)}
                else:
                    # في حالة Linear SVC نستخدم الـ Decision Function
                    d_func = current_model.decision_function(vec)[0]
                    exp_scores = np.exp(d_func - np.max(d_func))
                    probs = exp_scores / exp_scores.sum()
                    conf_scores = {c: p for c, p in zip(le.classes_, probs)}

                # عرض النتيجة
                color_class = {
                    "World": "cat-world", "Sports": "cat-sports",
                    "Business": "cat-biz", "Sci/Tech": "cat-tech"
                }.get(category, "")

                st.markdown(f"""
                <div class="card">
                    <p style="font-size:0.7rem; color:#5a567a;">RESULT · {selected_model_name}</p>
                    <div class="result-category {color_class}">{category}</div>
                    <p style="font-family:monospace; color:#7c6af7;">Confidence: {conf_scores[category]*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

                # الرسم البياني
                fig, ax = plt.subplots(figsize=(5, 3))
                fig.patch.set_alpha(0); ax.patch.set_alpha(0)
                cats = list(conf_scores.keys())
                vals = list(conf_scores.values())
                colors = ['#f5a623', '#e060f0', '#4ecf8a', '#60b0ff'] # ألوان التصنيفات

                ax.barh(cats, vals, color=colors)
                ax.set_xlim(0, 1)
                ax.tick_params(axis='both', colors='#8884a0', labelsize=10)
                for spine in ax.spines.values(): spine.set_visible(False)

                st.pyplot(fig)
        else:
            st.error("تأكد من وجود ملفات الموديلات في مجلد المشروع.")
    else:
        st.markdown("""
        <div class="card" style="text-align:center; opacity:0.4; padding:80px 20px;">
            <p style="font-size:3rem;">🧠</p>
            <p>تظهر النتائج هنا بعد الضغط على Analyze</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<hr><div style='text-align:center; color:#3a3850; font-size:0.7rem;'>NEWSIQ · AG NEWS PROJECT · TEAM 1</div>", unsafe_allow_html=True)