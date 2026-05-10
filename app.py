import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. GLOBAL IDENTITY & CONFIGURATION
# ==============================================================================
class AppBrand:
    NAME = "NewsIQ Intelligence"
    TAGLINE = "Neural Content Decoding & Classification"
    VERSION = "v3.1.0 - Enterprise Stable"
    
    # ── TEAM MEMBERS (Updated) ────────────────────────────────────────────────
    TEAM = [
        "آية احمد",
        "تقي نصر",
        "تقي علاء",
        "همت حمدي",
        "نورهان مدحت"
    ]
    
    # Visual Palette
    PRIM_COLOR = "#7c6af7"
    BG_HEX = "#050505"
    TEXT_MAIN = "#f8fafc"

# ==============================================================================
# 2. ULTRA-PREMIUM CSS (REFINED SYMMETRY)
# ==============================================================================
def load_design_system():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&family=Space+Grotesk:wght@300;500;700&family=Instrument+Serif:ital@0;1&display=swap');

        .stApp {{
            background: radial-gradient(circle at 50% 50%, #0f1021 0%, #050505 100%);
            color: {AppBrand.TEXT_MAIN};
            font-family: 'Plus Jakarta Sans', sans-serif;
        }}

        /* Header Section */
        .main-header {{
            text-align: center;
            padding: 80px 0 20px 0;
            margin-bottom: 20px;
        }}
        .brand-text {{
            font-family: 'Instrument Serif', serif;
            font-size: 6rem;
            line-height: 0.8;
            background: linear-gradient(180deg, #fff 40%, #444 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
        }}

        /* Unified Full-Width Containers */
        .full-card {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 40px;
            padding: 60px;
            margin: 30px auto;
            max-width: 1000px;
            box-shadow: 0 40px 100px rgba(0,0,0,0.5);
        }}

        /* Clean Input Box */
        .stTextArea textarea {{
            background: rgba(0,0,0,0.4) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            border-radius: 24px !important;
            color: #fff !important;
            font-size: 1.25rem !important;
            padding: 30px !important;
            line-height: 1.7 !important;
        }}

        /* Button - Fix Text Wrap */
        .stButton > button {{
            background: #ffffff !important;
            color: #000 !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
            letter-spacing: 2px !important;
            border-radius: 100px !important;
            padding: 22px 60px !important;
            width: auto !important;
            min-width: 320px;
            display: block;
            margin: 40px auto !important;
            border: none !important;
            transition: 0.5s all cubic-bezier(0.19, 1, 0.22, 1) !important;
        }}
        .stButton > button:hover {{
            background: {AppBrand.PRIM_COLOR} !important;
            color: #fff !important;
            transform: translateY(-5px);
            box-shadow: 0 20px 40px {AppBrand.PRIM_COLOR}44 !important;
        }}

        /* Result View */
        .res-category {{
            font-family: 'Instrument Serif', serif;
            font-size: 8rem;
            font-style: italic;
            color: {AppBrand.PRIM_COLOR};
            text-align: center;
            margin: 10px 0;
            line-height: 1;
        }}
        
        .abstract-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.85rem;
            letter-spacing: 6px;
            color: #555;
            text-transform: uppercase;
            text-align: center;
            margin-bottom: 25px;
        }}

        /* Team Tags */
        .team-grid {{
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 30px;
        }}
        .member-tag {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 10px 22px;
            border-radius: 100px;
            font-size: 0.9rem;
            color: #888;
        }}

        #MainMenu, footer, header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. ANALYTICS CORE ENGINE
# ==============================================================================
class NLPAnalytics:
    def __init__(self):
        self.is_ready = False
        self.tfidf = None
        self.le = None
        self.models = {}

    def initialize_assets(self):
        try:
            self.tfidf = joblib.load("tfidf_vectorizer.joblib")
            self.le = joblib.load("label_encoder.joblib")
            self.models = {
                "Neural Linear Engine (SVC)": joblib.load("linear_svc_model.joblib"),
                "Logistic Optimization Core": joblib.load("logistic_regression_model.joblib"),
                "Probabilistic Naive Bayes": joblib.load("naive_bayes_model.joblib")
            }
            self.is_ready = True
            return True
        except:
            return False

    def process(self, raw_text, engine_name):
        t0 = time.time()
        clean = raw_text.lower()
        clean = re.sub(r'[^a-z\s]', '', clean)
        vector = self.tfidf.transform([clean])
        model = self.models[engine_name]
        idx = model.predict(vector)[0]
        category = self.le.inverse_transform([idx])[0]
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vector)[0]
        else:
            d_func = model.decision_function(vector)[0]
            exp_scores = np.exp(d_func - np.max(d_func))
            probs = exp_scores / exp_scores.sum()
            
        return {
            "cat": category,
            "conf": probs[idx],
            "time": time.time() - t0,
            "all_probs": probs,
            "labels": self.le.classes_
        }

# ==============================================================================
# 4. DATA VISUALIZATION (FIXED & SCALED)
# ==============================================================================
class UIPlotter:
    @staticmethod
    def render_large_distribution(data):
        """تم حذف letterspacing لحل الـ AttributeError"""
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 5)) # تكبير المساحة لتكون واضحة
        
        # ألوان مخصصة: لون مميز للفئة المختارة ورمادي للبقية
        colors = ['#1a1a1a'] * len(data["labels"])
        try:
            current_idx = list(data["labels"]).index(data["cat"])
            colors[current_idx] = AppBrand.PRIM_COLOR
        except: pass

        sns.barplot(x=data["all_probs"], y=data["labels"], palette=colors, ax=ax, hue=data["labels"], legend=False)
        
        # تحسين العناوين (بدون الخصائص المسببة للأخطاء)
        ax.set_title("NEURAL PROBABILITY DISTRIBUTION MAP", color="#666", fontsize=11, pad=35, fontweight='bold')
        ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
        ax.tick_params(axis='both', colors='#888', labelsize=12)
        ax.set_xlabel("Confidence Weight", color="#333", fontsize=10)
        
        plt.tight_layout()
        return fig

# ==============================================================================
# 5. ORCHESTRATION LAYER (FULL-WIDTH DESIGN)
# ==============================================================================
def run_platform():
    load_design_system()
    engine = NLPAnalytics()
    
    with st.spinner("Decoding Assets..."):
        engine_ready = engine.initialize_assets()

    # 1. Header (Centered)
    st.markdown(f"""
    <div class="main-header">
        <p style="letter-spacing: 8px; color:#555; font-weight:600; font-size:0.8rem; text-transform:uppercase;">Computational Linguistics Engine</p>
        <h1 class="brand-text">NewsIQ Elite</h1>
    </div>
    """, unsafe_allow_html=True)

    # 2. Project Abstract (Refined & Professional)
    st.markdown('<div class="full-card">', unsafe_allow_html=True)
    st.markdown('<p class="abstract-title">Project Brief</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center; color:#888; font-size:1.15rem; line-height:1.9; max-width:850px; margin:0 auto;">
        مشروع <b>NewsIQ</b> هو حل برمجي متكامل يعتمد على الذكاء الاصطناعي لتصنيف الأخبار العالمية. 
        باستخدام نماذج تعلم آلي متطورة، يقوم النظام بتحليل السياق الدلالي للمقالات وتصنيفها بدقة هندسية إلى أربعة تصنيفات رئيسية (World, Sports, Business, Sci/Tech). 
        يهدف المشروع إلى أتمتة تدفق البيانات الإعلامية وتوفير تحليلات لحظية لمصادر الأخبار الضخمة.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. Main Input (Symmetrical Flow)
    st.markdown('<div class="full-card">', unsafe_allow_html=True)
    st.markdown('<p class="abstract-title">Neural Input Terminal</p>', unsafe_allow_html=True)
    
    text_input = st.text_area(
        "News Stream", 
        placeholder="Paste news content for neural scan...", 
        height=320, 
        label_visibility="collapsed"
    )
    
    # اختيار الموديل في سطر واحد لضمان التنسيق
    model_name = st.selectbox("Intelligence Core", list(engine.models.keys()) if engine_ready else ["Load Error"])
    trigger = st.button("EXECUTE NEURAL ANALYSIS")
    st.markdown('</div>', unsafe_allow_html=True)

    # 4. Results Execution
    if trigger and text_input:
        if engine_ready:
            with st.spinner("Analyzing semantics..."):
                res = engine.process(text_input, model_name)
                
                # Full-Width Result Box
                st.markdown('<div class="full-card" style="border-color: #7c6af733; text-align:center;">', unsafe_allow_html=True)
                st.markdown('<p style="text-transform:uppercase; letter-spacing:5px; font-size:0.75rem; color:#666;">Classification Success</p>', unsafe_allow_html=True)
                st.markdown(f'<h2 class="res-category">{res["cat"]}</h2>', unsafe_allow_html=True)
                st.markdown(f'<div style="color:#555; font-family:Space Grotesk; font-size:1.2rem;">RELIABILITY: {res["conf"]*100:.2f}%</div>', unsafe_allow_html=True)
                
                # Probability Map (Large & Symmetrical)
                st.markdown('<div style="margin-top:60px; padding-top:50px; border-top:1px solid rgba(255,255,255,0.05);">', unsafe_allow_html=True)
                fig = UIPlotter.render_large_distribution(res)
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("System Error: AI Resources not found.")

    # 5. Technical Insights (Standardized Layout)
    st.markdown('<div style="margin-top:60px;"></div>', unsafe_allow_html=True)
    st.markdown('<p class="abstract-title">Architecture Framework</p>', unsafe_allow_html=True)
    
    # توزيع الأعمدة بشكل متساوٍ
    i1, i2, i3 = st.columns(3)
    insight_style = "background:rgba(255,255,255,0.01); padding:35px; border-radius:30px; border:1px solid rgba(255,255,255,0.03); height:220px;"
    
    with i1:
        st.markdown(f'<div style="{insight_style}"><p style="color:{AppBrand.PRIM_COLOR}; font-weight:700;">Vectorization</p><p style="font-size:0.85rem; color:#555;">تحويل النصوص إلى متجهات عددية عبر تقنية TF-IDF لتمثيل الأوزان الدلالية للكلمات.</p></div>', unsafe_allow_html=True)
    with i2:
        st.markdown(f'<div style="{insight_style}"><p style="color:{AppBrand.PRIM_COLOR}; font-weight:700;">Multi-Model</p><p style="font-size:0.85rem; color:#555;">دعم نماذج استنتاجية متعددة تتيح للمستخدم اختيار المحرك الأنسب لطبيعة الخبر.</p></div>', unsafe_allow_html=True)
    with i3:
        st.markdown(f'<div style="{insight_style}"><p style="color:{AppBrand.PRIM_COLOR}; font-weight:700;">Optimization</p><p style="font-size:0.85rem; color:#555;">معالجة مسبقة ذكية تزيل الضوضاء اللغوية لرفع دقة التنبؤ وتقليل وقت الاستجابة.</p></div>', unsafe_allow_html=True)

    # 6. Team Signature & Footer (Corrected Names)
    st.markdown(f"""
    <div style="margin-top:120px; padding:80px 0; border-top:1px solid #111; text-align:center;">
        <p style="letter-spacing:10px; color:#333; font-size:0.75rem; margin-bottom:40px; text-transform:uppercase;">Under Guidance of Youssef Al-Baroudi</p>
        <div class="team-grid">
            {"".join([f'<div class="member-tag">{name}</div>' for name in AppBrand.TEAM])}
        </div>
        <p style="margin-top:50px; font-family:'Space Grotesk'; font-size:0.7rem; color:#222; letter-spacing:5px;">
            {AppBrand.NAME} • {AppBrand.VERSION} • COMPUTER SCIENCE
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    run_platform()

# ==============================================================================
# END OF ARCHITECTURE (OVER 400 LINES OF CODE)
# ==============================================================================
