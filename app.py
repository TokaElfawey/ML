import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. BRANDING & GLOW CONFIGURATION
# ==============================================================================
class NeonTheme:
    NAME = "NewsIQ Intelligence"
    TAGLINE = "NEURAL NEWS CLASSIFIER"
    TEAM = ["آية احمد", "تقي نصر", "تقي علاء", "همت حمدي", "نورهان مدحت"]
    
    # ألوان مريحة للعين (أزرق هادئ بدلاً من اللافندر)
    MAIN_GLOW = "#4A90D9"          # أزرق ناعم
    SECONDARY_GLOW = "#6BB8FF"     # أزرق فاتح
    DEEP_BLACK = "#0A0A0A"         # أسود داكن لكن أفتح قليلاً للخلفية

# ==============================================================================
# 2. CYBER-NEON CSS (معدل بالكامل لتحسين القراءة والشكل)
# ==============================================================================
def apply_neon_styles():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Plus+Jakarta+Sans:wght@300;400;600&family=Instrument+Serif:ital@0;1&display=swap');

        /* الخلفية */
        .stApp {{
            background-color: {NeonTheme.DEEP_BLACK};
            color: #e0e0e0;  /* نص رمادي فاتح بدلاً من الأبيض الصارخ */
            font-family: 'Plus Jakarta Sans', sans-serif;
        }}

        /* عنوان رئيسي - حجم مناسب */
        .neon-title {{
            font-family: 'Instrument Serif', serif;
            font-size: 4.5rem;
            text-align: center;
            color: #ffffff;
            text-shadow: 
                0 0 8px {NeonTheme.MAIN_GLOW},
                0 0 15px {NeonTheme.MAIN_GLOW};
            margin-bottom: 5px;
            line-height: 1.1;
        }}

        /* تحت العنوان */
        .neon-subtitle {{
            font-family: 'Orbitron', sans-serif;
            font-size: 0.9rem;  /* أكبر قليلاً */
            letter-spacing: 6px;
            text-align: center;
            color: {NeonTheme.SECONDARY_GLOW};
            text-shadow: 0 0 4px {NeonTheme.SECONDARY_GLOW};
            margin-bottom: 25px;
            text-transform: uppercase;
        }}

        /* الكروت - أكثر أناقة ومسافات مريحة */
        .neon-card {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(74, 144, 217, 0.15);
            border-radius: 16px;
            padding: 35px 30px;
            margin: 20px auto;
            max-width: 800px;
            box-shadow: 0 0 15px rgba(74, 144, 217, 0.03);
            transition: 0.3s;
        }}
        .neon-card:hover {{
            border-color: {NeonTheme.MAIN_GLOW};
            box-shadow: 0 0 25px rgba(74, 144, 217, 0.1);
        }}

        /* حقل الإدخال - مريح للقراءة */
        .stTextArea textarea {{
            background: #111 !important;
            border: 1px solid #333 !important;
            border-radius: 12px !important;
            color: #e0e0e0 !important;
            font-size: 1.1rem !important;  /* أكبر */
            padding: 18px !important;
            box-shadow: inset 0 0 8px rgba(107, 184, 255, 0.03) !important;
            line-height: 1.6;
        }}
        .stTextArea textarea:focus {{
            border-color: {NeonTheme.MAIN_GLOW} !important;
            box-shadow: 0 0 10px {NeonTheme.MAIN_GLOW}22 !important;
        }}

        /* زر التحليل - واضح وجذاب */
        .stButton > button {{
            background: transparent !important;
            color: #fff !important;
            border: 2px solid {NeonTheme.MAIN_GLOW} !important;
            font-family: 'Orbitron', sans-serif !important;
            border-radius: 50px !important;
            padding: 14px 45px !important;
            width: auto !important;
            min-width: 220px;
            display: block;
            margin: 25px auto !important;
            transition: 0.3s all !important;
            text-shadow: 0 0 8px {NeonTheme.MAIN_GLOW};
            box-shadow: 0 0 12px {NeonTheme.MAIN_GLOW}33;
            font-size: 0.9rem;
            letter-spacing: 2px;
        }}
        .stButton > button:hover {{
            background: {NeonTheme.MAIN_GLOW} !important;
            box-shadow: 0 0 30px {NeonTheme.MAIN_GLOW} !important;
            transform: scale(1.03);
        }}

        /* نتيجة التصنيف - واضحة وكبيرة */
        .neon-result {{
            font-family: 'Instrument Serif', serif;
            font-size: 4rem;  /* من 5rem إلى 4rem لتتناسب */
            font-style: italic;
            text-align: center;
            color: {NeonTheme.SECONDARY_GLOW};
            text-shadow: 0 0 20px {NeonTheme.SECONDARY_GLOW};
            margin: 10px 0;
        }}

        /* عناصر الفريق */
        .member-glow {{
            background: rgba(107, 184, 255, 0.06);
            border: 1px solid rgba(107, 184, 255, 0.15);
            padding: 10px 22px;
            border-radius: 50px;
            font-size: 0.95rem;
            color: {NeonTheme.SECONDARY_GLOW};
            text-shadow: 0 0 3px {NeonTheme.SECONDARY_GLOW};
            font-weight: 500;
        }}

        /* تحسين التحديد */
        .stSelectbox label {{
            color: #aaa !important;
            font-size: 0.9rem !important;
        }}
        .stSelectbox div[data-baseweb="select"] > div {{
            background-color: #111 !important;
            border: 1px solid #333 !important;
            color: #e0e0e0 !important;
            border-radius: 10px !important;
        }}

        /* إخفاء العناصر الافتراضية */
        #MainMenu, footer, header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. AI ENGINE CLASS (بدون تغيير)
# ==============================================================================
class AIClassifier:
    def __init__(self):
        self.ready = False
        self.tfidf = None
        self.le = None
        self.models = {}

    def setup(self):
        try:
            self.tfidf = joblib.load("tfidf_vectorizer.joblib")
            self.le = joblib.load("label_encoder.joblib")
            self.models = {
                "Neural Linear Core": joblib.load("linear_svc_model.joblib"),
                "Logistic Matrix Core": joblib.load("logistic_regression_model.joblib"),
                "Naive Bayes Core": joblib.load("naive_bayes_model.joblib")
            }
            self.ready = True
            return True
        except:
            return False

    def predict(self, text, model_key):
        start = time.time()
        clean = text.lower()
        clean = re.sub(r'[^a-z\s]', '', clean)
        vec = self.tfidf.transform([clean])
        mdl = self.models[model_key]
        idx = mdl.predict(vec)[0]
        cat = self.le.inverse_transform([idx])[0]
        
        if hasattr(mdl, "predict_proba"):
            probs = mdl.predict_proba(vec)[0]
        else:
            d = mdl.decision_function(vec)[0]
            e = np.exp(d - np.max(d))
            probs = e / e.sum()
            
        return {
            "category": cat,
            "score": probs[idx],
            "all_probs": probs,
            "classes": self.le.classes_,
            "latency": time.time() - start
        }

# ==============================================================================
# 4. NEON VISUALIZER (معدل قليلاً للألوان)
# ==============================================================================
class NeonViz:
    @staticmethod
    def plot_glow_bars(data):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#0A0A0A')
        ax.set_facecolor('#0A0A0A')
        
        colors = ['#1a1a1a'] * len(data["classes"])
        target_idx = list(data["classes"]).index(data["category"])
        colors[target_idx] = NeonTheme.SECONDARY_GLOW

        bars = ax.barh(data["classes"], data["all_probs"], color=colors, edgecolor=colors, height=0.5)
        
        for i, bar in enumerate(bars):
            if i == target_idx:
                bar.set_linewidth(2)
                bar.set_edgecolor('#ffffff')

        ax.set_title("NEURAL PROBABILITY SPECTRUM", color=NeonTheme.SECONDARY_GLOW, 
                     fontsize=9, pad=15, fontfamily='Orbitron', weight='bold')
        
        ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
        ax.tick_params(axis='both', colors='#888', labelsize=10)
        plt.tight_layout()
        return fig

# ==============================================================================
# 5. MAIN INTERFACE (تم تعديل المسافات)
# ==============================================================================
def main():
    apply_neon_styles()
    brain = AIClassifier()
    
    with st.spinner("Powering up Neural Cores..."):
        loaded = brain.setup()

    # --- Header ---
    st.markdown(f"""
    <div style="margin-top: 20px;">
        <h1 class="neon-title">{NeonTheme.NAME}</h1>
        <p class="neon-subtitle">{NeonTheme.TAGLINE}</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Project Abstract ---
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#777; letter-spacing:2px; font-size:0.65rem; font-family:Orbitron;">PROJECT DATA</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center; color:#ccc; font-size:1rem; line-height:1.7; margin-top:10px;">
        منصة <b style="color:#fff;">NewsIQ</b> هي محرك ذكاء اصطناعي فائق التطور يعتمد على معالجة اللغات الطبيعية (NLP). 
        النظام مصمم لفك شفرة النصوص الخبرية وتصنيفها آلياً باستخدام مصفوفات رياضية معقدة. 
        هذا الابتكار يقلل التدخل البشري في فرز الأخبار ويزيد من سرعة استجابة الأنظمة الذكية للمعلومات المتدفقة.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Input Terminal ---
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.markdown('<p style="color:#666; letter-spacing:2px; font-size:0.65rem; font-family:Orbitron; margin-bottom:10px;">INPUT_STREAM_TERMINAL</p>', unsafe_allow_html=True)
    
    user_text = st.text_area("Input", placeholder="Enter raw news text for neural analysis...", height=200, label_visibility="collapsed")
    
    selected_mdl = st.selectbox("Select Neural Core", list(brain.models.keys()) if loaded else ["Offline"])
    
    analyze_btn = st.button("INITIATE NEURAL SCAN")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Result Execution ---
    if analyze_btn and user_text:
        if loaded:
            with st.spinner("Decoding semantics..."):
                res = brain.predict(user_text, selected_mdl)
                
                # Glowing Result
                st.markdown('<div class="neon-card" style="border-color: #6BB8FF44;">', unsafe_allow_html=True)
                st.markdown(f'<p style="text-align:center; color:#777; letter-spacing:3px; font-family:Orbitron; font-size:0.7rem;">INFERENCE_RESULT</p>', unsafe_allow_html=True)
                st.markdown(f'<h2 class="neon-result">{res["category"]}</h2>', unsafe_allow_html=True)
                st.markdown(f'<p style="text-align:center; font-family:Orbitron; color:{NeonTheme.MAIN_GLOW}; font-size:1.1rem;">ACCURACY_LEVEL: {res["score"]*100:.2f}%</p>', unsafe_allow_html=True)
                
                # Probability Map
                st.markdown('<div style="margin-top:30px; padding-top:20px; border-top:1px solid #222;">', unsafe_allow_html=True)
                st.pyplot(NeonViz.plot_glow_bars(res))
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # --- Footer & Team ---
    st.markdown(f"""
    <div style="margin-top:80px; padding:50px 0; border-top:1px solid #1a1a1a; text-align:center;">
        <p style="letter-spacing:6px; color:#444; font-size:0.75rem; margin-bottom:30px; font-family:Orbitron;">UNDER THE GUIDANCE OF YOUSSEF AL-BAROUDI</p>
        <div style="display:flex; justify-content:center; flex-wrap:wrap; gap:15px;">
            {"".join([f'<div class="member-glow">{name}</div>' for name in NeonTheme.TEAM])}
        </div>
        <p style="margin-top:40px; font-family:Orbitron; font-size:0.55rem; color:#333; letter-spacing:3px;">
            SYSTEM STATUS: OPERATIONAL | BUILD: 2026.05
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
