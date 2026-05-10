import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. BRANDING & GLOW CONFIGURATION (معدلة بألوان هادئة)
# ==============================================================================
class NeonTheme:
    NAME = "NewsIQ Intelligence"
    TAGLINE = "AI-Powered News Classification"  # تم إضافة TAGLINE
    TEAM = ["آية احمد", "تقي نصر", "تقي علاء", "همت حمدي", "نورهان مدحت"]
    
    # ألوان نيون هادئة ومريحة للعين
    MAIN_GLOW = "#5b8def"          # أزرق ناعم بدلاً من اللافندر
    SECONDARY_GLOW = "#00d4ff"     # سماوي أقل حدة
    DEEP_BLACK = "#020202"

# ==============================================================================
# 2. CYBER-NEON CSS (تم تعديل القيم لتقليل المسافات والأحجام)
# ==============================================================================
def apply_neon_styles():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Plus+Jakarta+Sans:wght@300;400;600&family=Instrument+Serif:ital@0;1&display=swap');

        .stApp {{
            background-color: {NeonTheme.DEEP_BLACK};
            color: #ffffff;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }}

        /* عنوان أصغر حجماً */
        .neon-title {{
            font-family: 'Instrument Serif', serif;
            font-size: 4rem;          /* من 7rem إلى 4rem */
            text-align: center;
            color: #fff;
            text-shadow: 
                0 0 10px {NeonTheme.MAIN_GLOW},
                0 0 20px {NeonTheme.MAIN_GLOW},
                0 0 40px {NeonTheme.MAIN_GLOW};
            margin-bottom: 5px;        /* تقليل المسافة */
            line-height: 1;
        }}

        .neon-subtitle {{
            font-family: 'Orbitron', sans-serif;
            font-size: 0.7rem;         /* أصغر قليلاً */
            letter-spacing: 8px;
            text-align: center;
            color: {NeonTheme.SECONDARY_GLOW};
            text-shadow: 0 0 5px {NeonTheme.SECONDARY_GLOW};
            margin-bottom: 30px;       /* تقليل المسافة */
            text-transform: uppercase;
        }}

        /* الكروت: padding و margin أصغر */
        .neon-card {{
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid rgba(91, 141, 239, 0.2);  /* لون معدل */
            border-radius: 20px;        /* من 30 إلى 20 */
            padding: 25px;              /* من 50px إلى 25px */
            margin: 15px auto;          /* من 30px إلى 15px */
            max-width: 850px;           /* أصغر قليلاً */
            box-shadow: 0 0 20px rgba(91, 141, 239, 0.05);
            transition: 0.5s;
        }}
        .neon-card:hover {{
            border-color: {NeonTheme.MAIN_GLOW};
            box-shadow: 0 0 30px rgba(91, 141, 239, 0.2);
        }}

        /* حقل الإدخال: ارتفاع أقل */
        .stTextArea textarea {{
            background: #000 !important;
            border: 1px solid #222 !important;
            border-radius: 15px !important;  /* من 20 إلى 15 */
            color: {NeonTheme.SECONDARY_GLOW} !important;
            font-size: 1rem !important;      /* أصغر */
            padding: 15px !important;         /* من 25px إلى 15px */
            box-shadow: inset 0 0 10px rgba(0, 212, 255, 0.05) !important;
            height: 180px !important;         /* إضافة ارتفاع محدد أصغر */
        }}

        /* الزر: عرض أقل */
        .stButton > button {{
            background: transparent !important;
            color: #fff !important;
            border: 2px solid {NeonTheme.MAIN_GLOW} !important;
            font-family: 'Orbitron', sans-serif !important;
            border-radius: 50px !important;   /* من 100 إلى 50 */
            padding: 12px 30px !important;     /* من 20px 60px إلى 12px 30px */
            width: auto !important;
            min-width: 250px;                  /* من 350px إلى 250px */
            display: block;
            margin: 20px auto !important;      /* من 40px إلى 20px */
            transition: 0.4s all !important;
            text-shadow: 0 0 10px {NeonTheme.MAIN_GLOW};
            box-shadow: 0 0 15px {NeonTheme.MAIN_GLOW}44;
            font-size: 0.8rem;                /* خط أصغر */
        }}
        .stButton > button:hover {{
            background: {NeonTheme.MAIN_GLOW} !important;
            box-shadow: 0 0 40px {NeonTheme.MAIN_GLOW} !important;
            transform: scale(1.05);
        }}

        /* نتيجة التصنيف: حجم أصغر */
        .neon-result {{
            font-family: 'Instrument Serif', serif;
            font-size: 5rem;              /* من 8rem إلى 5rem */
            font-style: italic;
            text-align: center;
            color: {NeonTheme.SECONDARY_GLOW};
            text-shadow: 0 0 30px {NeonTheme.SECONDARY_GLOW};
            margin: 10px 0;               /* تقليل المسافة */
        }}

        /* فريق العمل */
        .member-glow {{
            background: rgba(0, 212, 255, 0.05);
            border: 1px solid rgba(0, 212, 255, 0.1);
            padding: 8px 18px;             /* أصغر */
            border-radius: 50px;
            font-size: 0.8rem;
            color: {NeonTheme.SECONDARY_GLOW};
            text-shadow: 0 0 5px {NeonTheme.SECONDARY_GLOW};
            font-weight: 600;
        }}

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
# 4. NEON VISUALIZER (بدون تغيير جوهري)
# ==============================================================================
class NeonViz:
    @staticmethod
    def plot_glow_bars(data):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 4))  # أصغر قليلاً
        fig.patch.set_facecolor('#020202')
        ax.set_facecolor('#020202')
        
        colors = ['#111111'] * len(data["classes"])
        target_idx = list(data["classes"]).index(data["category"])
        colors[target_idx] = NeonTheme.SECONDARY_GLOW

        bars = ax.barh(data["classes"], data["all_probs"], color=colors, edgecolor=colors, height=0.5)
        
        for i, bar in enumerate(bars):
            if i == target_idx:
                bar.set_linewidth(2)
                bar.set_edgecolor('#fff')

        ax.set_title("NEURAL PROBABILITY SPECTRUM", color=NeonTheme.SECONDARY_GLOW, 
                     fontsize=8, pad=20, fontfamily='Orbitron', weight='bold')
        
        ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
        ax.tick_params(axis='both', colors='#444', labelsize=9)
        plt.tight_layout()
        return fig

# ==============================================================================
# 5. MAIN INTERFACE (تم تقليل المسافات)
# ==============================================================================
def main():
    apply_neon_styles()
    brain = AIClassifier()
    
    with st.spinner("Powering up Neural Cores..."):
        loaded = brain.setup()

    # --- Header (مسافة علوية أقل) ---
    st.markdown(f"""
    <div style="margin-top: 20px;">   <!-- من 50px إلى 20px -->
        <h1 class="neon-title">{NeonTheme.NAME}</h1>
        <p class="neon-subtitle">{NeonTheme.TAGLINE}</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Project Abstract (كارد أصغر) ---
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#555; letter-spacing:2px; font-size:0.6rem; font-family:Orbitron;">PROJECT DATA</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center; color:#aaa; font-size:0.9rem; line-height:1.6; margin-top:10px;">
        منصة <b>NewsIQ</b> هي محرك ذكاء اصطناعي فائق التطور يعتمد على معالجة اللغات الطبيعية (NLP). 
        النظام مصمم لفك شفرة النصوص الخبرية وتصنيفها آلياً باستخدام مصفوفات رياضية معقدة. 
        هذا الابتكار يقلل التدخل البشري في فرز الأخبار ويزيد من سرعة استجابة الأنظمة الذكية للمعلومات المتدفقة.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Input Terminal (كارد أصغر) ---
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.markdown('<p style="color:#444; letter-spacing:2px; font-size:0.6rem; font-family:Orbitron; margin-bottom:10px;">INPUT_STREAM_TERMINAL</p>', unsafe_allow_html=True)
    
    user_text = st.text_area("Input", placeholder="Enter raw news text for neural analysis...", height=180, label_visibility="collapsed")  # ارتفاع 180
    
    selected_mdl = st.selectbox("Select Neural Core", list(brain.models.keys()) if loaded else ["Offline"])
    
    analyze_btn = st.button("INITIATE NEURAL SCAN")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Result Execution (مع مسافات أقل) ---
    if analyze_btn and user_text:
        if loaded:
            with st.spinner("Decoding semantics..."):
                res = brain.predict(user_text, selected_mdl)
                
                st.markdown('<div class="neon-card" style="border-color: #00d4ff44;">', unsafe_allow_html=True)
                st.markdown(f'<p style="text-align:center; color:#555; letter-spacing:3px; font-family:Orbitron; font-size:0.7rem;">INFERENCE_RESULT</p>', unsafe_allow_html=True)
                st.markdown(f'<h2 class="neon-result">{res["category"]}</h2>', unsafe_allow_html=True)
                st.markdown(f'<p style="text-align:center; font-family:Orbitron; color:{NeonTheme.MAIN_GLOW}; font-size:1rem;">ACCURACY_LEVEL: {res["score"]*100:.2f}%</p>', unsafe_allow_html=True)
                
                st.markdown('<div style="margin-top:30px; padding-top:20px; border-top:1px solid #111;">', unsafe_allow_html=True)
                st.pyplot(NeonViz.plot_glow_bars(res))
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # --- Footer & Team (مسافات أقل) ---
    st.markdown(f"""
    <div style="margin-top:60px; padding:50px 0; border-top:1px solid #111; text-align:center;">
        <p style="letter-spacing:8px; color:#222; font-size:0.7rem; margin-bottom:30px; font-family:Orbitron;">UNDER THE GUIDANCE OF YOUSSEF AL-BAROUDI</p>
        <div style="display:flex; justify-content:center; flex-wrap:wrap; gap:15px;">
            {"".join([f'<div class="member-glow">{name}</div>' for name in NeonTheme.TEAM])}
        </div>
        <p style="margin-top:30px; font-family:Orbitron; font-size:0.5rem; color:#111; letter-spacing:3px;">
            SYSTEM STATUS: OPERATIONAL | BUILD: 2026.05
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
