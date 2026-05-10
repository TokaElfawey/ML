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
    # الأسماء المحدثة
    TEAM = ["آية احمد", "تقي نصر", "تقي علاء", "همت حمدي", "نورهان مدحت"]
    
    # ألوان النيون
    MAIN_GLOW = "#7c6af7"
    SECONDARY_GLOW = "#00f2ff"
    DEEP_BLACK = "#020202"

# ==============================================================================
# 2. CYBER-NEON CSS (THE GLOW ENGINE)
# ==============================================================================
def apply_neon_styles():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Plus+Jakarta+Sans:wght@300;400;600&family=Instrument+Serif:ital@0;1&display=swap');

        /* الخلفية المظلمة جداً */
        .stApp {{
            background-color: {NeonTheme.DEEP_BLACK};
            color: #ffffff;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }}

        /* تأثير التوهج لاسم المشروع */
        .neon-title {{
            font-family: 'Instrument Serif', serif;
            font-size: 7rem;
            text-align: center;
            color: #fff;
            text-shadow: 
                0 0 10px {NeonTheme.MAIN_GLOW},
                0 0 20px {NeonTheme.MAIN_GLOW},
                0 0 40px {NeonTheme.MAIN_GLOW};
            margin-bottom: 10px;
            line-height: 1;
        }}

        .neon-subtitle {{
            font-family: 'Orbitron', sans-serif;
            font-size: 0.8rem;
            letter-spacing: 10px;
            text-align: center;
            color: {NeonTheme.SECONDARY_GLOW};
            text-shadow: 0 0 5px {NeonTheme.SECONDARY_GLOW};
            margin-bottom: 50px;
            text-transform: uppercase;
        }}

        /* الكروت المتوهجة */
        .neon-card {{
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid rgba(124, 106, 247, 0.2);
            border-radius: 30px;
            padding: 50px;
            margin: 30px auto;
            max-width: 950px;
            box-shadow: 0 0 20px rgba(124, 106, 247, 0.05);
            transition: 0.5s;
        }}
        .neon-card:hover {{
            border-color: {NeonTheme.MAIN_GLOW};
            box-shadow: 0 0 30px rgba(124, 106, 247, 0.2);
        }}

        /* المدخلات */
        .stTextArea textarea {{
            background: #000 !important;
            border: 1px solid #222 !important;
            border-radius: 20px !important;
            color: {NeonTheme.SECONDARY_GLOW} !important;
            font-size: 1.2rem !important;
            padding: 25px !important;
            box-shadow: inset 0 0 10px rgba(0, 242, 255, 0.05) !important;
        }}

        /* الزرار المضيء */
        .stButton > button {{
            background: transparent !important;
            color: #fff !important;
            border: 2px solid {NeonTheme.MAIN_GLOW} !important;
            font-family: 'Orbitron', sans-serif !important;
            border-radius: 100px !important;
            padding: 20px 60px !important;
            width: auto !important;
            min-width: 350px;
            display: block;
            margin: 40px auto !important;
            transition: 0.4s all !important;
            text-shadow: 0 0 10px {NeonTheme.MAIN_GLOW};
            box-shadow: 0 0 15px {NeonTheme.MAIN_GLOW}44;
        }}
        .stButton > button:hover {{
            background: {NeonTheme.MAIN_GLOW} !important;
            box-shadow: 0 0 40px {NeonTheme.MAIN_GLOW} !important;
            transform: scale(1.05);
        }}

        /* نتيجة التصنيف */
        .neon-result {{
            font-family: 'Instrument Serif', serif;
            font-size: 8rem;
            font-style: italic;
            text-align: center;
            color: {NeonTheme.SECONDARY_GLOW};
            text-shadow: 0 0 30px {NeonTheme.SECONDARY_GLOW};
            margin: 20px 0;
        }}

        /* فريق العمل */
        .member-glow {{
            background: rgba(0, 242, 255, 0.05);
            border: 1px solid rgba(0, 242, 255, 0.1);
            padding: 12px 25px;
            border-radius: 100px;
            font-size: 0.9rem;
            color: {NeonTheme.SECONDARY_GLOW};
            text-shadow: 0 0 5px {NeonTheme.SECONDARY_GLOW};
            font-weight: 600;
        }}

        #MainMenu, footer, header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. AI ENGINE CLASS
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
# 4. NEON VISUALIZER
# ==============================================================================
class NeonViz:
    @staticmethod
    def plot_glow_bars(data):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#020202')
        ax.set_facecolor('#020202')
        
        # ألوان النيون للبارات
        colors = ['#111111'] * len(data["classes"])
        target_idx = list(data["classes"]).index(data["category"])
        colors[target_idx] = NeonTheme.SECONDARY_GLOW

        bars = ax.barh(data["classes"], data["all_probs"], color=colors, edgecolor=colors, height=0.6)
        
        # إضافة توهج بسيط للبار المختار (محاكاة)
        for i, bar in enumerate(bars):
            if i == target_idx:
                bar.set_linewidth(2)
                bar.set_edgecolor('#fff')

        ax.set_title("NEURAL PROBABILITY SPECTRUM", color=NeonTheme.SECONDARY_GLOW, 
                     fontsize=10, pad=30, fontfamily='Orbitron', weight='bold')
        
        ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
        ax.tick_params(axis='both', colors='#444', labelsize=11)
        plt.tight_layout()
        return fig

# ==============================================================================
# 5. MAIN INTERFACE
# ==============================================================================
def main():
    apply_neon_styles()
    brain = AIClassifier()
    
    with st.spinner("Powering up Neural Cores..."):
        loaded = brain.setup()

    # --- Header ---
    st.markdown(f"""
    <div style="margin-top: 50px;">
        <h1 class="neon-title">{NeonTheme.NAME}</h1>
        <p class="neon-subtitle">{NeonTheme.TAGLINE}</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Project Abstract ---
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#555; letter-spacing:3px; font-size:0.7rem; font-family:Orbitron;">PROJECT DATA</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center; color:#aaa; font-size:1.1rem; line-height:1.8; margin-top:20px;">
        منصة <b>NewsIQ</b> هي محرك ذكاء اصطناعي فائق التطور يعتمد على معالجة اللغات الطبيعية (NLP). 
        النظام مصمم لفك شفرة النصوص الخبرية وتصنيفها آلياً باستخدام مصفوفات رياضية معقدة. 
        هذا الابتكار يقلل التدخل البشري في فرز الأخبار ويزيد من سرعة استجابة الأنظمة الذكية للمعلومات المتدفقة.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Input Terminal ---
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.markdown('<p style="color:#444; letter-spacing:3px; font-size:0.7rem; font-family:Orbitron; margin-bottom:20px;">INPUT_STREAM_TERMINAL</p>', unsafe_allow_html=True)
    
    user_text = st.text_area("Input", placeholder="Enter raw news text for neural analysis...", height=280, label_visibility="collapsed")
    
    selected_mdl = st.selectbox("Select Neural Core", list(brain.models.keys()) if loaded else ["Offline"])
    
    analyze_btn = st.button("INITIATE NEURAL SCAN")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Result Execution ---
    if analyze_btn and user_text:
        if loaded:
            with st.spinner("Decoding semantics..."):
                res = brain.predict(user_text, selected_mdl)
                
                # Glowing Result
                st.markdown('<div class="neon-card" style="border-color: #00f2ff44;">', unsafe_allow_html=True)
                st.markdown(f'<p style="text-align:center; color:#555; letter-spacing:5px; font-family:Orbitron; font-size:0.8rem;">INFERENCE_RESULT</p>', unsafe_allow_html=True)
                st.markdown(f'<h2 class="neon-result">{res["category"]}</h2>', unsafe_allow_html=True)
                st.markdown(f'<p style="text-align:center; font-family:Orbitron; color:{NeonTheme.MAIN_GLOW}; font-size:1.2rem;">ACCURACY_LEVEL: {res["score"]*100:.2f}%</p>', unsafe_allow_html=True)
                
                # Probability Map (The Glow Map)
                st.markdown('<div style="margin-top:60px; padding-top:40px; border-top:1px solid #111;">', unsafe_allow_html=True)
                st.pyplot(NeonViz.plot_glow_bars(res))
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # --- Footer & Team ---
    st.markdown(f"""
    <div style="margin-top:150px; padding:100px 0; border-top:1px solid #111; text-align:center;">
        <p style="letter-spacing:10px; color:#222; font-size:0.8rem; margin-bottom:50px; font-family:Orbitron;">UNDER THE GUIDANCE OF YOUSSEF AL-BAROUDI</p>
        <div style="display:flex; justify-content:center; flex-wrap:wrap; gap:20px;">
            {"".join([f'<div class="member-glow">{name}</div>' for name in NeonTheme.TEAM])}
        </div>
        <p style="margin-top:60px; font-family:Orbitron; font-size:0.6rem; color:#111; letter-spacing:5px;">
            SYSTEM STATUS: OPERATIONAL | BUILD: 2026.05
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
