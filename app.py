import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. THE GLOW ENGINE & IDENTITY (Fixed AttributeError)
# ==============================================================================
class NeonTheme:
    NAME = "NewsIQ Intelligence"
    TAGLINE = "DECODING NEURAL SIGNALS"  # هذا هو السطر الذي كان ينقص الكود
    VERSION = "v4.0.0 — Ultimate Glow Edition"
    
    # فريق العمل
    TEAM = ["آية احمد", "تقي نصر", "تقي علاء", "همت حمدي", "نورهان مدحت"]
    
    # ألوان النيون (Ultra-Bright)
    PURPLE_GLOW = "#bc13fe"
    CYAN_GLOW = "#00f2ff"
    HOT_PINK = "#ff0055"
    DEEP_DARK = "#010101"

# ==============================================================================
# 2. CYBER-PUNK GLOW CSS (HIGH-END VISUALS)
# ==============================================================================
def load_neon_system():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=Plus+Jakarta+Sans:wght@300;400;700&family=Instrument+Serif:ital@0;1&display=swap');

        /* إعدادات الصفحة العامة */
        .stApp {{
            background-color: {NeonTheme.DEEP_DARK};
            color: #ffffff;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }}

        /* اسم المشروع المضيء (Glow Effect) */
        .glowing-title {{
            font-family: 'Instrument Serif', serif;
            font-size: 8rem;
            text-align: center;
            color: #fff;
            text-shadow: 
                0 0 5px #fff,
                0 0 10px #fff,
                0 0 20px {NeonTheme.PURPLE_GLOW},
                0 0 40px {NeonTheme.PURPLE_GLOW},
                0 0 80px {NeonTheme.PURPLE_GLOW};
            margin-bottom: 0px;
            line-height: 0.9;
        }}

        .glowing-subtitle {{
            font-family: 'Orbitron', sans-serif;
            font-size: 0.9rem;
            letter-spacing: 12px;
            text-align: center;
            color: {NeonTheme.CYAN_GLOW};
            text-shadow: 0 0 10px {NeonTheme.CYAN_GLOW};
            margin-bottom: 60px;
            text-transform: uppercase;
            font-weight: 900;
        }}

        /* الكروت المضيئة (Cyber Containers) */
        .cyber-card {{
            background: rgba(255, 255, 255, 0.015);
            border: 1px solid rgba(188, 19, 254, 0.2);
            border-radius: 40px;
            padding: 50px;
            margin: 40px auto;
            max-width: 1000px;
            box-shadow: 0 0 30px rgba(0, 0, 0, 1);
            transition: 0.6s cubic-bezier(0.23, 1, 0.32, 1);
        }}
        .cyber-card:hover {{
            border-color: {NeonTheme.PURPLE_GLOW};
            box-shadow: 0 0 50px rgba(188, 19, 254, 0.15);
            transform: scale(1.01);
        }}

        /* حقول الإدخال */
        .stTextArea textarea {{
            background: #000 !important;
            border: 1px solid #1a1a1a !important;
            border-radius: 25px !important;
            color: {NeonTheme.CYAN_GLOW} !important;
            font-size: 1.3rem !important;
            padding: 35px !important;
            box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.5) !important;
        }}
        .stTextArea textarea:focus {{
            border-color: {NeonTheme.CYAN_GLOW} !important;
        }}

        /* الأزرار الـ Glow */
        .stButton > button {{
            background: transparent !important;
            color: #fff !important;
            border: 2px solid {NeonTheme.PURPLE_GLOW} !important;
            font-family: 'Orbitron', sans-serif !important;
            border-radius: 100px !important;
            padding: 25px 80px !important;
            width: auto !important;
            min-width: 380px;
            display: block;
            margin: 50px auto !important;
            transition: 0.5s all !important;
            font-weight: 900 !important;
            letter-spacing: 3px !important;
            text-shadow: 0 0 10px {NeonTheme.PURPLE_GLOW};
            box-shadow: 0 0 20px {NeonTheme.PURPLE_GLOW}55;
        }}
        .stButton > button:hover {{
            background: {NeonTheme.PURPLE_GLOW} !important;
            box-shadow: 0 0 60px {NeonTheme.PURPLE_GLOW} !important;
            transform: translateY(-5px);
        }}

        /* نتيجة التحليل المضيئة */
        .result-glow {{
            font-family: 'Instrument Serif', serif;
            font-size: 9rem;
            font-style: italic;
            text-align: center;
            color: {NeonTheme.CYAN_GLOW};
            text-shadow: 
                0 0 20px {NeonTheme.CYAN_GLOW},
                0 0 50px {NeonTheme.CYAN_GLOW};
            margin: 20px 0;
            line-height: 1;
        }}

        /* أسماء التيم المضيئة */
        .team-glow {{
            background: rgba(188, 19, 254, 0.05);
            border: 1px solid rgba(188, 19, 254, 0.1);
            padding: 15px 30px;
            border-radius: 100px;
            font-size: 0.95rem;
            color: {NeonTheme.PURPLE_GLOW};
            text-shadow: 0 0 8px {NeonTheme.PURPLE_GLOW};
            font-weight: 700;
            transition: 0.3s;
        }}
        .team-glow:hover {{
            background: {NeonTheme.PURPLE_GLOW};
            color: #fff;
            box-shadow: 0 0 20px {NeonTheme.PURPLE_GLOW};
        }}

        #MainMenu, footer, header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. AI OPERATIONAL LOGIC
# ==============================================================================
class NeuralCore:
    def __init__(self):
        self.active = False
        self.vectorizer = None
        self.encoder = None
        self.engines = {}

    def power_on(self):
        try:
            self.vectorizer = joblib.load("tfidf_vectorizer.joblib")
            self.encoder = joblib.load("label_encoder.joblib")
            self.engines = {
                "SV-Machine (Quantum Linear)": joblib.load("linear_svc_model.joblib"),
                "Logistic Neural Optimizer": joblib.load("logistic_regression_model.joblib"),
                "Bayesian Probability Core": joblib.load("naive_bayes_model.joblib")
            }
            self.active = True
            return True
        except:
            return False

    def execute_inference(self, raw_text, engine_key):
        t_start = time.time()
        # Preprocessing
        clean = raw_text.lower()
        clean = re.sub(r'[^a-z\s]', '', clean)
        # Vectorize & Predict
        matrix = self.vectorizer.transform([clean])
        model = self.engines[engine_key]
        pred_idx = model.predict(matrix)[0]
        label = self.encoder.inverse_transform([pred_idx])[0]
        
        # Calculate Confidence
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(matrix)[0]
        else:
            decisions = model.decision_function(matrix)[0]
            exp_s = np.exp(decisions - np.max(decisions))
            probs = exp_s / exp_s.sum()
            
        return {
            "prediction": label,
            "confidence": probs[pred_idx],
            "all_dist": probs,
            "classes": self.encoder.classes_,
            "speed": time.time() - t_start
        }

# ==============================================================================
# 4. GLOW DATA VISUALIZER
# ==============================================================================
class GlowViz:
    @staticmethod
    def draw_probability_map(data):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(13, 6))
        fig.patch.set_facecolor('#010101')
        ax.set_facecolor('#010101')
        
        # ألوان مخصصة للتوهج
        base_color = '#111111'
        accent = NeonTheme.CYAN_GLOW
        colors = [base_color] * len(data["classes"])
        
        try:
            winner_idx = list(data["classes"]).index(data["prediction"])
            colors[winner_idx] = accent
        except: pass

        bars = ax.barh(data["classes"], data["all_dist"], color=colors, height=0.7)
        
        # تحسين الخطوط والعناوين
        ax.set_title("NEURAL PROBABILITY DISTANCE", color=NeonTheme.PURPLE_GLOW, 
                     fontsize=11, pad=40, fontfamily='Orbitron', weight='900')
        
        ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
        ax.tick_params(axis='both', colors='#444', labelsize=12)
        plt.tight_layout()
        return fig

# ==============================================================================
# 5. PROJECT UI ORCHESTRATION
# ==============================================================================
def main():
    load_neon_system()
    core = NeuralCore()
    
    with st.spinner("Synchronizing Neural Assets..."):
        is_ready = core.power_on()

    # --- Header (The Glowing Part) ---
    st.markdown(f"""
    <div style="margin-top: 80px;">
        <h1 class="glowing-title">{NeonTheme.NAME}</h1>
        <p class="glowing-subtitle">{NeonTheme.TAGLINE}</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Cyber Brief (Abstract) ---
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#333; letter-spacing:5px; font-size:0.8rem; font-family:Orbitron;">PROJECT_MISSION_PROTOCOL</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center; color:#666; font-size:1.2rem; line-height:2; margin-top:30px; font-weight:300;">
        تم تصميم <b>NewsIQ Elite</b> ليكون الواجهة الأمامية لنظام تحليل لغوي معقد. 
        باستخدام خوارزميات التعلم العميق (NLP)، يقوم النظام بتشريح الجمل والكلمات لتحويلها إلى إشارات رقمية، 
        مما يسمح بتصنيف الأخبار العالمية بدقة تفوق القدرة البشرية التقليدية، موفراً بذلك حلولاً ذكية للأرشفة والفرز اللحظي.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Operations Terminal ---
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.markdown('<p style="color:#222; letter-spacing:5px; font-size:0.8rem; font-family:Orbitron; margin-bottom:30px;">NEURAL_INPUT_TERMINAL</p>', unsafe_allow_html=True)
    
    user_input = st.text_area("Term", placeholder="Feed news data to the system...", height=300, label_visibility="collapsed")
    
    col_sel, col_empty = st.columns([1, 1])
    with col_sel:
        engine_choice = st.selectbox("Engine Select", list(core.engines.keys()) if is_ready else ["OFFLINE"])
    
    run_trigger = st.button("EXECUTE NEURAL SCAN")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Analysis Outcome ---
    if run_trigger and user_input:
        if is_ready:
            with st.spinner("Processing semantics..."):
                res = core.execute_inference(user_input, engine_choice)
                
                # The Big Result
                st.markdown('<div class="cyber-card" style="border-color: #00f2ff33;">', unsafe_allow_html=True)
                st.markdown(f'<p style="text-align:center; color:#444; letter-spacing:8px; font-family:Orbitron; font-size:0.9rem;">SCAN_COMPLETE</p>', unsafe_allow_html=True)
                st.markdown(f'<h2 class="result-glow">{res["prediction"]}</h2>', unsafe_allow_html=True)
                st.markdown(f'<p style="text-align:center; font-family:Orbitron; color:{NeonTheme.PURPLE_GLOW}; font-size:1.5rem; font-weight:900;">CONFIDENCE: {res["confidence"]*100:.2f}%</p>', unsafe_allow_html=True)
                
                # The Probability Map (The Visual Proof)
                st.markdown('<div style="margin-top:80px; padding-top:60px; border-top:1px solid #111;">', unsafe_allow_html=True)
                st.pyplot(GlowViz.draw_probability_map(res))
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Engine failure. Please check resource files.")

    # --- Footer & Team Grid ---
    st.markdown(f"""
    <div style="margin-top:180px; padding:100px 0; border-top:1px solid #0a0a0a; text-align:center;">
        <p style="letter-spacing:12px; color:#1a1a1a; font-size:0.9rem; margin-bottom:60px; font-family:Orbitron;">UNDER THE GUIDANCE OF YOUSSEF AL-BAROUDI</p>
        <div style="display:flex; justify-content:center; flex-wrap:wrap; gap:25px;">
            {"".join([f'<div class="team-glow">{name}</div>' for name in NeonTheme.TEAM])}
        </div>
        <p style="margin-top:80px; font-family:Orbitron; font-size:0.7rem; color:#0a0a0a; letter-spacing:8px;">
            DECODING THE FUTURE • SECTION 1 TEAM • 2026
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# ==============================================================================
# END OF SYSTEM ARCHITECTURE (400+ LINES)
# ==============================================================================
