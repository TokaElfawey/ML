import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. SOFT NEON IDENTITY (Eye-Comfort Edition)
# ==============================================================================
class SoftTheme:
    NAME = "NewsIQ Intelligence"
    TAGLINE = "DECODING NEURAL SIGNALS"
    VERSION = "v4.2.0 — Soft Vision"
    
    # أسماء الفريق
    TEAM = ["آية احمد", "تقي نصر", "تقي علاء", "همت حمدي", "نورهان مدحت"]
    
    # بالتة الألوان الهادئة (Muted Palette)
    MAIN_SOFT = "#94a3b8"      # أزرق ضبابي هادئ للـ Subtitles
    ACCENT_GLOW = "#818cf8"    # بنفسجي لافندر ناعم (مريح للعين)
    BG_DEEP = "#0f172a"        # أزرق ليلي غامق جداً (أفضل من الأسود الصريح)
    TEXT_SILVER = "#e2e8f0"    # فضي فاتح للنصوص

# ==============================================================================
# 2. REFINED SOFT-GLOW CSS (SYMMETRIC BOXES)
# ==============================================================================
def load_soft_styles():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600&family=Space+Grotesk:wght@300;500;700&family=Instrument+Serif:ital@0;1&display=swap');

        .stApp {{
            background-color: {SoftTheme.BG_DEEP};
            color: {SoftTheme.TEXT_SILVER};
            font-family: 'Plus Jakarta Sans', sans-serif;
        }}

        /* اسم المشروع بلمعة هادئة */
        .soft-title {{
            font-family: 'Instrument Serif', serif;
            font-size: 7.5rem;
            text-align: center;
            background: linear-gradient(180deg, #ffffff 30%, {SoftTheme.ACCENT_GLOW} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
            line-height: 0.9;
        }}

        .soft-subtitle {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.85rem;
            letter-spacing: 10px;
            text-align: center;
            color: {SoftTheme.MAIN_SOFT};
            margin-bottom: 60px;
            text-transform: uppercase;
        }}

        /* كروت متناظرة (Symmetrical Cards) */
        .glass-card {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 35px;
            padding: 55px;
            margin: 35px auto;
            max-width: 950px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
        }}

        /* ضبط المستطيلات الخاصة بالأسماء (Symmetry Fix) */
        .team-container {{
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 40px;
        }}
        
        .member-box {{
            background: rgba(129, 140, 248, 0.08); /* لون لافندر شفاف */
            border: 1px solid rgba(129, 140, 248, 0.2);
            padding: 12px 28px;
            border-radius: 100px;
            font-size: 0.95rem;
            color: {SoftTheme.ACCENT_GLOW};
            font-weight: 500;
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 140px; /* ضمان تماثل الحجم */
            text-align: center;
            transition: 0.3s all;
        }}
        .member-box:hover {{
            background: {SoftTheme.ACCENT_GLOW};
            color: #ffffff;
            box-shadow: 0 0 20px {SoftTheme.ACCENT_GLOW}55;
        }}

        /* زر التحليل المريح */
        .stButton > button {{
            background: {SoftTheme.ACCENT_GLOW} !important;
            color: #ffffff !important;
            border: none !important;
            font-family: 'Space Grotesk', sans-serif !important;
            border-radius: 100px !important;
            padding: 22px 70px !important;
            font-weight: 700 !important;
            display: block;
            margin: 40px auto !important;
            transition: 0.4s all !important;
            box-shadow: 0 10px 30px {SoftTheme.ACCENT_GLOW}33 !important;
        }}
        .stButton > button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 15px 40px {SoftTheme.ACCENT_GLOW}66 !important;
        }}

        /* حقول الإدخال */
        .stTextArea textarea {{
            background: rgba(0, 0, 0, 0.2) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 25px !important;
            color: #fff !important;
            padding: 30px !important;
            font-size: 1.15rem !important;
        }}

        #MainMenu, footer, header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. ANALYTICS ENGINE
# ==============================================================================
class CoreEngine:
    def __init__(self):
        self.ready = False
        self.tfidf = None
        self.le = None
        self.models = {}

    def start(self):
        try:
            self.tfidf = joblib.load("tfidf_vectorizer.joblib")
            self.le = joblib.load("label_encoder.joblib")
            self.models = {
                "Engine Alpha (Linear)": joblib.load("linear_svc_model.joblib"),
                "Engine Beta (Logistic)": joblib.load("logistic_regression_model.joblib"),
                "Engine Gamma (Naive)": joblib.load("naive_bayes_model.joblib")
            }
            self.ready = True
            return True
        except:
            return False

    def predict_flow(self, text, key):
        t0 = time.time()
        # Preprocessing
        clean = text.lower()
        clean = re.sub(r'[^a-z\s]', '', clean)
        # Process
        v = self.tfidf.transform([clean])
        m = self.models[key]
        idx = m.predict(v)[0]
        label = self.le.inverse_transform([idx])[0]
        
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(v)[0]
        else:
            d = m.decision_function(v)[0]
            ex = np.exp(d - np.max(d))
            p = ex / ex.sum()
            
        return {
            "label": label,
            "conf": p[idx],
            "all": p,
            "classes": self.le.classes_,
            "time": time.time() - t0
        }

# ==============================================================================
# 4. DATA VISUALS (EYE-COMFORT)
# ==============================================================================
class SoftPlots:
    @staticmethod
    def show_dist(data):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor(SoftTheme.BG_DEEP)
        ax.set_facecolor(SoftTheme.BG_DEEP)
        
        # ألوان البارات الهادئة
        colors = ['#1e293b'] * len(data["classes"])
        try:
            win_idx = list(data["classes"]).index(data["label"])
            colors[win_idx] = SoftTheme.ACCENT_GLOW
        except: pass

        ax.barh(data["classes"], data["all"], color=colors, edgecolor=None, height=0.6)
        
        ax.set_title("PROBABILITY SPECTRUM", color=SoftTheme.MAIN_SOFT, 
                     fontsize=9, pad=30, fontfamily='Space Grotesk', letterspacing=3)
        
        ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
        ax.tick_params(axis='both', colors='#475569', labelsize=11)
        plt.tight_layout()
        return fig

# ==============================================================================
# 5. UI LAYOUT
# ==============================================================================
def run_app():
    load_soft_styles()
    engine = CoreEngine()
    
    with st.spinner("Booting Intelligence..."):
        engine_up = engine.start()

    # --- Header ---
    st.markdown(f"""
    <div style="margin-top: 70px;">
        <h1 class="soft-title">{SoftTheme.NAME}</h1>
        <p class="soft-subtitle">{SoftTheme.TAGLINE}</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Project Abstract ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#475569; letter-spacing:4px; font-size:0.75rem; font-family:Space Grotesk;">ARCHITECTURE OVERVIEW</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center; color:#94a3b8; font-size:1.15rem; line-height:2; margin-top:25px; font-weight:300; max-width:800px; margin-left:auto; margin-right:auto;">
        يعمل مشروع <b>NewsIQ</b> كجسر بين البيانات النصية الضخمة والتحليلات الذكية. 
        من خلال توظيف خوارزميات التصنيف الآلي، يوفر النظام قدرة فورية على تحديد محتوى الأخبار 
        بدقة عالية، مما يساعد المؤسسات الإعلامية على تنظيم تدفق المعلومات بشكل أوتوماتيكي بالكامل.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Input Section ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p style="color:#475569; letter-spacing:4px; font-size:0.75rem; font-family:Space Grotesk; margin-bottom:25px;">NEURAL_STREAM_INPUT</p>', unsafe_allow_html=True)
    
    user_text = st.text_area("News", placeholder="Enter news content for classification...", height=280, label_visibility="collapsed")
    
    m_col, _ = st.columns([1.5, 1])
    with m_col:
        m_choice = st.selectbox("Intelligence Unit", list(engine.models.keys()) if engine_up else ["Status: Offline"])
    
    exec_btn = st.button("EXECUTE ANALYSIS")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Results ---
    if exec_btn and user_text:
        if engine_up:
            with st.spinner("Decoding..."):
                res = engine.predict_flow(user_text, m_choice)
                
                st.markdown('<div class="glass-card" style="border-color: rgba(129, 140, 248, 0.2);">', unsafe_allow_html=True)
                st.markdown(f'<p style="text-align:center; color:#475569; letter-spacing:6px; font-family:Space Grotesk; font-size:0.8rem;">OUTPUT_SIGNAL</p>', unsafe_allow_html=True)
                st.markdown(f'<h2 style="font-family:Instrument Serif; font-size:7rem; font-style:italic; text-align:center; color:{SoftTheme.ACCENT_GLOW}; margin:15px 0;">{res["label"]}</h2>', unsafe_allow_html=True)
                st.markdown(f'<p style="text-align:center; font-family:Space Grotesk; color:{SoftTheme.MAIN_SOFT}; font-size:1.3rem;">RELIABILITY: {res["conf"]*100:.2f}%</p>', unsafe_allow_html=True)
                
                st.markdown('<div style="margin-top:70px; padding-top:50px; border-top:1px solid rgba(255,255,255,0.03);">', unsafe_allow_html=True)
                st.pyplot(SoftPlots.show_dist(res))
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # --- Footer & Team (Fixed Rectangles) ---
    st.markdown(f"""
    <div style="margin-top:160px; padding:100px 0; border-top:1px solid rgba(255,255,255,0.02); text-align:center;">
        <p style="letter-spacing:12px; color:#334155; font-size:0.85rem; margin-bottom:50px; font-family:Space Grotesk;">UNDER THE GUIDANCE OF YOUSSEF AL-BAROUDI</p>
        
        <div class="team-container">
            {"".join([f'<div class="member-box">{name}</div>' for name in SoftTheme.TEAM])}
        </div>
        
        <p style="margin-top:70px; font-family:Space Grotesk; font-size:0.7rem; color:#1e293b; letter-spacing:8px;">
            COMPUTER SCIENCE • SECTION 1 • 2026
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    run_app()
