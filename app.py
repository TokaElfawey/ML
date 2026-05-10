import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import time
import matplotlib.pyplot as plt

# ==============================================================================
# 1. BRANDING & REFINED THEME (UI/UX Optimized)
# ==============================================================================
class TechTheme:
    NAME = "NewsIQ Intelligence"
    TAGLINE = "Advanced Neural NLP Classifier" # تم إضافة التاج لاين المفقود
    TEAM = ["آية احمد", "تقي نصر", "تقي علاء", "همت حمدي", "نورهان مدحت"]
    
    # ألوان مريحة للعين (Deep Navy & Soft Mint)
    BG_COLOR = "#0E1117"        # رمادي داكن جداً (ليس أسود مطلق)
    CARD_BG = "#1A1C24"         # لون الكروت
    PRIMARY = "#4DEEB2"         # أخضر مينت مريح
    SECONDARY = "#5865F2"       # أزرق هادئ
    TEXT_COLOR = "#E0E0E0"      # أبيض مطفي

# ==============================================================================
# 2. SOPHISTICATED CSS
# ==============================================================================
def apply_custom_styles():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=JetBrains+Mono&display=swap');

        .stApp {{
            background-color: {TechTheme.BG_COLOR};
            color: {TechTheme.TEXT_COLOR};
            font-family: 'Inter', sans-serif;
        }}

        .main-title {{
            font-size: 3.5rem;
            font-weight: 800;
            text-align: center;
            background: linear-gradient(45deg, {TechTheme.PRIMARY}, {TechTheme.SECONDARY});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
        }}

        .sub-title {{
            text-align: center;
            color: #888;
            letter-spacing: 2px;
            text-transform: uppercase;
            font-size: 0.9rem;
            margin-bottom: 40px;
        }}

        .custom-card {{
            background: {TechTheme.CARD_BG};
            border: 1px solid #2D3139;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }}

        .stTextArea textarea {{
            background-color: #0B0D11 !important;
            color: #D1D1D1 !important;
            border: 1px solid #2D3139 !important;
            border-radius: 12px !important;
        }}

        .stButton > button {{
            width: 100%;
            background: linear-gradient(90deg, {TechTheme.SECONDARY}, {TechTheme.PRIMARY}) !important;
            color: white !important;
            border: none !important;
            padding: 12px !important;
            font-weight: bold !important;
            border-radius: 8px !important;
            transition: 0.3s all;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(77, 238, 178, 0.3) !important;
        }}

        .result-text {{
            font-size: 2.5rem;
            color: {TechTheme.PRIMARY};
            text-align: center;
            font-weight: bold;
        }}

        .member-badge {{
            background: #252932;
            padding: 5px 15px;
            border-radius: 20px;
            border: 1px solid #3E4451;
            font-size: 0.85rem;
            color: #BBB;
        }}
        
        /* Hide Streamlit elements */
        #MainMenu, footer, header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. AI ENGINE (Optimized with Caching)
# ==============================================================================
class AIClassifier:
    def __init__(self):
        self.ready = False
        self.tfidf = None
        self.le = None
        self.models = {}

    @st.cache_resource
    def _load_resources(_self): # استخدام الكاش لتسريع الأداء
        try:
            tfidf = joblib.load("tfidf_vectorizer.joblib")
            le = joblib.load("label_encoder.joblib")
            models = {
                "Neural Linear Core": joblib.load("linear_svc_model.joblib"),
                "Logistic Matrix Core": joblib.load("logistic_regression_model.joblib"),
                "Naive Bayes Core": joblib.load("naive_bayes_model.joblib")
            }
            return tfidf, le, models, True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return None, None, {}, False

    def setup(self):
        self.tfidf, self.le, self.models, self.ready = self._load_resources()
        return self.ready

    def predict(self, text, model_key):
        start = time.time()
        # تنظيف يدعم العربي والإنجليزي بشكل مبسط
        clean = re.sub(r'[^\w\s]', '', text.lower())
        vec = self.tfidf.transform([clean])
        mdl = self.models[model_key]
        
        idx = mdl.predict(vec)[0]
        cat = self.le.inverse_transform([idx])[0]
        
        if hasattr(mdl, "predict_proba"):
            probs = mdl.predict_proba(vec)[0]
        else:
            d = mdl.decision_function(vec)[0]
            probs = np.exp(d) / np.sum(np.exp(d))
            
        return {
            "category": cat,
            "score": probs[idx],
            "all_probs": probs,
            "classes": self.le.classes_,
            "latency": time.time() - start
        }

# ==============================================================================
# 4. MAIN INTERFACE
# ==============================================================================
def main():
    apply_custom_styles()
    brain = AIClassifier()
    loaded = brain.setup()

    # --- Header ---
    st.markdown(f'<h1 class="main-title">{TechTheme.NAME}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-title">{TechTheme.TAGLINE}</p>', unsafe_allow_html=True)

    # --- Layout ---
    col1, col2 = st.columns([1.2, 0.8], gap="large")

    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Analysis Terminal")
        user_text = st.text_area("", placeholder="Paste news content here for deep analysis...", height=250)
        
        c_load1, c_load2 = st.columns(2)
        with c_load1:
            selected_mdl = st.selectbox("Intelligence Core", list(brain.models.keys()) if loaded else ["Offline"])
        with c_load2:
            st.write(" ") # Alignment
            st.write(" ")
            analyze_btn = st.button("RUN ANALYSIS")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="custom-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown("### ℹ️ About Project")
        st.info("NewsIQ uses Natural Language Processing (NLP) to classify news in real-time with high precision.")
        st.markdown(f"""
        **System Specs:**
        - Build: 2026.05
        - Cores: {len(brain.models)} Active
        - Status: {'🟢 Online' if loaded else '🔴 Error'}
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Results ---
    if analyze_btn and user_text:
        if loaded:
            res = brain.predict(user_text, selected_mdl)
            
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            res_c1, res_c2 = st.columns([1, 1])
            
            with res_c1:
                st.markdown(f'<p style="color:#888;">PREDICTED CATEGORY</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-text">{res["category"]}</p>', unsafe_allow_html=True)
                st.metric("Confidence Score", f"{res['score']*100:.2f}%")
                st.caption(f"Processing time: {res['latency']*1000:.2f}ms")

            with res_c2:
                # Simple Plot
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor(TechTheme.CARD_BG)
                ax.set_facecolor(TechTheme.CARD_BG)
                
                y_pos = np.arange(len(res["classes"]))
                ax.barh(y_pos, res["all_probs"], color=TechTheme.SECONDARY)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(res["classes"], color="white")
                ax.tick_params(axis='x', colors='#555')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Footer ---
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center;">
        <p style="color:#666; font-size:0.8rem; margin-bottom:20px;">GUIDED BY YOUSSEF AL-BAROUDI</p>
        <div style="display:flex; justify-content:center; gap:10px; flex-wrap:wrap;">
            {''.join([f'<span class="member-badge">{n}</span>' for n in TechTheme.TEAM])}
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
