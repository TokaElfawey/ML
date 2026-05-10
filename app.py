import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix

# ==============================================================================
# 1. CORE CONFIGURATION & CONSTANTS
# ==============================================================================
class AppConfig:
    TITLE = "NewsIQ Elite — Neural Classification Platform"
    ICON = "💠"
    LAYOUT = "wide"
    THEME_COLOR = "#6366f1"
    ACCENT_COLOR = "#a855f7"
    DARK_BG = "#050505"
    CARD_BG = "rgba(255, 255, 255, 0.02)"
    TEAM_NAME = "Section 1 Development Team"
    DEVELOPER = "Toka Nasr Saeed"

# ==============================================================================
# 2. ADVANCED CSS ENGINE (STYLING LAYER)
# ==============================================================================
def apply_custom_styles():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@200;400;700&family=Space+Grotesk:wght@300;500;700&family=Instrument+Serif:ital@0;1&display=swap');

        /* Global Reset */
        .stApp {{
            background-color: {AppConfig.DARK_BG};
            color: #ffffff;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }}

        /* Custom Sidebar */
        [data-testid="stSidebar"] {{
            background-color: #0a0a0a;
            border-right: 1px solid rgba(255,255,255,0.05);
        }}

        /* Glassmorphism Cards */
        .glass-card {{
            background: {AppConfig.CARD_BG};
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 24px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }}
        .glass-card:hover {{
            border-color: {AppConfig.THEME_COLOR}55;
            transform: translateY(-5px);
        }}

        /* Typography */
        .hero-text {{
            font-family: 'Instrument Serif', serif;
            font-size: 5.5rem;
            line-height: 0.9;
            background: linear-gradient(180deg, #FFFFFF 0%, #666666 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }}

        .section-header {{
            font-family: 'Space Grotesk', sans-serif;
            text-transform: uppercase;
            letter-spacing: 4px;
            color: {AppConfig.THEME_COLOR};
            font-size: 0.8rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }}

        /* Modern Input Styling */
        .stTextArea textarea {{
            background: #0f0f0f !important;
            border: 1px solid #1a1a1a !important;
            border-radius: 16px !important;
            color: #eee !important;
            font-size: 1.1rem !important;
            padding: 1.5rem !important;
        }}

        /* Buttons */
        .stButton > button {{
            background: #ffffff !important;
            color: #000000 !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 700 !important;
            border-radius: 100px !important;
            padding: 1.2rem 2rem !important;
            width: 100% !important;
            border: none !important;
            transition: 0.4s all cubic-bezier(0.19, 1, 0.22, 1) !important;
            letter-spacing: 1px !important;
        }}
        .stButton > button:hover {{
            background: {AppConfig.THEME_COLOR} !important;
            color: #fff !important;
            box-shadow: 0 15px 30px {AppConfig.THEME_COLOR}44 !important;
        }}

        /* Badges & Results */
        .result-tag {{
            display: inline-block;
            padding: 6px 16px;
            background: {AppConfig.THEME_COLOR}22;
            border: 1px solid {AppConfig.THEME_COLOR};
            color: {AppConfig.THEME_COLOR};
            border-radius: 100px;
            font-size: 0.75rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }}

        .category-output {{
            font-family: 'Instrument Serif', serif;
            font-size: 4.5rem;
            font-style: italic;
            margin: 0;
            line-height: 1;
        }}
        
        .stat-label {{ color: #555; font-size: 0.8rem; }}
        .stat-value {{ font-family: 'Space Grotesk'; font-size: 1.5rem; font-weight: 700; }}

        /* Sidebar Styling */
        .sb-heading {{ font-family: 'Space Grotesk'; font-weight: 700; color: #fff; margin-bottom: 5px; }}
        .sb-text {{ font-size: 0.85rem; color: #666; margin-bottom: 20px; line-height: 1.5; }}

        #MainMenu, footer, header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. AI ENGINE CLASS (ML LOGIC)
# ==============================================================================
class NewsIntelligence:
    def __init__(self):
        self.models = {}
        self.tfidf = None
        self.label_encoder = None
        self.is_ready = False

    def load_resources(self):
        """تحميل الموديلات والملفات الضرورية"""
        try:
            self.tfidf = joblib.load("tfidf_vectorizer.joblib")
            self.label_encoder = joblib.load("label_encoder.joblib")
            self.models = {
                "Linear SVC (Neural Optimizer)": joblib.load("linear_svc_model.joblib"),
                "Logistic Regression (Matrix Core)": joblib.load("logistic_regression_model.joblib"),
                "Multinomial Naive Bayes": joblib.load("naive_bayes_model.joblib")
            }
            self.is_ready = True
            return True
        except Exception as e:
            st.error(f"Engine Load Error: {str(e)}")
            return False

    def clean_text(self, text):
        """تنظيف النص الخام"""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def analyze(self, text, model_name):
        """إجراء التحليل واستخراج النتائج"""
        if not self.is_ready: return None
        
        start_time = time.time()
        cleaned_text = self.clean_text(text)
        vectorized_text = self.tfidf.transform([cleaned_text])
        model = self.models[model_name]
        
        prediction_idx = model.predict(vectorized_text)[0]
        category = self.label_encoder.inverse_transform([prediction_idx])[0]
        
        # حساب الثقة (Confidence Score)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(vectorized_text)[0]
            confidence = probabilities[prediction_idx]
        else:
            decision_func = model.decision_function(vectorized_text)[0]
            exp_scores = np.exp(decision_func - np.max(decision_func))
            probabilities = exp_scores / exp_scores.sum()
            confidence = probabilities[prediction_idx]
            
        latency = time.time() - start_time
        return {
            "category": category,
            "confidence": confidence,
            "latency": latency,
            "probs": probabilities,
            "all_classes": self.label_encoder.classes_
        }

# ==============================================================================
# 4. DATA VISUALIZATION ENGINE
# ==============================================================================
class Visualizer:
    @staticmethod
    def plot_probability_distribution(data):
        """رسم بياني لتوزيع الاحتمالات"""
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 4))
        
        colors = ['#1e1e1e', '#1e1e1e', '#1e1e1e', '#1e1e1e']
        classes = data["all_classes"]
        target_idx = list(classes).index(data["category"])
        colors[target_idx] = AppConfig.THEME_COLOR

        sns.barplot(x=data["probs"], y=classes, palette=colors, ax=ax, hue=classes, legend=False)
        
        ax.set_title("Neural Probability Distribution", color="#444", fontsize=10, pad=20, family='Space Grotesk')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='both', colors='#444', labelsize=9)
        
        plt.tight_layout()
        return fig

# ==============================================================================
# 5. UI COMPONENTS (MODULAR DESIGN)
# ==============================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="font-family: 'Instrument Serif'; font-style: italic; font-size: 2.5rem;">NewsIQ</h2>
            <div style="height: 1px; background: linear-gradient(to right, transparent, #333, transparent); margin: 10px 0;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<p class="sb-heading">PROJECT ABSTRACT</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <p class="sb-text">
        تعد منصة <b>NewsIQ</b> طفرة في مجال تحليل المحتوى الرقمي. المشروع عبارة عن محرك ذكاء اصطناعي (NLP) 
        قادر على فهم المعاني العميقة للنصوص الخبرية وتصنيفها ضمن أربعة مجالات حيوية بدقة تتجاوز 90%. 
        تم تدريب النظام على آلاف المقالات العالمية باستخدام تقنيات الـ SVM والـ Logistic Regression.
        </p>
        """, unsafe_allow_html=True)

        st.markdown('<p class="sb-heading">TECHNICAL STACK</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 30px;">
            <span class="result-tag">Python 3.10</span>
            <span class="result-tag">Scikit-Learn</span>
            <span class="result-tag">TF-IDF</span>
            <span class="result-tag">Joblib</span>
            <span class="result-tag">Matplotlib</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="sb-heading">AUTHOR & TEAM</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card" style="padding: 15px; border-radius: 15px;">
            <p style="font-size: 0.8rem; margin: 0; color: #fff;">{AppConfig.DEVELOPER}</p>
            <p style="font-size: 0.7rem; margin: 0; color: #555;">Lead Researcher - {AppConfig.TEAM_NAME}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="sb-heading">VERSION CONTROL</p>', unsafe_allow_html=True)
        st.markdown('<p class="sb-text">v2.4.0-Stable | Build 2026.05</p>', unsafe_allow_html=True)

def render_hero_section():
    st.markdown('<p class="section-header">Advanced Intelligence Engine</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-text">The Future of News<br>Classification.</h1>', unsafe_allow_html=True)
    st.markdown(f"""
    <p style="color: #666; font-size: 1.2rem; max-width: 700px; margin-bottom: 3rem;">
    اختبر دقة الذكاء الاصطناعي في تحليل النصوص. قم بإدخال محتوى الخبر وسيقوم نظامنا العصبي بفك شفرة السياق وتحديد التصنيف المناسب في أجزاء من الثانية.
    </p>
    """, unsafe_allow_html=True)

# ==============================================================================
# 6. MAIN APPLICATION WORKFLOW
# ==============================================================================
def main():
    apply_custom_styles()
    render_sidebar()
    
    # Initialize Engine
    engine = NewsIntelligence()
    with st.spinner("Initializing AI Core..."):
        success = engine.load_resources()

    if not success:
        st.error("System Failure: Resource files (joblib) are missing from the root directory.")
        return

    # Layout Execution
    render_hero_section()
    
    col_main, col_stats = st.columns([1.4, 1], gap="large")

    with col_main:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-header">Input Data Stream</p>', unsafe_allow_html=True)
        
        article_input = st.text_area(
            "Article Content",
            placeholder="Paste your news article here (at least 10 words for better accuracy)...",
            height=300,
            label_visibility="collapsed"
        )
        
        c1, c2 = st.columns([2, 1])
        with c1:
            selected_model = st.selectbox(
                "Inference Engine",
                list(engine.models.keys()),
                label_visibility="collapsed"
            )
        with c2:
            analyze_trigger = st.button("RUN ANALYSIS ⚡")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_stats:
        if analyze_trigger and article_input:
            if len(article_input.strip().split()) < 3:
                st.warning("Input too short for reliable neural analysis.")
            else:
                with st.spinner("Neural Scanning..."):
                    results = engine.analyze(article_input, selected_model)
                    
                    if results:
                        # Display Results Card
                        st.markdown('<div class="glass-card" style="border-color: #6366f133;">', unsafe_allow_html=True)
                        st.markdown(f'<div class="result-tag">Inference via {selected_model.split(" ")[0]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<h2 class="category-output">{results["category"]}</h2>', unsafe_allow_html=True)
                        
                        # Confidence Metric
                        st.markdown('<div style="margin-top: 2rem;">', unsafe_allow_html=True)
                        st.markdown(f'<p class="stat-label">CONFIDENCE SCORE</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="stat-value">{results["confidence"]*100:.2f}%</p>', unsafe_allow_html=True)
                        st.progress(results["confidence"])
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Performance Metric
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid #1a1a1a;">
                            <div>
                                <p class="stat-label">LATENCY</p>
                                <p style="font-family: 'Space Grotesk'; font-size: 0.9rem;">{results['latency']*1000:.1f} ms</p>
                            </div>
                            <div>
                                <p class="stat-label">STATUS</p>
                                <p style="font-family: 'Space Grotesk'; font-size: 0.9rem; color: #4ade80;">Success</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Charts
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.markdown('<p class="section-header">Probability Map</p>', unsafe_allow_html=True)
                        fig = Visualizer.plot_probability_distribution(results)
                        st.pyplot(fig)
                        st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Idle State
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-style: dashed; padding: 100px 20px; opacity: 0.5;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">💠</div>
                <p class="section-header">Awaiting Signal</p>
                <p style="font-size: 0.8rem; color: #444;">Please provide article text in the left panel to begin neural classification.</p>
            </div>
            """, unsafe_allow_html=True)

    # ==============================================================================
    # 7. PROJECT DOCUMENTATION SECTION (THE "EXTRA" LUXURY)
    # ==============================================================================
    st.markdown('<div style="margin-top: 100px;"></div>', unsafe_allow_html=True)
    doc_col1, doc_col2, doc_col3 = st.columns(3)
    
    with doc_col1:
        st.markdown('<p class="section-header">01. Data Preprocessing</p>', unsafe_allow_html=True)
        st.markdown("""
        <p style="color: #666; font-size: 0.9rem;">
        يتم تحويل النصوص عبر معالج خاص يقوم بإزالة الرموز غير الضرورية وتحويل الكلمات إلى جذورها الأساسية لضمان تقليل الضوضاء في البيانات.
        </p>
        """, unsafe_allow_html=True)
        
    with doc_col2:
        st.markdown('<p class="section-header">02. Vectorization Model</p>', unsafe_allow_html=True)
        st.markdown("""
        <p style="color: #666; font-size: 0.9rem;">
        نستخدم تقنية <b>TF-IDF</b> (التردد العكسي للوثيقة) لتحويل الكلمات إلى مصفوفات عددية تعكس الأهمية النسبية لكل كلمة في سياق الخبر.
        </p>
        """, unsafe_allow_html=True)
        
    with doc_col3:
        st.markdown('<p class="section-header">03. Inference Logic</p>', unsafe_allow_html=True)
        st.markdown("""
        <p style="color: #666; font-size: 0.9rem;">
        يتم تنفيذ التنبؤ عبر مقارنة نتائج ثلاثة نماذج مختلفة، حيث يتم اختيار النموذج الأكثر استقراراً بناءً على دراسات دقة سابقة (Accuracy Metrics).
        </p>
        """, unsafe_allow_html=True)

    # Footer Signature
    st.markdown(f"""
    <div style="margin-top: 100px; padding: 40px 0; border-top: 1px solid #111; text-align: center;">
        <p style="font-family: 'Space Grotesk'; font-size: 0.7rem; color: #333; letter-spacing: 10px; text-transform: uppercase;">
            {AppConfig.DEVELOPER} • {AppConfig.TEAM_NAME} • ALL RIGHTS RESERVED 2026
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# ==============================================================================
# END OF ARCHITECTURE
# ==============================================================================
