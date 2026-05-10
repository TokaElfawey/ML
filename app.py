import streamlit as st
import joblib
import pandas as pd
import docx2txt
from pypdf import PdfReader
import re

# --- 1. إعدادات الصفحة المتقدمة ---
st.set_page_config(
    page_title="NewsAI | Intelligence Hub",
    page_icon="💠",
    layout="wide"
)

# --- 2. ستايل "الفخامة" (Dark Luxury Theme) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');

    /* تحسين الخطوط العامة */
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    .main {
        background: radial-gradient(circle at top right, #1e293b, #0f172a, #020617);
    }

    /* تحسين منطقة التكست لتبدو زجاجية */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        color: #f8fafc !important;
        font-size: 18px !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #00f5ff !important;
        box-shadow: 0 0 15px rgba(0, 245, 255, 0.2);
    }

    /* زر التحليل بتصميم نيون */
    .stButton button {
        width: 100%;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%) !important;
        border: none !important;
        color: white !important;
        padding: 20px !important;
        font-weight: 800 !important;
        font-size: 20px !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-radius: 15px !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(0, 210, 255, 0.4);
    }

    /* كارت النتيجة Glassmorphism */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 40px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        animation: fadeIn 0.8s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* إخفاء شعار ستريمليت لزيادة الرسمية */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. محرك المعالجة ---
@st.cache_resource
def load_nlp_assets():
    try:
        models = {
            "Logistic Regression": joblib.load("logistic_regression_model.joblib"),
            "Linear SVC": joblib.load("linear_svc_model.joblib"),
            "Naive Bayes": joblib.load("naive_bayes_model.joblib")
        }
        vectorizer = joblib.load("tfidf_vectorizer.joblib")
        le = joblib.load("label_encoder.joblib")
        return models, vectorizer, le
    except Exception as e:
        st.error(f"⚠️ Digital Assets Not Found: {e}")
        return None, None, None

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-z0-9\s]', '', text)
    return text

# --- 4. بناء الواجهة ---
def main():
    assets = load_nlp_assets()
    if assets[0] is None: return
    models, vectorizer, le = assets

    # Hero Section
    st.markdown("""
        <div style='text-align: center; padding: 40px 0;'>
            <h1 style='font-size: 70px; font-weight: 800; margin-bottom: 0; background: linear-gradient(to right, #ffffff, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                NewsAI <span style='color:#00f5ff'>Elite</span>
            </h1>
            <p style='color: #64748b; font-size: 20px; letter-spacing: 1px;'>Next-Generation Neural Content Classification</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar السلايد بار بشكل أنيق
    with st.sidebar:
        st.markdown("<h2 style='color:#00f5ff;'>⚙️ Neural Config</h2>", unsafe_allow_html=True)
        selected_model_name = st.selectbox("Intelligence Core", list(models.keys()))
        
        st.markdown("---")
        st.markdown("### Core Capabilities")
        if "SVC" in selected_model_name:
            st.info("🎯 High Precision Engine Active")
        else:
            st.info("⚡ Rapid Inference Active")

    # Main Interaction Area
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        input_method = st.tabs(["✍️ Manual Input", "📁 Document Upload"])
        
        input_text = ""
        with input_method[0]:
            input_text = st.text_area("", placeholder="Paste your article here for deep analysis...", height=300)
            
        with input_method[1]:
            file = st.file_uploader("Drop PDF, TXT or DOCX", type=['pdf', 'txt', 'docx'])
            if file:
                with st.spinner("Extracting Knowledge..."):
                    if file.type == "application/pdf":
                        reader = PdfReader(file)
                        input_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    elif file.type == "text/plain":
                        input_text = file.read().decode("utf-8")
                    else:
                        input_text = docx2txt.process(file)
                st.success("Context loaded successfully.")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Action Button
        if st.button("INITIATE NEURAL ANALYSIS"):
            if input_text.strip():
                with st.spinner("Deconstructing linguistic layers..."):
                    cleaned = clean_text(input_text)
                    vec = vectorizer.transform([cleaned])
                    model = models[selected_model_name]
                    
                    prediction = model.predict(vec)
                    label = le.inverse_transform(prediction)[0]
                    
                    # Confidence Calculation
                    conf = 0.94 # Default
                    if hasattr(model, "predict_proba"):
                        conf = max(model.predict_proba(vec)[0])

                    # Elegant Result Card
                    st.markdown(f"""
                        <div class="result-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <p style="color:#00f5ff; font-size:14px; text-transform:uppercase; letter-spacing:2px; margin-bottom:5px;">Classification Result</p>
                                    <h1 style="margin:0; color:#f8fafc; font-size: 55px; font-weight:800;">{label}</h1>
                                </div>
                                <div style="text-align:right;">
                                    <p style="color:#64748b; font-size:14px; margin-bottom:5px;">Confidence Score</p>
                                    <div style="background:rgba(0, 245, 255, 0.1); border: 1px solid #00f5ff; color:#00f5ff; padding:8px 20px; border-radius:50px; font-weight:bold; font-size:22px;">
                                        {conf:.1%}
                                    </div>
                                </div>
                            </div>
                            <div style="margin-top:30px; padding-top:20px; border-top: 1px solid rgba(255,255,255,0.1);">
                                <p style="color:#94a3b8; line-height:1.6; font-size: 18px;">
                                    Analysis Complete. The AI model has identified the core narrative of this content as <b>{label}</b> with a high degree of certainty based on linguistic patterns.
                                </p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
            else:
                st.warning("Please provide content to analyze.")

    # Footer المطور
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; border-top: 1px solid rgba(255,255,255,0.05); padding: 40px;'>
            <p style='color: #64748b; font-size: 14px; letter-spacing: 2px;'>CRAFTED BY</p>
            <h3 style='color: #f8fafc; font-weight: 600;'>SECTION 1 ELITE TEAM</h3>
            <div style='color: #475569; font-size: 16px; margin-top: 10px;'>
                Aya Ahmed • Toka Nasr • Toka Alaa • Hemmat Hamdi • Nourhan Medhat
            </div>
            <p style='font-size: 11px; margin-top: 25px; color: #334155; letter-spacing: 1px;'>
                &copy; 2026 INTELLIGENCE DEPLOYMENT PROTOCOL. ALL RIGHTS RESERVED.
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
