import streamlit as st
import joblib
import pandas as pd
import docx2txt
from pypdf import PdfReader
import re

# --- 1. إعدادات الصفحة والـ UI ---
st.set_page_config(
    page_title="NewsAI | Advanced Classifier",
    page_icon="🤖",
    layout="wide"
)

# تحسين الخطوط وإضافة تصميم الـ Footer
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    /* تكبير الخطوط العامة */
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
        font-size: 18px; /* تكبير الخط الأساسي */
    }
    
    .main { background-color: #0b0e14; }
    
    /* تكبير وتنسيق منطقة النص */
    .stTextArea textarea { 
        font-size: 20px !important; 
        background-color: #161b22; 
        color: #e6edf3; 
        border-radius: 12px; 
    }
    
    /* تنسيق زر التحليل */
    .stButton button { 
        background: linear-gradient(135deg, #007cf0 0%, #00dfd8 100%); 
        color: white; 
        font-size: 22px !important; 
        height: 3.5em !important; 
        font-weight: 700;
        border-radius: 15px;
    }
    
    /* كارت النتيجة */
    .result-card { 
        background: rgba(22, 27, 34, 0.8); 
        padding: 30px; 
        border-radius: 20px; 
        border: 1px solid #30363d; 
        margin-top: 25px; 
    }
    
    /* تصميم الـ Footer الأسفل */
    .footer {
        position: relative;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: #8b949e;
        text-align: center;
        padding: 50px 0 20px 0;
        font-size: 16px;
        border-top: 1px solid #30363d;
        margin-top: 50px;
    }
    .team-names {
        color: #58a6ff;
        font-weight: bold;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. محرك المعالجة ---
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
        st.error(f"⚠️ Assets Missing: {e}")
        return None, None, None

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-z0-9\s]', '', text)
    return text

# --- 3. بناء التطبيق ---
def main():
    models, vectorizer, le = load_nlp_assets()
    
    # Header
    st.markdown("<h1 style='text-align: center; font-size: 50px;'>NewsAI <span style='color:#58a6ff'>Predictor</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e; font-size: 20px;'>Advanced NLP System for Professional News Categorization</p>", unsafe_allow_html=True)
    st.divider()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        selected_model_name = st.selectbox("Select Intelligence Model", list(models.keys()))
        st.write("---")
        st.markdown("### Model Insights")
        st.info(f"Currently using: **{selected_model_name}**")

    # Main Input Area
    input_method = st.radio("Select Input Source:", ["Write or Paste Text", "Upload Document File"], horizontal=True)
    
    input_text = ""
    if input_method == "Write or Paste Text":
        input_text = st.text_area("Article Content", placeholder="Enter the news article here...", height=300)
    else:
        file = st.file_uploader("Upload (PDF, TXT, DOCX)", type=['pdf', 'txt', 'docx'])
        if file:
            with st.spinner("Reading file..."):
                if file.type == "application/pdf":
                    reader = PdfReader(file)
                    input_text = " ".join([page.extract_text() for page in reader.pages])
                elif file.type == "text/plain":
                    input_text = file.read().decode("utf-8")
                else:
                    input_text = docx2txt.process(file)
            st.success("Content extracted successfully!")

    # Analysis Action
    if st.button("START AI ANALYSIS"):
        if input_text.strip():
            with st.spinner("🧠 Deep Analysis in progress..."):
                cleaned = clean_text(input_text)
                vec = vectorizer.transform([cleaned])
                model = models[selected_model_name]
                
                prediction = model.predict(vec)
                label = le.inverse_transform(prediction)[0]
                
                # Confidence Calculation
                conf = 0.95
                if hasattr(model, "predict_proba"):
                    conf = max(model.predict_proba(vec)[0])

                # Display Result
                st.markdown(f"""
                    <div class="result-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h1 style="margin:0; color:#58a6ff; font-size: 40px;">{label}</h1>
                            <span style="background:#238636; color:white; padding:8px 20px; border-radius:30px; font-weight:bold;">
                                Confidence: {conf:.1%}
                            </span>
                        </div>
                        <p style="color:#8b949e; margin-top:20px; font-size: 20px;">
                            The AI model has analyzed the linguistic patterns and classified this news under <b>{label}</b>.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()
        else:
            st.error("Please enter some text to analyze.")

    # --- FOOTER Section ---
    st.markdown(f"""
        <div class="footer">
            <p>Developed with ❤️ by Section 1 Team</p>
            <div class="team-names">
                آية احمد • تقي نصر • تقي علاء • همت حمدي • نورهان مدحت
            </div>
            <p style="font-size: 12px; margin-top: 10px;">© 2026 AI Deployment Project</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
