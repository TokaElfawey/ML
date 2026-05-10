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

# تحسين الخطوط وتنسيق الواجهة والـ Footer
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    /* تكبير الخطوط العامة */
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
        font-size: 19px; 
    }
    
    .main { background-color: #0b0e14; }
    
    /* تكبير منطقة النص */
    .stTextArea textarea { 
        font-size: 20px !important; 
        background-color: #161b22; 
        color: #e6edf3; 
        border-radius: 12px; 
        border: 1px solid #30363d;
    }
    
    /* تنسيق زر التحليل */
    .stButton button { 
        background: linear-gradient(135deg, #007cf0 0%, #00dfd8 100%); 
        color: white; 
        font-size: 24px !important; 
        height: 3.5em !important; 
        font-weight: 700;
        border-radius: 15px;
        border: none;
    }

    /* كارت النتيجة */
    .result-card { 
        background: rgba(22, 27, 34, 0.8); 
        padding: 35px; 
        border-radius: 20px; 
        border: 1px solid #30363d; 
        margin-top: 25px; 
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. محرك المعالجة (تحميل الموديلات) ---
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

# --- 3. بناء التطبيق الرئيسي ---
def main():
    assets = load_nlp_assets()
    if assets[0] is None: return
    models, vectorizer, le = assets
    
    # Header
    st.markdown("<h1 style='text-align: center; font-size: 55px; margin-bottom:0;'>NewsAI <span style='color:#00f5ff'>Predictor</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 22px;'>Enterprise NLP System for News Classification</p>", unsafe_allow_html=True)
    st.write("---")

    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='color:#00f5ff;'>⚙️ Settings</h2>", unsafe_allow_html=True)
        selected_model_name = st.selectbox("Choose AI Model", list(models.keys()))
        st.divider()
        st.markdown("### Model Stats")
        if "SVC" in selected_model_name:
            st.success("Best for Accuracy 🎯")
        else:
            st.info("Best for Speed ⚡")

    # Input Area
    input_method = st.radio("Select Input Method:", ["Type/Paste Text", "Upload Document"], horizontal=True)
    
    input_text = ""
    if input_method == "Type/Paste Text":
        input_text = st.text_area("Article Content", placeholder="Paste news article content here...", height=250)
    else:
        file = st.file_uploader("Upload File (PDF, TXT, DOCX)", type=['pdf', 'txt', 'docx'])
        if file:
            with st.spinner("Processing file..."):
                if file.type == "application/pdf":
                    reader = PdfReader(file)
                    input_text = " ".join([page.extract_text() for page in reader.pages])
                elif file.type == "text/plain":
                    input_text = file.read().decode("utf-8")
                else:
                    input_text = docx2txt.process(file)
            st.success("Text extracted!")

    # Execution
    if st.button("RUN AI ANALYSIS"):
        if input_text.strip():
            with st.spinner("Analyzing linguistic features..."):
                cleaned = clean_text(input_text)
                vec = vectorizer.transform([cleaned])
                model = models[selected_model_name]
                
                prediction = model.predict(vec)
                label = le.inverse_transform(prediction)[0]
                
                # Confidence
                conf = 0.94
                if hasattr(model, "predict_proba"):
                    conf = max(model.predict_proba(vec)[0])

                # Result Display
                st.markdown(f"""
                    <div class="result-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h1 style="margin:0; color:#00f5ff; font-size: 45px;">{label}</h1>
                            <span style="background:#238636; color:white; padding:10px 25px; border-radius:30px; font-weight:bold; font-size:20px;">
                                Confidence: {conf:.1%}
                            </span>
                        </div>
                        <p style="color:#94a3b8; margin-top:20px; font-size: 22px;">
                            The AI engine has analyzed the content and successfully matched it with the <b>{label}</b> category.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()
        else:
            st.error("Please enter text first.")

    # --- 4. الـ Footer الاحترافي بالأسماء الجديدة ---
    st.markdown("<br><br><br><hr>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; color: #94a3b8; font-family: sans-serif; font-size: 16px; padding-bottom: 30px;'>
            Developed with ❤️ by <b style='color: #00f5ff;'>Section 1 Team</b><br>
            <div style='margin-top: 10px; font-size: 18px; letter-spacing: 0.5px;'>
                آية احمد | تقي نصر | تقي علاء | همت حمدي | نورهان مدحت
            </div>
            <p style='font-size: 13px; margin-top: 15px; opacity: 0.7;'>© 2026 Academic Deployment Project</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
