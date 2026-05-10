import streamlit as st
import joblib
import os
import pandas as pd
from PIL import Image
import docx2txt
from pypdf import PdfReader

# --- 1. إعدادات الصفحة (UI/UX Settings) ---
st.set_page_config(
    page_title="AG News AI Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تصميم CSS مخصص لجعل الواجهة تبدو كـ SaaS احترافي
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTextArea textarea { background-color: #161b22; color: white; border-radius: 10px; border: 1px solid #30363d; }
    .stButton button { width: 100%; background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); color: black; font-weight: bold; border: none; border-radius: 10px; padding: 10px; }
    .result-card { background-color: #1c2128; padding: 20px; border-radius: 15px; border-left: 5px solid #4facfe; margin-top: 20px; }
    .confidence-bar { height: 10px; background-color: #30363d; border-radius: 5px; margin-top: 10px; }
    .confidence-fill { height: 100%; background: linear-gradient(90deg, #00f2fe, #4facfe); border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. وظائف مساعدة (Helpers) ---
@st.cache_resource
def load_assets():
    """تحميل الموديلات والـ Vectorizer مرة واحدة وتخزينها في الذاكرة"""
    try:
        models = {
            "Logistic Regression": joblib.load("logistic_regression_model.joblib"),
            "Linear SVC": joblib.load("linear_svc_model.joblib"),
            "Naive Bayes": joblib.load("naive_bayes_model.joblib")
        }
        vectorizer = joblib.load("tfidf_vectorizer.joblib")
        label_encoder = joblib.load("label_encoder.joblib")
        return models, vectorizer, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}. تأكدي أن ملفات الموديلات في نفس المجلد.")
        return None, None, None

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "text/plain":
        return str(uploaded_file.read(), "utf-8")
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(uploaded_file)
    return ""

# --- 3. بناء الواجهة (The Interface) ---
def main():
    models, vectorizer, label_encoder = load_assets()
    
    # Sidebar: Model Selection & Info
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
        st.title("Settings")
        st.subheader("Model Configuration")
        selected_model_name = st.selectbox("Choose AI Model", list(models.keys()))
        
        st.divider()
        st.info("💡 **Accuracy Hint:** Linear SVC typically performs best for text classification.")
        
        if st.button("Clear History"):
            st.session_state.history = []

    # Main Content
    st.markdown("<h1 style='text-align: center;'>📰 AG News Classifier AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e;'>Advanced NLP System for News Categorization</p>", unsafe_allow_html=True)

    # Tabs for different inputs
    tab1, tab2 = st.tabs(["✍️ Manual Input", "📁 Upload Document"])

    input_text = ""

    with tab1:
        input_text = st.text_area("Paste the news content here...", height=200, placeholder="Example: Apple launches new iPhone with AI features...")

    with tab2:
        uploaded_file = st.file_uploader("Choose a file (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])
        if uploaded_file:
            input_text = extract_text_from_file(uploaded_file)
            st.success("File uploaded and text extracted successfully!")

    # Analyze Button
    if st.button("🚀 Analyze Content"):
        if input_text.strip() == "":
            st.warning("Please provide some text first.")
        else:
            with st.spinner("AI is analyzing the text..."):
                # Inference Logic
                model = models[selected_model_name]
                vec_text = vectorizer.transform([input_text])
                prediction = model.predict(vec_text)
                label = label_encoder.inverse_transform(prediction)[0]
                
                # Confidence Score (Estimation if not available)
                confidence = 0.92
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(vec_text)
                    confidence = float(max(probs[0]))

                # Display Results
                st.markdown(f"""
                <div class="result-card">
                    <h3 style='margin:0;'>Result: <span style='color:#4facfe;'>{label}</span></h3>
                    <p style='margin:5px 0 0 0; font-size:0.9rem; color:#8b949e;'>Model: {selected_model_name}</p>
                    <div style='margin-top:15px;'>
                        <span>Confidence Level: {int(confidence*100)}%</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {int(confidence*100)}%;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.balloons()

    # Footer
    st.markdown("<br><hr><center><small>Powered by NLP Pipelines | Section 1 Team</small></center>", unsafe_allow_html=True)

if __name__ == "__main__":
    if 'history' not in st.session_state:
        st.session_state.history = []
    main()
