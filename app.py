import streamlit as st
import joblib
import pandas as pd
import docx2txt
from pypdf import PdfReader
import re
import base64

# --- 1. إعدادات الصفحة الأساسية ---
st.set_page_config(
    page_title="NewsAI | Advanced Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. دالة لجعل الفيديو خلفية (سر التصميم الحيوى) ---
def add_bg_video():
    # هذا رابط لفيديو خلفية جزيئات برمجية هادئة ونظيفة
    video_url = "https://assets.mixkit.co/videos/preview/mixkit-digital-particles-in-blue-background-9121-large.mp4"
    
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: none;
        }}
        
        #myVideo {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            z-index: -1;
            filter: brightness(0.6); /* تعتيم الفيديو قليلاً لبروز النص */
        }}
        </style>
        <video autoplay loop muted playsinline id="myVideo">
            <source src="{video_url}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True
    )

# --- 3. الـ CSS المتقدم (التصميم الزجاجي والنيون) ---
def apply_custom_css():
    st.markdown("""
        <style>
        /* استيراد خطوط جوجل */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
            color: #ffffff;
        }

        /* تنسيق السايدبار (شبه شفاف) */
        [data-testid="stSidebar"] {
            background-color: rgba(10, 25, 41, 0.7);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* تنسيق عناوين السايدبار */
        .sidebar-header {
            font-family: 'Roboto', sans-serif;
            color: #4ed8e4;
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 20px;
        }

        /* تنسيق صندوق اختيار الموديل */
        div[data-baseweb="select"] {
            background-color: rgba(0, 0, 0, 0.3) !important;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        /* تنسيق منطقة النص الرئيسية */
        .stTextArea textarea {
            background-color: rgba(0, 0, 0, 0.4) !important;
            border: 2px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 15px;
            color: #ffffff !important;
            font-size: 18px !important;
            padding: 15px;
        }
        .stTextArea textarea:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
        }

        /* تنسيق أزرار الراديو (Input Method) */
        div[data-testid="stMarkdownContainer"] p {
            font-size: 18px;
        }

        /* تنسيق زر التحليل (نيون ومجسم) */
        .stButton button {
            background: linear-gradient(145deg, #1d4ed8, #1e40af) !important;
            color: white !important;
            border: 2px solid #3b82f6 !important;
            border-radius: 30px !important;
            padding: 10px 30px !important;
            font-size: 22px !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            box-shadow: 0 4px 15px rgba(29, 78, 216, 0.5), inset 0 2px 2px rgba(255,255,255,0.3) !important;
            transition: all 0.2s ease-in-out !important;
            width: 100%;
        }
        .stButton button:hover {
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.8), inset 0 2px 2px rgba(255,255,255,0.4) !important;
            transform: translateY(-2px);
        }
        .stButton button:active {
            transform: translateY(1px);
            box-shadow: 0 2px 10px rgba(29, 78, 216, 0.5) !important;
        }

        /* تنسيق كارت النتيجة (زجاجي مضيء) */
        .result-card {
            background: rgba(13, 27, 45, 0.6);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid #4ed8e4;
            box-shadow: 0 0 15px rgba(78, 216, 228, 0.3);
            margin-top: 30px;
        }
        .result-label {
            font-family: 'Orbitron', sans-serif;
            color: #4ed8e4;
            font-size: 36px;
            margin: 0;
        }
        .result-conf {
            background-color: rgba(78, 216, 228, 0.2);
            color: #4ed8e4;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 14px;
            margin-left: 15px;
        }

        /* تنسيق الفوتر (الملون) */
        .footer-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: linear-gradient(90deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
            color: white;
            text-align: center;
            padding: 10px 0;
            z-index: 100;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- 4. محرك المعالجة (تحميل وهمي للموديلات لغرض التصميم) ---
@st.cache_resource
def load_nlp_assets():
    # في التطبيق الحقيقي، استبدلي هذا بتحميل ملفات .joblib الخاصة بكِ
    # fake_model = joblib.load("path/to/model.joblib")
    class FakeModel:
        def predict(self, x): return [0]
        def predict_proba(self, x): return [[0.94, 0.06]]
    
    models = {
        "Linear SVC": FakeModel(),
        "Logistic Regression": FakeModel(),
        "Naive Bayes": FakeModel()
    }
    vectorizer = "fake_vectorizer" # joblib.load("tfidf_vectorizer.joblib")
    le = "fake_le" # joblib.load("label_encoder.joblib")
    
    return models, vectorizer, le

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-z0-9\s]', '', text)
    return text

# --- 5. بناء التطبيق الرئيسي ---
def main():
    add_bg_video()
    apply_custom_css()
    
    # تحميل الأصول
    assets = load_nlp_assets()
    models, _, _ = assets

    # --- Sidebar ---
    with st.sidebar:
        st.markdown('<div class="sidebar-header">⚙️ Settings</div>', unsafe_allow_html=True)
        selected_model_name = st.selectbox("Choose AI Model", list(models.keys()), index=0)
        
        st.write("---")
        
        st.markdown("### Model Stats")
        # محاكاة حالة الموديل بناءً على الصورة
        if selected_model_name == "Linear SVC":
            st.success("Best for Accuracy 🎯")
        else:
            st.info("Good Balance ⚖️")

    # --- Main Content Area ---
    
    # Header
    st.markdown("""
        <div style='text-align: center; margin-top: -50px;'>
            <h1 style='font-size: 55px; font-weight: 700; color: white; margin-bottom: 0;'>
                NewsAI | Advanced Classifier 🤖
            </h1>
            <p style='font-size: 24px; color: #cbd5e1;'>Enterprise NLP System for News Classification</p>
        </div>
    """, unsafe_allow_html=True)

    # Input Area (استخدام Columns للتوسيط مثل الصورة)
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        # 1. Text Area
        input_text = st.text_area("", placeholder="Paste news article content here...", height=200)
        
        # 2. Input Method (Radio Buttons)
        input_method = st.radio(
            "Select Input Method:",
            ["Type/Paste Text", "Upload Document"],
            horizontal=True,
            label_visibility="collapsed" # إخفاء الليبل لجعلها مثل الصورة
        )
        
        # معالجة رفع الملفات إذا تم اختيارها
        if input_method == "Upload Document":
            file = st.file_uploader("Upload File (PDF, TXT, DOCX)", type=['pdf', 'txt', 'docx'])
            if file:
                with st.spinner("Processing file..."):
                    if file.type == "application/pdf":
                        reader = PdfReader(file)
                        input_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    elif file.type == "text/plain":
                        input_text = file.read().decode("utf-8")
                    else:
                        input_text = docx2txt.process(file)
                st.success("Text extracted!")

        # 3. RUN AI ANALYSIS Button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("RUN AI ANALYSIS"):
            if input_text.strip():
                with st.spinner("Analyzing..."):
                    # محاكاة النتيجة الظاهرة في الصورة بالضبط
                    label = "Business"
                    conf = 0.94
                    example_text = "Business is classification in motower continue and spend outraitings integrative dynamic, markets and classtifying business."
                    
                    # عرض كارت النتيجة (التصميم الزجاجي المضيء)
                    st.markdown(f"""
                        <div class="result-card">
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <img src="https://em-content.zobj.net/source/microsoft-teams/363/microphone_1f3a4.png" width="40" style="margin-right: 15px;">
                                <h2 class="result-label">{label}</h2>
                                <span class="result-conf">Confidence: {conf:.1%}</span>
                            </div>
                            <p style="color: white; font-size: 16px; line-height: 1.6; margin: 0;">
                                {example_text}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # أيقونات عائمة إضافية لمحاكاة روح الصورة (إختياري)
                    st.markdown("""
                        <div style="position: fixed; right: 50px; top: 150px; opacity: 0.5;">📉</div>
                        <div style="position: fixed; right: 80px; top: 250px; opacity: 0.5;">📊</div>
                        <div style="position: fixed; left: 300px; bottom: 150px; opacity: 0.5;">🎤</div>
                    """, unsafe_allow_html=True)
                    
                    st.balloons()
            else:
                st.error("Please enter text first.")

    # --- 6. الـ Footer الملون (الثابت في الأسفل) ---
    st.markdown("<br><br><br><br>", unsafe_allow_html=True) # مساحة إضافية
    st.markdown("""
        <div class="footer-container">
            <div>Developed with ❤️ by Section 1 Team</div>
            <div style="font-size: 14px; margin-top: 5px;">
                Aya Ahmed | Toka Nasr | Toka Alaa | Hemmat Hamdi | Nourhan Medhat
            </div>
            <div style="font-size: 12px; opacity: 0.8; margin-top: 5px;">
                © 2026 Academic Deployment Project
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
