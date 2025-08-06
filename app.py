
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Sleep Disorder Predictor",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        background-attachment: fixed;
        font-family: 'Poppins', sans-serif;
    }
    
    .main .block-container {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        margin: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        border-right: 3px solid #3498db;
    }
    
    .css-1d391kg .sidebar-content {
        background: transparent;
    }
    
    /* Sidebar text styling - comprehensive approach */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
    }
    
    /* Alternative sidebar selectors */
    .css-1d391kg p, .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6,
    .css-1d391kg div, .css-1d391kg span, .css-1d391kg label {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
    }
    
    /* Force all sidebar text to be white */
    .css-1d391kg * {
        color: white !important;
    }
    
    /* Additional sidebar selectors */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    .css-1d391kg strong, .css-1d391kg b {
        color: #74b9ff !important;
        font-weight: 600;
    }
    
    /* Additional specific sidebar text targeting */
    .css-1d391kg .element-container,
    .css-1d391kg .stMarkdown,
    .css-1d391kg .stText {
        color: white !important;
    }
    
    /* Target specific text elements */
    .css-1d391kg .stMarkdown p,
    .css-1d391kg .stMarkdown div,
    .css-1d391kg .stMarkdown span {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
    }
    
    /* Radio button labels in sidebar */
    .css-1d391kg .stRadio > div > div > div > label,
    [data-testid="stSidebar"] .stRadio > div > div > div > label {
        color: white !important;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
    }
    
    .css-1d391kg .stRadio > div > div > div > label:hover,
    [data-testid="stSidebar"] .stRadio > div > div > div > label:hover {
        background: rgba(255, 255, 255, 0.2);
        color: white !important;
        transform: scale(1.02);
    }
    
    /* Main header */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 3s ease-in-out infinite;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        letter-spacing: 2px;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Sub headers */
    .sub-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3498db;
        padding-left: 15px;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        border: none;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    /* Prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .prediction-box:hover::before {
        left: 100%;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        border: none;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(116, 185, 255, 0.3);
        border-left: 5px solid #00b894;
    }
    
    /* Form styling */
    .stForm {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Form label styling - increased text size for prediction features */
    .stSelectbox > div > div > div > label,
    .stSlider > div > div > div > label {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
        margin-bottom: 0.8rem !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1) !important;
        letter-spacing: 0.5px !important;
    }
    
    /* Additional form label targeting for better coverage */
    .stSelectbox label,
    .stSlider label,
    .stNumberInput label,
    .stTextInput label {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
        margin-bottom: 0.8rem !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1) !important;
        letter-spacing: 0.5px !important;
    }
    
    /* More comprehensive form label targeting */
    .stForm label,
    .stForm .stSelectbox > div > div > div > label,
    .stForm .stSlider > div > div > div > label,
    .stForm .stNumberInput > div > div > div > label,
    .stForm .stTextInput > div > div > div > label {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
        margin-bottom: 0.8rem !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1) !important;
        letter-spacing: 0.5px !important;
    }
    
    /* Target all label elements within the form */
    form label,
    .stForm label,
    [data-testid="stForm"] label {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
        margin-bottom: 0.8rem !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1) !important;
        letter-spacing: 0.5px !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(255, 107, 107, 0.4);
        background: linear-gradient(45deg, #4ecdc4, #ff6b6b);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    

    
    /* Radio button styling */
    .stRadio > div > div > div > label {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 10px;
        padding: 10px 15px;
        margin: 5px 0;
        border: 2px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .stRadio > div > div > div > label:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Success, warning, error, info messages */
    .stSuccess {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
        border-radius: 15px;
        padding: 1rem;
        border: none;
        box-shadow: 0 10px 30px rgba(0, 184, 148, 0.3);
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        color: white;
        border-radius: 15px;
        padding: 1rem;
        border: none;
        box-shadow: 0 10px 30px rgba(253, 203, 110, 0.3);
    }
    
    .stError {
        background: linear-gradient(135deg, #e17055 0%, #d63031 100%);
        color: white;
        border-radius: 15px;
        padding: 1rem;
        border: none;
        box-shadow: 0 10px 30px rgba(225, 112, 85, 0.3);
    }
    
    .stInfo {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        border-radius: 15px;
        padding: 1rem;
        border: none;
        box-shadow: 0 10px 30px rgba(116, 185, 255, 0.3);
    }
    
    /* Plotly chart styling */
    .js-plotly-plot {
        border-radius: 15px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        background: white;
        padding: 10px;
    }
    
    /* Footer styling */
    .footer-text {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white !important;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 10px 30px rgba(44, 62, 80, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    .footer-text p {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
        font-weight: 500;
        margin: 0.5rem 0;
    }
    
    /* Additional footer text targeting */
    .footer-text * {
        color: white !important;
    }
    
    /* Force all text in footer to be white */
    div[class*="footer-text"] p,
    div[class*="footer-text"] div,
    div[class*="footer-text"] span {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
        font-weight: 500;
    }
    
    /* Text styling */
    p, li {
        color: #2c3e50;
        font-weight: 400;
        line-height: 1.6;
    }
    
    strong, b {
        color: #34495e;
        font-weight: 600;
    }
    
    /* Prediction result text styling */
    .main h1, .main h2, .main h3 {
        color: #2c3e50 !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    /* Ensure all main content text is visible */
    .main p, .main div, .main span {
        color: #2c3e50 !important;
    }
    
    /* Success, warning, error, info text */
    .stSuccess p, .stWarning p, .stError p, .stInfo p {
        color: white !important;
        font-weight: 500;
    }
    
    /* Animation for elements */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .main-header, .sub-header, .metric-card, .prediction-box, .info-box {
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        .sub-header {
            font-size: 1.4rem;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and encoders"""
    try:
        model = joblib.load('xgb_sleep_model.pkl')
        scaler = joblib.load('scaler.pkl')
        bmi_encoder = joblib.load('BMI Category_encoder.pkl')
        gender_encoder = joblib.load('Gender_encoder.pkl')
        disorder_mapping = joblib.load('sleep_disorder_mapping_encoder.pkl')
        reverse_disorder_mapping = {v: k for k, v in disorder_mapping.items()}
        return model, scaler, bmi_encoder, gender_encoder, reverse_disorder_mapping
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None, None, None

def predict_sleep_disorder(user_input, model, scaler, bmi_encoder, gender_encoder):
    """Make prediction using the loaded model"""
    try:
        # Calculate Sleep Efficiency
        sleep_efficiency = user_input['Quality of Sleep'] / user_input['Sleep Duration']
        sleep_efficiency = np.clip(sleep_efficiency, 0.1, 2.0)
        
        # Encode categorical features
        bmi_encoded = bmi_encoder.transform([user_input['BMI Category']])[0]
        gender_encoded = gender_encoder.transform([user_input['Gender']])[0]
        
        # Arrange features in correct order
        X_input = np.array([
            gender_encoded,
            user_input['Age'],
            user_input['Sleep Duration'],
            user_input['Quality of Sleep'],
            user_input['Physical Activity Level'],
            user_input['Stress Level'],
            bmi_encoded,
            user_input['Heart Rate'],
            user_input['Daily Steps'],
            sleep_efficiency
        ]).reshape(1, -1)
        
        # Scale features
        X_input_scaled = scaler.transform(X_input)
        
        # Predict
        prediction = model.predict(X_input_scaled)[0]
        probabilities = model.predict_proba(X_input_scaled)[0]
        
        return prediction, probabilities, sleep_efficiency
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None



def show_home(model, scaler, bmi_encoder, gender_encoder, reverse_disorder_mapping):
    st.markdown('<h1 class="main-header">😴 Sleep Disorder Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    col1, _ = st.columns([2, 1])
    with col1:
        st.markdown('<h2 class="sub-header">🔍 Enter Your Information</h2>', unsafe_allow_html=True)
        with st.form("prediction_form"):
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                age = st.slider("Age", 18, 100, 30)
                sleep_duration = st.slider("Sleep Duration (hours)", 3, 12, 7)
                quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 7)
                physical_activity = st.slider("Physical Activity Level (minutes/day)", 0, 300, 60)
            with col1_2:
                stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
                bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight"])
                heart_rate = st.slider("Heart Rate (bpm)", 40, 120, 70)
                daily_steps = st.slider("Daily Steps", 1000, 20000, 8000)
            submitted = st.form_submit_button("🔮 Predict Sleep Disorder", use_container_width=True)
        if submitted:
            user_input = {
                'Gender': gender,
                'Age': age,
                'Sleep Duration': sleep_duration,
                'Quality of Sleep': quality_of_sleep,
                'Physical Activity Level': physical_activity,
                'Stress Level': stress_level,
                'BMI Category': bmi_category,
                'Heart Rate': heart_rate,
                'Daily Steps': daily_steps
            }
            prediction, probabilities, sleep_efficiency = predict_sleep_disorder(
                user_input, model, scaler, bmi_encoder, gender_encoder
            )
            if prediction is not None:
                st.session_state["show_result"] = True
                st.session_state["predicted_disorder"] = reverse_disorder_mapping[prediction]
                st.session_state["probabilities"] = probabilities
                st.session_state["sleep_efficiency"] = sleep_efficiency
                st.session_state["user_input"] = user_input
                st.session_state["go_to_result"] = True
                st.experimental_rerun()



def show_prediction_result():
    # Scroll to top when this page is rendered
    st.markdown(
        """
        <script>
            window.scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<h1 class="main-header">🎯 Prediction Result</h1>', unsafe_allow_html=True)
    predicted_disorder = st.session_state.get("predicted_disorder", "Unknown")
    st.markdown(f"### {predicted_disorder}")
    st.markdown('<h3 class="sub-header">💡 Recommendations</h3>', unsafe_allow_html=True)
    if predicted_disorder == "No disorder":
        st.success("🎉 Great! Your sleep patterns appear healthy. Keep maintaining your current lifestyle!")
        st.markdown("""
        **Tips to maintain good sleep:**
        - Maintain consistent sleep schedule
        - Keep your bedroom cool and dark
        - Avoid screens before bedtime
        - Regular exercise (but not too close to bedtime)
        """)
    elif predicted_disorder == "Insomnia":
        st.warning("⚠️ You may be experiencing insomnia. Consider consulting a healthcare provider.")
        st.markdown("""
        **Suggestions for insomnia:**
        - Establish a regular sleep routine
        - Avoid caffeine and alcohol before bed
        - Practice relaxation techniques
        - Consider cognitive behavioral therapy for insomnia (CBT-I)
        """)
    elif predicted_disorder == "Sleep Apnea":
        st.error("🚨 You may have sleep apnea. Please consult a healthcare provider for proper diagnosis.")
        st.markdown("""
        **Important for sleep apnea:**
        - Consult a sleep specialist
        - Consider a sleep study
        - Maintain healthy weight
        - Avoid sleeping on your back
        """)
    else:
        st.info("No recommendations available.")
    if st.button("Back to Home"):
        st.session_state["show_result"] = False
        st.session_state["go_to_home"] = True
        st.experimental_rerun()

def main():
    # Handle navigation redirection before rendering widgets
    if st.session_state.get("go_to_result"):
        st.session_state["nav_radio"] = "Prediction Result"
        del st.session_state["go_to_result"]
    if st.session_state.get("go_to_home"):
        st.session_state["nav_radio"] = "Home"
        del st.session_state["go_to_home"]
    model, scaler, bmi_encoder, gender_encoder, reverse_disorder_mapping = load_model_and_encoders()
    if model is None:
        st.error("Failed to load the model. Please check if all model files are present.")
        return
    # Sidebar
    st.sidebar.markdown("## 📊 About")
    st.sidebar.markdown("""
    This app predicts sleep disorders using machine learning based on various health and lifestyle factors.

    **Disorders predicted:**
    - No Disorder
    - Insomnia  
    - Sleep Apnea
    """)
    page = st.sidebar.radio(
        "Navigation",
        ["Home"] + (["Prediction Result"] if st.session_state.get("show_result") else []),
        key="nav_radio"
    )
    if page == "Home":
        show_home(model, scaler, bmi_encoder, gender_encoder, reverse_disorder_mapping)
    elif page == "Prediction Result":
        show_prediction_result()
    st.markdown("---")
    st.markdown("""
    <div class="footer-text">
        <p>⚠️ This is a predictive model and should not replace professional medical advice.</p>
        <p>Always consult with healthcare professionals for proper diagnosis and treatment.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 