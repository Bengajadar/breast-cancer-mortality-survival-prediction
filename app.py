# Importing Libraries
import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go  

# Loading ML Models and Scaler
try:
    clf = joblib.load(r"/content/mortality_status_classifier.pkl")
    reg = joblib.load(r"/content/survival_months_regressor.pkl")
    scaler = joblib.load(r"/content/scaler.pkl")
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Streamlit Page Configuration
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

# Custom CSS 
st.markdown("""
    <style>
        .stApp {
            background-color: black;
            color: white;
        }
        div[data-testid="stSidebar"] {
            background-color: #222222;
        }
        .banner {
            background-color: rgb(97, 86, 92);
            padding: 20px;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
            transition: background-color 0.3s;
        }
        .banner:hover {
            background-color: rgb(63, 57, 60);
        }
        .subtitle {
            font-size: 22px;
            font-weight: bold;
            color: white;
            text-align: center;
            margin-bottom: 20px;
        }
        .section-header {
            font-size: 20px;
            font-weight: bold;
            color: #ff66b2;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .stButton>button {
            font-size: 22px;
            font-weight: bold;
            color: white;
            background-color:rgb(97, 86, 92);
            border-radius: 10px;
            padding: 15px;
            border: none;
            width: 100%;
            text-align: center;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color:rgb(63, 57, 60);
        }
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: white !important;
        }
        .result-card {
            background-color: #222222;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
            border: 2px solid rgb(97, 86, 92);
        }
    </style>
""", unsafe_allow_html=True)

# Title and Subtitle
st.markdown("<div class='banner'>Breast Cancer Survival Prediction</div>", unsafe_allow_html=True)

st.markdown('<p class="subtitle">Enter patient details below to predict survival outcome.</p>', unsafe_allow_html=True)

st.markdown('<p class="subtitle">This AI is experimental and not a substitute for professional medical advice. Consult a healthcare provider for any health concerns.</p>', unsafe_allow_html=True)

# Input Form
col1, col2 = st.columns([1, 1])

# Input fields (1st Column)
with col1:
    age = st.number_input("*Age*", min_value=1, max_value=120)
    race = st.radio("*Race*", ["White", "Black", "Other"])
    t_stage = st.radio("*T Stage*", ["T1", "T2", "T3", "T4"])
    n_stage = st.radio("*N Stage*", ["N1", "N2", "N3"])
    sixth_stage = st.radio("*6th Stage*", ["IIA", "IIB", "IIIA", "IIIB", "IIIC"])

# Input fields (2nd Column)
with col2:
    marital_status = st.radio("*Marital Status*", ["Married", "Divorced", "Single", "Widowed", "Separated"])
    grade = st.radio("*Cancer Grade*", ["1", "2", "3", "4"])
    a_stage = st.radio("*A Stage*", ["Regional", "Distant"])
    tumor_size = st.number_input("*Tumor Size (in cm)*", min_value=0.1, max_value=50.0)
    estrogen_status = st.radio("*Estrogen Status*", ["Positive", "Negative"])
    progesterone_status = st.radio("*Progesterone Status*", ["Positive", "Negative"])

# Additional Information (Regional Nodes Examined / Regional Nodes Positive)
st.markdown('<p class="section-header">Additional Information</p>', unsafe_allow_html=True)
col3, col4 = st.columns([1, 1])

with col3:
    regional_node_examined = st.number_input("*Regional Nodes Examined*", min_value=0, max_value=50)

with col4:
    regional_node_positive = st.number_input("*Regional Nodes Positive*", min_value=0, max_value=50)

# Mapping Inputs into Numerical
race_mapping = {"White": 0, "Black": 1, "Other": 2}
t_stage_mapping = {"T1": 0, "T2": 1, "T3": 2, "T4": 3}
n_stage_mapping = {"N1": 0, "N2": 1, "N3": 2}
sixth_stage_mapping = {"IIA": 0, "IIB": 1, "IIIA": 2, "IIIB": 3, "IIIC": 4}
marital_mapping = {"Married": 0, "Divorced": 1, "Single": 2, "Widowed": 3, "Separated": 4}
grade_mapping = {"1": 0, "2": 1, "3": 2, "4": 3}
a_stage_mapping = {"Regional": 0, "Distant": 1}
estrogen_mapping = {"Positive": 0, "Negative": 1}
progesterone_mapping = {"Positive": 0, "Negative": 1}

# Final Patient Data for Prediction
patient_data = np.array([[age, race_mapping[race], t_stage_mapping[t_stage], n_stage_mapping[n_stage],
                          sixth_stage_mapping[sixth_stage], marital_mapping[marital_status], grade_mapping[grade],
                          a_stage_mapping[a_stage], tumor_size, estrogen_mapping[estrogen_status],
                          progesterone_mapping[progesterone_status], regional_node_examined, regional_node_positive]])

# PREDICT Button
if st.button("*PREDICT*", use_container_width=True):
    try:
        # Mortality Status Prediction
        predicted_status = clf.predict(patient_data)[0]
        status_mapping = {0: "Alive", 1: "Dead"}
        predicted_status_mapped = status_mapping[predicted_status]

        # Survival Time in Months Prediction
        patient_data_scaled = scaler.transform(patient_data)
        predicted_survival = reg.predict(patient_data_scaled)[0]

        # Convert Survival Months to Years & Months
        years = int(predicted_survival // 12)
        months = int(predicted_survival % 12)
        survival_text = f"{years} years and {months} months ({int(predicted_survival)} months)"

        # Set Color Based on Prediction
        status_class = "status-alive" if predicted_status == 0 else "status-dead"

        # Custom CSS for Prediction Results
        st.markdown("""
            <style>
                .status-alive { color: #28a745; font-weight: bold; }
                .status-dead { color: #dc3545; font-weight: bold; }
            </style>
        """, unsafe_allow_html=True)

        # Display Prediction Results 
        st.markdown('<p class="section-header">Prediction Results</p>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class='result-card'>
                <h2>Predicted Mortality Status: <span class='{status_class}'>{predicted_status_mapped}</span></h2>
                <h3>Predicted Survival Time: {survival_text}</h3>
            </div>
        """, unsafe_allow_html=True)

        # Gauge Chart for Survival Time Predictions
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=int(predicted_survival),
            title={'text': "Survival Time (Months)"},
            gauge={
                "axis": {"range": [0, 120]},  
                "bar": {"color": "#ff66b2"},
                "steps": [
                    {"range": [0, 40], "color": "#ff9999"},
                    {"range": [40, 80], "color": "#ffcc99"},
                    {"range": [80, 120], "color": "#99ff99"},
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}")