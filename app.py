import streamlit as st 
import numpy as np 
import joblib 
# Load model 
model = joblib.load('best_model.pkl') 
st.set_page_config(page_title="Health Risk Detector", page_icon="💡", layout="centered") 
st.title("💡 AI-Powered Health Risk Detector") 
st.write("Use this tool to assess the risk of heart disease based on patient data.") 
# Sidebar Inputs 
st.sidebar.header("Patient Information") 
age = st.sidebar.slider("Age", 20, 100, 50) 
sex = st.sidebar.selectbox("Sex", ["Male", "Female"]) 
chest_pain_type = st.sidebar.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]) 
resting_bp = st.sidebar.number_input("Resting BP (mm Hg)", 80, 200, 120) 
cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", 100, 600, 200) 
fasting_blood_sugar = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"]) 
rest_ecg = st.sidebar.selectbox("Resting ECG Results", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"]) 
thalach = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, 150) 
exang = st.sidebar.selectbox("Exercise-Induced Angina", ["Yes", "No"]) 
oldpeak = st.sidebar.number_input("ST Depression (Oldpeak)", 0.0, 6.0, step=0.1) 
slope = st.sidebar.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"]) 
ca = st.sidebar.selectbox("Number of Major Vessels Colored", [0, 1, 2, 3]) 
thal = st.sidebar.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"]) 
# --- Encode Inputs --- 
sex = 1 if sex == "Male" else 0 
fasting_blood_sugar = 1 if fasting_blood_sugar == "Yes" else 0 
exang = 1 if exang == "Yes" else 0 
chest_pain_map = { "Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3 } 
rest_ecg_map = { "Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2 } 
slope_map = { "Upsloping": 0, "Flat": 1, "Downsloping": 2 } 
thal_map = { "Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3 } 
features = np.array([[age, sex, chest_pain_map[chest_pain_type], resting_bp, cholesterol, fasting_blood_sugar, rest_ecg_map[rest_ecg], thalach, exang, oldpeak, slope_map[slope], ca, thal_map[thal]]]) 
# Prediction 
if st.button("🩺 Predict Risk"):
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        st.markdown('<div class="prediction-card" style="background:#ffcccc; border-left: 5px solid #d32f2f;">⚠️ High Risk Detected</div>', unsafe_allow_html=True)
        st.warning("Consult a healthcare provider immediately.")
    else:
        st.markdown('<div class="prediction-card" style="background:#ccffcc; border-left: 5px solid #388E3C;">✅ Low Risk Detected</div>', unsafe_allow_html=True)
        st.info("Maintain a healthy lifestyle to prevent future risk.")

    if proba is not None:
        st.progress(proba)  
        st.write(f"🧠 **Prediction Confidence:** `{proba:.2%}`")

    # Save result
    if st.download_button("📄 Download Report", f"Risk: {'High' if prediction==1 else 'Low'}\nConfidence: {proba:.2%}", file_name="health_risk_report.txt"):
        st.success("Report download started.")
