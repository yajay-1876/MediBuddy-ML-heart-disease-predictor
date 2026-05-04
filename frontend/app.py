from dotenv import load_dotenv
import os
from pathlib import Path
import streamlit as st
import requests

load_dotenv()

API_URL=os.getenv("API_URL")

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥Heart Disease Predictor")
st.subheader("Please enter your diagnosis results to predict:")

col1, col2, col3 =st.columns(3)

with col1:
    age=st.number_input("Age", 1, 120, 52)
    sex=st.selectbox("Sex(1=Male, 0=Female)",[0,1])
    cp=st.number_input("Chest Pain Type(cp)",0, 3, 0)
    trestbps=st.number_input("Resting Blood Pressure", 0, 250, 125)
    chol=st.number_input("Cholestrol",0, 600, 212)
with col2:
    fbs=st.selectbox("Fasting Blood Sugar > 120mg/dl",[0,1])
    restecg=st.selectbox("Resting ECG (restecg)",[0,1,2])
    thalach=st.number_input("Max Heart Rate (thalach)",0 ,250,168)
    exang=st.selectbox("Exercise Induced Agina",[0,1])
with col3:
    oldpeak=st.number_input("Old peak (ST depression)",0.0, 10.0, 1.0)
    slope=st.number_input("Slope",0,2,2)
    ca=st.number_input("Number of vessels (ca)",0, 4, 0)
    thal=st.number_input("Thal",0, 3, 2)

if st.button("🔍 Predict"):
    input_data={
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    response=requests.post(API_URL,json=input_data)

    if response.status_code !=200:
        st.error("something went wrong")
    else:
        result=response.json()
        #{'prediction': 1, 'probability': 0.9152384753359162, 'diagnosis': 'Heart Disease Detected'}
        prediction=result['prediction']
        probability=result['probability']
        diagnosis=result['diagnosis']

        st.divider()

        st.metric(
            label="Heart disease probability",
            value=f"{probability:.2%}"
        )
        if prediction==1:
            st.error(f"⚠️Model Prediction: {diagnosis}")
        else:
            st.success(f"✅ Model Prediction: {diagnosis}")
     



