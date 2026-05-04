from fastapi import FastAPI
from pydantic import BaseModel

from backend.predictor import predict_heart_disease



app=FastAPI(
    title="Heart disease predictor",
    version="1.0.0"
)

class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int 

@app.get("/health")
def api_health_check():
    return {"status : OK"}

@app.post("/predict-heart-disease")
def prediction(input_data: HeartDiseaseInput):
    input_dict=input_data.model_dump()
    result=predict_heart_disease(input_features=input_dict)

    return {
        "prediction": result["prediction"],
        "probability":result["probability"],
        "diagnosis": (
            "Heart Disease Detected"
            if result["prediction"] == 1
            else "No Heart Disease Detected"
        )
    }