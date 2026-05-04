import os
from pathlib import Path
from dotenv import load_dotenv
from joblib import load
import logging
import pandas as pd


load_dotenv()

PROJECT_FOLDER=Path(os.getenv("PROJECT_FOLDER")).resolve()
LOG_PATH=PROJECT_FOLDER / os.getenv("LOG_DIR") / os.getenv("LOG_FILE")
MODEL_PATH=PROJECT_FOLDER / os.getenv("MODEL_DIR")  / os.getenv("MODEL_NAME")

LOG_PATH.parent.mkdir(parents=True,exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH)
    ]
)
model=load(filename=MODEL_PATH)
logging.info("Model loaded successfully")

def predict_heart_disease(input_features):
    input_df=pd.DataFrame([input_features])
    prediction=int(model.predict(input_df)[0])
    probability=float(model.predict_proba(input_df)[0][1])

    logging.info(f"\nModel Prediction :{prediction}\nHeart disease risk probability: {probability:.2%}")
    return {
        "prediction":prediction,
        "probability":f"{probability:.2%}"
    }

#example usage
# sample_input = {
#     "age": 52,
#     "sex": 1,
#     "cp": 0,
#     "trestbps": 125,
#     "chol": 212,
#     "fbs": 0,
#     "restecg": 1,
#     "thalach": 168,
#     "exang": 0,
#     "oldpeak": 1.0,
#     "slope": 2,
#     "ca": 0,
#     "thal": 2
# }
# result = predict_heart_disease(sample_input)
# print(result)