from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logistic_model = joblib.load(
    os.path.join(BASE_DIR, "model", "logistic_heart_model.joblib")
)

dt_model = joblib.load(
    os.path.join(BASE_DIR, "model", "decision_tree_heart_model.joblib")
)


class FeaturesInput(BaseModel):
    features: list[float]

@app.post("/predict/logistic")
def predict_logistic(data: FeaturesInput):
    try:
        if len(data.features) != 13:
            return {"error": "Expected 13 features."}
        prediction = logistic_model.predict([data.features])
        return {"heart_disease": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict/decision-tree")
def predict_decision_tree(data: FeaturesInput):
    try:
        if len(data.features) != 13:
            return {"error": "Expected 13 features."}
        prediction = dt_model.predict([data.features])
        return {"heart_disease": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}
