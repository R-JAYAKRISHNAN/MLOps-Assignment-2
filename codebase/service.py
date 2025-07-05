from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import mlflow
import pandas as pd
import os 
import json

app = FastAPI()
# model = joblib.load("../codebase/model.joblib")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
model = joblib.load(MODEL_PATH)


class InputText(BaseModel):
    review: str

@app.get("/best_model_parameter")
def get_best_params():
    # client = mlflow.tracking.MlflowClient()
    # latest_run = client.search_runs(experiment_ids=["0"], order_by=["start_time DESC"])[0]
    param_path = os.path.join(os.path.dirname(__file__),"best_model_params.json")
    if not os.path.exists(param_path):
        return {"error": "No trained model found yet."}
    with open(param_path) as f:
        params = json.load(f)
    return {"best_model_parameter": params}
    # return latest_run.data.params

@app.get("/train")
def trigger_training():
    # import subprocess
    # subprocess.run(["python3", "train.py"])
    return {"status": "Training complete"}

@app.post("/predict")
def predict_sentiment(data: InputText):
    prediction = model.predict([data.review])
    sentiment = "positive" if prediction[0] == 1 else "negative"
    return {"sentiment": sentiment}
