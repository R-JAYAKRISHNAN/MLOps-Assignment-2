from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
import joblib
import mlflow
import pandas as pd

app = FastAPI()
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model.joblib")
model = joblib.load(model_path)

class InputText(BaseModel):
    review: str

app.get("/")
def root():
    return {"message": "API is up and running!"}

@app.get("/best_model_parameter")
def get_best_params():
    client = mlflow.tracking.MlflowClient()
    latest_run = client.search_runs(experiment_ids=["0"], order_by=["start_time DESC"])[0]
    return latest_run.data.params

@app.post("/train")
def trigger_training():
    import subprocess
    subprocess.run(["python3", "train.py"])
    return {"status": "Training complete"}

@app.post("/predict")
def predict_sentiment(data: InputText):
    prediction = model.predict([data.review])
    sentiment = "positive" if prediction[0] == 1 else "negative"
    return {"sentiment": sentiment}
