from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import mlflow
import pandas as pd

app = FastAPI()
model = joblib.load("../codebase/model.joblib")

class InputText(BaseModel):
    review: str

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
