import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model_artifacts')

def load_model(name):
    path = os.path.join(MODEL_DIR, f'foul_detector_{name}.pkl')
    path = os.path.abspath(path)
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise RuntimeError(f"Model file not found at {path}")

models = {
    "low": load_model("low"),
    "medium": load_model("medium"),
    "high": load_model("high")
}

#FastAPI
app = FastAPI(title="Tweet Foul-Language Detection API")

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    level: str = Field(..., pattern="^(low|medium|high)$")

class PredictResponse(BaseModel):
    label: int
    probability: float
    threshold: float
    level: str


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    level = req.level
    if level not in models:
        raise HTTPException(status_code=400, detail="Invalid level")
    artifact = models[level]
    vectorizer = artifact['vectorizer']
    model = artifact['model']
    threshold = float(artifact.get('threshold', 0.5))

    try:
        xvec = vectorizer.transform([req.text])
        prob = float(model.predict_proba(xvec)[0,1])
        label = int(prob >= threshold)
        return PredictResponse(label=label, probability=prob, threshold=threshold, level=level)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Tweet Foul-Language Detection API is running!"}
