from __future__ import annotations
import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from src.features import TextFeaturizer
from src.lr_from_scratch import OneVsRestLogReg
import time

app = FastAPI(title="Document Classifier API", version="1.1")

# CORS configuration for local development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://document-classifier-api-1.onrender.com",
        "https://document-classifier-ui.onrender.com",
        "*"  # Allow all origins (remove in production for security)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=5)
    top_k: int = 3
    threshold: float = 0.65

class PredictResponse(BaseModel):
    label: str
    confidence: float
    abstained: bool
    top_k: list[dict]

# Load artifacts
featurizer = TextFeaturizer.load("artifacts/tfidf.joblib")
W = np.load("artifacts/lr_weights.npy")
b = np.load("artifacts/lr_bias.npy")

with open("artifacts/label_map.json", "r") as f:
    raw = json.load(f)
    id_to_label = {int(k): v for k, v in raw.items()}

model = OneVsRestLogReg()
model.W = W
model.b = b

# Track startup time
startup_time = time.time()

@app.get("/")
def root():
    return {
        "message": "Document Classifier API",
        "version": "1.1",
        "status": "running",
        "uptime_seconds": round(time.time() - startup_time, 2),
        "endpoints": {
            "health": "/health",
            "ping": "/ping",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint for Render"""
    return {
        "ok": True,
        "status": "healthy",
        "uptime_seconds": round(time.time() - startup_time, 2),
        "model_loaded": model.W is not None,
        "features_loaded": featurizer.vectorizer is not None
    }

@app.get("/ping")
def ping():
    """Simple ping endpoint to keep service alive"""
    return {"ping": "pong", "timestamp": time.time()}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    X = featurizer.transform([req.text])
    proba = model.predict_proba(X)[0]
    
    pred_id = int(np.argmax(proba))
    pred_label = id_to_label[pred_id]
    conf = float(proba[pred_id])
    
    # top-k
    k = max(1, min(req.top_k, len(proba)))
    top_ids = np.argsort(-proba)[:k]
    top = [{"label": id_to_label[int(i)], "confidence": float(proba[int(i)])} for i in top_ids]
    
    abstain = conf < float(req.threshold)
    final_label = "needs_review" if abstain else pred_label
    
    return PredictResponse(
        label=final_label,
        confidence=conf,
        abstained=abstain,
        top_k=top
    )