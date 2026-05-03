from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from datetime import datetime

from src.schemas import RecommendResponse, RecommendationItem, BatchRequest

app = FastAPI()

# Load model
model = joblib.load("models/model.pkl")
features = joblib.load("models/rating_features.pkl")


@app.get("/")
def root():
    return {"message": "API running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "time": datetime.now().isoformat()
    }


@app.get("/recommend", response_model=RecommendResponse)
def recommend(user_id: int, n: int = 5):

    if user_id not in features.user_ids:
        raise HTTPException(status_code=404, detail="User not found")

    similar = features.get_similar_users(user_id, n)

    recommendations = [
        RecommendationItem(
            movie_id=uid,
            predicted_rating=score,
            rank=i+1
        )
        for i, (uid, score) in enumerate(similar)
    ]

    return {
        "user_id": user_id,
        "recommendations": recommendations
    }


@app.post("/predict_batch")
def predict_batch(req: BatchRequest):

    results = []

    for item in req.predictions:
        pred = model.predict(item.user_id, item.movie_id)

        results.append({
            "user_id": item.user_id,
            "movie_id": item.movie_id,
            "predicted_rating": pred
        })

    return {"predictions": results}