from pydantic import BaseModel
from typing import List

class RecommendationItem(BaseModel):
    movie_id: int
    predicted_rating: float
    rank: int

class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItem]

class PredictionItem(BaseModel):
    user_id: int
    movie_id: int

class BatchRequest(BaseModel):
    predictions: List[PredictionItem]