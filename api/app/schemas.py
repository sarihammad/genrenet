"""
Pydantic schemas for API responses.
"""
from typing import List
from pydantic import BaseModel


class GenrePrediction(BaseModel):
    """Single genre prediction."""
    label: str
    score: float


class PredictResponse(BaseModel):
    """Response for genre prediction endpoint."""
    topk: List[GenrePrediction]
