from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    predictions: List[float]
    rmse: float
    batch_name: str