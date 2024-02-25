from fastapi import FastAPI
from pydantic import BaseModel, conlist
from typing import List, Any

from app.model import TrainedMnistModel

model = TrainedMnistModel()

app = FastAPI()


class InputDataModel(BaseModel):
    matrices: List[conlist(conlist(item_type=int, min_length=28, max_length=28), min_length=28, max_length=28)]


class PredictionResponse(BaseModel):
    predictions: List[List[float]]


@app.get("/")
async def root():
    return {"status": "online"}


@app.get("/score")
async def score():
    return {"score": model.score}


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: InputDataModel):
    predictions = model.predict(input_data.matrices)
    return {"predictions": predictions}
