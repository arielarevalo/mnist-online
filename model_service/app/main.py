from fastapi import FastAPI

from app.model import TrainedMnistModel

app = FastAPI()

model = TrainedMnistModel()


@app.get("/")
async def root():
    return {"status": "online"}


@app.get("/score")
async def score():
    return {"score": model.score}


@app.post("/predict")
async def predict():
    pass
