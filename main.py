from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

model = RandomForestClassifier(n_estimators=100)
trained = False

class TrainData(BaseModel):
    data: list

class PredictData(BaseModel):
    data: list

@app.get("/")
def root():
    return {"status":"ok"}

@app.post("/train")
def train(d: TrainData):
    global model, trained

    X = []
    y = []

    for item in d.data:
        X.append(item["features"])
        y.append(item["result"])

    if len(X) < 20:
        return {"error":"not enough data"}

    model.fit(X, y)
    trained = True

    return {"trained":True,"samples":len(X)}

@app.post("/predict")
def predict(d: PredictData):

    if not trained:
        return {"confidence":0.5}

    X = np.array([d.data])

    prob = model.predict_proba(X)[0][1]

    return {"confidence":float(prob)}
