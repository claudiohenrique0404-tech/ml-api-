from fastapi import FastAPI
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# modelo fake inicial (depois evoluímos)
model = RandomForestClassifier()

X = np.random.rand(100,10)
y = np.random.randint(0,2,100)

model.fit(X,y)

@app.post("/predict")
def predict(data: dict):
    arr = np.array(data["data"][-10:])
    score = model.predict_proba([arr])[0][1]

    return {
        "confidence": float(score)
    }
