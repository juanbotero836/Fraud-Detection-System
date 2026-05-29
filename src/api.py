from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import FraudPredictor

app = FastAPI(
    title = " Fraud Detection API",
    description = "Detect fraudulent credit card transactions",
    version = "1.0.0"
)

predictor = FraudPredictor()

class TransactionRequest(BaseModel):
    Time : float
    V1 : float
    V2 : float
    V3 : float
    V4 : float
    V5 : float
    V6 : float
    V7 : float
    V8 : float
    V9 : float
    V10 : float
    V11 : float
    V12 : float
    V13 : float
    V14 : float
    V15 : float
    V16 : float
    V17 : float
    V18 : float
    V19 : float
    V20 : float
    V21 : float
    V22 : float
    V23 : float
    V24 : float 
    V25 : float
    V26 : float
    V27 : float
    V28 : float
    Amount : float

class PredictionResponse(BaseModel):
    is_fraud : bool
    fraud_probability : float
    
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    try:
        result = predictor.predict(transaction.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

