from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

class FraudPredictor:
    
    def __init__(self):
        
        self.model_path = (BASE_DIR / 'models' / 'XGBoost.pkl')
        self.pipeline_path = (BASE_DIR / 'models' / 'pipeline.pkl')
        
        self.model = joblib.load(self.model_path)
        self.pipeline = joblib.load(self.pipeline_path)

    def predict(self, data: dict) -> dict:
        df = pd.DataFrame([data])
        df_scaled =self.pipeline.transform(df)
        prediction = self.model.predict(df_scaled)[0]
        probability = self.model.predict_proba(df_scaled)[0][1]
        return {
            'is_fraud' : int(prediction),
            'fraud_probability' : round(float(probability), 4)
        }
        