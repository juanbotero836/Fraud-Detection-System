import pickle
import pandas as pd

class FraudPredictor:
    
    def __init__(self, model_path : str = '../models/model.pkl',
                 pipeline_path : str = '../models/pipeline.pkl'):
        
        self.model = self._load(model_path)
        self.pipeline = self._load(pipeline_path)
        
    def _load(self, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)
        
    def predict(self, data: dict) -> dict:
        df = pd.DataFrame([data])
        df_scaled =self.pipeline.transform(df)
        prediction = self.model.predict(df_scaled)[0]
        probability = self.model.predict_proba(df_scaled)[0][1]
        return {
            'is_fraud' : bool(prediction),
            'fraud_probability' : round(float(probability), 4)
        }
        