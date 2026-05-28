from pathlib import Path
import pickle
import pandas as pd

class FraudPredictor:
    
    def __init__(self, model_path : str = None,
                 pipeline_path : str = None):
        BASE_DIR = Path(__file__).resolve().parent.parent
        self.model_path = Path(model_path) if model_path else BASE_DIR / 'models' / 'model.pkl'
        self.pipeline_path = Path(pipeline_path) if pipeline_path else BASE_DIR / 'models' / 'pipeline.pkl'
        
        self.model = self._load(self.model_path)
        self.pipeline = self._load(self.pipeline_path)
        
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
        