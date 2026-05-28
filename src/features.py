from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class FeaturePipeline:
    def __init__(self, test_size: float = 0.2, random_state:int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.pipeline = self._built_pipeline()
        
    def _built_pipeline(self) -> Pipeline:
        """
        Construye el preprocesamiento de los datos
        """
        return Pipeline([
            ('scaler', StandardScaler())
        ])
        
    def load_data(self, path : str) -> pd.DataFrame:
        """
        Cargar el dataset y convertirlo en df
        """
        return pd.read_csv(path)
        
    def split_features_target(self, df : pd.DataFrame):
        X = df.drop(columns=['Class'])
        y = df['Class']
        return X, y
    
    def get_train_test_split(self, X, y):
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
    def fit_transform_train(self, X_train):
        return self.pipeline.fit_transform(X_train)
    
    def transform_test(self, X_test):
        """
        Solo transform en test. El Pipeline ya fue ajustado en train.
        """
        return self.pipeline.transform(X_test)
    
    def save_pipeline(self, path = 'models/pipeline.pkl'):
        models_dir = Path(path).parent
        models_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.pipeline, path)
        
        print(f'Pipeline Guardado en: {path}')
        
    