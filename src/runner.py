from features import FeaturePipeline
from train import FraudDetectionTrainer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'raw' / 'creditcard.csv'

def main():
    # 1. Cragar y preparar los datos
    feature_pipeline = FeaturePipeline()
    df = feature_pipeline.load_data(DATA_PATH)
    X, y = feature_pipeline.split_features_target(df)
    X_train_raw, X_test_raw, y_train, y_test = feature_pipeline.get_train_test_split(X, y)
    
    # 2. Transformar 
    X_train = feature_pipeline.fit_transform_train(X_train_raw)
    X_test = feature_pipeline.transform_test(X_test_raw)
    
    # 3. Entrenar y loggear los experimentos
    trainer = FraudDetectionTrainer()
    trainer.run_all_experiments(X_train, y_train, X_test, y_test)
    
if __name__ == '__main__':
    main()
    