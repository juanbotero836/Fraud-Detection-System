from features import FeaturePipeline
from train import FraudDetectionTrainer


DATA_PATH = 'data/raw/creditcard.csv'

def main():
    # 1. Cragar y preparar los datos
    pipeline = FeaturePipeline()
    df = pipeline.load_data(DATA_PATH)
    X, y = pipeline.split_features_target(df)
    X_train_raw, X_test_raw, y_train, y_test = pipeline.get_train_test_split(X, y)
    
    # 2. Transformar 
    X_train = pipeline.fit_transform_train(X_train_raw)
    X_test = pipeline.transform_test(X_test_raw)
    
    pipeline.save_pipeline()
    
    # 3. Entrenar y loggear los experimentos
    trainer = FraudDetectionTrainer()
    trainer.run_all_experiments(X_train, y_train, X_test, y_test)
    
    
if __name__ == '__main__':
    main()
    