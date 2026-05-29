import pandas as pd
from src.features import FeaturePipeline

def test_split_features_target():
    df = pd.DataFrame({
        'V1' : [1.0, 2.0],
        'Amount' : [100.0, 200.0],
        'Class' : [0, 1]
    })
    
    pipeline = FeaturePipeline()
    X, y = pipeline.split_features_target(df)
    
    assert 'Class' not in X.columns
    assert len(y) == 2
    
def test_train_test_split_proportions():
    df = pd.DataFrame({
        'V1' : range(100),
        'Amount' : range(100),
        'Class' : [0]*90 + [1]*10
    })
    
    pipeline = FeaturePipeline()
    X, y = pipeline.split_features_target(df)
    
    X_train, X_test, y_train, y_test = pipeline.get_train_test_split(X, y)
    
    assert len(X_test) == 20
    assert len(X_train) == 80
    
    
    
    
    