from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class FraudDetectionTrainer:
    
    MODELS = {
        'XGBoost' : XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM' : LGBMClassifier(random_state=42)
    }
    
    SAMPLERS = {
        'smote' : SMOTE(random_state=42),
        'adasyn' : ADASYN(random_state=42)
    }
    
    def __init__(self, experiment_name : str = 'Fraud Detection'):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
    def _apply_oversampling(self, X_train, y_train, method : str):
        sampler = self.SAMPLERS.get(method)
        if sampler is None:
            raise ValueError(f'Metodo Desconocido: {method}. Usar SMOTE o ADASYN')
        return sampler.fit_resample(X_train, y_train)
    
    def _save_model(self, model, model_name):
        models_dir = Path('models')
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f'{model_name}.pkl'
        
        joblib.dump(model, model_path)
        
        print(f'Modelo Guardado en: {model_path}')
    
    def _log_run(self, model, model_name, sampling_method, X_train, y_train, X_test, y_test):
        
        run_name = f'{model_name}_{sampling_method}'
        
        with mlflow.start_run(run_name=run_name):
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            report = classification_report(y_test, y_pred, output_dict=True)
            roc_auc = roc_auc_score(y_test, y_prob)
            
            mlflow.log_param('model', model_name)
            mlflow.log_param('sampling_method', sampling_method)
            mlflow.log_metric('roc_auc', roc_auc)
            mlflow.log_metric('precision_fraud', report['1']['precision'])
            mlflow.log_metric('recall_fraud', report['1']['recall'])
            mlflow.log_metric('f1_fraud', report['1']['f1-score'])
            mlflow.sklearn.log_model(model, name='model')
            
            self._save_model(model, model_name)
            
            print(f"{model_name} + {sampling_method} | ROC-AUC: {roc_auc:.4f} | Recall: {report['1']['recall']:.4f}")
            
    def run_all_experiments(self, X_train, y_train, X_test, y_test):
        for sampling_method in self.SAMPLERS:
            X_res, y_res = self._apply_oversampling(X_train, y_train, sampling_method)
            for model_name, model in self.MODELS.items():
                self._log_run(model, model_name, sampling_method, X_res, y_res, X_test, y_test)
                

