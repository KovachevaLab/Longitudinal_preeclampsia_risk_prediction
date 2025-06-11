


import xgboost as xgb 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


weeks_list = [14, 20, 24, 28, 32, 34, 36, 38]
acog_features = ['nulliparous', 'multifetal_gestations', 'preeclampsia_in_previous_pregnancy', 'chronic_hypertension', 'pregestational_diabetes', 'gestational_diabetes', 'systemic_lupus_erythematosus', 'prepregnancy_bmi_over_30', 'antiphospholipid_antibody_syndrome', 'maternal_age_over_35', 'kidney_disease', 'assisted_reproductive_technology']
model_dict = {
    'xgb': xgb.XGBClassifier,
    'rf': RandomForestClassifier,
    'logistic': LogisticRegression,
    'elastic': ElasticNet,
    'naive_bayes': GaussianNB,
    'mlp': MLPClassifier
}

model_params_dict = {
    'rf': {
        'random_state': 0,
        'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 10), 
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 200), 
        'min_samples_split': lambda trial: trial.suggest_int("min_samples_split", 2, 20, log=True)
        },
    'xgb': {
            'seed': 0,
            'eval_metric': 'logloss',
            'max_depth': lambda trial: trial.suggest_int('max_depth', 1, 4),
            'n_estimators': lambda trial: trial.suggest_categorical("n_estimators", [50, 75, 100, 200]),
            'learning_rate': lambda trial: trial.suggest_loguniform('lr', 0.001, 0.1),
        },
    'logistic': {
        'random_state': 0,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 100,
        'C': lambda trial: trial.suggest_loguniform('logistic_C', 0.001, 10.0)
    },
    'elastic': {
        'random_state': 0,
        'max_iter': 100,
        'alpha': lambda trial: trial.suggest_loguniform('alpha', 0.001, 0.1)
    },
    'naive_bayes': {
        'var_smoothing': lambda trial: trial.suggest_loguniform('naive_bayes_var_smoothing', 1e-9, 1e-7)
    },
    'mlp': {
        'random_state': 0,
        'hidden_layer_sizes': lambda trial: tuple([trial.suggest_int(f'mlp_hidden_layer_{i}', 50, 200) for i in range(trial.suggest_int('mlp_n_layers', 1, 3))]),
        'activation': lambda trial: trial.suggest_categorical('mlp_activation', ['identity', 'logistic', 'tanh', 'relu']),
        'solver': lambda trial: trial.suggest_categorical('mlp_solver', ['lbfgs', 'sgd', 'adam']),
        'alpha': lambda trial: trial.suggest_loguniform('mlp_alpha', 1e-5, 1e-3),
        'learning_rate': lambda trial: trial.suggest_categorical('mlp_learning_rate', ['constant', 'invscaling', 'adaptive']),
        'max_iter': 20,
        'tol': 1e-4,
        'verbose':False
    }
}