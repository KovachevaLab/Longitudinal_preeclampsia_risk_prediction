import optuna
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from config_column_groups import *
from config_model_parameters import *
import pandas as pd
from sklearn.neural_network import MLPClassifier
import shap
from datetime import timedelta
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

def attempt_delete(study_name, storage):
    """
    Attempts to delete an Optuna study from the specified storage. 
    Ignores error if the study does not exist.
    """
    try:
        optuna.delete_study(study_name, storage)
    except KeyError:
        pass
    
def load_data(weeks, comparisons=False):
    """
    Loads preprocessed datasets for a given number of gestational weeks.
    
    Args:
        weeks (int): The gestational week threshold for data filtering.
        comparisons (bool): Whether to load comparison datasets.
    
    Returns:
        dict: A dictionary with train, test, and numom datasets (X, y).
    """
    filepath = f"../processed_data/modelling_data/{weeks}_data.pkl"
    if comparisons:
        filepath = f"../processed_data/modelling_data/{weeks}_old_comparisons.pkl"
    datasets = joblib.load(filepath)
    X_train = datasets['train']['X']
    y_train = datasets['train']['y']
    X_test = datasets['test']['X']
    y_test = datasets['test']['y']
    X_numom = datasets['numom']['X']
    y_numom = datasets['numom']['y']
    datasets = {'train': (X_train, y_train), 'test': (X_test, y_test), 'numom': (X_numom, y_numom)}
    return datasets

def apply_variance_threshold(datasets, threshold=0.0):
    """
    Applies variance threshold feature selection to remove low-variance features.
    Forces inclusion of 'proteinuria' feature.
    
    Args:
        datasets (dict): Dictionary of datasets with (X, y) tuples.
        threshold (float): Variance threshold below which features are removed.
    
    Returns:
        dict: Filtered datasets with reduced feature sets.
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(datasets['train'][0])
    mask = selector.get_support()

    assert 'proteinuria' in datasets['train'][0].columns
    
    proteinuria_index = datasets['train'][0].columns.get_loc('proteinuria')
    mask[proteinuria_index] = True

    for key in datasets.keys():
        datasets[key] = (datasets[key][0].loc[:, mask], datasets[key][1])
    return datasets

def apply_feature_selection(datasets, k=10):
    """
    Selects top-k features based on ANOVA F-statistic between label and feature.
    
    Args:
        datasets (dict): Dictionary of datasets with (X, y) tuples.
        k (int): Number of top features to select.
    
    Returns:
        dict: Datasets with only selected features.
    """
    X_train, y_train = datasets['train']
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)
    mask = selector.get_support()
    
    for key in datasets.keys():
        X, y = datasets[key]
        datasets[key] = (X.loc[:, mask], y)
    return datasets

def cross_validate_model(datasets, params, model):
    """
    Performs 5-fold stratified cross-validation with random oversampling.
    
    Args:
        datasets (dict): Dictionary with 'train' key for training data.
        params (dict): Model hyperparameters.
        model (class): Class of the ML model to be trained.
    
    Returns:
        list: ROC-AUC scores for each fold.
    """
    cv, scores = StratifiedKFold(n_splits=5, random_state=42, shuffle=True), []
    X_train, y_train = datasets['train']

    for train_idxs, test_idxs in cv.split(X_train, y_train):
        X_train_i, y_train_i = np.array(X_train)[train_idxs], np.array(y_train)[train_idxs]
        X_test_i, y_test_i = np.array(X_train)[test_idxs], np.array(y_train)[test_idxs]
        
        X_train_i, y_train_i = RandomOverSampler(random_state=42).fit_resample(X_train_i, y_train_i)
        
        model_i = model(**params)
        model_i.fit(X_train_i, y_train_i)
        
        y_hat_test_i = model_i.predict_proba(X_test_i)[:, 1] if hasattr(model_i, 'predict_proba') else model_i.predict(X_test_i)
        scores.append(roc_auc_score(y_test_i, y_hat_test_i))
    
    return scores

def final_model(datasets, params, model):
    """
    Trains the final model on oversampled full training data.
    
    Args:
        datasets (dict): Dictionary with 'train' data.
        params (dict): Hyperparameters for the model.
        model (class): ML model class.
    
    Returns:
        object: Trained model.
    """
    X_train, y_train = datasets['train']
    X_sampled, y_sampled = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
    model_f = model(**params)
    model_f.fit(X_sampled, y_sampled)
    return model_f


def get_trial_params_xgb(trial):
    """
    Suggests hyperparameters for XGBoost using Optuna trial object.
    
    Args:
        trial (optuna.trial.Trial): Optuna trial instance.
    
    Returns:
        dict: Dictionary of suggested hyperparameters.
    """
    max_depth = trial.suggest_int("max_depth", 1, 4)
    n_estimators = trial.suggest_categorical("n_estimators", [50, 75, 100, 200])
    lr = trial.suggest_categorical("lr", [0.1, 0.01, 0.001])
    return {'max_depth':max_depth, 'n_estimators':n_estimators, 'lr':lr}

def add_predictions(datasets, model):
    """
    Adds model predictions to each dataset split.
    
    Args:
        datasets (dict): Datasets with (X, y) pairs.
        model (object): Trained model with predict or predict_proba method.
    
    Returns:
        dict: Datasets with (X, y, y_pred) tuples.
    """
    updated_datasets = {}
    for key, (X, y) in datasets.items():
        y_pred_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)
        updated_datasets[key] = (X, y, y_pred_prob)
    return updated_datasets

def calculate_metrics(y_true, y_pred):
    """
    Calculates sensitivity, specificity, PPV, and NPV from true and predicted labels.
    
    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted probabilities or binary labels.
    
    Returns:
        tuple: (sensitivity, specificity, PPV, NPV)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, [1 if x > 0.5 else 0 for x in y_pred]).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return sensitivity, specificity, ppv, npv


def set_user_attr(trial, datasets):
    """
    Stores evaluation metrics as user attributes on the Optuna trial.
    
    Args:
        trial (optuna.trial.Trial): Optuna trial instance.
        datasets (dict): Datasets with (X, y, y_pred) tuples.
    """
    for name, (_, y_true, y_pred) in datasets.items():
        trial.set_user_attr(f"{name}_auc", roc_auc_score(y_true, y_pred))
        trial.set_user_attr(f"{name}_accuracy", accuracy_score(y_true, [1 if x > 0.5 else 0 for x in y_pred]))

        sensitivity, specificity, ppv, npv = calculate_metrics(y_true, y_pred)
        trial.set_user_attr(f"{name}_sensitivity", sensitivity)
        trial.set_user_attr(f"{name}_specificity", specificity)
        trial.set_user_attr(f"{name}_ppv", ppv)
        trial.set_user_attr(f"{name}_npv", npv)

def get_predictions(model, datasets):
    """
    Gets model predictions for each dataset split.
    
    Args:
        model (object): Trained model with predict_proba.
        datasets (dict): Dictionary of datasets.
    
    Returns:
        dict: Dictionary with (y_true, y_pred) for each dataset.
    """
    predictions = {}
    for name, (X, y) in datasets.items():
        y_pred = model.predict_proba(X)[:, 1]
        predictions[name] = (y, y_pred)
    return predictions

def get_best_metric(study_name, storage, attribute="train_auc"):
    """
    Retrieves the specified metric from the best Optuna trial.
    
    Args:
        study_name (str): Name of the Optuna study.
        storage (str): Storage backend URL.
        attribute (str): Metric name to retrieve.
    
    Returns:
        float: Metric value from best trial.
    """
    study = optuna.load_study(study_name=study_name, storage=storage)
    best_trial = study.best_trial
    return best_trial.user_attrs.get(attribute)

def remove_collinear_features(datasets, vif_threshold=5):
    """
    Removes collinear features based on Variance Inflation Factor (VIF).
    
    Args:
        datasets (dict): Dictionary of datasets with (X, y).
        vif_threshold (float): VIF threshold for feature removal.
    
    Returns:
        dict: Datasets with collinear features removed.
    """
    # Extract the train dataset
    X_train, y_train = datasets['train']
    features_train = X_train.columns
    
    # Calculate VIF for each feature in the train dataset
    vif_data_train = pd.DataFrame()
    vif_data_train["feature"] = features_train
    vif_data_train["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(len(features_train))]
    
    # Identify features to remove based on VIF
    features_to_remove = vif_data_train.loc[vif_data_train["VIF"] > vif_threshold, "feature"].tolist()
    
    # Remove identified features from all datasets
    for key, (X, y) in datasets.items():
        # Apply feature removal only to X (input features)
        X_filtered = X.drop(columns=features_to_remove, errors='ignore')
        datasets[key] = (X_filtered, y)
    
    return datasets


def objective(trial, weeks=36, model_type='xgb', eval=False):
    """
    Optuna objective function to evaluate or train a model pipeline.
    
    Args:
        trial (optuna.trial.Trial): Optuna trial instance.
        weeks (int): Gestational age threshold for data selection.
        model_type (str): Model type key from config.
        eval (bool): Whether to return model and predictions instead of score.
    
    Returns:
        float or tuple: Mean CV score or (model, datasets) if eval=True.
    """
    # get data and trial params
    datasets_preprocessed = load_data(weeks)
    for key in datasets_preprocessed.keys():
        info_cols = ['epic_pmrn', 'delivery_date', 'pregnancy_start', 'limit_date', 'delivery_hospital']
        datasets_preprocessed[key] = (datasets_preprocessed[key][0].drop(info_cols, axis=1), datasets_preprocessed[key][1])
    datasets = apply_variance_threshold(datasets_preprocessed, threshold=0.001)
    if model_type in ['elastic', 'logistic']: 
        datasets = remove_collinear_features(datasets, vif_threshold=10)
    
    params = {key: value(trial) if callable(value) else value for key, value in model_params_dict[model_type].items()}
    
    if not eval:
        scores = cross_validate_model(datasets, params, model_dict[model_type])
    
    # refit on all data
    model_f = final_model(datasets, params, model_dict[model_type])
    datasets = add_predictions(datasets, model_f)
    if eval:
        return model_f, datasets
    set_user_attr(trial, datasets)
    return np.mean(scores)

def compute_shap_values(model, datasets):
    """
    Computes SHAP values for model explainability.
    
    Args:
        model (object): Trained model.
        datasets (dict): Dictionary with dataset splits.
    
    Returns:
        tuple: SHAP explanation object and optional index list.
    """
    indexes = None
    if isinstance(model, (MLPClassifier)):
        test_data = datasets['test'][0].sample(frac=1, random_state=0).iloc[:1000]
        indexes = test_data.index
        explainer = shap.Explainer(model.predict, test_data)
        shap_values = explainer.shap_values(test_data)
        shap_values = shap.Explanation(shap_values, test_data)
    else:
        explainer = shap.Explainer(model)
        shap_values = explainer(datasets['test'][0])
    return shap_values, indexes

def get_acog_features(df):
    """
    Merges dataset with ACOG-related clinical features from external CSV.
    
    Args:
        df (pd.DataFrame): Input dataset with patient identifiers.
    
    Returns:
        pd.DataFrame: Dataset with merged ACOG features.
    """
    df_base = pd.read_csv('../processed_data/processing_data/base_processing.csv')
    df = pd.merge(df, df_base[['multiple_gestation', 'past_medical_history_preeclampsia', 'past_medical_history_gestational_diabetes', 'past_medical_history_antiphospholipid_syndrome', 'current_pregnancy_sle', 'past_medical_history_sle', 'bmi_before_pregnancy']+['epic_pmrn', 'delivery_date']].drop_duplicates(subset=['epic_pmrn', 'delivery_date']), on=['epic_pmrn', 'delivery_date'], how='left')

    df['multifetal_gestations'] = df['multiple_gestation']
    df['preeclampsia_in_previous_pregnancy'] = df['past_medical_history_preeclampsia']
    df['chronic_hypertension'] = (df.current_pregnancy_chypertension.astype(bool) | df.past_medical_history_chronic_hypertension.astype(bool))
    df['pregestational_diabetes'] = df.past_medical_history_gestational_diabetes
    df['gestational_diabetes'] = df.current_pregnancy_gestational_diabetes
    df['systemic_lupus_erythematosus'] = (pd.to_datetime(df.current_pregnancy_sle) < (pd.to_datetime(df.pregnancy_start) + timedelta(weeks=20))) | df.past_medical_history_sle
    df['prepregnancy_bmi_over_30'] = df.bmi_before_pregnancy.apply(lambda x: float(max(str(x).split('|'))) >= 30 if pd.notna(x) else x)
    df['antiphospholipid_antibody_syndrome'] = df.past_medical_history_antiphospholipid_syndrome
    df['maternal_age_over_35'] = df.maternal_age >= 35
    df['assisted_reproductive_technology'] = df.ivf
    return df