# Standard
import os
import math
import json
import random
from datetime import datetime
from typing import Dict, Literal

# General
import numpy as np
import pandas as pd
from tqdm import tqdm

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, PowerTransformer
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

# PyTorch
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch.utils.data import DataLoader, Dataset

# Hyperparameter Optimization
import optuna

# Tabular Deep Learning Models
from rtdl_revisiting_models import MLP, ResNet, FTTransformer
import xgboost as xgb

# Other
import delu
import typing as ty


# Seed
def set_random_seed(seed):
    # Set Python random seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
def main(hparams, run_id):
    
    # Define Hyperparameter
    experiment_repeat   = 15
    n_classes           = 4
    batch_size          = 128

    epochs = 200
    warmup_epochs = 5

    test_accs   = []

    # Get current time to name the log directory
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_log_dir = os.path.join(script_dir, current_time)

    for experiment_number in range(experiment_repeat):
            
        # Set random seed for reproducibility
        set_random_seed(experiment_number)

        # --- Load OpenML HIGGS SMALL dataset ---
        df = fetch_openml(data_id=41168, as_frame=True)['frame']
        df['label'] = df['class'].astype(int)
        df = df.drop(columns=['class'])  # drop the original label column

        NUM_COLS = df.drop(columns='label').columns.tolist()
        CAT_COLS = []  # No categorical columns

        df = df.dropna()

        cat_cardinalities = [df[col].nunique() for col in CAT_COLS]
        
        # Train / Test Split
        train_data, test_data = train_test_split(
            df,
            test_size=1/5,
            stratify=df['label'],
            random_state=experiment_number
        )

        train_data, valid_data = train_test_split(
            train_data,
            test_size=1/5,
            stratify=train_data['label'],
            random_state=experiment_number
        )

        # train_data = df.loc[train_idx].copy()
        # valid_data = df.loc[valid_idx].copy()
        # test_data  = df.loc[test_idx].copy()

        # Power transformation
        power_transformer       = PowerTransformer(method='yeo-johnson', standardize=True)
        train_data[NUM_COLS]    = power_transformer.fit_transform(train_data[NUM_COLS])
        valid_data[NUM_COLS]    = power_transformer.transform(valid_data[NUM_COLS])
        test_data[NUM_COLS]     = power_transformer.transform(test_data[NUM_COLS])

        # One-Hot encoding
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        onehot_encoder.fit(train_data[CAT_COLS])
        train_cat_ohe = onehot_encoder.transform(train_data[CAT_COLS])
        valid_cat_ohe = onehot_encoder.transform(valid_data[CAT_COLS])
        test_cat_ohe  = onehot_encoder.transform(test_data[CAT_COLS])

        y_train = train_data['label'].values
        y_valid = valid_data['label'].values
        y_test  = test_data['label'].values

        train_data_encoded = np.concatenate([train_data[NUM_COLS].values, train_cat_ohe], axis=1)
        valid_data_encoded = np.concatenate([valid_data[NUM_COLS].values, valid_cat_ohe], axis=1)
        test_data_encoded  = np.concatenate([test_data[NUM_COLS].values,  test_cat_ohe], axis=1)
        
        # === Optuna Hyperparameters (from figure) ===
        params = {
            "booster": "gbtree",
            "max_depth": hparams['max_depth'],
            "min_child_weight": hparams['min_child_weight'],
            "subsample": hparams['subsample'],
            "learning_rate": hparams['learning_rate'],
            "colsample_bylevel": hparams['colsample_bylevel'],
            "colsample_bytree": hparams['colsample_bytree'],
            "gamma": hparams['gamma'],
            "lambda": hparams['lambda'],
            "alpha": hparams['alpha'],
            "objective": "multi:softprob",
            "eval_metric": "merror",
            "num_class": n_classes,
            "verbosity": 0,
            "n_estimators": 2000,
            "tree_method": "hist",
            "n_jobs": 8
        }
        
        dtrain = xgb.DMatrix(train_data_encoded, label=y_train)
        dvalid = xgb.DMatrix(valid_data_encoded, label=y_valid)
        dtest = xgb.DMatrix(test_data_encoded, label=y_test)
        model = xgb.train(
            params,
            dtrain,
            evals=[(dvalid, "val")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # Predict on test set
        y_test_pred = np.argmax(model.predict(dtest), axis=1)
        
        # Evaluate
        test_acc = accuracy_score(y_test, y_test_pred)
        test_accs.append(test_acc)

    mean_acc = np.mean(test_accs)
    std_acc = np.std(test_accs)
    print(f"Mean ACC: {mean_acc:.4f}, Std Dev ACC: {std_acc:.4f}")
    
    return mean_acc, std_acc
    
if __name__ == '__main__':
    with open("optuna_pareto_trials.json", "r") as f:
        trials = json.load(f)

    results = []
    for run_id, trial in enumerate(trials):
        hparams = trial["params"]
        print(f"\n====== Running trial {run_id} with params: {hparams} ======\n")
        test_mean_acc, test_std_acc = main(hparams, run_id)
        results.append({
            "trial_id": run_id,
            "params": hparams,
            "test_mean_accuarcy": test_mean_acc,
            "test_std_accuarcy": test_std_acc
        })

    with open("optuna_pareto_results.json", "w") as f:
        json.dump(results, f, indent=4)