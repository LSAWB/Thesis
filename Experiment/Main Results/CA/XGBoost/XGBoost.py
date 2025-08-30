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
from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.metrics import root_mean_squared_error

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
    n_classes           = 2
    batch_size          = 128

    epochs = 200
    warmup_epochs = 5

    test_rmses   = []

    # Get current time to name the log directory
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_log_dir = os.path.join(script_dir, current_time)
    os.makedirs(base_log_dir, exist_ok=True)

    for experiment_number in range(experiment_repeat):
            
        # Set random seed for reproducibility
        set_random_seed(experiment_number)

        # 取得資料
        df = fetch_california_housing(as_frame=True).frame
        X = df.drop(columns=["MedHouseVal"])
        y_orig = df["MedHouseVal"].values.astype(np.float32).reshape(-1, 1)

        # Train / Val / Test split
        X_temp, X_test, y_temp, y_test = train_test_split(X, y_orig, test_size=0.2, random_state=experiment_number)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=experiment_number)

        # 對 y 做標準化 (僅用 train 的 mean 和 std)
        y_mean = y_train.mean()
        y_std  = y_train.std()
        y_train = (y_train - y_mean) / y_std
        y_val   = (y_val   - y_mean) / y_std
        y_test  = (y_test  - y_mean) / y_std

        # 選定要標準化的欄位
        NUM_COLS = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

        # 注意：直接用 DataFrame 進行轉換
        scaler_power = PowerTransformer(method='yeo-johnson', standardize=True)
        train_data_encoded = scaler_power.fit_transform(X_train[NUM_COLS])
        valid_data_encoded = scaler_power.transform(X_val[NUM_COLS])
        test_data_encoded  = scaler_power.transform(X_test[NUM_COLS])

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
            "objective": "reg:squarederror",     
            "eval_metric": "rmse",             
            "verbosity": 0,
            "n_estimators": 2000,
            "tree_method": "hist",
            "n_jobs": 8,
        }

        dtrain = xgb.DMatrix(train_data_encoded, label=y_train)
        dvalid = xgb.DMatrix(valid_data_encoded, label=y_val)
        dtest = xgb.DMatrix(test_data_encoded, label=y_test)
        model = xgb.train(
            params,
            dtrain,
            evals=[(dvalid, "val")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # Predict on test set
        y_test_pred = model.predict(dtest)

        # Evaluate
        test_rmse = root_mean_squared_error(y_test, y_test_pred)
        test_rmses.append(test_rmse)

    mean_rmse = np.mean(test_rmses)
    std_rmse = np.std(test_rmses)
    print(f"Mean RMSE: {mean_rmse:.4f}, Std Dev RMSE: {std_rmse:.4f}")
    
    return mean_rmse, std_rmse


if __name__ == '__main__':
    with open("optuna_pareto_trials.json", "r") as f:
        trials = json.load(f)

    results = []
    for run_id, trial in enumerate(trials):
        hparams = trial["params"]
        print(f"\n====== Running trial {run_id} with params: {hparams} ======\n")
        test_mean_rmse, test_std_rmse = main(hparams, run_id)
        results.append({
            "trial_id": run_id,
            "params": hparams,
            "test_mean_rmse": test_mean_rmse,
            "test_std_rmse": test_std_rmse
        })

    with open("optuna_pareto_results.json", "w") as f:
        json.dump(results, f, indent=4)