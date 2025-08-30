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

def objective(trial):
    
    # Define Hyperparameter
    seed = experiment_number = 42
    n_classes = 1
    batch_size = 128

    warmup_epochs = 5
    epochs = 200
        
     # Set random seed
    set_random_seed(seed)

    # 取得資料
    df = fetch_california_housing(as_frame=True).frame
    X = df.drop(columns=["MedHouseVal"])
    y_orig = df["MedHouseVal"].values.astype(np.float32).reshape(-1, 1)

    # Train / Val / Test split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_orig, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=seed)

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
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-8, 1e5, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1.0, log=True),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 1e2, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 1e2, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1e2, log=True),
        "objective": "reg:squarederror",     
        "eval_metric": "rmse",             
        "verbosity": 0,
        "n_estimators": 2000,
        "tree_method": "hist",
        "n_jobs": 8,
    }

    dtrain = xgb.DMatrix(train_data_encoded, label=y_train)
    dvalid = xgb.DMatrix(valid_data_encoded, label=y_val)
    model = xgb.train(
        params,
        dtrain,
        evals=[(dvalid, "val")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    # Predict & evaluate
    y_valid_pred = model.predict(dvalid)
    y_train_pred = model.predict(dtrain)
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    valid_rmse = root_mean_squared_error(y_val, y_valid_pred)

    return train_rmse, valid_rmse


if __name__ == '__main__':
    
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=50)

    pareto_results = []
    for t in study.best_trials:
        result = {
            "train_rmse": t.values[0],
            "valid_rmse": t.values[1],
            "params": t.params
        }
        pareto_results.append(result)

    with open("optuna_pareto_trials.json", "w") as f:
        json.dump(pareto_results, f, indent=4)

    print("✅ Saved Pareto trials to optuna_pareto_trials.json")