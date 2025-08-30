# Standard imports
import random
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.datasets import fetch_openml, fetch_california_housing

# XGBoost
import xgboost as xgb

# PyTorch (for reproducibility)
import torch


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    
def main():
    experiment_repeat = 15
    test_accs = []

    for experiment_number in range(experiment_repeat):
        
        # Set random seed
        set_random_seed(experiment_number)

        # 取得資料
        df = fetch_california_housing(as_frame=True).frame
        X = df.drop(columns=["MedHouseVal"])
        y_orig = df["MedHouseVal"].values.astype(np.float32).reshape(-1, 1)

        # Define numerical and categorical columns
        NUM_COLS = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        CAT_COLS = []
    
        # Train / Val / Test split
        X_temp, X_test, y_temp, y_test = train_test_split(X, y_orig, test_size=0.2, random_state=experiment_number)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=experiment_number)

        # 對 y 做標準化 (僅用 train 的 mean 和 std)
        y_mean = y_train.mean()
        y_std  = y_train.std()
        y_train = (y_train - y_mean) / y_std
        y_valid   = (y_val - y_mean) / y_std
        y_test  = (y_test  - y_mean) / y_std
    
        train_data = X_train.copy()
        train_data['label'] = y_train

        valid_data = X_val.copy()
        valid_data['label'] = y_val

        test_data = X_test.copy()
        test_data['label'] = y_test

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
        
        x_train_num = train_data[NUM_COLS].values.astype(np.float32)
        x_valid_num = valid_data[NUM_COLS].values.astype(np.float32)
        x_test_num  = test_data[NUM_COLS].values.astype(np.float32)
        
        X_train = np.hstack([x_train_num, train_cat_ohe])
        X_valid = np.hstack([x_valid_num, valid_cat_ohe])
        X_test  = np.hstack([x_test_num,  test_cat_ohe])
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        dtest = xgb.DMatrix(X_test, label=y_test)

        xgb_params = {
            "booster": "gbtree",
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_jobs": 8
        }

        model = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=2000,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        y_pred = model.predict(dtest)
        rmse = root_mean_squared_error(y_test, y_pred)
        print(f"Experiment #{experiment_number + 1} - Test RMSE: {rmse:.4f}")
        test_accs.append(rmse)

    print(f"\nAverage Test RMSE over {experiment_repeat} runs: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")

if __name__ == "__main__":
    main()
