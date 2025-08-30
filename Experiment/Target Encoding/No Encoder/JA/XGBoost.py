# Standard imports
import random
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

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
    n_classes = 4
    
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
        
        # Step 6: Column stats
        cat_cardinalities = [df[col].nunique() for col in CAT_COLS]
        
        # Step 1: 先從 df 拿索引做切分
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=1/5,
            stratify=df['label'],
            random_state=experiment_number
        )

        # Step 2: 用 index 做第二次切分，但 stratify 要根據原始 df 對應的 label
        train_idx, valid_idx = train_test_split(
            train_idx,
            test_size=1/5,
            stratify=df.loc[train_idx, 'label'],
            random_state=experiment_number
        )

        train_data = df.loc[train_idx].copy()
        valid_data = df.loc[valid_idx].copy()
        test_data  = df.loc[test_idx].copy()
    
        y_train = train_data['label'].values
        y_valid = valid_data['label'].values
        y_test  = test_data['label'].values

        # Power transformation
        power_transformer       = PowerTransformer(method='yeo-johnson', standardize=True)
        train_data[NUM_COLS]    = power_transformer.fit_transform(train_data[NUM_COLS])
        valid_data[NUM_COLS]    = power_transformer.transform(valid_data[NUM_COLS])
        test_data[NUM_COLS]     = power_transformer.transform(test_data[NUM_COLS])
        
        x_train_num = train_data[NUM_COLS].values.astype(np.float32)
        x_valid_num = valid_data[NUM_COLS].values.astype(np.float32)
        x_test_num  = test_data[NUM_COLS].values.astype(np.float32)
        
        y_train = train_data['label'].values
        y_valid = valid_data['label'].values
        y_test  = test_data['label'].values
        
        dtrain = xgb.DMatrix(x_train_num, label=y_train)
        dvalid = xgb.DMatrix(x_valid_num, label=y_valid)
        dtest = xgb.DMatrix(x_test_num, label=y_test)

        xgb_params = {
            "booster": "gbtree",
            "objective": "multi:softprob",  # ✔️ correct for multi-class classification
            "eval_metric": "merror",        # ✔️ classification error
            "n_jobs": 8,
            "num_class": n_classes          # ✔️ specify total number of classes
        }

        model = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=2000,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # Predict on test set
        y_test_pred = np.argmax(model.predict(dtest), axis=1)
        
        # Evaluate
        test_acc = accuracy_score(y_test, y_test_pred)
        test_accs.append(test_acc)
        
        print(f"This is test_acc: {test_acc}")

    mean_acc = np.mean(test_accs)
    std_acc = np.std(test_accs)
    print(f"Mean ACC: {mean_acc:.4f}, Std Dev ACC: {std_acc:.4f}")
    print(f"\nAverage Test Accuracy over {experiment_repeat} runs: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")


if __name__ == "__main__":
    main()
