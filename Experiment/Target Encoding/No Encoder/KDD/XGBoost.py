# Standard imports
import random
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, OneHotEncoder
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

    for experiment_number in range(experiment_repeat):
        
        # Set random seed for reproducibility
        set_random_seed(experiment_number)

        # Step 1: Load dataset
        df = fetch_openml(data_id=981, as_frame=True)['frame']
        df['label'] = df['Who_Pays_for_Access_Work'].astype(int)
        df = df.drop(columns=['Who_Pays_for_Access_Work'])

        # Step 2: Convert 'Age' to numeric and bin into groups
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['Age_group'] = (df['Age'] // 10).astype('Int64')
        df['Age_group'] = df['Age_group'].astype('category')
        df = df.drop(columns=['Age'])  # Drop original age column

        # Step 3: Convert object columns to categorical & fill NA as 'NAN'
        for col in df.columns:
            if df[col].dtype == "object" or pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].astype("category")
                if "NAN" not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories("NAN")
                df[col] = df[col].fillna("NAN")

        # Step 4: Rare category consolidation
        rare_threshold = len(df) * 0.005
        categorical_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()

        for cat_col in categorical_cols:
            value_counts = df[cat_col].value_counts()
            rare_values = value_counts[value_counts < rare_threshold].index
            df[cat_col] = df[cat_col].apply(lambda x: 'Others_DATA' if x in rare_values else x)

        # Step 5: Update CAT_COLS and NUM_COLS
        CAT_COLS = df.select_dtypes(include=['category', 'object']).columns.tolist()
        NUM_COLS = df.select_dtypes(include=['number']).drop(columns=['label']).columns.tolist()

        for col in CAT_COLS:
            df[col] = df[col].astype(str)
        
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

        # # Power transformation
        # power_transformer       = PowerTransformer(method='yeo-johnson', standardize=True)
        # train_data[NUM_COLS]    = power_transformer.fit_transform(train_data[NUM_COLS])
        # valid_data[NUM_COLS]    = power_transformer.transform(valid_data[NUM_COLS])
        # test_data[NUM_COLS]     = power_transformer.transform(test_data[NUM_COLS])

        # One-Hot encoding
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        onehot_encoder.fit(train_data[CAT_COLS])
        train_cat_ohe = onehot_encoder.transform(train_data[CAT_COLS])
        valid_cat_ohe = onehot_encoder.transform(valid_data[CAT_COLS])
        test_cat_ohe  = onehot_encoder.transform(test_data[CAT_COLS])
        
        # x_train_num = train_data[NUM_COLS].values.astype(np.float32)
        # x_valid_num = valid_data[NUM_COLS].values.astype(np.float32)
        # x_test_num  = test_data[NUM_COLS].values.astype(np.float32)
        
        X_train = train_cat_ohe
        X_valid = valid_cat_ohe
        X_test  = test_cat_ohe

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        dtest = xgb.DMatrix(X_test, label=y_test)

        xgb_params = {
            "booster": "gbtree",
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": "8",
        }

        model = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=2000,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        y_pred = (model.predict(dtest) > 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        print(f"Experiment #{experiment_number + 1} - Test Accuracy: {acc:.4f}")
        test_accs.append(acc)

    print(f"\nAverage Test Accuracy over {experiment_repeat} runs: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")


if __name__ == "__main__":
    main()
