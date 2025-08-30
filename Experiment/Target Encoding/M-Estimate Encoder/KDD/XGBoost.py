# Standard imports
import random
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

class NumericalEstimateEncoder:
    def __init__(self, feature_col, target_col, bin_size=0.001, m=10):
        self.feature_col = feature_col
        self.target_col = target_col
        self.bin_size = bin_size
        self.m = m

        self.scaler = MinMaxScaler()
        self.bin_to_y = defaultdict(list)
        self.bin_to_mean = {}
        self.bin_to_count = {}
        self.default_mean = None
        self.std_bin = None

    def fit(self, df):
        x = df[self.feature_col].values.reshape(-1, 1)
        y = df[self.target_col].values

        x_scaled = self.scaler.fit_transform(x).flatten()
        bin_indices = (x_scaled / self.bin_size).astype(int)

        for i, bin_idx in enumerate(bin_indices):
            self.bin_to_y[bin_idx].append(y[i])

        self.default_mean = np.mean(y)
        self.std_bin = int(np.std(bin_indices))
        all_bin_indices = list(self.bin_to_y.keys())

        for bin_idx in all_bin_indices:
            neighbor_y = []
            for neighbor in range(bin_idx - self.std_bin, bin_idx + self.std_bin + 1):
                if neighbor in self.bin_to_y:
                    neighbor_y.extend(self.bin_to_y[neighbor])

            n_b = len(neighbor_y)
            mu_b = np.mean(neighbor_y)
            smooth_mu = (n_b * mu_b + self.m * self.default_mean) / (n_b + self.m)

            self.bin_to_mean[bin_idx] = smooth_mu
            self.bin_to_count[bin_idx] = n_b

    def transform(self, df):
        x = df[self.feature_col].values.reshape(-1, 1)
        x_scaled = self.scaler.transform(x).flatten()
        bin_indices = (x_scaled / self.bin_size).astype(int)

        known_bins = np.array(list(self.bin_to_mean.keys()))
        result = []

        for bin_idx in bin_indices:
            if bin_idx in self.bin_to_mean:
                result.append(self.bin_to_mean[bin_idx])
            else:
                distances = np.abs(known_bins - bin_idx)
                min_dist = np.min(distances)
                closest_bins = known_bins[distances == min_dist]

                total_weight = sum(self.bin_to_count[b] for b in closest_bins)
                weighted_sum = sum(self.bin_to_mean[b] * self.bin_to_count[b] for b in closest_bins)

                result.append(weighted_sum / total_weight)

        return pd.Series(result, index=df.index)


class CategoricalEstimateEncoder:
    def __init__(self, feature_col, target_col, m=10):
        self.feature_col = feature_col
        self.target_col = target_col
        self.m = m

        self.category_to_mean = {}
        self.category_to_count = {}
        self.category_to_smoothed = {}
        self.default_mean = None

    def fit(self, df):
        grouped = df.groupby(self.feature_col, observed=False)[self.target_col]
        mean_per_cat = grouped.mean()
        count_per_cat = grouped.count()

        self.default_mean = df[self.target_col].mean()

        for category in mean_per_cat.index:
            n = count_per_cat[category]
            mu_c = mean_per_cat[category]
            smoothed = (n * mu_c + self.m * self.default_mean) / (n + self.m)
            self.category_to_smoothed[category] = smoothed

    def transform(self, df):
        encoded = df[self.feature_col].map(self.category_to_smoothed)
        return encoded.fillna(self.default_mean)
    
    
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

        X_train_num_list, X_valid_num_list, X_test_num_list = [], [], []
        for col in NUM_COLS:
            encoder = NumericalEstimateEncoder(feature_col=col, target_col='label', m=len(train_data)*0.01)
            encoder.fit(train_data)
            X_train_num_list.append(encoder.transform(train_data).rename(col + '_enc'))
            X_valid_num_list.append(encoder.transform(valid_data).rename(col + '_enc'))
            X_test_num_list.append(encoder.transform(test_data).rename(col + '_enc'))

        X_train_cat_list, X_valid_cat_list, X_test_cat_list = [], [], []
        for col in CAT_COLS:
            encoder = CategoricalEstimateEncoder(feature_col=col, target_col='label', m=len(train_data)*0.01)
            encoder.fit(train_data)
            X_train_cat_list.append(encoder.transform(train_data).rename(col + '_enc'))
            X_valid_cat_list.append(encoder.transform(valid_data).rename(col + '_enc'))
            X_test_cat_list.append(encoder.transform(test_data).rename(col + '_enc'))

        X_train = pd.concat(X_train_num_list + X_train_cat_list, axis=1)
        X_valid = pd.concat(X_valid_num_list + X_valid_cat_list, axis=1)
        X_test = pd.concat(X_test_num_list + X_test_cat_list, axis=1)

        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

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
