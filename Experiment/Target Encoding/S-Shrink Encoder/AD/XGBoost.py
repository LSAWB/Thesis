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


class NumericalShrinkEncoder:
    def __init__(self, feature_col, target_col, bin_size=0.001, S1=20, S2=10):
        self.feature_col = feature_col
        self.target_col = target_col
        self.bin_size = bin_size
        self.S1 = S1
        self.S2 = S2

        self.scaler = MinMaxScaler()
        self.bin_to_y = defaultdict(list)
        self.bin_to_phi = {}
        self.bin_to_count = {}
        self.default_mean = None
        self.std_bin = None

    def fit(self, df):
        x = df[self.feature_col].values.reshape(-1, 1)
        y = df[self.target_col].values

        # Step 1: MinMax Scaling + Binning
        x_scaled = self.scaler.fit_transform(x).flatten()
        bin_indices = (x_scaled / self.bin_size).astype(int)

        # Step 2: Collect bin-to-y mapping
        for i, bin_idx in enumerate(bin_indices):
            self.bin_to_y[bin_idx].append(y[i])

        self.default_mean = np.mean(y)
        self.std_bin = int(np.std(bin_indices))
        all_bin_indices = list(self.bin_to_y.keys())

        # Step 3: Compute smoothed value for each bin
        for bin_idx in all_bin_indices:
            neighbor_y = []
            for neighbor in range(bin_idx - self.std_bin, bin_idx + self.std_bin + 1):
                if neighbor in self.bin_to_y:
                    neighbor_y.extend(self.bin_to_y[neighbor])
            m_k = len(neighbor_y)
            mu_k = np.mean(neighbor_y)

            # Compute B_k using sigmoid function
            B_k = 1 / (1 + np.exp(-(m_k - self.S1) / self.S2))

            # Apply smoothing formula
            phi = B_k * mu_k + (1 - B_k) * self.default_mean
            self.bin_to_phi[bin_idx] = phi
            self.bin_to_count[bin_idx] = m_k

    def transform(self, df):
        x = df[self.feature_col].values.reshape(-1, 1)
        x_scaled = self.scaler.transform(x).flatten()
        bin_indices = (x_scaled / self.bin_size).astype(int)

        known_bins = np.array(list(self.bin_to_phi.keys()))
        result = []

        for bin_idx in bin_indices:
            if bin_idx in self.bin_to_phi:
                result.append(self.bin_to_phi[bin_idx])
            else:
                distances   = np.abs(known_bins - bin_idx)
                min_dist    = np.min(distances)
                closest_bins = known_bins[distances == min_dist]

                total_weight = sum(self.bin_to_count[b] for b in closest_bins)
                weighted_sum = sum(self.bin_to_phi[b] * self.bin_to_count[b] for b in closest_bins)

                result.append(weighted_sum / total_weight)

        return pd.Series(result, index=df.index)


class CategoricalShrinkEncoder:
    def __init__(self, feature_col, target_col, S1=20, S2=10):
        self.feature_col = feature_col
        self.target_col = target_col
        self.S1 = S1
        self.S2 = S2

        self.category_to_smoothed = {}
        self.default_mean = None

    def fit(self, df):
        grouped = df.groupby(self.feature_col, observed=False)[self.target_col]
        mean_per_cat = grouped.mean()
        count_per_cat = grouped.count()
        self.default_mean = df[self.target_col].mean()

        for category in mean_per_cat.index:
            mk = count_per_cat[category]
            mu_k = mean_per_cat[category]

            # 計算 S-shrink 權重 B_k
            Bk = 1 / (1 + np.exp(-(mk - self.S1) / self.S2))

            # 最終平滑編碼
            smoothed = Bk * mu_k + (1 - Bk) * self.default_mean
            self.category_to_smoothed[category] = smoothed

    def transform(self, df):
        encoded = df[self.feature_col].map(self.category_to_smoothed)
        return encoded.fillna(self.default_mean)
    

def main():
    experiment_repeat = 15
    test_accs = []

    for experiment_number in range(experiment_repeat):
        set_random_seed(experiment_number)
        df = fetch_openml(data_id=1590, as_frame=True)['frame']
        df['label'] = (df['class'] == '>50K').astype(int)
        df = df.drop(columns=['class'])

        CAT_COLS = df.select_dtypes(include=['category', 'object']).columns.tolist()
        NUM_COLS = df.select_dtypes(include=['number']).drop(columns=['label']).columns.tolist()

        # Handle missing values in categorical columns
        for col in CAT_COLS:
            if df[col].dtype.name == 'category':
                if 'Missing' not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories('Missing')
            df[col] = df[col].fillna('Missing')

        # Handle rare categories
        threshold = len(df) * 0.005 
        for col in CAT_COLS:
            value_counts = df[col].value_counts()
            rare_categories = value_counts[value_counts < threshold].index
            df[col] = df[col].apply(lambda x: 'Others' if x in rare_categories else x)

        train_data, test_data = train_test_split(df, test_size=1/3, stratify=df['label'], random_state=experiment_number)
        train_data, valid_data = train_test_split(train_data, test_size=1/5, stratify=train_data['label'], random_state=experiment_number)

        y_train = train_data['label'].values
        y_valid = valid_data['label'].values
        y_test = test_data['label'].values

        X_train_num_list, X_valid_num_list, X_test_num_list = [], [], []
        for col in NUM_COLS:
            encoder = NumericalShrinkEncoder(feature_col=col, target_col='label')
            encoder.fit(train_data)
            X_train_num_list.append(encoder.transform(train_data).rename(col + '_enc'))
            X_valid_num_list.append(encoder.transform(valid_data).rename(col + '_enc'))
            X_test_num_list.append(encoder.transform(test_data).rename(col + '_enc'))

        X_train_cat_list, X_valid_cat_list, X_test_cat_list = [], [], []
        for col in CAT_COLS:
            encoder = CategoricalShrinkEncoder(feature_col=col, target_col='label')
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
