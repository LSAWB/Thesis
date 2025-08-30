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

class NumericalMeanEncoderMultiClass:
    def __init__(self, feature_col, target_col, bin_size=0.001):
        self.feature_col = feature_col
        self.target_col = target_col
        self.bin_size = bin_size

        self.scaler = MinMaxScaler()
        self.bin_to_y = defaultdict(list)
        self.bin_to_probs = {}
        self.bin_to_count = {}
        self.default_probs = None
        self.std_bin = None
        self.class_labels = None

    def fit(self, df):
        x = df[self.feature_col].values.reshape(-1, 1)
        y = df[self.target_col].values

        self.class_labels = np.unique(y)
        num_classes = len(self.class_labels)
        x_scaled = self.scaler.fit_transform(x).flatten()
        bin_indices = (x_scaled / self.bin_size).astype(int)

        for i, bin_idx in enumerate(bin_indices):
            self.bin_to_y[bin_idx].append(y[i])

        # Default probs: global distribution
        label_counts = np.bincount(y, minlength=num_classes)
        self.default_probs = label_counts / np.sum(label_counts)
        self.std_bin = int(np.std(bin_indices))

        all_bin_indices = list(self.bin_to_y.keys())

        for bin_idx in all_bin_indices:
            neighbor_y = []
            for neighbor in range(bin_idx - self.std_bin, bin_idx + self.std_bin + 1):
                if neighbor in self.bin_to_y:
                    neighbor_y.extend(self.bin_to_y[neighbor])
            neighbor_y = np.array(neighbor_y)

            # Get probabilities for this bin
            counts = np.bincount(neighbor_y, minlength=num_classes)
            probs = counts / np.sum(counts)
            self.bin_to_probs[bin_idx] = probs
            self.bin_to_count[bin_idx] = np.sum(counts)

    def transform(self, df):
        x = df[self.feature_col].values.reshape(-1, 1)
        x_scaled = self.scaler.transform(x).flatten()
        bin_indices = (x_scaled / self.bin_size).astype(int)

        known_bins = np.array(list(self.bin_to_probs.keys()))
        result = []

        for bin_idx in bin_indices:
            if bin_idx in self.bin_to_probs:
                result.append(self.bin_to_probs[bin_idx])
            else:
                distances = np.abs(known_bins - bin_idx)
                min_dist = np.min(distances)
                closest_bins = known_bins[distances == min_dist]

                total_weight = sum(self.bin_to_count[b] for b in closest_bins)
                weighted_sum = sum(self.bin_to_probs[b] * self.bin_to_count[b] for b in closest_bins)

                result.append(weighted_sum / total_weight if total_weight > 0 else self.default_probs)

        return pd.DataFrame(result, index=df.index, columns=[f'{self.feature_col}_enc_class_{c}' for c in self.class_labels])
    
    
# class NumericalMeanEncoder:
#     def __init__(self, feature_col, target_col, bin_size=0.001):
#         self.feature_col = feature_col
#         self.target_col = target_col
#         self.bin_size = bin_size

#         self.scaler = MinMaxScaler()
#         self.bin_to_y = defaultdict(list)
#         self.bin_to_mean = {}
#         self.bin_to_count = {}
#         self.default_mean = None
#         self.std_bin = None

#     def fit(self, df):
#         x = df[self.feature_col].values.reshape(-1, 1)
#         y = df[self.target_col].values

#         x_scaled = self.scaler.fit_transform(x).flatten()
#         bin_indices = (x_scaled / self.bin_size).astype(int)

#         for i, bin_idx in enumerate(bin_indices):
#             self.bin_to_y[bin_idx].append(y[i])

#         self.default_mean = np.mean(y)
#         self.std_bin = int(np.std(bin_indices))

#         all_bin_indices = list(self.bin_to_y.keys())

#         for bin_idx in all_bin_indices:
#             neighbor_y = []
#             for neighbor in range(bin_idx - self.std_bin, bin_idx + self.std_bin + 1):
#                 if neighbor in self.bin_to_y:
#                     neighbor_y.extend(self.bin_to_y[neighbor])
#             self.bin_to_mean[bin_idx] = np.mean(neighbor_y)
#             self.bin_to_count[bin_idx] = len(neighbor_y)

#     def transform(self, df):
#         x = df[self.feature_col].values.reshape(-1, 1)
#         x_scaled = self.scaler.transform(x).flatten()
#         bin_indices = (x_scaled / self.bin_size).astype(int)

#         known_bins = np.array(list(self.bin_to_mean.keys()))
#         result = []

#         for bin_idx in bin_indices:
#             if bin_idx in self.bin_to_mean:
#                 result.append(self.bin_to_mean[bin_idx])
#             else:
#                 distances = np.abs(known_bins - bin_idx)
#                 min_dist = np.min(distances)
#                 closest_bins = known_bins[distances == min_dist]

#                 total_weight = sum(self.bin_to_count[b] for b in closest_bins)
#                 weighted_sum = sum(self.bin_to_mean[b] * self.bin_to_count[b] for b in closest_bins)

#                 result.append(weighted_sum / total_weight)

#         return pd.Series(result, index=df.index)


# class CategoricalMeanEncoder:
#     def __init__(self, feature_col, target_col):
#         self.feature_col = feature_col
#         self.target_col = target_col
#         self.category_to_mean = {}
#         self.category_to_count = {}
#         self.default_mean = None

#     def fit(self, df):
#         grouped = df.groupby(self.feature_col, observed=False)[self.target_col]
#         self.category_to_mean = grouped.mean().to_dict()
#         self.category_to_count = grouped.count().to_dict()
#         self.default_mean = df[self.target_col].mean()

#     def transform(self, df):
#         encoded = df[self.feature_col].map(self.category_to_mean)
#         return encoded.fillna(self.default_mean)
    
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

        numerical_encoders = {}
        X_train_num_list, X_valid_num_list, X_test_num_list = [], [], []
        for col in NUM_COLS:
            encoder = NumericalMeanEncoderMultiClass(feature_col=col, target_col='label', bin_size=0.001)
            encoder.fit(train_data)
            numerical_encoders[col] = encoder

            X_train_num_list.append(encoder.transform(train_data).add_prefix(col + '_enc_'))
            X_valid_num_list.append(encoder.transform(valid_data).add_prefix(col + '_enc_'))
            X_test_num_list.append(encoder.transform(test_data).add_prefix(col + '_enc_'))

        # X_train_cat_list, X_valid_cat_list, X_test_cat_list = [], [], []
        # for col in CAT_COLS:
        #     encoder = CategoricalMeanEncoder(feature_col=col, target_col='label')
        #     encoder.fit(train_data)
        #     X_train_cat_list.append(encoder.transform(train_data).rename(col + '_enc'))
        #     X_valid_cat_list.append(encoder.transform(valid_data).rename(col + '_enc'))
        #     X_test_cat_list.append(encoder.transform(test_data).rename(col + '_enc'))

        X_train = pd.concat(X_train_num_list, axis=1)
        X_valid = pd.concat(X_valid_num_list, axis=1)
        X_test = pd.concat(X_test_num_list, axis=1)

        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        dtest = xgb.DMatrix(X_test, label=y_test)

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
