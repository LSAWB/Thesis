import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import config
from sklearn.neighbors import KDTree
from typing import List, Dict
from collections import defaultdict

# class Cat_BetaEncoder:
#     """Encodes categorical features using Bayesian statistics derived from the target."""
    
#     def __init__(self, group):
#         """
#         Initializes the encoder with the specified group (categorical feature).
        
#         Parameters:
#             group (str): The name of the categorical feature to encode.
#         """
#         self.group = group
#         self.stats = None
#         self.prior_mean = None


#     def fit(self, df, target_col):
#         """
#         Fits the encoder using the DataFrame and target column, calculating necessary statistics.
        
#         Parameters:
#             df (pd.DataFrame): The input DataFrame containing features and target.
#             target_col (str): The name of the target column.
#         """
#         # Compute the prior mean from the target column
#         self.prior_mean = np.mean(df[target_col])
        
#         # Aggregate sum and count for the target by the group (categorical feature)
#         stats = df.groupby(self.group)[target_col].agg(['sum', 'count'])
#         stats.rename(columns={'sum': 'n', 'count': 'N'}, inplace=True)
#         stats.reset_index(level=0, inplace=True)
        
#         self.stats = stats


#     def transform(self, df, stat_type, N_min=1):
#         """
#         Transforms the DataFrame by adding encoded features based on the chosen statistical method.
        
#         Parameters:
#             df (pd.DataFrame): The input DataFrame to transform.
#             stat_type (str): The type of statistic to calculate ('mean', 'mode', etc.).
#             N_min (int): Minimum sample size for considering statistics reliable.
        
#         Returns:
#             pd.Series: The transformed feature column.
#         """
#         # Merge original DataFrame with the precomputed stats
#         df_stats = pd.merge(df[[self.group]], self.stats, on=self.group, how='left')
        
#         # Handle missing values
#         n, N = df_stats['n'].fillna(self.prior_mean), df_stats['N'].fillna(1.0)
        
#         # Calculate prior parameters
#         N_prior = np.maximum(N_min - N, 0)
#         alpha_prior = self.prior_mean * N_prior
#         beta_prior = (1 - self.prior_mean) * N_prior

#         # Calculate posterior parameters
#         alpha, beta = alpha_prior + n, beta_prior + N - n
        
#         # Calculate the requested statistic
#         value = self._calculate_stat(alpha, beta, stat_type)
        
#         # Handle missing values in the result
#         value[np.isnan(value)] = np.nanmedian(value)
        
#         return value


#     def _calculate_stat(self, alpha, beta, stat_type):
#         """Helper method to calculate a specific statistic from alpha and beta parameters."""
#         if stat_type == 'mean':
#             return alpha / (alpha + beta)
#         elif stat_type == 'mode':
#             return (alpha - 1) / (alpha + beta - 2)
#         elif stat_type == 'median':
#             return (alpha - 1/3) / (alpha + beta - 2/3)
#         elif stat_type == 'var':
#             return alpha * beta / ((alpha + beta)**2 * (alpha + beta + 1))
#         elif stat_type == 'skewness':
#             return 2 * (beta - alpha) * np.sqrt(alpha + beta + 1) / ((alpha + beta + 2) * np.sqrt(alpha * beta))
#         elif stat_type == 'kurtosis':
#             num = 6 * (alpha - beta)**2 * (alpha + beta + 1) - alpha * beta * (alpha + beta + 2)
#             return num / (alpha * beta * (alpha + beta + 2) * (alpha + beta + 3))
#         else:
#             raise ValueError(f"Unsupported stat_type: {stat_type}")


#     @staticmethod
#     def label_encoding(data, cat_cols):
#         """Encodes categorical columns in the DataFrame using Label Encoding."""
#         for col in cat_cols:
#             data[col] = LabelEncoder().fit_transform(data[col])
#         return data


#     @staticmethod
#     def beta_target_encoding(data, cat_cols, target='label'):
#         """Applies Beta Target Encoding to specified categorical columns."""
#         N_min = config.BETA_N_MIN
#         feature_cols = []
        
#         for col in cat_cols:
#             encoder = Cat_BetaEncoder(col)
#             encoder.fit(data, target)
#             for stat in config.STATISTIC:
#                 feature_name = f'{col}_{stat}'
#                 data[feature_name] = encoder.transform(data, stat, N_min)
#                 feature_cols.append(feature_name)
        
#         return data, feature_cols


# class Num_BetaEncoder:
#     def __init__(self, label_col: str, target_cols: List[str], stat_types: List[str], 
#                  range_ratios: Dict[str, float], N_min: int = 5, n_jobs: int = -1):
#         self.label_col = label_col
#         self.target_cols = target_cols
#         self.stat_types = stat_types
#         self.range_ratios = range_ratios 
#         self.N_min = N_min
#         self.n_jobs = n_jobs

#         self.df_train = None
#         self.p_global = None
#         self.trees = {}
#         self.radii = {}  # actual radius per column

#     def _calculate_stat(self, alpha, beta, stat_type):
#         if stat_type == 'mean':
#             return alpha / (alpha + beta)
#         elif stat_type == 'mode':
#             return (alpha - 1) / (alpha + beta - 2) if alpha > 1 and beta > 1 else np.nan
#         elif stat_type == 'median':
#             return (alpha - 1/3) / (alpha + beta - 2/3)
#         elif stat_type == 'var':
#             return alpha * beta / ((alpha + beta)**2 * (alpha + beta + 1))
#         elif stat_type == 'skewness':
#             return 2 * (beta - alpha) * np.sqrt(alpha + beta + 1) / ((alpha + beta + 2) * np.sqrt(alpha * beta))
#         elif stat_type == 'kurtosis':
#             num = 6 * (alpha - beta)**2 * (alpha + beta + 1) - alpha * beta * (alpha + beta + 2)
#             return num / (alpha * beta * (alpha + beta + 2) * (alpha + beta + 3))
#         else:
#             raise ValueError(f"Unsupported stat_type: {stat_type}")

#     def fit(self, df_train):
#         self.df_train = df_train
#         self.p_global = df_train[self.label_col].mean()
#         self.label_array = df_train[self.label_col].values

#         for col in self.target_cols:
#             X = df_train[[col]].values
#             self.trees[col] = KDTree(X)

#             # Standard Deviation
#             std = df_train[col].std()
#             self.radii[col] = self.range_ratios.get(col, 1.0) * std

#             # # Constant 
#             # self.radii[col] = 0.05

#     def transform(self, df_apply):
#         all_results = {}

#         for col in self.target_cols:
            
#             # 一次性取得所有 query points
#             X_apply = df_apply[[col]].values
#             tree = self.trees[col]
#             radius = self.radii[col]

#             # 一次查詢所有行的鄰居 index
#             all_neighbors_idx = tree.query_radius(X_apply, r=radius)
            
#             # 建立每列的統計結果列表
#             col_stats = {f'{col}_{stat}': [] for stat in self.stat_types}

#             for idx in all_neighbors_idx:
#                 labels = self.label_array[idx]
                
#                 N = len(labels)
#                 n = labels.sum()
                
#                 # 計算先驗（平滑）
#                 N_prior = max(0, self.N_min - N)
#                 alpha_prior = self.p_global * N_prior
#                 beta_prior = (1 - self.p_global) * N_prior

#                 alpha = alpha_prior + n
#                 beta = beta_prior + (N - n)

#                 # 計算所需的統計量
#                 for stat in self.stat_types:
#                     col_stats[f'{col}_{stat}'].append(self._calculate_stat(alpha, beta, stat))

#             # 將每列結果存入結果 dict
#             for stat in self.stat_types:
#                 all_results[f'{col}_{stat}'] = col_stats[f'{col}_{stat}']

#         return pd.DataFrame(all_results, index=df_apply.index)


# class Cat_BetaEncoder:
#     def __init__(self, cat_cols: List[str], stat_types: List[str], N_min: int = 1):
#         self.cat_cols = cat_cols
#         self.stat_types = stat_types
#         self.N_min = N_min
#         self.fitted_stats = defaultdict(dict)
#         self.prior_mean = None

#     def fit(self, df: pd.DataFrame, target_col: str):
#         self.prior_mean = df[target_col].mean()

#         for col in self.cat_cols:
#             group_stats = df.groupby(col)[target_col].agg(['sum', 'count'])
#             group_stats.rename(columns={'sum': 'n', 'count': 'N'}, inplace=True)

#             for category, row in group_stats.iterrows():
#                 n = row['n']
#                 N = row['N']

#                 # Compute prior
#                 N_prior     = max(self.N_min - N, 0)
#                 alpha_prior = self.prior_mean * N_prior
#                 beta_prior  = (1 - self.prior_mean) * N_prior

#                 # Posterior ( PAPER: 0 or 1 )
#                 alpha = alpha_prior + n
#                 beta  = beta_prior  + (N - n)

#                 self.fitted_stats[col][category] = {
#                     'alpha': alpha,
#                     'beta': beta
#                 }

#     def transform(self, df: pd.DataFrame) -> pd.DataFrame:
#         encoded_df = pd.DataFrame(index=df.index)

#         for col in self.cat_cols:
#             for stat in self.stat_types:
#                 values = []
#                 for val in df[col]:
#                     if val in self.fitted_stats[col]:
#                         alpha = self.fitted_stats[col][val]['alpha']
#                         beta = self.fitted_stats[col][val]['beta']
#                         values.append(self._calculate_stat(alpha, beta, stat))
#                     else:
#                         values.append(self.prior_mean)
#                 encoded_df[f'{col}_{stat}'] = values

#         return encoded_df

#     def _calculate_stat(self, alpha, beta, stat_type):
#         if stat_type == 'mean':
#             return alpha / (alpha + beta)
#         elif stat_type == 'mode':
#             return (alpha - 1) / (alpha + beta - 2) if alpha > 1 and beta > 1 else self.prior_mean
#         elif stat_type == 'median':
#             return (alpha - 1/3) / (alpha + beta - 2/3)
#         elif stat_type == 'var':
#             return alpha * beta / ((alpha + beta)**2 * (alpha + beta + 1))
#         elif stat_type == 'skewness':
#             return 2 * (beta - alpha) * np.sqrt(alpha + beta + 1) / ((alpha + beta + 2) * np.sqrt(alpha * beta))
#         elif stat_type == 'kurtosis':
#             num = 6 * (alpha - beta)**2 * (alpha + beta + 1) - alpha * beta * (alpha + beta + 2)
#             return num / (alpha * beta * (alpha + beta + 2) * (alpha + beta + 3))
#         else:
#             raise ValueError(f"Unsupported stat_type: {stat_type}")


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