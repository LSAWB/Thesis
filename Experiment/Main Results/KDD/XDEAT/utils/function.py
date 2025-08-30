import torch
import random
import numpy as np
import pandas as pd
import math

def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def round_tokenization(df, num_columns, rounding=2):
    """
    Apply value_to_token_rounded to selected numerical columns with custom rounding.
    """
    def value_to_token_rounded(value):
        # Skip processing for non-numeric or NaN values
        if pd.isnull(value) or not isinstance(value, (int, float)):
            return None
        
        # Round and scale
        rounded_value = round(value, rounding)
        token = int(rounded_value * (10 ** rounding))
        return token

    # Apply to specified columns
    for col in num_columns:
        df[col] = df[col].apply(value_to_token_rounded)
    
    return df


def encode_train_data(X_train, y_train, columns, cat_col, encoded_suffix):
    """
    Perform Beta Target Encoding on training data and generate mappings.

    Parameters:
    - X_train (pd.DataFrame): Training feature data.
    - y_train (pd.Series): Training labels.
    - columns (list): Columns to encode.
    - cat_col (list): Categorical columns for label encoding.
    - encoded_suffix (list): List of suffixes for encoding.

    Returns:
    - pd.DataFrame: Encoded training data.
    - dict: Mapping dictionary for validation and test encoding.
    """
    # Label Encoding
    train_groups = X_train[columns].copy()
    train_group = Cat_BetaEncoder.label_encoding(train_groups, cat_col)
    
    # Combine features with target for encoding
    train_data = pd.concat([train_group, y_train], axis=1)
    train_data.reset_index(inplace=True, drop=True)

    # Perform Beta Target Encoding
    train_data, _ = Cat_BetaEncoder.beta_target_encoding(train_data, columns)
    
    # Drop label column after encoding
    train_data = train_data.drop(columns=['label'], axis=1)

    # Create mapping dictionary
    mappings = {}
    for col in columns:
        mappings[col] = {}
        for suffix in encoded_suffix:
            
            # Extract the unique pairs of original and encoded values
            encoded_col = col + suffix
            if encoded_col in train_data.columns:  
                mapping = train_data[[col, encoded_col]].drop_duplicates().reset_index(drop=True)
                mappings[col][suffix] = mapping

    return train_data, mappings


def encode_other_data(X_data, y_data, columns, cat_col, encoded_suffix, mappings):
    """
    Apply Beta Target Encoding to validation or test data using predefined mappings.

    Parameters:
    - X_data (pd.DataFrame): Validation or test feature data.
    - y_data (pd.Series): Validation or test labels.
    - columns (list): Columns to encode.
    - cat_col (list): Categorical columns for label encoding.
    - encoded_suffix (list): List of suffixes for encoding.
    - mappings (dict): Mapping dictionary generated from training data.

    Returns:
    - pd.DataFrame: Encoded validation or test data.
    """
    # Label Encoding
    data_groups = X_data[columns].copy()
    encoded_group = Cat_BetaEncoder.label_encoding(data_groups, cat_col)

    # Combine features with target for encoding
    encoded_data = pd.concat([encoded_group, y_data], axis=1)
    encoded_data.reset_index(inplace=True, drop=True)

    # Apply mappings for encoding
    for col in columns:
        for suffix in encoded_suffix:
            encoded_data = apply_mapping(encoded_data, col, suffix, mappings)

    # Drop label column after encoding
    encoded_data = encoded_data.drop(columns=['label'], axis=1)
    
    return encoded_data


def apply_mapping(data, column, statistic, mappings):
    # Create a dictionary from the mappings DataFrame for the specific statistic
    mapping_dict = dict(zip(mappings[column][statistic][column], mappings[column][statistic][f'{column}{statistic}']))
    
    # Apply the mapping to the data
    data[f'{column}{statistic}'] = data[column].map(mapping_dict)
    return data


def reorder_columns(data, columns):
    temp = data.drop(columns, axis=1) 
    final_columns_order = []

    for feature in columns:
        if feature in data.columns:
            final_columns_order.append(feature)

        feature_stats_columns = [
            col for col in temp.columns
            if col.startswith(feature + "_")
        ]
        final_columns_order.extend(feature_stats_columns)

    return data[final_columns_order]


def _calculate_stat(alpha, beta, stat_type):
    if stat_type == 'mean':
        return alpha / (alpha + beta)
    elif stat_type == 'mode':
        return (alpha - 1) / (alpha + beta - 2) if (alpha > 1 and beta > 1) else np.nan
    elif stat_type == 'median':
        return (alpha - 1/3) / (alpha + beta - 2/3) if (alpha > 1 and beta > 1) else np.nan
    elif stat_type == 'skewness':
        return 2 * (beta - alpha) * np.sqrt(alpha + beta + 1) / ((alpha + beta + 2) * np.sqrt(alpha * beta))
    elif stat_type == 'kurtosis':
        num = 6 * (alpha - beta)**2 * (alpha + beta + 1) - alpha * beta * (alpha + beta + 2)
        denom = alpha * beta * (alpha + beta + 2) * (alpha + beta + 3)
        return num / denom
    else:
        raise ValueError(f"Unsupported stat_type: {stat_type}")

def _compute_stats_for_row(i, df, label_col, col, stat_types, range_radius, N_min, p_global):
    target_value = df.iloc[i][col]
    mask = (df[col] >= target_value - range_radius) & (df[col] <= target_value + range_radius)
    neighbors = df[mask]
    N = len(neighbors)
    n = neighbors[label_col].sum()

    N_prior = max(0, N_min - N)
    alpha_prior = p_global * N_prior
    beta_prior = (1 - p_global) * N_prior

    alpha = alpha_prior + n
    beta = beta_prior + (N - n)

    return {
        f'{col}_{stat}': _calculate_stat(alpha, beta, stat) for stat in stat_types
    }

def compute_numerical_bte(df, label_col, target_cols, stat_types, range_radius=0.05, N_min=5, n_jobs=-1):
    results = {}
    for col in target_cols:
        p_global = df[label_col].mean()
        column_results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_stats_for_row)(i, df, label_col, col, stat_types, range_radius, N_min, p_global)
            for i in range(len(df))
        )
        for stat in stat_types:
            results[f'{col}_{stat}'] = [row[f'{col}_{stat}'] for row in column_results]
    return pd.DataFrame(results)


def get_scheduler(optimizer, total_epochs, warmup_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        return 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)