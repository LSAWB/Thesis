import config
from utils import (set_random_seed, reorder_columns, NumericalEstimateEncoder, CategoricalEstimateEncoder, get_scheduler)
from dataset import CustomDataset
from model import DualTransformer
from trainer import EarlyStopping, ModelTrainer

# ========== Standard Library ==========
import os
import math
import random
from datetime import datetime

# ========== Third-Party Libraries ==========
import numpy as np
import pandas as pd
from tqdm import tqdm
import openml

# ========== PyTorch ==========
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

# ========== Sklearn ==========
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    MinMaxScaler,
    PowerTransformer,
    StandardScaler
)

# ========== Visualization ==========
from tensorboardX import SummaryWriter

# ========== Other ==========
import optuna

import json

# Global cache variables
CACHED_DATA = None


def objective(trial):

    experiment_number = 42

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

    # Step 5: Update CAT_COLS and NUM_COLS
    CAT_COLS = df.select_dtypes(include=['category', 'object']).columns.tolist()
    NUM_COLS = df.select_dtypes(include=['number']).drop(columns=['label']).columns.tolist()
    
    for col in CAT_COLS:
        df[col] = df[col].astype(str)
        
    # Handle rare categories
    threshold = len(df) * config.RARE_CATEGORIES_RATIO
    for col in CAT_COLS:
        value_counts = df[col].value_counts()
        rare_categories = value_counts[value_counts < threshold].index
        df[col] = df[col].apply(lambda x: 'Others' if x in rare_categories else x)
    
    # Dataset Split
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
    y_train = train_data['label'].values
    y_val = valid_data['label'].values
    y_test  = test_data['label'].values
    
    cat_cardinalities = [train_data[col].nunique() for col in CAT_COLS]
    
    # # Encoding
    # numerical_encoders = {}
    # X_train_num_list, X_valid_num_list, X_test_num_list = [], [], []
    # for col in NUM_COLS:
    #     encoder = NumericalEstimateEncoder(feature_col=col, target_col='label', bin_size=config.ENCODING_BINSIZE, m=len(train_data)*config.ENCODING_SMOOTHING)
    #     encoder.fit(train_data)
    #     numerical_encoders[col] = encoder
        
    #     X_train_num_list.append(encoder.transform(train_data).rename(col + '_enc'))
    #     X_valid_num_list.append(encoder.transform(valid_data).rename(col + '_enc'))
    #     X_test_num_list.append(encoder.transform(test_data).rename(col + '_enc'))

    # X_train_num = pd.concat(X_train_num_list, axis=1)
    # X_valid_num = pd.concat(X_valid_num_list, axis=1)
    # X_test_num  = pd.concat(X_test_num_list, axis=1)

    categorical_encoders = {}
    X_train_cat_list, X_valid_cat_list, X_test_cat_list = [], [], []
    for col in CAT_COLS:
        encoder = CategoricalEstimateEncoder(feature_col=col, target_col='label', m=len(train_data)*config.ENCODING_SMOOTHING)
        encoder.fit(train_data)
        categorical_encoders[col] = encoder
        
        X_train_cat_list.append(encoder.transform(train_data).rename(col + '_enc'))
        X_valid_cat_list.append(encoder.transform(valid_data).rename(col + '_enc'))
        X_test_cat_list.append(encoder.transform(test_data).rename(col + '_enc'))

    X_train_cat = pd.concat(X_train_cat_list, axis=1)
    X_valid_cat = pd.concat(X_valid_cat_list, axis=1)
    X_test_cat  = pd.concat(X_test_cat_list, axis=1)
            
    # X_train_bte_scaled  = pd.concat([X_train_num, X_train_cat],axis=1) 
    # X_valid_bte_scaled  = pd.concat([X_valid_num, X_valid_cat],axis=1) 
    # X_test_bte_scaled   = pd.concat([X_test_num, X_test_cat],axis=1)
        
    X_train_bte_scaled  = X_train_cat
    X_valid_bte_scaled  = X_valid_cat
    X_test_bte_scaled   = X_test_cat

    scaler = StandardScaler()
    BTE_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_bte_scaled),
        columns=X_train_bte_scaled.columns,
        index=X_train_bte_scaled.index
    )
    BTE_valid_scaled = pd.DataFrame(
        scaler.transform(X_valid_bte_scaled),
        columns=X_valid_bte_scaled.columns,
        index=X_valid_bte_scaled.index
    )
    BTE_test_scaled = pd.DataFrame(
        scaler.transform(X_test_bte_scaled),
        columns=X_test_bte_scaled.columns,
        index=X_test_bte_scaled.index
    )
    
    # Categorical
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    cat_train_array = ordinal_encoder.fit_transform(train_data[CAT_COLS])
    cat_val_array   = ordinal_encoder.transform(valid_data[CAT_COLS])
    cat_test_array  = ordinal_encoder.transform(test_data[CAT_COLS])

    cat_train_tensor = torch.tensor(cat_train_array, dtype=torch.long)
    cat_val_tensor  = torch.tensor(cat_val_array, dtype=torch.long)
    cat_test_tensor = torch.tensor(cat_test_array, dtype=torch.long)
    
    # Numerical
    # power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    # num_train_array = power_transformer.fit_transform(train_data[NUM_COLS])
    # num_val_array = power_transformer.transform(valid_data[NUM_COLS])
    # num_test_array = power_transformer.transform(test_data[NUM_COLS])

    # num_train_tensor = torch.tensor(num_train_array, dtype=torch.float)
    # num_val_tensor = torch.tensor(num_val_array, dtype=torch.float)
    # num_test_tensor = torch.tensor(num_test_array, dtype=torch.float)

    train_dataset = CustomDataset(cat_train_tensor, None, BTE_train_scaled, y_train)
    val_dataset   = CustomDataset(cat_val_tensor, None, BTE_valid_scaled, y_val)
    test_dataset  = CustomDataset(cat_test_tensor, None, BTE_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Model 
    dim_model   = trial.suggest_int('dim_model', 64, 512, step=8)
    ffn_factor  = trial.suggest_float('ffn_factor', 2/3, 8/3)
    dim_ff      = int(dim_model * ffn_factor)
    
    num_layers_cross    = trial.suggest_int("cross_blocks", 1, 6)
    
    res_dropout = trial.suggest_float('res_dropout', 0.0, 0.2)
    att_dropout = trial.suggest_float('att_dropout', 0.0, 0.5)
    ffn_dropout = trial.suggest_float('ffn_dropout', 0.0, 0.5)
    
    learning_rate   = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay    = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # num_layers_vit      = trial.suggest_int("vit_blocks", 1, 4)

    model = DualTransformer(
        ori_cardinalities=cat_cardinalities,
        num_features=len(NUM_COLS),
        cat_features=len(CAT_COLS),
        statistic_features=(len(NUM_COLS) + len(CAT_COLS)),
        num_heads=config.NUM_HEADS,
        dim_model=dim_model,
        dim_ff=dim_ff,
        num_layers_cross=num_layers_cross,
        res_dropout=res_dropout,
        att_dropout=att_dropout,
        ffn_dropout=ffn_dropout,
        return_attention=True,
        num_labels=config.NUM_LABELS
    ).to(config.DEVICE)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS * 10)
    scheduler = get_scheduler(optimizer, config.EPOCHS, config.WARMUP_EPOCHS)

    # Early Stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, delta=config.EARLY_STOPPING_DELTA)

    # Trainer
    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.DEVICE,
        no_of_classes=config.NUM_LABELS,
    )

    # Start Training
    best_valid_acc = 0
    all_epoch_metrics = []

    for epoch in range(config.EPOCHS):
        
        # Train
        train_loss, train_accuracy = trainer.train(train_loader, epoch)

        # Evaluate
        valid_loss, valid_accuracy = trainer.evaluate(val_loader, epoch)
        
        # Info
        print(f"Epoch: {epoch + 1}/{config.EPOCHS}")
        print(f"Training Accuracy Score: {train_accuracy:.4f}")
        print(f"Validation Accuracy Score: {valid_accuracy:.4f}")

        # === 儲存每個 epoch 的 AUROC ===
        all_epoch_metrics.append({
            "epoch": epoch,
            "train_max_acc": train_accuracy,
            "valid_max_acc": valid_accuracy
        })

        # Save model
        if valid_accuracy > best_valid_acc:
            save_file = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }
            best_valid_acc = valid_accuracy
            # torch.save(save_file, best_model_path)

        # Early stopping
        early_stopping(valid_accuracy)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        best_epoch = max(all_epoch_metrics, key=lambda x: x["valid_max_acc"])
        train_acc = best_epoch["train_max_acc"]
        valid_acc = best_epoch["valid_max_acc"]

    return train_acc, valid_acc 


if __name__ == "__main__":
    
    study = optuna.create_study(directions=["maximize", "maximize"])
    study.optimize(objective, n_trials=50)

    pareto_results = []
    for t in study.best_trials:
        result = {
            "train_auroc": t.values[0],
            "valid_auroc": t.values[1],
            "params": t.params
        }
        pareto_results.append(result)

    with open("optuna_pareto_trials.json", "w") as f:
        json.dump(pareto_results, f, indent=4)

    print("✅ Saved Pareto trials to optuna_pareto_trials.json")
    