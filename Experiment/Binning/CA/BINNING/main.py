import config
from utils import (set_random_seed, NumericalEstimateEncoder, CategoricalEstimateEncoder, get_scheduler)
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
from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    MinMaxScaler,
    PowerTransformer,
    StandardScaler
)

import json
import time

def main():

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_log_dir = os.path.join(config.BASE_LOG_DIR, f"{current_time}")
    os.makedirs(base_log_dir, exist_ok=True)
    
    valid_rmses      = []
    test_rmses       = []
    all_preprocess_times = []
    
    for experiment_number in range(config.EXPERIMENT_REPEAT):
        
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
        y_val   = (y_val - y_mean) / y_std
        y_test  = (y_test  - y_mean) / y_std
    
        train_data = X_train.copy()
        train_data['label'] = y_train

        valid_data = X_val.copy()
        valid_data['label'] = y_val

        test_data = X_test.copy()
        test_data['label'] = y_test

        # Start timing for NumericalEstimateEncoder
        start_time = time.time()
                
        numerical_encoders = {}
        X_train_num_list, X_valid_num_list, X_test_num_list = [], [], []
        for col in NUM_COLS:
            encoder = NumericalEstimateEncoder(feature_col=col, target_col='label', bin_size=0.001, m=len(train_data)*config.ENCODING_SMOOTHING)
            encoder.fit(train_data)
            numerical_encoders[col] = encoder
            
            X_train_num_list.append(encoder.transform(train_data).rename(col + '_enc'))
            X_valid_num_list.append(encoder.transform(valid_data).rename(col + '_enc'))
            X_test_num_list.append(encoder.transform(test_data).rename(col + '_enc'))
        
        X_train_num = pd.concat(X_train_num_list, axis=1)
        X_valid_num = pd.concat(X_valid_num_list, axis=1)
        X_test_num  = pd.concat(X_test_num_list, axis=1)

        # Stop timing and save duration
        preprocess_duration = time.time() - start_time
        print(f"[Experiment {experiment_number+1}] Numerical Preprocessing Time: {preprocess_duration:.4f} seconds")
        all_preprocess_times.append(preprocess_duration)
        
        categorical_encoders = {}
        X_train_cat_list, X_valid_cat_list, X_test_cat_list = [], [], []
        for col in CAT_COLS:
            encoder = CategoricalEstimateEncoder(feature_col=col, target_col='label', m=len(train_data)*config.ENCODING_SMOOTHING)
            encoder.fit(train_data)
            categorical_encoders[col] = encoder
            
            X_train_cat_list.append(encoder.transform(train_data).rename(col + '_enc'))
            X_valid_cat_list.append(encoder.transform(valid_data).rename(col + '_enc'))
            X_test_cat_list.append(encoder.transform(test_data).rename(col + '_enc'))

        X_train_bte_scaled  = X_train_num
        X_valid_bte_scaled  = X_valid_num
        X_test_bte_scaled   = X_test_num
        
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
        
        # # Categorical
        # ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        # cat_train_array = ordinal_encoder.fit_transform(train_data[CAT_COLS])
        # cat_val_array   = ordinal_encoder.transform(valid_data[CAT_COLS])
        # cat_test_array  = ordinal_encoder.transform(test_data[CAT_COLS])

        # cat_train_tensor = torch.tensor(cat_train_array, dtype=torch.long)
        # cat_val_tensor  = torch.tensor(cat_val_array, dtype=torch.long)
        # cat_test_tensor = torch.tensor(cat_test_array, dtype=torch.long)
        
        # Numerical
        power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        num_train_array = power_transformer.fit_transform(train_data[NUM_COLS])
        num_val_array = power_transformer.transform(valid_data[NUM_COLS])
        num_test_array = power_transformer.transform(test_data[NUM_COLS])

        num_train_tensor = torch.tensor(num_train_array, dtype=torch.float)
        num_val_tensor = torch.tensor(num_val_array, dtype=torch.float)
        num_test_tensor = torch.tensor(num_test_array, dtype=torch.float)
        
        train_dataset = CustomDataset(None, num_train_tensor, BTE_train_scaled, y_train)
        val_dataset   = CustomDataset(None, num_val_tensor, BTE_valid_scaled, y_val)
        test_dataset  = CustomDataset(None, num_test_tensor, BTE_test_scaled, y_test)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        # =========================== Model Training ===========================
        
        model = DualTransformer(
            ori_cardinalities=[],
            num_features=len(NUM_COLS),
            cat_features=len(CAT_COLS),
            statistic_features=(len(NUM_COLS) + len(CAT_COLS)),
            dim_model=144,
            num_heads=config.NUM_HEADS,
            dim_ff=int(144 * 2.194723548402206),
            num_layers_cross=4,
            num_labels=config.NUM_LABELS,
            return_attention=True,
            att_dropout=0.23098690183628157,
            res_dropout=0.017294919182791336,
            ffn_dropout=0.4158676676923419
        ).to(config.DEVICE)

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005403927782950507, weight_decay=0.0009793464546267137)
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
            y_std=y_std
        )
        
        # Start Training
        best_valid_rmse = float("inf")
        all_epoch_metrics = []
        print(f"Experiment {experiment_number+1} with random seed {experiment_number}")
        
        best_model_path = os.path.join(base_log_dir, f"best_model_seed_{experiment_number}.pth")

        for epoch in range(config.EPOCHS):
            
            train_loss, train_rmse = trainer.train(train_loader, epoch)
            valid_loss, valid_rmse = trainer.evaluate(val_loader, epoch)

            # Info
            print(f"Epoch: {epoch + 1}/{config.EPOCHS}")
            print(f"Training RMSE: {train_rmse:.4f}")
            print(f"Validation RMSE: {valid_rmse:.4f}")

            # Save model
            if valid_rmse < best_valid_rmse:
                save_file = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                }
                best_valid_rmse = valid_rmse
                torch.save(save_file, best_model_path)

            # Early stopping
            early_stopping(valid_rmse, model) 
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # After training, load the best model and test
        checkpoint = torch.load(best_model_path)

        # Restore model, optimizer, scheduler, and epoch
        model.load_state_dict(checkpoint["model"])
        
        test_loss, test_rmse = trainer.test(test_loader, epoch)
        print(f"Test RMSE : {test_rmse:.4f}")
        test_rmses.append(test_rmse)
        
        # Delete the best model file after loading
        if os.path.exists(best_model_path):  
            os.remove(best_model_path)  # Deletes the file
            print(f"Deleted best model checkpoint: {best_model_path}")
        else:
            print(f"Checkpoint file not found: {best_model_path}")

    test_rmses = [rmse_tensor.cpu().item() for rmse_tensor in test_rmses]
    print(f"Test RMSE Mean: {np.mean(test_rmses)}")
    print(f"Test RMSE STD : {np.std(test_rmses)}")
    print(f"Numerical Preprocessing Avg Time: {np.mean(all_preprocess_times):.4f} seconds")
    
if __name__ == "__main__":
    main()

