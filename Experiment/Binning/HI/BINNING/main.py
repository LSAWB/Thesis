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
from sklearn.datasets import fetch_openml
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
    
    valid_accs      = []
    test_accs       = []
    all_preprocess_times = []
    
    for experiment_number in range(config.EXPERIMENT_REPEAT):
        
        # Set random seed for reproducibility
        set_random_seed(experiment_number)

        # --- Load OpenML HIGGS SMALL dataset ---
        df = fetch_openml(data_id=23512, as_frame=True)['frame']
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
        y_val   = valid_data['label'].values
        y_test  = test_data['label'].values
        
        # Start timing for NumericalEstimateEncoder
        start_time = time.time()
        
        # Encoding
        numerical_encoders = {}
        X_train_num_list, X_valid_num_list, X_test_num_list = [], [], []
        for col in NUM_COLS:
            encoder = NumericalEstimateEncoder(feature_col=col, target_col='label', bin_size=config.ENCODING_BINSIZE, m=len(train_data)*config.ENCODING_SMOOTHING)
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
        
        # categorical_encoders = {}
        # X_train_cat_list, X_valid_cat_list, X_test_cat_list = [], [], []
        # for col in CAT_COLS:
        #     encoder = CategoricalEstimateEncoder(feature_col=col, target_col='label', m=len(train_data)*config.ENCODING_SMOOTHING)
        #     encoder.fit(train_data)
        #     categorical_encoders[col] = encoder
            
        #     X_train_cat_list.append(encoder.transform(train_data).rename(col + '_enc'))
        #     X_valid_cat_list.append(encoder.transform(valid_data).rename(col + '_enc'))
        #     X_test_cat_list.append(encoder.transform(test_data).rename(col + '_enc'))

        # X_train_cat = pd.concat(X_train_cat_list, axis=1)
        # X_valid_cat = pd.concat(X_valid_cat_list, axis=1)
        # X_test_cat  = pd.concat(X_test_cat_list, axis=1)
                
        # X_train_bte_scaled  = pd.concat([X_train_num, X_train_cat],axis=1) 
        # X_valid_bte_scaled  = pd.concat([X_valid_num, X_valid_cat],axis=1) 
        # X_test_bte_scaled   = pd.concat([X_test_num, X_test_cat],axis=1)
         
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
        # cat_val_array = ordinal_encoder.transform(valid_data[CAT_COLS])
        # cat_test_array = ordinal_encoder.transform(test_data[CAT_COLS])

        # cat_train_tensor = torch.tensor(cat_train_array, dtype=torch.long)
        # cat_val_tensor = torch.tensor(cat_val_array, dtype=torch.long)
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
            ori_cardinalities=cat_cardinalities,
            num_features=len(NUM_COLS),
            cat_features=len(CAT_COLS),
            statistic_features=(len(NUM_COLS) + len(CAT_COLS)),
            dim_model=456,
            num_heads=config.NUM_HEADS,
            dim_ff=int(456 * 2.5679347000680206),
            num_layers_cross=6,
            num_labels=config.NUM_LABELS,
            return_attention=True,
            att_dropout=0.1717876000472725,
            res_dropout=0.04776200794444128,
            ffn_dropout=0.20127223443220832
        ).to(config.DEVICE)
        
        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=5.23659189978151e-05, weight_decay=2.0637691093199376e-06)
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
        print(f"Experiment {experiment_number+1} with random seed {experiment_number}")
        
        best_model_path = os.path.join(base_log_dir, f"best_model_seed_{experiment_number}.pth")

        for epoch in range(config.EPOCHS):
            
            # Train
            train_loss, train_accuracy = trainer.train(train_loader, epoch)

            # Evaluate
            valid_loss, valid_accuracy = trainer.evaluate(val_loader, epoch)
            
            # Info
            print(f"Epoch: {epoch + 1}/{config.EPOCHS}")
            print(f"Training Accuracy Score: {train_accuracy:.4f}")
            print(f"Validation Accuracy Score: {valid_accuracy:.4f}")

            # Save model
            if valid_accuracy > best_valid_acc:
                save_file = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                }
                best_valid_acc = valid_accuracy
                torch.save(save_file, best_model_path)

            # Early stopping
            early_stopping(valid_accuracy)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # After training, load the best model and test
        checkpoint = torch.load(best_model_path)

        # Restore model, optimizer, scheduler, and epoch
        model.load_state_dict(checkpoint["model"])
        
        test_loss, test_accuracy = trainer.test(test_loader, epoch)

        print(f"Test Accuracy : {test_accuracy:.4f}")
        
        # Store the best AUROC for this experiment
        valid_accs.append(best_valid_acc)
        test_accs.append(test_accuracy)

        # Delete the best model file after loading
        if os.path.exists(best_model_path):  
            os.remove(best_model_path)  # Deletes the file
            print(f"Deleted best model checkpoint: {best_model_path}")
        else:
            print(f"Checkpoint file not found: {best_model_path}")

    mean_acc = np.mean(test_accs)
    std_acc = np.std(test_accs)
    print(f"Test : Mean ACC: {mean_acc:.4f}, Std Dev ACC: {std_acc:.4f}")
    print(f"Numerical Preprocessing Avg Time: {np.mean(all_preprocess_times):.4f} seconds")

if __name__ == "__main__":
    main()
