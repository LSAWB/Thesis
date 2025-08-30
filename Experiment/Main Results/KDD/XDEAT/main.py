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

def main(hparams, run_id):

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_log_dir = os.path.join(config.BASE_LOG_DIR, f"{current_time}")
    os.makedirs(base_log_dir, exist_ok=True)
    
    valid_accs      = []
    test_accs       = []
    
    for experiment_number in range(config.EXPERIMENT_REPEAT):
        
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
        
        print("This is NUM COLS: ")
        print(NUM_COLS)
        
        print("This is CAT_COLS: ")
        print(CAT_COLS)
        
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
            test_size=1/3,
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
        
        print(train_data)
        print(valid_data)
        print(test_data)
        print(len(CAT_COLS))
        
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
        cat_val_array = ordinal_encoder.transform(valid_data[CAT_COLS])
        cat_test_array = ordinal_encoder.transform(test_data[CAT_COLS])

        cat_train_tensor = torch.tensor(cat_train_array, dtype=torch.long)
        cat_val_tensor = torch.tensor(cat_val_array, dtype=torch.long)
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
        # =========================== Model Training ===========================
        
        model = DualTransformer(
            ori_cardinalities=cat_cardinalities,
            num_features=len(NUM_COLS),
            cat_features=len(CAT_COLS),
            statistic_features=(len(NUM_COLS) + len(CAT_COLS)),
            dim_model=hparams["dim_model"],
            num_heads=config.NUM_HEADS,
            dim_ff=int(hparams["dim_model"] * hparams["ffn_factor"]),
            num_layers_cross=hparams["cross_blocks"],
            num_labels=config.NUM_LABELS,
            return_attention=True,
            att_dropout=hparams["att_dropout"],
            res_dropout=hparams["res_dropout"],
            ffn_dropout=hparams["ffn_dropout"]
        ).to(config.DEVICE)

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"])
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

    return mean_acc, std_acc

if __name__ == "__main__":
    with open("optuna_pareto_trials.json", "r") as f:
        trials = json.load(f)

    results = []
    for run_id, trial in enumerate(trials):
        hparams = trial["params"]
        print(f"\n====== Running trial {run_id} with params: {hparams} ======\n")
        test_mean_acc, test_std_acc = main(hparams, run_id)
        results.append({
            "trial_id": run_id,
            "params": hparams,
            "test_mean_accuarcy": test_mean_acc,
            "test_std_accuarcy": test_std_acc
        })

    # 儲存結果
    with open("optuna_pareto_results.json", "w") as f:
        json.dump(results, f, indent=4)
