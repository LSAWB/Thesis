# Standard
import os
import math
import json
import random
from datetime import datetime
from typing import Dict, Literal
from collections import defaultdict

# General
import numpy as np
import pandas as pd
from tqdm import tqdm

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, PowerTransformer, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

# PyTorch
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch.utils.data import DataLoader, Dataset

# Hyperparameter Optimization
import optuna

# Tabular Deep Learning Models
from rtdl_revisiting_models import MLP, ResNet, FTTransformer

# Other
import delu
import typing as ty

# Scheduler
def get_scheduler(optimizer, total_epochs, warmup_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        return 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Seed
def set_random_seed(seed):
    # Set Python random seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Dataset
class CustomDataset(Dataset):
    def __init__(self, x_num=None, x_cat=None, y=None):
        if x_num is not None:
            self.x_num = torch.tensor(x_num.values, dtype=torch.float32)
        else:
            self.x_num = None

        if x_cat is not None:
            self.x_cat = torch.tensor(x_cat.values, dtype=torch.long)
        else:
            self.x_cat = None

        self.labels = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        output = {}
        if self.x_num is not None:
            output["x_num"] = self.x_num[idx]
        if self.x_cat is not None:
            output["x_cat"] = self.x_cat[idx]
        return output, self.labels[idx]
    

# Early Stopping 
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.best_model_state = model.state_dict()
        elif val_score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.best_model_state = model.state_dict()
            self.counter = 0
    
            
# Model Trainer
class ModelTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, train_loader, epoch):
        self.model.train()
        train_loss = 0.0
        ground_truths, preds_logits = [], []

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            self.optimizer.zero_grad()

            (batch_data, labels) = batch
            
            if "x_num" in batch_data:
                x_num = batch_data["x_num"].to(self.device)
            else:
                x_num = None

            if "x_cat" in batch_data:
                x_cat = batch_data["x_cat"].to(self.device)
            else:
                x_cat = None
                
            labels = labels.squeeze(dim=-1).long().to(self.device)

            # Forward pass for DCNv2: (x_num, x_cat)
            logits = self.model(x_num, x_cat)
            class_loss = self.criterion(logits.float(), labels)

            total_loss = class_loss
            total_loss.backward()
            train_loss += total_loss.item()

            self.optimizer.step()
            self.scheduler.step()

            # probs = torch.sigmoid(logits).detach()
            preds_logits.append(logits.detach())
            ground_truths.append(labels.detach())

        return self._evaluate(ground_truths, preds_logits, train_loader, train_loss)

    def evaluate(self, val_loader, epoch):
        self.model.eval()
        valid_loss = 0.0
        ground_truths, preds_logits = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Evaluating Epoch {epoch+1}", leave=False):

                (batch_data, labels) = batch
                
                if "x_num" in batch_data:
                    x_num = batch_data["x_num"].to(self.device)
                else:
                    x_num = None

                if "x_cat" in batch_data:
                    x_cat = batch_data["x_cat"].to(self.device)
                else:
                    x_cat = None
                    
                labels = labels.squeeze(dim=-1).long().to(self.device)

                # Forward pass for DCNv2: (x_num, x_cat)
                logits = self.model(x_num, x_cat)
                class_loss = self.criterion(logits.float(), labels)
                    
                total_loss = class_loss
                valid_loss += total_loss.item()

                # probs = torch.sigmoid(logits).detach()
                preds_logits.append(logits.detach())
                ground_truths.append(labels.detach())

        return self._evaluate(ground_truths, preds_logits, val_loader, valid_loss)  

    
    def test(self, test_loader, epoch):
        self.model.eval()
        test_loss = 0.0
        ground_truths, preds_logits = [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Testing Epoch {epoch+1}", leave=False):
                
                (batch_data, labels) = batch
                
                if "x_num" in batch_data:
                    x_num = batch_data["x_num"].to(self.device)
                else:
                    x_num = None

                if "x_cat" in batch_data:
                    x_cat = batch_data["x_cat"].to(self.device)
                else:
                    x_cat = None
                    
                labels = labels.squeeze(dim=-1).long().to(self.device)

                # Forward pass for DCNv2: (x_num, x_cat)
                logits = self.model(x_num, x_cat)
                class_loss = self.criterion(logits.float(), labels)
                    
                total_loss = class_loss
                test_loss += total_loss.item()

                # probs = torch.sigmoid(logits).detach()
                preds_logits.append(logits.detach())
                ground_truths.append(labels.detach())

        return self._evaluate(ground_truths, preds_logits, test_loader, test_loss)  


    def _prepare_data(self, data):
        tabular_data, labels = data
        batch_size = len(tabular_data)

        return tabular_data.to(self.device, dtype=torch.long), labels.to(self.device, dtype=torch.float)  


    def _evaluate(self, ground_truths, preds_logits, loader, loss):
        ground_truths = torch.cat(ground_truths)
        preds_logits = torch.cat(preds_logits)
        loss /= len(loader)

        # Extract predicted labels
        preds_probs = torch.sigmoid(preds_logits)
        preds_labels = preds_probs.argmax(dim=1)

        # Convert tensors to numpy arrays
        y_true = ground_truths.cpu().numpy()
        y_pred = preds_labels.cpu().numpy()

        # Use sklearn to compute accuracy
        accuracy = accuracy_score(y_true, y_pred)

        return loss, accuracy


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
    
    # Define Hyperparameter
    experiment_repeat = 15
    n_classes = 2
    batch_size = 128
    epochs = 200

    # Get current time to name the log directory
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_log_dir = os.path.join(script_dir, current_time)

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

        numerical_encoders = {}
        X_train_num_list, X_valid_num_list, X_test_num_list = [], [], []
        for col in NUM_COLS:
            encoder = NumericalEstimateEncoder(feature_col=col, target_col='label', bin_size=0.001, m=100)
            encoder.fit(train_data)
            numerical_encoders[col] = encoder
            
            X_train_num_list.append(encoder.transform(train_data).rename(col + '_enc'))
            X_valid_num_list.append(encoder.transform(valid_data).rename(col + '_enc'))
            X_test_num_list.append(encoder.transform(test_data).rename(col + '_enc'))
        
        # X_train_num = pd.concat(X_train_num_list, axis=1)
        # X_valid_num = pd.concat(X_valid_num_list, axis=1)
        # X_test_num  = pd.concat(X_test_num_list, axis=1)

        categorical_encoders = {}
        X_train_cat_list, X_valid_cat_list, X_test_cat_list = [], [], []
        for col in CAT_COLS:
            encoder = CategoricalEstimateEncoder(feature_col=col, target_col='label', m=100)
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
        X_train_bte_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_bte_scaled),
            columns=X_train_bte_scaled.columns,
            index=X_train_bte_scaled.index
        )
        X_valid_bte_scaled = pd.DataFrame(
            scaler.transform(X_valid_bte_scaled),
            columns=X_valid_bte_scaled.columns,
            index=X_valid_bte_scaled.index
        )
        X_test_bte_scaled = pd.DataFrame(
            scaler.transform(X_test_bte_scaled),
            columns=X_test_bte_scaled.columns,
            index=X_test_bte_scaled.index
        )
        
        # Dataset and Dataloader
        train_dataset   = CustomDataset(X_train_bte_scaled, None, y_train)
        val_dataset     = CustomDataset(X_valid_bte_scaled, None, y_valid)
        test_dataset    = CustomDataset(X_test_bte_scaled, None, y_test)
        train_loader    = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader      = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader     = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FTTransformer(
            n_cont_features=len(CAT_COLS) + len(NUM_COLS),
            cat_cardinalities=[],
            d_out=n_classes,
            **FTTransformer.get_default_kwargs(),
        ).to(device)

        optimizer = model.make_default_optimizer()
        criterion  = F.cross_entropy

        # Define total steps and warmup steps
        total_steps = epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        early_stopping = EarlyStopping(patience=16, delta=0)

        best_valid_acc = 0
        all_epoch_metrics = []
        
        # Create a new directory for this experiment
        experiment_log_dir = os.path.join(base_log_dir, str(experiment_number+1))
        os.makedirs(experiment_log_dir, exist_ok=True)
        best_model_path = os.path.join(base_log_dir, f"best_model_seed_{experiment_number}.pth")

        # Initialize trainer
        trainer = ModelTrainer(model, criterion, optimizer, scheduler, device)

        for epoch in range(epochs):
            
            # Train
            train_loss, train_accuracy = trainer.train(train_loader, epoch)

            # Evaluate
            valid_loss, valid_accuracy = trainer.evaluate(val_loader, epoch)

            # Info
            print(f"Epoch: {epoch + 1}/{epochs}")
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Validation Accuracy: {valid_accuracy:.4f}")

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
                torch.save(save_file, best_model_path)

            # Early stopping
            early_stopping(valid_accuracy, model) 
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # After training, load the best model and test
        checkpoint = torch.load(best_model_path)

        # Restore model, optimizer, scheduler, and epoch
        model.load_state_dict(checkpoint["model"])
        
        test_loss, test_accuracy = trainer.test(test_loader, epoch)
        print(f"Test Accuracy : {test_accuracy:.4f}")
        test_accs.append(test_accuracy)
        
        # Delete the best model file after loading
        if os.path.exists(best_model_path):  
            os.remove(best_model_path)  # Deletes the file
            print(f"Deleted best model checkpoint: {best_model_path}")
        else:
            print(f"Checkpoint file not found: {best_model_path}")

    print(f"Test Accuarcy Mean: {np.mean(test_accs)}")
    print(f"Test Accuarcy STD : {np.std(test_accs)}")

if __name__ == '__main__':
    main()