# Standard
import os
import math
import json
import random
from datetime import datetime
from typing import Dict, Literal, List
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


class NumericalBetaEncoder:
    def __init__(self, feature_col: str, target_col: str, bin_size: float = 0.01, N_min: int = 5):
        self.feature_col = feature_col
        self.target_col = target_col
        self.bin_size = bin_size
        self.N_min = N_min

        self.scaler = MinMaxScaler()
        self.p_global = None
        self.bin_to_stat = {}
        self.std_bin = None
        self.bin_to_y = defaultdict(list)

    def fit(self, df: pd.DataFrame):
        x = df[self.feature_col].values.reshape(-1, 1)
        y = df[self.target_col].values
        x_scaled = self.scaler.fit_transform(x).flatten()
        bin_indices = (x_scaled / self.bin_size).astype(int)

        for i, bin_idx in enumerate(bin_indices):
            self.bin_to_y[bin_idx].append(y[i])

        self.p_global = np.mean(y)
        self.std_bin = int(np.std(bin_indices))

        for bin_idx in self.bin_to_y:
            neighbor_y = []
            for i in range(bin_idx - self.std_bin, bin_idx + self.std_bin + 1):
                if i in self.bin_to_y:
                    neighbor_y.extend(self.bin_to_y[i])

            N = len(neighbor_y)
            n = sum(neighbor_y)

            N_prior = max(self.N_min - N, 0)
            alpha = n + self.p_global * N_prior
            beta  = (N - n) + (1 - self.p_global) * N_prior

            smoothed_mean = alpha / (alpha + beta)
            self.bin_to_stat[bin_idx] = (smoothed_mean, N)

    def transform(self, df: pd.DataFrame) -> pd.Series:
        x = df[self.feature_col].values.reshape(-1, 1)
        x_scaled = self.scaler.transform(x).flatten()
        bin_indices = (x_scaled / self.bin_size).astype(int)

        known_bins = np.array(list(self.bin_to_stat.keys()))
        values = []

        for bin_idx in bin_indices:
            if bin_idx in self.bin_to_stat:
                values.append(self.bin_to_stat[bin_idx][0])
            else:
                dists = np.abs(known_bins - bin_idx)
                min_dist = np.min(dists)
                closest_bins = known_bins[dists == min_dist]

                weighted_sum = 0
                total_weight = 0
                for b in closest_bins:
                    mean, count = self.bin_to_stat[b]
                    weighted_sum += mean * count
                    total_weight += count

                values.append(weighted_sum / total_weight)

        return pd.Series(values, index=df.index)


class CategoricalBetaEncoder:
    def __init__(self, feature_col: str, target_col: str, N_min: int = 5):
        self.feature_col = feature_col
        self.target_col = target_col
        self.N_min = N_min

        self.p_global = None
        self.cat_to_mean = {}
        self.cat_to_count = {}

    def fit(self, df: pd.DataFrame):
        y = df[self.target_col]
        x = df[self.feature_col]
        self.p_global = y.mean()

        group = df.groupby(self.feature_col)[self.target_col].agg(['sum', 'count'])

        for category, row in group.iterrows():
            n = row['sum']
            N = row['count']

            N_prior = max(self.N_min - N, 0)
            alpha = n + self.p_global * N_prior
            beta  = (N - n) + (1 - self.p_global) * N_prior
            mean = alpha / (alpha + beta)

            self.cat_to_mean[category] = mean
            self.cat_to_count[category] = N

    def transform(self, df: pd.DataFrame) -> pd.Series:
        values = []
        for val in df[self.feature_col]:
            if val in self.cat_to_mean:
                values.append(self.cat_to_mean[val])
            else:
                values.append(self.p_global)
        return pd.Series(values, index=df.index)
        
    
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

        # Load OpenML Adult dataset 
        df = fetch_openml(data_id=1590, as_frame=True)['frame']
        df['label'] = (df['class'] == '>50K').astype(int)
        df = df.drop(columns=['class'])  # drop the original label column

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
            
        # Train / Test Split
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
        y_valid = valid_data['label'].values
        y_test  = test_data['label'].values
        
        cat_cardinalities = [train_data[col].nunique() for col in CAT_COLS]
        
        numerical_encoders = {}
        X_train_num_list, X_valid_num_list, X_test_num_list = [], [], []
        for col in NUM_COLS:
            encoder = NumericalBetaEncoder(feature_col=col, target_col='label', bin_size=0.001, N_min=len(train_data)*0.3)
            encoder.fit(train_data)
            numerical_encoders[col] = encoder
            
            X_train_num_list.append(encoder.transform(train_data).rename(col + '_enc'))
            X_valid_num_list.append(encoder.transform(valid_data).rename(col + '_enc'))
            X_test_num_list.append(encoder.transform(test_data).rename(col + '_enc'))

        X_train_num = pd.concat(X_train_num_list, axis=1)
        X_valid_num = pd.concat(X_valid_num_list, axis=1)
        X_test_num  = pd.concat(X_test_num_list, axis=1)

        categorical_encoders = {}
        X_train_cat_list, X_valid_cat_list, X_test_cat_list = [], [], []
        for col in CAT_COLS:
            encoder = CategoricalBetaEncoder(feature_col=col, target_col='label', N_min=len(train_data)*0.3)
            encoder.fit(train_data)
            categorical_encoders[col] = encoder
            
            X_train_cat_list.append(encoder.transform(train_data).rename(col + '_enc'))
            X_valid_cat_list.append(encoder.transform(valid_data).rename(col + '_enc'))
            X_test_cat_list.append(encoder.transform(test_data).rename(col + '_enc'))

        X_train_cat = pd.concat(X_train_cat_list, axis=1)
        X_valid_cat = pd.concat(X_valid_cat_list, axis=1)
        X_test_cat  = pd.concat(X_test_cat_list, axis=1)
        
        X_train_bte_scaled  = pd.concat([X_train_num, X_train_cat],axis=1) 
        X_valid_bte_scaled  = pd.concat([X_valid_num, X_valid_cat],axis=1) 
        X_test_bte_scaled   = pd.concat([X_test_num, X_test_cat],axis=1) 

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