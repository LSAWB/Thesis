# Standard
import os
import math
import json
import random
from datetime import datetime
from typing import Dict, Literal

# General
import numpy as np
import pandas as pd
from tqdm import tqdm

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, PowerTransformer
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml, fetch_california_housing

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
import node


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
# class CustomDataset(Dataset):
#     def __init__(self, x, y):
#         if isinstance(x, np.ndarray):
#             self.x = torch.tensor(x, dtype=torch.float32)
#         else:
#             self.x = torch.tensor(x.values, dtype=torch.float32)

#         if isinstance(y, np.ndarray):
#             self.labels = torch.tensor(y, dtype=torch.float32)
#         else:
#             self.labels = torch.tensor(y.values, dtype=torch.float32)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return self.x[idx], self.labels[idx]

class CustomDataset(Dataset):
    def __init__(self, x_num=None, x_cat=None, y=None):
        if x_num is not None:
            self.x_num = torch.tensor(x_num, dtype=torch.float32)
        else:
            self.x_num = None

        if x_cat is not None:
            self.x_cat = torch.tensor(x_cat, dtype=torch.long)
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
# class EarlyStopping:
#     def __init__(self, patience=5, delta=0):
#         self.patience = patience
#         self.delta = delta
#         self.best_score = None
#         self.early_stop = False
#         self.counter = 0
#         self.best_model_state = None

#     def __call__(self, val_score, model):
#         if self.best_score is None:
#             self.best_score = val_score
#             self.best_model_state = model.state_dict()
#         elif val_score < self.best_score + self.delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = val_score
#             self.best_model_state = model.state_dict()
#             self.counter = 0


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_rmse, model):
        if self.best_score is None:
            self.best_score = val_rmse
            self.best_model_state = model.state_dict()
        elif val_rmse > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_rmse
            self.best_model_state = model.state_dict()
            self.counter = 0
            
# Model 
class NODE(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        num_layers: int,
        layer_dim: int,
        depth: int,
        tree_dim: int,
        choice_function: str,
        bin_function: str,
        d_out: int,
        categories: ty.Optional[ty.List[int]],
        d_embedding: int,
    ) -> None:
        super().__init__()

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        self.d_out = d_out
        self.block = node.DenseBlock(
            input_dim=d_in,
            num_layers=num_layers,
            layer_dim=layer_dim,
            depth=depth,
            tree_dim=tree_dim,
            bin_function=getattr(node, bin_function),
            choice_function=getattr(node, choice_function),
            flatten_output=False,
        )

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        if x_cat is not None:
            x_cat = self.category_embeddings(x_cat + self.category_offsets[None])
            x = torch.cat([x_num, x_cat.view(x_cat.size(0), -1)], dim=-1)
        else:
            x = x_num

        x = self.block(x)
        x = x[..., : self.d_out].mean(dim=-2)
        x = x.squeeze(-1)
        return x


# Model Trainer
class ModelTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, y_std):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.std    = y_std

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

            labels = labels.squeeze(dim=-1).float().to(self.device)

            # Forward pass for DCNv2: (x_num, x_cat)
            logits = self.model(x_num, x_cat).squeeze()
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

                labels = labels.squeeze(dim=-1).float().to(self.device)

                # Forward pass for DCNv2: (x_num, x_cat)
                logits = self.model(x_num, x_cat).squeeze()
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

                labels = labels.squeeze(dim=-1).float().to(self.device)

                # Forward pass for DCNv2: (x_num, x_cat)
                logits = self.model(x_num, x_cat).squeeze()
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


    def _evaluate(self, ground_truths, preds, loader, loss):
        ground_truths = torch.cat(ground_truths)
        preds = torch.cat(preds)

        loss /= len(loader)
        rmse = torch.sqrt(F.mse_loss(preds, ground_truths)) * self.std.item()

        return loss, rmse.cpu()


def main():
    
    # Define Hyperparameter
    experiment_repeat = 15
    n_classes = 1
    batch_size = 128

    warmup_epochs = 5
    epochs = 200
    
    test_rmses = []
    
    # Get current time to name the log directory
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_log_dir = os.path.join(script_dir, current_time)
    os.makedirs(base_log_dir, exist_ok=True)
    
    for experiment_number in range(experiment_repeat):
        
        # Set random seed
        set_random_seed(experiment_number)

        # 取得資料
        df = fetch_california_housing(as_frame=True).frame
        X = df.drop(columns=["MedHouseVal"])
        y_orig = df["MedHouseVal"].values.astype(np.float32).reshape(-1, 1)

        # Train / Val / Test split
        X_temp, X_test, y_temp, y_test = train_test_split(X, y_orig, test_size=0.2, random_state=experiment_number)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=experiment_number)

        # 對 y 做標準化 (僅用 train 的 mean 和 std)
        y_mean = y_train.mean()
        y_std  = y_train.std()
        y_train = (y_train - y_mean) / y_std
        y_val   = (y_val - y_mean) / y_std
        y_test  = (y_test  - y_mean) / y_std

        # 選定要標準化的欄位
        NUM_COLS = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

        # 注意：直接用 DataFrame 進行轉換
        scaler_power = PowerTransformer(method='yeo-johnson', standardize=True)
        train_data_encoded = scaler_power.fit_transform(X_train[NUM_COLS])
        valid_data_encoded = scaler_power.transform(X_val[NUM_COLS])
        test_data_encoded  = scaler_power.transform(X_test[NUM_COLS])
        
        # Dataset and Dataloader
        train_dataset = CustomDataset(train_data_encoded, None, y_train)
        val_dataset   = CustomDataset(valid_data_encoded, None, y_val)
        test_dataset  = CustomDataset(test_data_encoded, None,  y_test)
        train_loader    = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader      = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader     = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NODE(
            d_in=len(NUM_COLS),
            num_layers=1,
            layer_dim=2048,
            depth=6,
            tree_dim=1,
            choice_function="entmax15",
            bin_function="sparsemax",
            d_out=n_classes,
            categories=[],
            d_embedding=192,
        ).to(device)
        criterion = torch.nn.MSELoss()

        # Define total steps and warmup steps
        # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = get_scheduler(optimizer, epochs, warmup_epochs)
        early_stopping = EarlyStopping(patience=16, delta=0)

        # For output path
        # Create a new directory for this experiment
        experiment_log_dir = os.path.join(base_log_dir, str(experiment_number+1))
        os.makedirs(experiment_log_dir, exist_ok=True)

        # For output path
        best_model_path = os.path.join(base_log_dir, f"best_model_seed_{experiment_number}.pth")
            
        best_valid_rmse = float("inf")
        all_epoch_metrics = []

        # Initialize trainer
        trainer = ModelTrainer(model, criterion, optimizer, scheduler, device, y_std)

        for epoch in range(epochs):
            
            train_loss, train_rmse = trainer.train(train_loader, epoch)
            valid_loss, valid_rmse = trainer.evaluate(val_loader, epoch)

            # Info
            print(f"Epoch: {epoch + 1}/{epochs}")
            print(f"Training RMSE: {train_rmse:.4f} | Validation RMSE: {valid_rmse:.4f}")

            # Save model if improved
            if valid_rmse < best_valid_rmse:
                best_valid_rmse = valid_rmse
                save_file = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                }
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

        print(f"Test RMSE: {test_rmse:.4f}")

        # Store the best AUROC for this experiment
        test_rmses.append(test_rmse)

        # Delete the best model file after loading
        if os.path.exists(best_model_path):  
            os.remove(best_model_path)  # Deletes the file
            print(f"Deleted best model checkpoint: {best_model_path}")
        else:
            print(f"Checkpoint file not found: {best_model_path}")

    # After all experiments, calculate the mean and std of AUROCs
    mean_rmse = np.mean(test_rmses)
    std_rmse = np.std(test_rmses)
    print(f"Mean ACC: {mean_rmse:.4f}, Std Dev ACC: {std_rmse:.4f}")


if __name__ == '__main__':
    main()