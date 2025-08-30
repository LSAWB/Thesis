import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


class ModelTrainer:
    def __init__(self, model, optimizer, scheduler, device, no_of_classes, y_std):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.no_of_classes = no_of_classes
        self.std = y_std

    def train(self, train_loader, epoch):
        
        self.model.train()
        train_loss = 0.0
        ground_truths, preds_logits = [], []

        for data in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            self.optimizer.zero_grad()

            cat_data, num_data, bte_data, labels = self._prepare_data(data)
            logits = self.model(cat_data, num_data, bte_data)
            labels = labels.squeeze().float()

            criterion = torch.nn.MSELoss()
            classification_loss = criterion(logits.float(), labels)
            total_loss = classification_loss

            total_loss.backward()
            train_loss += total_loss.item()

            self.optimizer.step()
            self.scheduler.step()

            preds_logits.append(logits.detach())
            ground_truths.append(labels.detach())

        return self._evaluate(ground_truths, preds_logits, train_loader, train_loss)


    def evaluate(self, val_loader, epoch):
        
        self.model.eval()
        valid_loss = 0.0
        ground_truths, preds_logits = [], []

        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Evaluating Epoch {epoch+1}", leave=False):

                cat_data, num_data, bte_data, labels = self._prepare_data(data)
                logits = self.model(cat_data, num_data, bte_data)
                labels = labels.squeeze().float()

                criterion = torch.nn.MSELoss()
                classification_loss = criterion(logits.float(), labels)
                total_loss = classification_loss

                valid_loss += total_loss.item()

                preds_logits.append(logits.detach())
                ground_truths.append(labels.detach())

        return self._evaluate(ground_truths, preds_logits, val_loader, valid_loss)  

    
    def test(self, test_loader, epoch):
        self.model.eval()
        test_loss = 0.0
        ground_truths, preds_logits = [], []

        with torch.no_grad():
            for data in tqdm(test_loader, desc=f"Testing Epoch {epoch+1}", leave=False):

                cat_data, num_data, bte_data, labels = self._prepare_data(data)
                logits = self.model(cat_data, num_data, bte_data)
                labels = labels.squeeze().float()

                criterion = torch.nn.MSELoss()
                classification_loss = criterion(logits.float(), labels)
                total_loss = classification_loss

                test_loss += total_loss.item()

                preds_logits.append(logits.detach())
                ground_truths.append(labels.detach())

        return self._evaluate(ground_truths, preds_logits, test_loader, test_loss)    


    def _prepare_data(self, data):
        cat_data, num_data, bte_data, labels = data
        return (
            cat_data.to(self.device, dtype=torch.long),
            num_data.to(self.device, dtype=torch.float32),
            bte_data.to(self.device, dtype=torch.float32),
            labels.to(self.device, dtype=torch.float32)
        )


    # def _evaluate(self, ground_truths, preds_logits, loader, loss):
    #     ground_truths = torch.cat(ground_truths)
    #     preds_logits = torch.cat(preds_logits)
    #     loss /= len(loader)

    #     # Extract positive class scores for compatibility with ground_truths
    #     preds_probs = torch.sigmoid(preds_logits)  # Use torch.sigmoid instead of F.sigmoid (deprecated)
    #     preds_labels = preds_probs.argmax(dim=1)

    #     # Compute accuracy
    #     correct_preds = (preds_labels == ground_truths).sum().item()
    #     total_preds = ground_truths.size(0)
    #     accuracy = correct_preds / total_preds
         
    #     return loss, accuracy

    def _evaluate(self, ground_truths, preds, loader, loss):
        ground_truths = torch.cat(ground_truths)
        preds = torch.cat(preds)

        loss /= len(loader)
        rmse = torch.sqrt(F.mse_loss(preds, ground_truths)) * self.std.item()

        return loss, rmse