import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score

class ModelTrainer:
    def __init__(self, model, optimizer, scheduler, device, no_of_classes):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.no_of_classes = no_of_classes

    def train(self, train_loader, epoch):
        
        self.model.train()
        train_loss = 0.0
        ground_truths, preds_logits = [], []

        for data in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            self.optimizer.zero_grad()

            cat_data, num_data, bte_data, labels = self._prepare_data(data)
            logits = self.model(cat_data, num_data, bte_data)
            labels = labels.squeeze(dim=-1).long() 

            classification_loss = F.cross_entropy(logits, labels)
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
                labels = labels.squeeze(dim=-1).long() 

                classification_loss = F.cross_entropy(logits, labels)
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
                labels = labels.squeeze(dim=-1).long() 

                classification_loss = F.cross_entropy(logits, labels)
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


    def _evaluate(self, ground_truths, preds_logits, loader, loss):
        ground_truths = torch.cat(ground_truths)
        preds_logits = torch.cat(preds_logits)
        loss /= len(loader)

        # Extract positive class scores for compatibility with ground_truths
        preds_probs = torch.sigmoid(preds_logits)  # Use torch.sigmoid instead of F.sigmoid (deprecated)
        preds_labels = preds_probs.argmax(dim=1)

        # Compute accuracy
        correct_preds = (preds_labels == ground_truths).sum().item()
        total_preds = ground_truths.size(0)
        accuracy = correct_preds / total_preds
         
        return loss, accuracy


    # def imbalanced_evaluate(self, ground_truths, preds_logits, loader, loss):
    #     """
    #     Evaluates the model performance based on multiple metrics including loss, accuracy, AUROC, AUPRC, F1 Score, Recall, and Precision.
        
    #     Args:
    #         ground_truths (list): Ground truth labels for the batch.
    #         preds_logits (list): Model logits for the batch.
    #         loader (DataLoader): DataLoader for the evaluation set.
    #         loss (float): Initial loss value.

    #     Returns:
    #         tuple: Loss, accuracy, AUROC, AUPRC, F1 Score, Recall, Precision
    #     """
    #     # Concatenate all ground truths and predictions
    #     ground_truths = torch.cat(ground_truths)
    #     preds_logits = torch.cat(preds_logits)
    #     loss /= len(loader)

    #     # Get prediction probabilities
    #     preds_probs = torch.softmax(preds_logits, dim=1)  # Apply softmax for multi-class classification

    #     # For multi-class classification, use argmax to get predicted class
    #     preds_labels = preds_probs.argmax(dim=1)

    #     # Calculate Accuracy
    #     correct_preds = (preds_labels == ground_truths).sum().item()
    #     total_preds = ground_truths.size(0)
    #     accuracy = correct_preds / total_preds

    #     # Convert to numpy for sklearn metrics
    #     ground_truths_np = ground_truths.cpu().numpy()
    #     preds_probs_np = preds_probs.cpu().detach().numpy()

    #     # AUROC (Area Under ROC Curve) for multi-class
    #     auroc = roc_auc_score(ground_truths_np, preds_probs_np, multi_class='ovr', average='macro')

    #     # AUPRC (Area Under Precision-Recall Curve) for multi-class
    #     auprc = average_precision_score(ground_truths_np, preds_probs_np, average='macro')

    #     # F1 Score (Macro average for multi-class)
    #     f1 = f1_score(ground_truths_np, preds_labels.cpu().numpy(), average='macro')

    #     # Recall (Macro average for multi-class)
    #     recall = recall_score(ground_truths_np, preds_labels.cpu().numpy(), average='macro')

    #     # Precision (Macro average for multi-class)
    #     precision = precision_score(ground_truths_np, preds_labels.cpu().numpy(), average='macro')

    #     # Return all metrics
    #     return loss, accuracy, auroc, auprc, f1, recall, precision