import torch
import torch.nn.functional as F
import numpy as np
import config

# class CBLoss:
#     def __init__(self):
#         self.num_classes = config.NUM_LABELS
#         self.num_tokens = config.NUM_TOKENS
#         self.beta = config.BETA
#         self.gamma = config.GAMMA
#         self.classification_loss_type = config.C_LOSS_TYPE
#         self.reconstruction_loss_type = config.R_LOSS_TYPE

#     def compute_cb_weights(self, samples_per_cls):
#         effective_num = 1.0 - torch.pow(self.beta, samples_per_cls)
#         effective_num[effective_num == 0] = 1e-8  # Prevent division by zero

#         weights = (1.0 - self.beta) / effective_num
#         weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8) * self.num_classes  # Normalize
#         return torch.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0).float()

#     def compute_classification_loss(self, labels, logits, samples_per_cls):
#         weights = self.compute_cb_weights(torch.tensor(samples_per_cls, dtype=torch.float32)).to(logits.device)
#         labels_one_hot = F.one_hot(labels.long(), num_classes=self.num_classes).float().to(logits.device)

#         weights = weights.unsqueeze(0).repeat(labels.shape[0], 1) * labels_one_hot
#         weights = weights.sum(dim=1, keepdim=True).repeat(1, self.num_classes)

#         if self.classification_loss_type == "focal":
#             return self.focal_loss(labels_one_hot, logits, weights)
#         elif self.classification_loss_type == "sigmoid":
#             return F.binary_cross_entropy_with_logits(logits, labels_one_hot, weight=weights).mean()
#         elif self.classification_loss_type == "softmax":
#             return F.binary_cross_entropy(logits.softmax(dim=1), labels_one_hot, weight=weights).mean()

#     def compute_reconstruction_loss(self, labels, logits, samples_per_cls):
#         weights = self.compute_cb_weights(samples_per_cls).to(logits.device)
        
#         labels_one_hot = F.one_hot(labels.long(), num_classes=self.num_tokens+2).float().to(logits.device)

#         weights = weights.unsqueeze(0).expand(labels.shape[0], labels.shape[1], -1)

#         weights = (weights * labels_one_hot).sum(dim=-1, keepdim=True).expand(-1, -1, self.num_tokens+2)

#         if self.reconstruction_loss_type == "focal":
#             return self.focal_loss(labels_one_hot, logits, weights)
#         elif self.reconstruction_loss_type == "sigmoid":
#             return F.binary_cross_entropy_with_logits(logits, labels_one_hot, weight=weights).mean()
#         elif self.reconstruction_loss_type == "softmax":
#             return F.binary_cross_entropy(logits.softmax(dim=-1), labels_one_hot, weight=weights).mean()


#     def focal_loss(self, labels_one_hot, logits, weights):
#         """
#         Compute Focal Loss.
#         """
#         bce_loss = F.binary_cross_entropy_with_logits(logits, labels_one_hot, reduction='none')
#         pt = torch.exp(-bce_loss)  # Compute pt = exp(-BCE)
#         focal_loss = ((1 - pt) ** self.gamma * bce_loss * weights).mean()
#         return focal_loss


