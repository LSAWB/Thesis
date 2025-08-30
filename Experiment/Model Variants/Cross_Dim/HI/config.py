import os
import torch

# Paths
# BASE_LOG_DIR = '/home/willie/LAB/2025/0613/Experiment_32/Project'
BASE_LOG_DIR = os.getcwd()

# Experiment Settings ( Seed )
EXPERIMENT_REPEAT = 15

# Training Parameters
EPOCHS          = 200
WARMUP_EPOCHS   = 5

BATCH_SIZE      = 512

LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-5

NUM_LABELS      = 2

# Early Stopping
EARLY_STOPPING_PATIENCE = 16
EARLY_STOPPING_DELTA = 0

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Preprocessing 
RARE_CATEGORIES_RATIO   = 0.005
ENCODING_BINSIZE        = 0.001
ENCODING_SMOOTHING      = 0.01

# Model Hyperparameters
DIM_MODEL = DIM_EMB = 192
DIM_FEEDFORWARD     = 256

NUM_HEADS           = 8

NUM_LAYERS_CROSS    = 3

RES_DROPOUT         = 0.0
FFN_DROPOUT         = 0.1
ATT_DROPOUT         = 0.2
