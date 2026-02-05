import torch
import os
import numpy as np
import random

class Config:
    DATA_ROOT = './dataset'
    FEATURE_DIR = './extract-frame'

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    SEQ_LEN = 32
    INPUT_DIM = 1029 
    HIDDEN_DIM = 256
    NUM_HEADS = 8
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    SEED = 42
    FOLDS = 5
    EPOCHS = 30
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    
    MODEL_TYPE = "LSTM" #change this to RNN, CNN, or LSTM

    FACE_DETECTION_CONFIDENCE = 0.5
    FACE_MESH_CONFIDENCE = 0.5
    FACE_MARGIN = 0.2

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False