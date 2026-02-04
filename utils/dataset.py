import torch
import numpy as np
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        features = np.load(path).astype(np.float32)

        if self.augment:
            noise = np.random.normal(0, 0.01, features.shape).astype(np.float32)
            features += noise

        return torch.FloatTensor(features), torch.tensor(self.labels[idx], dtype=torch.long)