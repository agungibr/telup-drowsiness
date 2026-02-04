import os
import glob
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from utils.config import Config
from utils.dataset import FeatureDataset
from train import get_model_instance

def evaluate():    
    files = glob.glob(os.path.join(Config.FEATURE_DIR, "*.npy"))
    if not files: files = glob.glob(os.path.join(Config.FEATURE_DIR, "**/*.npy"), recursive=True)
    
    df = pd.DataFrame([{
        'path': f, 
        'label': 1 if 'microsleep' in f or 'yawning' in f else 0,
        'subject': os.path.basename(f).split('_')[0] + "_" + os.path.basename(f).split('_')[1]
    } for f in files])
    
    test_subjs = ['subject_16', 'subject_17', 'subject_18', 'subject_19', 'subject_20']
    test_df = df[df['subject'].isin(test_subjs)]
    
    test_ds = FeatureDataset(test_df['path'].values, test_df['label'].values, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    model = get_model_instance(Config.MODEL_TYPE)
    model_path = f"models/best_{Config.MODEL_TYPE}_fold1.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded {model_path}")
    else:
        print(f"Model not found")
        return

    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(Config.DEVICE), y.to(Config.DEVICE)
            pred = torch.argmax(model(X), dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    print(f"Test F1 Score: {f1_score(all_labels, all_preds, average='macro'):.4f}")

if __name__ == "__main__":
    evaluate()