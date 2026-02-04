import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

from utils.config import Config, seed_everything
from utils.dataset import FeatureDataset

from models.rnn import RNN
from models.lstm import LSTM
from models.cnn import CNN
from models.hybrid import TemporalReasoner

def get_model_instance(model_type):
    if model_type == "RNN": return RNN().to(Config.DEVICE)
    if model_type == "LSTM": return LSTM().to(Config.DEVICE)
    if model_type == "CNN": return CNN().to(Config.DEVICE)
    return TemporalReasoner().to(Config.DEVICE)

def train():
    seed_everything(Config.SEED)
    
    files = glob.glob(os.path.join(Config.FEATURE_DIR, "*.npy"))
    if not files: files = glob.glob(os.path.join(Config.FEATURE_DIR, "**/*.npy"), recursive=True)
    
    df = pd.DataFrame([{
        'path': f, 
        'label': 1 if 'microsleep' in f or 'yawning' in f else 0,
        'subject': os.path.basename(f).split('_')[0] + "_" + os.path.basename(f).split('_')[1]
    } for f in files])
    
    test_subjs = ['subject_16', 'subject_17', 'subject_18', 'subject_19', 'subject_20']
    train_df = df[~df['subject'].isin(test_subjs)].reset_index(drop=True)
    
    gkf = GroupKFold(n_splits=Config.FOLDS)
    folds = list(gkf.split(train_df, groups=train_df['subject']))

    all_fold_train_losses = [] 
    all_fold_val_f1 = []
    
    print(f"Training {Config.MODEL_TYPE} Model ({Config.FOLDS}-Fold CV)...")
        
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n--- Fold {fold+1}/{Config.FOLDS} ---")
        
        train_ds = FeatureDataset(train_df.iloc[train_idx]['path'].values, train_df.iloc[train_idx]['label'].values, augment=True)
        val_ds = FeatureDataset(train_df.iloc[val_idx]['path'].values, train_df.iloc[val_idx]['label'].values, augment=False)
        
        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        model = get_model_instance(Config.MODEL_TYPE)
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-2)
        criterion = torch.nn.CrossEntropyLoss()
        
        best_f1 = 0
        fold_train_losses = []
        fold_val_f1s = []
        
        for epoch in range(Config.EPOCHS):
            model.train()
            t_loss = 0
            for X, y in train_loader:
                X, y = X.to(Config.DEVICE), y.to(Config.DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
                t_loss += loss.item()
                
            model.eval()
            preds, true_labels = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(Config.DEVICE), y.to(Config.DEVICE)
                    preds.extend(torch.argmax(model(X), dim=1).cpu().numpy())
                    true_labels.extend(y.cpu().numpy())
            
            f1 = f1_score(true_labels, preds, average='macro')
            
            fold_train_losses.append(t_loss/len(train_loader))
            fold_val_f1s.append(f1)
            
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), f"models/best_{Config.MODEL_TYPE}_fold{fold+1}.pth")
        
        print(f"   Best F1: {best_f1:.4f}")
        
        all_fold_train_losses.append(fold_train_losses)
        all_fold_val_f1.append(fold_val_f1s)
    
    avg_loss = np.mean(all_fold_train_losses, axis=0)
    std_loss = np.std(all_fold_train_losses, axis=0)
    
    avg_f1 = np.mean(all_fold_val_f1, axis=0)
    std_f1 = np.std(all_fold_val_f1, axis=0)
    
    epochs = range(1, Config.EPOCHS + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, avg_loss, color='red', label='Mean Train Loss')
    plt.fill_between(epochs, avg_loss - std_loss, avg_loss + std_loss, color='red', alpha=0.1)
    plt.title(f'Average Training Loss ({Config.MODEL_TYPE})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, avg_f1, color='blue', label='Mean Val F1')
    plt.fill_between(epochs, avg_f1 - std_f1, avg_f1 + std_f1, color='blue', alpha=0.1)
    plt.title(f'Average Validation F1 ({Config.MODEL_TYPE})')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = f"result/{Config.MODEL_TYPE}_average_performance.png"
    plt.savefig(save_path)
    
    print(f"\nOverall Mean F1 Score: {np.mean([max(f) for f in all_fold_val_f1]):.4f}")

if __name__ == "__main__":
    os.makedirs("result", exist_ok=True)
    train()