import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

class RNASeqDataset(Dataset):
    def __init__(self, X, y, device):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y.values, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {'input_ids': self.X[idx], 'labels': self.y[idx]}

def create_datasets(X_train, X_val, y_train, y_val, device):
    train_dataset = RNASeqDataset(X_train, y_train, device)
    val_dataset = RNASeqDataset(X_val, y_val, device)
    return train_dataset, val_dataset

def load_data(train_df_path, train_targets_path):
    X = pd.read_csv(train_df_path)
    y = pd.read_csv(train_targets_path)['AAC']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def load_test_data(test_df_path, scaler):
    X_test = pd.read_csv(test_df_path)
    X_test_scaled = scaler.transform(X_test)  # Use the same scaler from training
    return torch.tensor(X_test_scaled, dtype=torch.float32)