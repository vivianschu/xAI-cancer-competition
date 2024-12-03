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
    X = pd.read_csv(train_df_path, index_col=0)
    y = pd.read_csv(train_targets_path)
    
    y = y.set_index('sample').loc[X.index]['AAC']  # align on index and select 'AAC' column

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values) 
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns) 

    # train-validation split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train.values, X_val.values, y_train, y_val

def create_datasets(X_train, X_val, y_train, y_val, device):
    train_dataset = RNASeqDataset(X_train, y_train, device)
    val_dataset = RNASeqDataset(X_val, y_val, device)
    return train_dataset, val_dataset

def load_test_data(test_df_path, scaler):
    # load and scale test data
    X_test = pd.read_csv(test_df_path, index_col=0)  # ensure sample names are the index
    X_test_scaled = scaler.transform(X_test)
    return torch.tensor(X_test_scaled, dtype=torch.float32)