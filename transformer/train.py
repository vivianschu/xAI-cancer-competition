import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

def train_model(model, train_dataset, val_dataset, epochs=10, batch_size=32, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation step
        model.eval()
        val_loss = 0.0
        preds = []
        true_vals = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                preds.extend(outputs.cpu().squeeze().numpy())
                true_vals.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        mse = mean_squared_error(true_vals, preds)
        r2 = r2_score(true_vals, preds)
        spearman_corr, _ = spearmanr(true_vals, preds)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Validation MSE: {mse:.4f}")
        print(f"  Validation R2: {r2:.4f}")
        print(f"  Validation Spearman's Rank Correlation: {spearman_corr:.4f}")

    return model

def predict_on_test(model, test_dataset, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            outputs = model(input_ids)
            predictions.extend(outputs.cpu().squeeze().numpy())

    return predictions