import wandb
import torch
import pandas as pd
from model import TransformerForRegression
from sklearn.preprocessing import StandardScaler
from process import load_data, create_datasets, load_test_data
from train import train_model, predict_on_test

base = "/home/vivian.chu/vivian-sandbox/other/xAI-cancer-competition/.data"

def main():
    # File paths
    train_df_path = f"{base}/train_subset.csv"
    train_targets_path = f"{base}/train_targets.csv"
    test_df_path = f"{base}/test.csv"

    # Determine the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize W&B
    wandb.init(project="rna-seq-regression", name="transformer-regression", config={
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "model": "TransformerForRegression"
    })
    config = wandb.config

    # Load and preprocess training data
    X_train, X_val, y_train, y_val = load_data(train_df_path, train_targets_path)
    train_dataset, val_dataset = create_datasets(X_train, X_val, y_train, y_val, device)

    # Load and preprocess test data
    scaler = StandardScaler()
    scaler.fit(X_train)  # Use the same scaler as training
    X_test = load_test_data(test_df_path, scaler).to(device)

    # Initialize model
    input_dim = X_train.shape[1]
    model = TransformerForRegression(input_dim)

    # Train the model
    model = train_model(model, train_dataset, val_dataset, device, epochs=config.epochs)

    # Predict on the test set
    test_predictions = predict_on_test(model, X_test, device)

    # Log predictions to W&B
    wandb.log({"test_predictions": test_predictions})

    # Save predictions
    pd.DataFrame(test_predictions, columns=["Predicted_AAC"]).to_csv("test_predictions.csv", index=False)
    print("Predictions saved to 'test_predictions.csv'")

    # Finish W&B run
    wandb.finish()

if __name__ == "__main__":
    main()
