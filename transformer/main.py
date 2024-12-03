from process import load_data, create_datasets
from model import TransformerForRegression
from train import train_model

base = "/home/vivian.chu/vivian-sandbox/other/xAI-cancer-competition/.data"

def main():
    # File paths
    train_df_path = f"{base}/train_subset.csv"
    train_targets_path = f"{base}/train_targets.csv"

    # Load and preprocess data
    X_train, X_val, y_train, y_val = load_data(train_df_path, train_targets_path)
    train_dataset, val_dataset = create_datasets(X_train, X_val, y_train, y_val)

    # Initialize model
    input_dim = X_train.shape[1]
    model = TransformerForRegression(input_dim)

    # Train the model
    model = train_model(model, train_dataset, val_dataset, epochs=10)

if __name__ == "__main__":
    main()