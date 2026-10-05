import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["preprocess"]

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_raw = np.load(os.path.join(RAW_DIR, "train.npz"))
    test_raw = np.load(os.path.join(RAW_DIR, "test.npz"))

    X_train_full = train_raw["images"].astype("float32") / 255.0
    y_train_full = train_raw["labels"]
    X_test = test_raw["images"].astype("float32") / 255.0
    y_test = test_raw["labels"]

    X_train_full = train_raw["images"].astype("float32") / 127.5 - 1.0
    X_test = test_raw["images"].astype("float32") / 127.5 - 1.0

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=y_train_full,
    )

    np.savez(os.path.join(PROCESSED_DIR, "train.npz"), images=X_train, labels=y_train)
    np.savez(os.path.join(PROCESSED_DIR, "val.npz"), images=X_val, labels=y_val)
    np.savez(os.path.join(PROCESSED_DIR, "test.npz"), images=X_test, labels=y_test)

    print(f"train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

if __name__ == "__main__":
    main()