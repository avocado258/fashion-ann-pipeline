import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    os.makedirs(MODELS_DIR, exist_ok=True)

    train = np.load(os.path.join(PROCESSED_DIR, "train.npz"))
    val = np.load(os.path.join(PROCESSED_DIR, "val.npz"))

    X_train, y_train = train["images"], train["labels"]
    X_val, y_val = val["images"], val["labels"]

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(params["dense_units"], activation="relu"),
        tf.keras.layers.Dropout(params["dropout_rate"]),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
    )

    model.save(os.path.join(MODELS_DIR, "model.h5"))
    pd.DataFrame(history.history).to_csv(os.path.join(MODELS_DIR, "history.csv"), index=False)

    print(f"Saved model to {MODELS_DIR}/model.h5 and history to {MODELS_DIR}/history.csv")

if __name__ == "__main__":
    main()