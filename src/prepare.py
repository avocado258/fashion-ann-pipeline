import os
import numpy as np
import tensorflow as tf

RAW_DIR = "data/raw"

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    np.savez(
        os.path.join(RAW_DIR, "train.npz"),
        images=train_images,
        labels=train_labels,
    )
    np.savez(
        os.path.join(RAW_DIR, "test.npz"),
        images=test_images,
        labels=test_labels,
    )

    print(f"Saved train: {train_images.shape}, test: {test_images.shape} to {RAW_DIR}/")

if __name__ == "__main__":
    main()