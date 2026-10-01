import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
METRICS_PATH = "metrics.json"
CM_PATH = "confusion_matrix.png"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

def main():
    model = tf.keras.models.load_model(f"{MODELS_DIR}/model.h5")

    test = np.load(f"{PROCESSED_DIR}/test.npz")
    X_test, y_test = test["images"], test["labels"]

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    plt.savefig(CM_PATH)
    plt.close(fig)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"test_loss={test_loss:.4f}, test_accuracy={test_accuracy:.4f}")
    print(f"Saved metrics to {METRICS_PATH} and confusion matrix to {CM_PATH}")

if __name__ == "__main__":
    main()