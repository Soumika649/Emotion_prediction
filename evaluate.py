"""
Generates a proper evaluation report for the trained BiGRU model:
- per-class precision / recall / F1 (classification_report)
- a confusion matrix plot

This goes beyond the single "accuracy" number in the notebook and is
what most interviewers actually want to see for an imbalanced
multi-class problem.

Usage:
    python evaluate.py
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "Artifacts/BiGRU_Modle.keras"
TOKENIZER_PATH = "Artifacts/tokenizer.pkl"
MAX_SEQUENCE_LENGTH = 50
OUTPUT_DIR = "Artifacts"


def main():
    print("Loading test split of dair-ai/emotion...")
    dataset = load_dataset("dair-ai/emotion")
    test_text = dataset["test"]["text"]
    test_label = dataset["test"]["label"]
    label_names = dataset["test"].features["label"].names

    print("Loading model and tokenizer...")
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    print("Running inference on test set...")
    sequences = tokenizer.texts_to_sequences(test_text)
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    predictions = np.argmax(model.predict(padded), axis=1)

    report = classification_report(test_label, predictions, target_names=label_names, digits=3)
    print("\n" + report)

    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved classification report to {report_path}")

    cm = confusion_matrix(test_label, predictions)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — BiGRU")
    plt.tight_layout()

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()
