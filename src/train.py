from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.features import TextFeaturizer
from src.lr_from_scratch import OneVsRestLogReg



def main():
    # -------------------------
    # Load + clean data
    # -------------------------
    df = pd.read_csv("data/raw/docs.csv")
    df = df.dropna(subset=["text", "label"]).copy()

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str)

    print("Label counts (raw):")
    print(df["label"].value_counts())

    # drop classes with too few samples
    min_count = 5
    vc = df["label"].value_counts()
    keep = vc[vc >= min_count].index
    df = df[df["label"].isin(keep)].copy()

    print("\nLabel counts after filtering:")
    print(df["label"].value_counts())

    # -------------------------
    # Build label maps
    # -------------------------
    labels = sorted(df["label"].unique().tolist())
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    id_to_label = {i: lab for lab, i in label_to_id.items()}

    # -------------------------
    # Build X / y
    # -------------------------
    texts = df["text"].tolist()
    y = df["label"].map(label_to_id).astype(int).values

    # -------------------------
    # Train / test split
    # -------------------------
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -------------------------
    # Featurization
    # -------------------------
    featurizer = TextFeaturizer.fit(
        X_train_texts,
        max_features=20000,
        ngram_range=(1, 2)
    )
    X_train = featurizer.transform(X_train_texts)
    X_test = featurizer.transform(X_test_texts)

    # -------------------------
    # Train model
    # -------------------------
    model = OneVsRestLogReg(lr=0.7, epochs=350, reg_lambda=1e-5)
    model.fit(X_train, y_train, n_classes=len(labels))

    # -------------------------
    # Evaluation
    # -------------------------
    preds = model.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=labels, digits=3))

    # -------------------------
    # Save artifacts
    # -------------------------
    featurizer.save("artifacts/tfidf.joblib")
    np.save("artifacts/lr_weights.npy", model.W)
    np.save("artifacts/lr_bias.npy", model.b)
    with open("artifacts/label_map.json", "w") as f:
        json.dump(id_to_label, f, indent=2)

    print("\nSaved artifacts to ./artifacts")


if __name__ == "__main__":
    main()
