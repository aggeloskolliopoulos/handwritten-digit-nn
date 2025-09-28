#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data_from_csv(x_csv, y_csv):
    X = pd.read_csv(x_csv).to_numpy(dtype=float)
    y = pd.read_csv(y_csv).to_numpy().squeeze()
    if y.ndim > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    return (X, y)

def load_mnist():
    from tensorflow.keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(len(X_train), -1).astype("float32") / 255.0
    X_test  = X_test.reshape(len(X_test), -1).astype("float32") / 255.0
    return (X_train, y_train), (X_test, y_test)

def build_model(input_dim, num_classes, hidden_units=[128, 64], dropout=0.0):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    model = Sequential()
    model.add(Dense(hidden_units[0], activation="relu", input_shape=(input_dim,)))
    if dropout > 0: model.add(Dropout(dropout))
    for hu in hidden_units[1:]:
        model.add(Dense(hu, activation="relu"))
        if dropout > 0: model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def plot_history(history):
    plt.figure()
    plt.plot(history.history["loss"], label="loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(history.history["accuracy"], label="accuracy")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x_csv", type=str)
    ap.add_argument("--y_csv", type=str)
    ap.add_argument("--num_classes", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    args = ap.parse_args()

    if args.x_csv and args.y_csv:
        X, y = load_data_from_csv(args.x_csv, args.y_csv)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        (X_train, y_train), (X_test, y_test) = load_mnist()

    num_classes = args.num_classes or int(np.max(y_train)) + 1
    model = build_model(X_train.shape[1], num_classes, dropout=args.dropout)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=args.epochs, batch_size=args.batch_size, verbose=2)
    plot_history(history)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    model.save("multiclass_nn_clean.keras")
    print("Saved model: multiclass_nn_clean.keras")

if __name__ == "__main__":
    main()
