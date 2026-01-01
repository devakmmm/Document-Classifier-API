from __future__ import annotations
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    # stable sigmoid
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_log_loss(y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-12) -> float:
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return float(-np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))


class OneVsRestLogReg:
    """
    Multi-class logistic regression via One-vs-Rest.
    Trained with batch gradient descent.
    """

    def __init__(self, lr: float = 0.5, epochs: int = 400, reg_lambda: float = 0.0):
        self.lr = lr
        self.epochs = epochs
        self.reg_lambda = reg_lambda
        self.W: np.ndarray | None = None  # shape: (n_classes, n_features)
        self.b: np.ndarray | None = None  # shape: (n_classes,)

    def fit(self, X, y: np.ndarray, n_classes: int) -> None:
        """
        X: scipy sparse matrix or ndarray shape (n_samples, n_features)
        y: int labels shape (n_samples,)
        """
        n_samples = y.shape[0]
        n_features = X.shape[1]

        self.W = np.zeros((n_classes, n_features), dtype=np.float64)
        self.b = np.zeros((n_classes,), dtype=np.float64)

        # train each class vs rest
        for k in range(n_classes):
            yk = (y == k).astype(np.float64)  # binary targets

            wk = np.zeros((n_features,), dtype=np.float64)
            bk = 0.0

            for _ in range(self.epochs):
                # scores -> probs
                z = X @ wk + bk
                p = sigmoid(np.asarray(z).reshape(-1))

                # gradients
                # dL/dw = (1/n) X^T (p - y) + lambda * w
                diff = (p - yk)
                grad_w = (X.T @ diff) / n_samples
                grad_w = np.asarray(grad_w).reshape(-1) + self.reg_lambda * wk
                grad_b = float(np.mean(diff))

                wk -= self.lr * grad_w
                bk -= self.lr * grad_b

            self.W[k, :] = wk
            self.b[k] = bk

    def predict_proba(self, X) -> np.ndarray:
        """
        Returns class probabilities shape (n_samples, n_classes)
        OvR probabilities are normalized to sum to 1.
        """
        assert self.W is not None and self.b is not None
        scores = X @ self.W.T + self.b  # (n_samples, n_classes)
        probs = sigmoid(np.asarray(scores))
        # normalize to a distribution
        denom = probs.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        return probs / denom

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
