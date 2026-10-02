"""
CEHSNF Hybrid Model

Combines the neural model with symbolic regression on the
neural prediction residual.

Hybrid workflow:

    Neural prediction
          ↓
    Residual = y - neural_prediction
          ↓
    Symbolic regression
          ↓
    Symbolic residual correction
          ↓
    Hybrid prediction
"""

from __future__ import annotations

import numpy as np

from cehsnf.models.symbolic import (
    fit_symbolic,
    predict_symbolic,
    get_equation,
    get_complexity,
)


def compute_residual(
    y_true: np.ndarray,
    neural_prediction: np.ndarray,
) -> np.ndarray:
    """Compute the residual left by the neural model."""

    y_true = np.asarray(y_true, dtype=float).ravel()
    neural_prediction = np.asarray(
        neural_prediction,
        dtype=float,
    ).ravel()

    if len(y_true) != len(neural_prediction):
        raise ValueError(
            "y_true and neural_prediction must have "
            "the same number of samples."
        )

    return y_true - neural_prediction


def hybrid_train(
    neural_prediction: np.ndarray,
    y_true: np.ndarray,
    X: np.ndarray,
    niterations: int = 100,
    maxsize: int = 15,
    parsimony: float = 1e-3,
):
    """
    Train symbolic regression on the neural residual.

    Parameters
    ----------
    neural_prediction:
        Predictions produced by the neural model.

    y_true:
        Ground-truth target values.

    X:
        Original input features.

    Returns
    -------
    PySRRegressor
        Symbolic model representing the residual correction.
    """

    X = np.asarray(X, dtype=float)

    residual = compute_residual(
        y_true,
        neural_prediction,
    )

    model = fit_symbolic(
        X,
        residual,
        niterations=niterations,
        maxsize=maxsize,
        parsimony=parsimony,
    )

    return model


def hybrid_predict(
    neural_prediction: np.ndarray,
    hybrid_model,
    X: np.ndarray,
) -> np.ndarray:
    """
    Generate the final hybrid prediction.

    Hybrid prediction = neural prediction
                       + symbolic residual correction.
    """

    neural_prediction = np.asarray(
        neural_prediction,
        dtype=float,
    ).ravel()

    symbolic_correction = predict_symbolic(
        hybrid_model,
        X,
    )

    return neural_prediction + symbolic_correction


if __name__ == "__main__":
    from cehsnf.systems.harmonic import make_dataset
    from cehsnf.models.neural import (
        neural_pipeline,
    )

    print("=" * 60)
    print("CEHSNF hybrid model test")
    print("=" * 60)

    X, y = make_dataset()

    # Use a deterministic split for this module test.
    split = int(0.8 * len(X))

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    # Train neural model.
    neural = neural_pipeline(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=100,
    )

    # Train symbolic residual model.
    hybrid_model = hybrid_train(
        neural["train_prediction"],
        y_train,
        X_train,
        niterations=20,
        maxsize=10,
    )

    # Generate hybrid prediction.
    hybrid_prediction = hybrid_predict(
        neural["test_prediction"],
        hybrid_model,
        X_test,
    )

    mse = float(
        np.mean(
            (y_test - hybrid_prediction) ** 2
        )
    )

    print()
    print("Input shape :", X.shape)
    print("Train shape :", X_train.shape)
    print("Test shape  :", X_test.shape)
    print("Hybrid MSE  :", f"{mse:.6f}")
    print(
        "Equation    :",
        get_equation(hybrid_model),
    )
    print(
        "Complexity  :",
        get_complexity(hybrid_model),
    )