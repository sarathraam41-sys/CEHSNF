"""
CEHSNF Symbolic Regression Module

Symbolic regression discovers compact mathematical expressions
that approximate the underlying physical relationship.

Based on the original CEHSNF methodology using PySR.
"""

from __future__ import annotations

import numpy as np
from pysr import PySRRegressor


DEFAULT_BINARY_OPERATORS = ["+", "-", "*", "/"]
DEFAULT_UNARY_OPERATORS = ["sin", "cos"]


def create_symbolic_model(
    niterations: int = 100,
    maxsize: int = 15,
    parsimony: float = 1e-3,
) -> PySRRegressor:
    """
    Create a PySR symbolic regression model.
    """

    return PySRRegressor(
        niterations=niterations,
        binary_operators=DEFAULT_BINARY_OPERATORS,
        unary_operators=DEFAULT_UNARY_OPERATORS,
        maxsize=maxsize,
        parsimony=parsimony,
        model_selection="best",
        elementwise_loss=(
            "loss(prediction, target) = "
            "(prediction - target)^2"
        ),
        verbosity=1,
    )


def fit_symbolic(
    X: np.ndarray,
    y: np.ndarray,
    niterations: int = 100,
    maxsize: int = 15,
    parsimony: float = 1e-3,
) -> PySRRegressor:
    """
    Fit symbolic regression to the supplied dataset.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")

    model = create_symbolic_model(
        niterations=niterations,
        maxsize=maxsize,
        parsimony=parsimony,
    )

    model.fit(X, y)

    return model


def predict_symbolic(
    model: PySRRegressor,
    X: np.ndarray,
) -> np.ndarray:
    """
    Generate predictions from a fitted symbolic model.
    """

    X = np.asarray(X, dtype=float)

    return np.asarray(model.predict(X), dtype=float)


def get_equation(model: PySRRegressor) -> str:
    return str(model.get_best()["equation"])


def get_complexity(model: PySRRegressor) -> int:
    """
    Return the complexity of the selected symbolic equation.
    """

    best = model.get_best()

    return int(best["complexity"])


def symbolic_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    niterations: int = 100,
    maxsize: int = 15,
    parsimony: float = 1e-3,
) -> dict:
    """
    Complete symbolic regression pipeline.

    Returns:
        model
        prediction
        equation
        complexity
    """

    model = fit_symbolic(
        X_train,
        y_train,
        niterations=niterations,
        maxsize=maxsize,
        parsimony=parsimony,
    )

    prediction = predict_symbolic(model, X_test)

    return {
        "model": model,
        "prediction": prediction,
        "equation": get_equation(model),
        "complexity": get_complexity(model),
    }


if __name__ == "__main__":
    from cehsnf.systems.harmonic import make_dataset

    print("=" * 60)
    print("CEHSNF symbolic regression test")
    print("=" * 60)

    X, y = make_dataset()

    # Small test configuration.
    # Increase niterations for the actual experiments.
    model = fit_symbolic(
        X,
        y,
        niterations=20,
        maxsize=10,
    )

    prediction = predict_symbolic(model, X)

    mse = float(np.mean((y - prediction) ** 2))

    print()
    print("Input shape :", X.shape)
    print("Target shape:", y.shape)
    print("MSE         :", f"{mse:.6f}")
    print("Equation    :", get_equation(model))
    print("Complexity  :", get_complexity(model))