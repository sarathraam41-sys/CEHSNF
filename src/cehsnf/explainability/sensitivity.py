"""
CEHSNF - Sensitivity Analysis

Complexity-Aware Explainable Hybrid Symbolic-Neural Framework

This module provides model-agnostic local sensitivity analysis.

The analysis measures how much a model's prediction changes when
one input feature is perturbed by a small amount.

Sensitivity_i =
    mean(|f(X + delta_i) - f(X)|)

This module is intentionally independent of the PINN implementation
so that it can be reused with:

    - Neural networks
    - PINNs
    - Hybrid models
    - Other CEHSNF predictors
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import numpy as np


# ---------------------------------------------------------------------
# Default feature names
# ---------------------------------------------------------------------

DEFAULT_FEATURE_NAMES = [
    "theta1",
    "omega1",
    "theta2",
    "omega2",
]


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

def _prepare_inputs(
    X: np.ndarray,
) -> np.ndarray:
    """
    Convert input data into a valid 2D floating-point array.
    """

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            f"X must be a 2D array. Received shape: {X.shape}"
        )

    if X.shape[0] == 0:
        raise ValueError(
            "X must contain at least one sample."
        )

    if X.shape[1] == 0:
        raise ValueError(
            "X must contain at least one feature."
        )

    return X


# ---------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------

def _predict(
    model_or_predictor,
    X: np.ndarray,
) -> np.ndarray:
    """
    Generate predictions from either:

        1. An object with a .predict() method
        2. A callable predictor function

    This makes the sensitivity module model-agnostic.
    """

    if callable(model_or_predictor):
        prediction = model_or_predictor(X)

    elif hasattr(model_or_predictor, "predict"):
        prediction = model_or_predictor.predict(X)

    else:
        raise TypeError(
            "model_or_predictor must either be callable "
            "or provide a predict() method."
        )

    prediction = np.asarray(
        prediction,
        dtype=float,
    ).reshape(-1)

    if prediction.shape[0] != X.shape[0]:
        raise ValueError(
            "The predictor returned an unexpected number "
            "of predictions."
        )

    return prediction


# ---------------------------------------------------------------------
# Global local-sensitivity analysis
# ---------------------------------------------------------------------

def sensitivity_analysis(
    model_or_predictor,
    X: np.ndarray,
    feature_names: Optional[Iterable[str]] = None,
    perturbation: float = 1e-4,
) -> Dict[str, float]:
    """
    Perform local feature sensitivity analysis.

    Each feature is independently perturbed by a small amount.

    Sensitivity is calculated as:

        mean(
            abs(
                prediction_perturbed
                -
                prediction_original
            )
        )

    Parameters
    ----------
    model_or_predictor:
        A model with a .predict() method or a callable predictor.

    X:
        Input feature matrix.

    feature_names:
        Optional names for the features.

    perturbation:
        Amount added to one feature at a time.

    Returns
    -------
    dict
        Mapping of feature names to sensitivity values.
    """

    X = _prepare_inputs(X)

    if perturbation <= 0:
        raise ValueError(
            "perturbation must be greater than zero."
        )

    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [
            f"x{i}"
            for i in range(n_features)
        ]
    else:
        feature_names = list(feature_names)

        if len(feature_names) != n_features:
            raise ValueError(
                "feature_names must contain exactly "
                f"{n_features} names."
            )

    # -------------------------------------------------------------
    # Baseline prediction
    # -------------------------------------------------------------

    baseline_prediction = _predict(
        model_or_predictor,
        X,
    )

    sensitivities: Dict[str, float] = {}

    # -------------------------------------------------------------
    # Perturb each feature independently
    # -------------------------------------------------------------

    for feature_index, feature_name in enumerate(
        feature_names
    ):

        perturbed_X = X.copy()

        perturbed_X[:, feature_index] += perturbation

        perturbed_prediction = _predict(
            model_or_predictor,
            perturbed_X,
        )

        difference = np.abs(
            perturbed_prediction
            - baseline_prediction
        )

        sensitivities[str(feature_name)] = float(
            np.mean(difference)
        )

    return sensitivities


# ---------------------------------------------------------------------
# Relative sensitivity
# ---------------------------------------------------------------------

def relative_sensitivity_analysis(
    model_or_predictor,
    X: np.ndarray,
    feature_names: Optional[Iterable[str]] = None,
    perturbation: float = 1e-4,
    epsilon: float = 1e-12,
) -> Dict[str, float]:
    """
    Calculate sensitivity normalized by the magnitude of the
    baseline prediction.

    Relative sensitivity:

        mean(
            |f(x + delta) - f(x)|
            /
            (|f(x)| + epsilon)
        )

    This can be useful when comparing sensitivity across models
    with different prediction scales.
    """

    X = _prepare_inputs(X)

    if perturbation <= 0:
        raise ValueError(
            "perturbation must be greater than zero."
        )

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be greater than zero."
        )

    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [
            f"x{i}"
            for i in range(n_features)
        ]
    else:
        feature_names = list(feature_names)

        if len(feature_names) != n_features:
            raise ValueError(
                "feature_names must contain exactly "
                f"{n_features} names."
            )

    baseline_prediction = _predict(
        model_or_predictor,
        X,
    )

    denominator = (
        np.abs(baseline_prediction)
        + epsilon
    )

    sensitivities: Dict[str, float] = {}

    for feature_index, feature_name in enumerate(
        feature_names
    ):

        perturbed_X = X.copy()

        perturbed_X[:, feature_index] += perturbation

        perturbed_prediction = _predict(
            model_or_predictor,
            perturbed_X,
        )

        absolute_change = np.abs(
            perturbed_prediction
            - baseline_prediction
        )

        relative_change = (
            absolute_change
            / denominator
        )

        sensitivities[str(feature_name)] = float(
            np.mean(relative_change)
        )

    return sensitivities


# ---------------------------------------------------------------------
# Feature ranking
# ---------------------------------------------------------------------

def rank_sensitivities(
    sensitivities: Dict[str, float],
    descending: bool = True,
) -> list[tuple[str, float]]:
    """
    Sort sensitivity values.

    Parameters
    ----------
    sensitivities:
        Dictionary returned by sensitivity_analysis().

    descending:
        If True, most sensitive features appear first.
    """

    return sorted(
        sensitivities.items(),
        key=lambda item: item[1],
        reverse=descending,
    )


# ---------------------------------------------------------------------
# Sensitivity report
# ---------------------------------------------------------------------

def sensitivity_report(
    model_or_predictor,
    X: np.ndarray,
    feature_names: Optional[Iterable[str]] = None,
    perturbation: float = 1e-4,
) -> Dict[str, object]:
    """
    Generate a complete sensitivity report.

    Returns
    -------
    dict
        Contains:

            sensitivities
            ranking
            perturbation
    """

    sensitivities = sensitivity_analysis(
        model_or_predictor=model_or_predictor,
        X=X,
        feature_names=feature_names,
        perturbation=perturbation,
    )

    ranking = rank_sensitivities(
        sensitivities
    )

    return {
        "sensitivities": sensitivities,
        "ranking": ranking,
        "perturbation": perturbation,
    }


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

__all__ = [
    "DEFAULT_FEATURE_NAMES",
    "sensitivity_analysis",
    "relative_sensitivity_analysis",
    "rank_sensitivities",
    "sensitivity_report",
]


# ---------------------------------------------------------------------
# Module test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    from cehsnf.systems.double_pendulum import (
        make_dataset,
    )
    from cehsnf.models.pinn import (
        train_pinn,
    )

    print("CEHSNF sensitivity analysis test")
    print("-" * 50)

    # -------------------------------------------------------------
    # Generate canonical double-pendulum dataset
    # -------------------------------------------------------------

    X, y = make_dataset()

    print(
        f"Input shape    : {X.shape}"
    )

    print(
        f"Target shape   : {y.shape}"
    )

    # -------------------------------------------------------------
    # Train a small PINN for testing
    # -------------------------------------------------------------

    model, history = train_pinn(
        X,
        y,
        epochs=100,
        learning_rate=1e-3,
        physics_weight=0.1,
        gravity=9.81,
        hidden_dim=64,
        verbose=False,
    )

    # -------------------------------------------------------------
    # PINN predictor
    #
    # The PINN internally accepts the six-feature CEHSNF dataset
    # and uses the four physical variables.
    # -------------------------------------------------------------

    from cehsnf.models.pinn import predict_pinn

    predictor = lambda data: predict_pinn(
        model,
        data,
    )

    # -------------------------------------------------------------
    # Sensitivity analysis
    #
    # The canonical dataset has six features, therefore we name
    # all six explicitly.
    # -------------------------------------------------------------

    feature_names = [
        "theta1",
        "omega1",
        "theta2",
        "omega2",
        "sin_delta",
        "cos_delta",
    ]

    sensitivities = sensitivity_analysis(
        predictor,
        X,
        feature_names=feature_names,
        perturbation=1e-4,
    )

    # -------------------------------------------------------------
    # Ranking
    # -------------------------------------------------------------

    ranking = rank_sensitivities(
        sensitivities
    )

    # -------------------------------------------------------------
    # Results
    # -------------------------------------------------------------

    print()
    print("Sensitivity values")
    print("-" * 50)

    for feature, value in sensitivities.items():

        print(
            f"{feature:10s} : {value:.8f}"
        )

    print()
    print("Sensitivity ranking")
    print("-" * 50)

    for position, (feature, value) in enumerate(
        ranking,
        start=1,
    ):

        print(
            f"{position}. "
            f"{feature:10s} : "
            f"{value:.8f}"
        )

    print()
    print(
        "Sensitivity analysis test completed successfully."
    )