"""
CEHSNF - Evaluation Metrics

Complexity-Aware Explainable Hybrid Symbolic-Neural Framework

Centralized regression evaluation utilities used across:

    - Neural models
    - Symbolic regression
    - Hybrid models
    - Physics-informed neural networks
    - Benchmark experiments
    - NASA C-MAPSS validation

Metrics:
    - MSE
    - RMSE
    - MAE
    - R²
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------

def _prepare_targets(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and normalize target and prediction arrays.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    ).reshape(-1)

    y_prediction = np.asarray(
        y_prediction,
        dtype=float,
    ).reshape(-1)

    if y_true.size == 0:
        raise ValueError(
            "y_true must contain at least one value."
        )

    if y_prediction.size == 0:
        raise ValueError(
            "y_prediction must contain at least one value."
        )

    if y_true.shape != y_prediction.shape:
        raise ValueError(
            "y_true and y_prediction must contain "
            "the same number of values. "
            f"Received {y_true.shape} and "
            f"{y_prediction.shape}."
        )

    if not np.all(np.isfinite(y_true)):
        raise ValueError(
            "y_true contains NaN or infinite values."
        )

    if not np.all(np.isfinite(y_prediction)):
        raise ValueError(
            "y_prediction contains NaN or infinite values."
        )

    return y_true, y_prediction


# ---------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------

def mse(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
) -> float:
    """
    Calculate Mean Squared Error.

        MSE = mean((y_true - y_prediction)^2)
    """

    y_true, y_prediction = _prepare_targets(
        y_true,
        y_prediction,
    )

    return float(
        mean_squared_error(
            y_true,
            y_prediction,
        )
    )


def rmse(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
) -> float:
    """
    Calculate Root Mean Squared Error.

        RMSE = sqrt(MSE)
    """

    return float(
        np.sqrt(
            mse(
                y_true,
                y_prediction,
            )
        )
    )


def mae(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
) -> float:
    """
    Calculate Mean Absolute Error.

        MAE = mean(|y_true - y_prediction|)
    """

    y_true, y_prediction = _prepare_targets(
        y_true,
        y_prediction,
    )

    return float(
        mean_absolute_error(
            y_true,
            y_prediction,
        )
    )


def r2(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
) -> float:
    """
    Calculate the coefficient of determination R².
    """

    y_true, y_prediction = _prepare_targets(
        y_true,
        y_prediction,
    )

    return float(
        r2_score(
            y_true,
            y_prediction,
        )
    )


# ---------------------------------------------------------------------
# Complete metric set
# ---------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate the complete CEHSNF regression metric set.

    Returns
    -------
    dict
        Dictionary containing:

            mse
            rmse
            mae
            r2
    """

    y_true, y_prediction = _prepare_targets(
        y_true,
        y_prediction,
    )

    mse_value = float(
        mean_squared_error(
            y_true,
            y_prediction,
        )
    )

    return {
        "mse": mse_value,
        "rmse": float(np.sqrt(mse_value)),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_prediction,
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                y_prediction,
            )
        ),
    }


# ---------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------

def compute_residuals(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
) -> np.ndarray:
    """
    Calculate prediction residuals.

        residual = y_true - y_prediction
    """

    y_true, y_prediction = _prepare_targets(
        y_true,
        y_prediction,
    )

    return y_true - y_prediction


# ---------------------------------------------------------------------
# Residual statistics
# ---------------------------------------------------------------------

def residual_statistics(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate descriptive statistics for model residuals.
    """

    residuals = compute_residuals(
        y_true,
        y_prediction,
    )

    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "min": float(np.min(residuals)),
        "max": float(np.max(residuals)),
        "median": float(np.median(residuals)),
        "absolute_mean": float(
            np.mean(np.abs(residuals))
        ),
    }


# ---------------------------------------------------------------------
# Prediction summary
# ---------------------------------------------------------------------

def prediction_summary(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
) -> Dict[str, object]:
    """
    Generate a complete prediction evaluation summary.

    Returns
    -------
    dict
        Contains metrics, residuals and residual statistics.
    """

    metrics = compute_metrics(
        y_true,
        y_prediction,
    )

    residuals = compute_residuals(
        y_true,
        y_prediction,
    )

    statistics = residual_statistics(
        y_true,
        y_prediction,
    )

    return {
        "metrics": metrics,
        "residuals": residuals,
        "residual_statistics": statistics,
    }


# ---------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------

def compare_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple models against the same target.

    Parameters
    ----------
    y_true:
        Ground-truth target.

    predictions:
        Dictionary:

            {
                "neural": neural_prediction,
                "symbolic": symbolic_prediction,
                "hybrid": hybrid_prediction,
                ...
            }

    Returns
    -------
    dict
        Metrics for every supplied model.
    """

    if not predictions:
        raise ValueError(
            "predictions must contain at least one model."
        )

    results: Dict[str, Dict[str, float]] = {}

    for model_name, prediction in predictions.items():

        results[str(model_name)] = compute_metrics(
            y_true,
            prediction,
        )

    return results


# ---------------------------------------------------------------------
# Metric table
# ---------------------------------------------------------------------

def metric_table(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> list[Dict[str, float | str]]:
    """
    Return model comparison results as a list of rows.

    This format is convenient for pandas/DataFrame conversion.
    """

    comparison = compare_models(
        y_true,
        predictions,
    )

    rows: list[Dict[str, float | str]] = []

    for model_name, metrics in comparison.items():

        rows.append(
            {
                "model": model_name,
                "mse": metrics["mse"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            }
        )

    return rows


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

__all__ = [
    "mse",
    "rmse",
    "mae",
    "r2",
    "compute_metrics",
    "compute_residuals",
    "residual_statistics",
    "prediction_summary",
    "compare_models",
    "metric_table",
]


# ---------------------------------------------------------------------
# Module test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("CEHSNF evaluation metrics test")
    print("-" * 50)

    # -------------------------------------------------------------
    # Example ground truth
    # -------------------------------------------------------------

    y_true = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ]
    )

    # -------------------------------------------------------------
    # Example predictions
    # -------------------------------------------------------------

    neural_prediction = np.array(
        [
            1.1,
            1.9,
            3.1,
            3.8,
            5.2,
        ]
    )

    symbolic_prediction = np.array(
        [
            0.9,
            2.1,
            2.9,
            4.1,
            4.9,
        ]
    )

    hybrid_prediction = np.array(
        [
            1.02,
            1.98,
            3.01,
            4.00,
            5.01,
        ]
    )

    # -------------------------------------------------------------
    # Individual metrics
    # -------------------------------------------------------------

    print(
        f"MSE          : "
        f"{mse(y_true, neural_prediction):.6f}"
    )

    print(
        f"RMSE         : "
        f"{rmse(y_true, neural_prediction):.6f}"
    )

    print(
        f"MAE          : "
        f"{mae(y_true, neural_prediction):.6f}"
    )

    print(
        f"R²           : "
        f"{r2(y_true, neural_prediction):.6f}"
    )

    # -------------------------------------------------------------
    # Complete metrics
    # -------------------------------------------------------------

    metrics = compute_metrics(
        y_true,
        neural_prediction,
    )

    print()
    print("Complete metrics")
    print("-" * 50)

    for name, value in metrics.items():

        print(
            f"{name.upper():8s} : {value:.6f}"
        )

    # -------------------------------------------------------------
    # Residuals
    # -------------------------------------------------------------

    residuals = compute_residuals(
        y_true,
        neural_prediction,
    )

    print()
    print("Residuals")
    print("-" * 50)

    print(residuals)

    # -------------------------------------------------------------
    # Residual statistics
    # -------------------------------------------------------------

    statistics = residual_statistics(
        y_true,
        neural_prediction,
    )

    print()
    print("Residual statistics")
    print("-" * 50)

    for name, value in statistics.items():

        print(
            f"{name:14s} : {value:.6f}"
        )

    # -------------------------------------------------------------
    # Model comparison
    # -------------------------------------------------------------

    predictions = {
        "neural": neural_prediction,
        "symbolic": symbolic_prediction,
        "hybrid": hybrid_prediction,
    }

    comparison = compare_models(
        y_true,
        predictions,
    )

    print()
    print("Model comparison")
    print("-" * 50)

    for model_name, model_metrics in comparison.items():

        print()
        print(model_name)

        for metric_name, value in model_metrics.items():

            print(
                f"  {metric_name.upper():6s} : "
                f"{value:.6f}"
            )

    # -------------------------------------------------------------
    # Metric table
    # -------------------------------------------------------------

    rows = metric_table(
        y_true,
        predictions,
    )

    print()
    print("Metric table")
    print("-" * 50)

    for row in rows:

        print(
            f"{row['model']:10s} | "
            f"MSE={row['mse']:.6f} | "
            f"RMSE={row['rmse']:.6f} | "
            f"MAE={row['mae']:.6f} | "
            f"R²={row['r2']:.6f}"
        )

    print()
    print(
        "Evaluation metrics test completed successfully."
    )