"""
CEHSNF - Physics-Informed Neural Network (PINN)

Complexity-Aware Explainable Hybrid Symbolic-Neural Framework

This module implements the physics-informed neural component of CEHSNF.

Pipeline:
    Physical System
        ↓
    Dataset
        ↓
    Physics-Informed Neural Network
        ↓
    Physics + Data Loss
        ↓
    Prediction
        ↓
    Explainability / Hybrid Pipeline

The canonical double-pendulum dataset contains six features:

    [theta1, omega1, theta2, omega2, sin(delta), cos(delta)]

The PINN uses only the four physical state variables:

    [theta1, omega1, theta2, omega2]

The additional engineered features remain available for the
symbolic and hybrid components of CEHSNF.
"""

from __future__ import annotations

import copy
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# PINN Network
# ---------------------------------------------------------------------

class CEHSNFPINN(nn.Module):
    """
    Physics-Informed Neural Network used by CEHSNF.

    Parameters
    ----------
    input_dim:
        Number of physical input variables used by the network.

    hidden_dim:
        Number of neurons in each hidden layer.

    The default architecture is:

        Input
          ↓
        Linear
          ↓
        Tanh
          ↓
        Linear
          ↓
        Tanh
          ↓
        Linear
          ↓
        Tanh
          ↓
        Output
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the PINN."""

        return self.network(x)


# ---------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------

def _prepare_inputs(
    X: np.ndarray,
) -> np.ndarray:
    """
    Prepare CEHSNF input features for the PINN.

    The canonical double-pendulum dataset contains:

        [theta1, omega1, theta2, omega2, sin(delta), cos(delta)]

    The PINN only consumes:

        [theta1, omega1, theta2, omega2]

    Parameters
    ----------
    X:
        Input feature matrix.

    Returns
    -------
    np.ndarray
        Four physical state variables.
    """

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(
            f"X must be a 2D array. Received shape: {X.shape}"
        )

    if X.shape[1] < 4:
        raise ValueError(
            "PINN requires at least 4 input features: "
            "[theta1, omega1, theta2, omega2]."
        )

    # Keep only the physical state variables.
    return X[:, :4]


# ---------------------------------------------------------------------
# Physics loss
# ---------------------------------------------------------------------

def physics_loss(
    X: torch.Tensor,
    prediction: torch.Tensor,
    gravity: float = 9.81,
) -> torch.Tensor:
    """
    Compute the physics constraint loss.

    The original CEHSNF PINN formulation uses the approximate
    physical relation:

        omega1_dot ≈ -g * sin(theta1)

    Therefore:

        L_physics =
            MSE(prediction, -g * sin(theta1))

    Parameters
    ----------
    X:
        Physical input tensor with columns:

            [theta1, omega1, theta2, omega2]

    prediction:
        Neural network prediction.

    gravity:
        Gravitational acceleration.

    Returns
    -------
    torch.Tensor
        Physics loss.
    """

    if X.ndim != 2 or X.shape[1] < 4:
        raise ValueError(
            "X must have at least four columns."
        )

    theta1 = X[:, 0:1]

    physical_target = -gravity * torch.sin(theta1)

    return nn.functional.mse_loss(
        prediction,
        physical_target,
    )


# ---------------------------------------------------------------------
# Data loss
# ---------------------------------------------------------------------

def data_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    """
    Compute supervised data loss.

    Uses mean squared error.
    """

    return nn.functional.mse_loss(
        prediction,
        target,
    )


# ---------------------------------------------------------------------
# Total PINN loss
# ---------------------------------------------------------------------

def total_loss(
    X: torch.Tensor,
    y: torch.Tensor,
    prediction: torch.Tensor,
    physics_weight: float = 0.1,
    gravity: float = 9.81,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the complete PINN objective.

        L_total =
            L_data + lambda * L_physics

    Returns
    -------
    total:
        Combined loss.

    supervised:
        Data loss.

    physics:
        Physics loss.
    """

    supervised = data_loss(
        y,
        prediction,
    )

    physics = physics_loss(
        X,
        prediction,
        gravity=gravity,
    )

    total = supervised + physics_weight * physics

    return total, supervised, physics


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_pinn(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 800,
    learning_rate: float = 1e-3,
    physics_weight: float = 0.1,
    gravity: float = 9.81,
    hidden_dim: int = 64,
    verbose: bool = True,
) -> Tuple[CEHSNFPINN, List[Dict[str, float]]]:
    """
    Train the CEHSNF Physics-Informed Neural Network.

    Parameters
    ----------
    X:
        Input dataset.

        The function accepts both:

            (n_samples, 4)

        and:

            (n_samples, 6)

        For six-feature double-pendulum data, only the first
        four physical variables are used.

    y:
        Target values.

    epochs:
        Number of training epochs.

    learning_rate:
        Adam learning rate.

    physics_weight:
        Weight assigned to the physics loss.

    gravity:
        Gravitational acceleration.

    hidden_dim:
        Hidden layer size.

    verbose:
        Whether to print training progress.

    Returns
    -------
    model:
        Trained CEHSNFPINN.

    history:
        List containing loss values for every epoch.
    """

    # -------------------------------------------------------------
    # Prepare input
    # -------------------------------------------------------------

    X = _prepare_inputs(X)

    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of samples."
        )

    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than zero."
        )

    if physics_weight < 0:
        raise ValueError(
            "physics_weight cannot be negative."
        )

    # -------------------------------------------------------------
    # Convert to tensors
    # -------------------------------------------------------------

    X_tensor = torch.from_numpy(X).to(DEVICE)
    y_tensor = torch.from_numpy(y).to(DEVICE)

    # -------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------

    model = CEHSNFPINN(
        input_dim=4,
        hidden_dim=hidden_dim,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # -------------------------------------------------------------
    # Best model tracking
    # -------------------------------------------------------------

    best_loss = float("inf")
    best_state = copy.deepcopy(
        model.state_dict()
    )

    history: List[Dict[str, float]] = []

    # -------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):

        model.train()

        optimizer.zero_grad()

        prediction = model(X_tensor)

        loss, supervised, physics = total_loss(
            X_tensor,
            y_tensor,
            prediction,
            physics_weight=physics_weight,
            gravity=gravity,
        )

        loss.backward()

        optimizer.step()

        loss_value = float(loss.item())
        supervised_value = float(supervised.item())
        physics_value = float(physics.item())

        history.append(
            {
                "epoch": epoch,
                "loss": loss_value,
                "data_loss": supervised_value,
                "physics_loss": physics_value,
            }
        )

        # ---------------------------------------------------------
        # Save best model
        # ---------------------------------------------------------

        if loss_value < best_loss:
            best_loss = loss_value
            best_state = copy.deepcopy(
                model.state_dict()
            )

        # ---------------------------------------------------------
        # Progress
        # ---------------------------------------------------------

        if verbose and (
            epoch == 1
            or epoch % 100 == 0
            or epoch == epochs
        ):
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Total: {loss_value:.8f} | "
                f"Data: {supervised_value:.8f} | "
                f"Physics: {physics_value:.8f}"
            )

    # -------------------------------------------------------------
    # Restore best model
    # -------------------------------------------------------------

    model.load_state_dict(best_state)

    elapsed = time.perf_counter() - start_time

    if verbose:
        print(
            f"Training time : {elapsed:.4f} seconds"
        )

    return model, history


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

def predict_pinn(
    model: CEHSNFPINN,
    X: np.ndarray,
) -> np.ndarray:
    """
    Generate predictions using a trained PINN.

    Accepts both four-feature and six-feature CEHSNF inputs.
    """

    X = _prepare_inputs(X)

    X_tensor = torch.from_numpy(X).to(DEVICE)

    model.eval()

    with torch.no_grad():
        prediction = model(X_tensor)

    return (
        prediction
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard regression metrics.
    """

    y_true = np.asarray(y_true).reshape(-1)
    y_prediction = np.asarray(y_prediction).reshape(-1)

    mse = mean_squared_error(
        y_true,
        y_prediction,
    )

    rmse = float(
        np.sqrt(mse)
    )

    mae = mean_absolute_error(
        y_true,
        y_prediction,
    )

    r2 = r2_score(
        y_true,
        y_prediction,
    )

    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": float(mae),
        "r2": float(r2),
    }


# ---------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------

def sensitivity_analysis(
    model: CEHSNFPINN,
    X: np.ndarray,
    perturbation: float = 1e-4,
) -> Dict[str, float]:
    """
    Perform local sensitivity analysis.

    Each physical input feature is perturbed independently.

    Sensitivity is measured as:

        mean(
            |f(x + delta) - f(x)|
        )

    Parameters
    ----------
    model:
        Trained PINN.

    X:
        Input dataset.

    perturbation:
        Small perturbation applied to each feature.

    Returns
    -------
    dict
        Mean absolute sensitivity for each physical variable.
    """

    X = _prepare_inputs(X)

    if perturbation <= 0:
        raise ValueError(
            "perturbation must be greater than zero."
        )

    base_prediction = predict_pinn(
        model,
        X,
    )

    feature_names = [
        "theta1",
        "omega1",
        "theta2",
        "omega2",
    ]

    sensitivities: Dict[str, float] = {}

    for feature_index, feature_name in enumerate(
        feature_names
    ):

        perturbed = X.copy()

        perturbed[:, feature_index] += perturbation

        perturbed_prediction = predict_pinn(
            model,
            perturbed,
        )

        difference = np.abs(
            perturbed_prediction
            - base_prediction
        )

        sensitivities[feature_name] = float(
            np.mean(difference)
        )

    return sensitivities


# ---------------------------------------------------------------------
# Complete pipeline
# ---------------------------------------------------------------------

def pinn_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 800,
    learning_rate: float = 1e-3,
    physics_weight: float = 0.1,
    gravity: float = 9.81,
    hidden_dim: int = 64,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Execute the complete CEHSNF PINN pipeline.

    Returns
    -------
    dict
        Contains:

        model
        prediction
        metrics
        loss_history
        sensitivity
    """

    model, history = train_pinn(
        X=X,
        y=y,
        epochs=epochs,
        learning_rate=learning_rate,
        physics_weight=physics_weight,
        gravity=gravity,
        hidden_dim=hidden_dim,
        verbose=verbose,
    )

    prediction = predict_pinn(
        model,
        X,
    )

    metrics = compute_metrics(
        y,
        prediction,
    )

    sensitivity = sensitivity_analysis(
        model,
        X,
    )

    return {
        "model": model,
        "prediction": prediction,
        "metrics": metrics,
        "loss_history": history,
        "sensitivity": sensitivity,
    }


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

__all__ = [
    "CEHSNFPINN",
    "physics_loss",
    "data_loss",
    "total_loss",
    "train_pinn",
    "predict_pinn",
    "compute_metrics",
    "sensitivity_analysis",
    "pinn_pipeline",
]


# ---------------------------------------------------------------------
# Module test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    from cehsnf.systems.double_pendulum import (
        make_dataset,
    )

    print("CEHSNF PINN test")
    print("-" * 50)

    # -------------------------------------------------------------
    # Generate canonical CEHSNF double-pendulum data
    # -------------------------------------------------------------

    X, y = make_dataset()

    print(
        f"Input shape    : {X.shape}"
    )

    print(
        f"Target shape   : {y.shape}"
    )

    # -------------------------------------------------------------
    # Train
    # -------------------------------------------------------------

    model, history = train_pinn(
        X,
        y,
        epochs=100,
        learning_rate=1e-3,
        physics_weight=0.1,
        gravity=9.81,
        hidden_dim=64,
        verbose=True,
    )

    # -------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------

    prediction = predict_pinn(
        model,
        X,
    )

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------

    metrics = compute_metrics(
        y,
        prediction,
    )

    # -------------------------------------------------------------
    # Sensitivity
    # -------------------------------------------------------------

    sensitivity = sensitivity_analysis(
        model,
        X,
    )

    # -------------------------------------------------------------
    # Results
    # -------------------------------------------------------------

    print()
    print("PINN results")
    print("-" * 50)

    print(
        f"MSE          : {metrics['mse']:.6f}"
    )

    print(
        f"RMSE         : {metrics['rmse']:.6f}"
    )

    print(
        f"MAE          : {metrics['mae']:.6f}"
    )

    print(
        f"R²           : {metrics['r2']:.6f}"
    )

    print()
    print("Sensitivity")
    print("-" * 50)

    for feature, value in sensitivity.items():
        print(
            f"{feature:8s} : {value:.8f}"
        )

    print()
    print(
        f"Final loss   : {history[-1]['loss']:.8f}"
    )

    print(
        f"Best loss    : "
        f"{min(item['loss'] for item in history):.8f}"
    )

    print()
    print("PINN test completed successfully.")