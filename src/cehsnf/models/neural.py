"""
CEHSNF - Neural Modeling

Reusable neural-network components for the
Complexity-Aware Explainable Hybrid Symbolic-Neural Framework.
"""

import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


SEED = 42
DEVICE = torch.device("cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)


class CEHSNFNetwork(nn.Module):
    """
    Fully connected neural network used by CEHSNF.

    Architecture:
        Input
          ↓
        Linear → Tanh
          ↓
        Linear → Tanh
          ↓
        Linear → Tanh
          ↓
        Output
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        output_dim=1,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),

            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)


def create_model(input_dim, hidden_dim=64):
    """Create a CEHSNF neural network."""

    model = CEHSNFNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    )

    return model.to(DEVICE)


def train_model(
    model,
    X_train,
    y_train,
    epochs=500,
    lr=0.001,
    patience=50,
):
    """
    Train the neural model using MSE loss and Adam.

    Returns:
        model
        loss_history
        training_time
    """

    X_tensor = torch.as_tensor(
        X_train,
        dtype=torch.float32,
        device=DEVICE,
    )

    y_tensor = torch.as_tensor(
        np.asarray(y_train).reshape(-1, 1),
        dtype=torch.float32,
        device=DEVICE,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )

    criterion = nn.MSELoss()

    loss_history = []

    best_loss = np.inf
    best_state = None
    wait = 0

    start_time = time.time()

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        prediction = model(X_tensor)

        loss = criterion(
            prediction,
            y_tensor,
        )

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        loss_history.append(current_loss)

        if current_loss < best_loss:

            best_loss = current_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            wait = 0

        else:
            wait += 1

        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    training_time = time.time() - start_time

    return (
        model,
        loss_history,
        training_time,
    )


def predict_model(model, X):
    """Generate predictions from a trained model."""

    model.eval()

    X_tensor = torch.as_tensor(
        X,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        prediction = model(X_tensor)

    return prediction.cpu().numpy().flatten()


def compute_metrics(y_true, y_pred):
    """Compute standard regression metrics."""

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "R2": r2_score(
            y_true,
            y_pred,
        ),
    }


def compute_residuals(y_true, y_pred):
    """Compute prediction residuals."""

    return np.asarray(y_true) - np.asarray(y_pred)


def neural_pipeline(
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=500,
):
    """
    Complete CEHSNF neural pipeline.

    Returns the trained model, predictions,
    residuals, training history and metrics.
    """

    model = create_model(
        input_dim=X_train.shape[1]
    )

    (
        model,
        history,
        training_time,
    ) = train_model(
        model,
        X_train,
        y_train,
        epochs=epochs,
    )

    train_prediction = predict_model(
        model,
        X_train,
    )

    test_prediction = predict_model(
        model,
        X_test,
    )

    train_metrics = compute_metrics(
        y_train,
        train_prediction,
    )

    test_metrics = compute_metrics(
        y_test,
        test_prediction,
    )

    return {
        "model": model,
        "history": history,
        "training_time": training_time,
        "train_prediction": train_prediction,
        "test_prediction": test_prediction,
        "train_residual": compute_residuals(
            y_train,
            train_prediction,
        ),
        "test_residual": compute_residuals(
            y_test,
            test_prediction,
        ),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


if __name__ == "__main__":

    from cehsnf.systems.harmonic import make_dataset

    X, y = make_dataset(
        num_points=300
    )

    split = int(0.8 * len(X))

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    result = neural_pipeline(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=100,
    )

    print("CEHSNF neural model test")
    print(f"Input shape    : {X.shape}")
    print(
        f"Training time  : "
        f"{result['training_time']:.4f} seconds"
    )
    print(
        f"Test MSE       : "
        f"{result['test_metrics']['MSE']:.6f}"
    )
    print(
        f"Test RMSE      : "
        f"{result['test_metrics']['RMSE']:.6f}"
    )
    print(
        f"Test MAE       : "
        f"{result['test_metrics']['MAE']:.6f}"
    )
    print(
        f"Test R²        : "
        f"{result['test_metrics']['R2']:.6f}"
    )