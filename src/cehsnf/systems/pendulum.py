"""
Nonlinear pendulum system.

System:
    theta'' = -(g / L) * sin(theta)

The module provides a numerical simulation and a supervised-learning
dataset for the CEHSNF framework.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


DEFAULT_GRAVITY = 9.81
DEFAULT_LENGTH = 1.0


def simulate(
    time_start: float = 0.0,
    time_end: float = 10.0,
    num_points: int = 1000,
    theta0: float = 1.0,
    omega0: float = 0.0,
    gravity: float = DEFAULT_GRAVITY,
    length: float = DEFAULT_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a nonlinear pendulum.

    Returns
    -------
    t : np.ndarray
        Time values.
    theta : np.ndarray
        Angular displacement.
    omega : np.ndarray
        Angular velocity.
    alpha : np.ndarray
        Angular acceleration.
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")

    if length <= 0:
        raise ValueError("length must be greater than zero.")

    def dynamics(
        _t: float,
        state: np.ndarray,
    ) -> list[float]:
        theta, omega = state

        alpha = -(gravity / length) * np.sin(theta)

        return [omega, alpha]

    t = np.linspace(time_start, time_end, num_points)

    solution = solve_ivp(
        dynamics,
        (time_start, time_end),
        [theta0, omega0],
        t_eval=t,
    )

    if not solution.success:
        raise RuntimeError(
            f"Pendulum simulation failed: {solution.message}"
        )

    theta = solution.y[0]
    omega = solution.y[1]

    alpha = -(gravity / length) * np.sin(theta)

    return t, theta, omega, alpha


def make_dataset(
    time_start: float = 0.0,
    time_end: float = 10.0,
    num_points: int = 1000,
    theta0: float = 1.0,
    omega0: float = 0.0,
    gravity: float = DEFAULT_GRAVITY,
    length: float = DEFAULT_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a supervised-learning dataset.

    Features:
        [angular displacement, angular velocity]

    Target:
        angular acceleration
    """
    _, theta, omega, alpha = simulate(
        time_start=time_start,
        time_end=time_end,
        num_points=num_points,
        theta0=theta0,
        omega0=omega0,
        gravity=gravity,
        length=length,
    )

    X = np.column_stack((theta, omega))
    y = alpha

    return X, y


if __name__ == "__main__":
    X, y = make_dataset()

    print("Nonlinear pendulum dataset")
    print(f"Features shape : {X.shape}")
    print(f"Target shape   : {y.shape}")
    print(f"First sample   : X={X[0]}, y={y[0]:.6f}")