"""
Harmonic oscillator system.

Provides simulation and dataset generation for the CEHSNF framework.

System:
    x'' + x = 0

Therefore:
    x(t) = sin(t)
    v(t) = cos(t)
    a(t) = -x(t)
"""

from __future__ import annotations

import numpy as np


def simulate(
    time_start: float = 0.0,
    time_end: float = 10.0,
    num_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate the harmonic oscillator.

    Returns
    -------
    t : np.ndarray
        Time values.
    x : np.ndarray
        Position values.
    v : np.ndarray
        Velocity values.
    a : np.ndarray
        Acceleration values.
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")

    t = np.linspace(time_start, time_end, num_points)

    x = np.sin(t)
    v = np.cos(t)
    a = -x

    return t, x, v, a


def make_dataset(
    time_start: float = 0.0,
    time_end: float = 10.0,
    num_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a supervised-learning dataset.

    Features:
        [position, velocity]

    Target:
        acceleration
    """
    _, x, v, a = simulate(
        time_start=time_start,
        time_end=time_end,
        num_points=num_points,
    )

    X = np.column_stack((x, v))
    y = a

    return X, y


if __name__ == "__main__":
    X, y = make_dataset()

    print("Harmonic oscillator dataset")
    print(f"Features shape : {X.shape}")
    print(f"Target shape   : {y.shape}")
    print(f"First sample   : X={X[0]}, y={y[0]:.6f}")