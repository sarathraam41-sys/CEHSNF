"""
Damped harmonic oscillator system.

System:
    x'' = -omega^2 * x - gamma * v

where:
    x      = displacement
    v      = velocity
    gamma  = damping coefficient
    omega  = natural angular frequency
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


DEFAULT_GAMMA = 0.5
DEFAULT_OMEGA = 2.0


def simulate(
    time_start: float = 0.0,
    time_end: float = 10.0,
    num_points: int = 1000,
    x0: float = 1.0,
    v0: float = 0.0,
    gamma: float = DEFAULT_GAMMA,
    omega: float = DEFAULT_OMEGA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate the damped harmonic oscillator.

    Returns
    -------
    t : np.ndarray
        Time values.
    x : np.ndarray
        Displacement.
    v : np.ndarray
        Velocity.
    a : np.ndarray
        Acceleration.
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")

    if omega <= 0:
        raise ValueError("omega must be greater than zero.")

    if gamma < 0:
        raise ValueError("gamma cannot be negative.")

    def dynamics(
        _t: float,
        state: np.ndarray,
    ) -> list[float]:
        x, v = state

        acceleration = -(omega**2) * x - gamma * v

        return [v, acceleration]

    t = np.linspace(time_start, time_end, num_points)

    solution = solve_ivp(
        dynamics,
        (time_start, time_end),
        [x0, v0],
        t_eval=t,
    )

    if not solution.success:
        raise RuntimeError(
            f"Damped oscillator simulation failed: {solution.message}"
        )

    x = solution.y[0]
    v = solution.y[1]

    acceleration = -(omega**2) * x - gamma * v

    return t, x, v, acceleration


def make_dataset(
    time_start: float = 0.0,
    time_end: float = 10.0,
    num_points: int = 1000,
    x0: float = 1.0,
    v0: float = 0.0,
    gamma: float = DEFAULT_GAMMA,
    omega: float = DEFAULT_OMEGA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a supervised-learning dataset.

    Features:
        [displacement, velocity]

    Target:
        acceleration
    """
    _, x, v, acceleration = simulate(
        time_start=time_start,
        time_end=time_end,
        num_points=num_points,
        x0=x0,
        v0=v0,
        gamma=gamma,
        omega=omega,
    )

    X = np.column_stack((x, v))
    y = acceleration

    return X, y


if __name__ == "__main__":
    X, y = make_dataset()

    print("Damped harmonic oscillator dataset")
    print(f"Features shape : {X.shape}")
    print(f"Target shape   : {y.shape}")
    print(f"First sample   : X={X[0]}, y={y[0]:.6f}")