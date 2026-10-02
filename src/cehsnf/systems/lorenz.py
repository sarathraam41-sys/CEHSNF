"""
CEHSNF - Lorenz Dynamical System

Chaotic Lorenz system used for nonlinear dynamical-system evaluation.
"""

import numpy as np
from scipy.integrate import solve_ivp


DEFAULT_SIGMA = 10.0
DEFAULT_RHO = 28.0
DEFAULT_BETA = 8.0 / 3.0


def dynamics(
    t,
    state,
    sigma=DEFAULT_SIGMA,
    rho=DEFAULT_RHO,
    beta=DEFAULT_BETA,
):
    """
    Compute Lorenz system derivatives.

    State:
        [x, y, z]
    """

    x, y, z = state

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return np.array([dx, dy, dz])


def simulate(
    t_span=(0.0, 20.0),
    num_points=2000,
    x0=1.0,
    y0=1.0,
    z0=1.0,
    sigma=DEFAULT_SIGMA,
    rho=DEFAULT_RHO,
    beta=DEFAULT_BETA,
):
    """
    Simulate the Lorenz system.
    """

    t = np.linspace(t_span[0], t_span[1], num_points)

    initial_state = [x0, y0, z0]

    solution = solve_ivp(
        lambda time, state: dynamics(
            time,
            state,
            sigma=sigma,
            rho=rho,
            beta=beta,
        ),
        t_span,
        initial_state,
        t_eval=t,
        rtol=1e-9,
        atol=1e-11,
    )

    if not solution.success:
        raise RuntimeError(
            f"Lorenz simulation failed: {solution.message}"
        )

    return solution.t, solution.y.T


def make_dataset(
    t_span=(0.0, 20.0),
    num_points=2000,
    x0=1.0,
    y0=1.0,
    z0=1.0,
    sigma=DEFAULT_SIGMA,
    rho=DEFAULT_RHO,
    beta=DEFAULT_BETA,
):
    """
    Build a supervised learning dataset.

    Features:
        x, y, z

    Target:
        dx/dt
    """

    t, states = simulate(
        t_span=t_span,
        num_points=num_points,
        x0=x0,
        y0=y0,
        z0=z0,
        sigma=sigma,
        rho=rho,
        beta=beta,
    )

    X = states.copy()

    targets = np.array([
        dynamics(
            time,
            state,
            sigma=sigma,
            rho=rho,
            beta=beta,
        )[0]
        for time, state in zip(t, states)
    ])

    return X, targets


if __name__ == "__main__":
    X, y = make_dataset()

    print("Lorenz system dataset")
    print(f"Features shape : {X.shape}")
    print(f"Target shape   : {y.shape}")
    print(f"First sample   : X={X[0]}, y={y[0]:.6f}")