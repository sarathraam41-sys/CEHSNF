"""
CEHSNF - Coupled Double Pendulum System

True nonlinear coupled double-pendulum dynamics.
"""

import numpy as np
from scipy.integrate import solve_ivp


DEFAULT_GRAVITY = 9.81
DEFAULT_LENGTH = 1.0


def dynamics(
    t,
    state,
    gravity=DEFAULT_GRAVITY,
    length=DEFAULT_LENGTH,
):
    """
    Compute the coupled double-pendulum derivatives.

    State:
        [theta1, omega1, theta2, omega2]
    """

    theta1, omega1, theta2, omega2 = state

    delta = theta2 - theta1

    denominator = 2.0 - np.cos(2.0 * delta)

    domega1 = (
        -gravity * (
            2.0 * np.sin(theta1)
            + np.sin(theta1 - 2.0 * theta2)
        )
        - np.sin(delta) * (
            2.0 * omega2**2 * length
            + omega1**2 * length * np.cos(delta)
        )
    ) / (length * denominator)

    domega2 = (
        2.0 * np.sin(delta) * (
            omega1**2 * length
            + gravity * np.cos(theta1)
            + omega2**2 * length * np.cos(delta)
        )
    ) / (length * denominator)

    return np.array([
        omega1,
        domega1,
        omega2,
        domega2,
    ])


def simulate(
    t_span=(0.0, 10.0),
    num_points=1000,
    theta1_0=1.0,
    omega1_0=0.0,
    theta2_0=1.2,
    omega2_0=0.0,
    gravity=DEFAULT_GRAVITY,
    length=DEFAULT_LENGTH,
):
    """
    Simulate the coupled double pendulum.
    """

    t = np.linspace(t_span[0], t_span[1], num_points)

    initial_state = [
        theta1_0,
        omega1_0,
        theta2_0,
        omega2_0,
    ]

    solution = solve_ivp(
        lambda time, state: dynamics(
            time,
            state,
            gravity=gravity,
            length=length,
        ),
        t_span,
        initial_state,
        t_eval=t,
        rtol=1e-9,
        atol=1e-11,
    )

    if not solution.success:
        raise RuntimeError(
            f"Double-pendulum simulation failed: {solution.message}"
        )

    return solution.t, solution.y.T


def make_dataset(
    t_span=(0.0, 10.0),
    num_points=1000,
    theta1_0=1.0,
    omega1_0=0.0,
    theta2_0=1.2,
    omega2_0=0.0,
    gravity=DEFAULT_GRAVITY,
    length=DEFAULT_LENGTH,
):
    """
    Build a supervised learning dataset.

    Features:
        theta1, omega1, theta2, omega2,
        sin(theta2-theta1), cos(theta2-theta1)

    Target:
        domega1
    """

    t, states = simulate(
        t_span=t_span,
        num_points=num_points,
        theta1_0=theta1_0,
        omega1_0=omega1_0,
        theta2_0=theta2_0,
        omega2_0=omega2_0,
        gravity=gravity,
        length=length,
    )

    theta1 = states[:, 0]
    omega1 = states[:, 1]
    theta2 = states[:, 2]
    omega2 = states[:, 3]

    delta = theta2 - theta1

    X = np.column_stack([
        theta1,
        omega1,
        theta2,
        omega2,
        np.sin(delta),
        np.cos(delta),
    ])

    targets = np.array([
        dynamics(
            time,
            state,
            gravity=gravity,
            length=length,
        )[1]
        for time, state in zip(t, states)
    ])

    return X, targets


if __name__ == "__main__":
    X, y = make_dataset()

    print("Coupled double pendulum dataset")
    print(f"Features shape : {X.shape}")
    print(f"Target shape   : {y.shape}")
    print(f"First sample   : X={X[0]}, y={y[0]:.6f}")