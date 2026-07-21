"""MCMC methods."""

from typing import Callable

import random

import numpy as np


def metropolis_hastings(
    x: float,
    sigma: float,
    pdf: Callable[[float], float]
) -> list[float]:
    """Draw samples from a distribution using Metropolis–Hastings.

    The function uses a Gaussian random-walk proposal distribution centered
    at the current state. At each iteration, the proposed state is accepted
    according to the ratio between the target densities of the proposed and
    current states.

    The target density does not need to be normalized, because its
    normalization constant cancels in the acceptance ratio.

    Args:
        x: Initial state of the Markov chain.
        sigma: Standard deviation of the Gaussian proposal distribution.
            Must be greater than zero.
        pdf: Function that evaluates the target probability density, or an
            unnormalized quantity proportional to it, at a given state.

    Returns:
        A list containing 100,000 samples from the Markov chain, including
        repeated states for rejected proposals.
    """
    x_list = []
    for _ in range(100_000):
        x_prime = random.gauss(mu=x, sigma=sigma)
        alpha = min(1.0, float(pdf(x_prime) / pdf(x)))

        prob = random.random()
        if alpha >= prob:
            x = x_prime
        x_list.append(x)
    return x_list


def ising_model(
    env: np.ndarray,
    x_loc: int,
    y_loc: int,
    state: float,
    h: float = 0.1,
    beta: float = 1.0,
    lam: float = 1.0
):
    """Calculate the unnormalized probability of a local Ising-model state.

    The function computes the contribution of a proposed state at a given
    lattice position. It considers the four directly adjacent neighbors,
    an external field, and the currently observed state at the selected
    position.

    Args:
        env: Two-dimensional array containing the current lattice states.
        x_loc: Row index of the selected lattice position.
        y_loc: Column index of the selected lattice position.
        state: Proposed state at the selected position, typically ``-1`` or
            ``1``.
        h: Strength of the external field.
        beta: Strength of the interaction with neighboring lattice states.
        lam: Strength of the interaction with the currently observed state.

    Returns:
        The unnormalized probability weight of the proposed state.
    """

    ext = 0
    if x_loc - 1 >= 0:
        ext += env[x_loc - 1, y_loc]
    if x_loc + 1 < len(env):
        ext += env[x_loc + 1, y_loc]
    if y_loc - 1 >= 0:
        ext += env[x_loc, y_loc - 1]
    if y_loc + 1 < len(env):
        ext += env[x_loc, y_loc + 1]

    o = env[x_loc, y_loc]
    return np.exp(state * (h + beta * ext + lam * o))
