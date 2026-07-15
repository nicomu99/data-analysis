"""Utility helper."""

import numpy as np


def _as_target_matrix(
    values: np.ndarray,
    *,
    name: str,
):
    """Convert regression targets to a two-dimensional floating-point array.

    Args:
        values: Target values with shape ``(num_samples,)`` or
            ``(num_samples, num_targets)``.
        name: Name used to identify the argument in error messages.

    Returns:
        Target values with shape ``(num_samples, num_targets)``.

    Raises:
        ValueError: If ``values`` is not one- or two-dimensional.
    """
    array = np.asarray(values, dtype=float)

    if array.ndim == 1:
        return array[:, np.newaxis]

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be one- or two-dimensional, "
            f"but received an array with {array.ndim} dimensions."
        )

    return array
