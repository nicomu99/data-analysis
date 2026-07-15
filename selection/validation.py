"""Cross-validation utilities for model evaluation."""

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


def cross_validation(
    model: Any,
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    scoring_fn: Callable,
    k: int = 10
) -> np.floating:
    """Evaluate a model using k-fold cross-validation.

    The samples are randomly shuffled and divided into ``k`` folds. For each
    fold, the model is fitted on the remaining folds and evaluated on the
    held-out fold using the supplied scoring function. The function returns
    the mean score across all folds.

    The model must provide the following methods:

    - ``fit(x, y)``, which fits the model and returns the fitted model.
    - ``predict(x)``, which returns a tuple whose first element contains the
      predictions.

    Args:
        model: Model instance to evaluate. The model must implement compatible
            ``fit`` and ``predict`` methods.
        x: Feature matrix with shape ``(num_samples, num_features)``.
        y: Target values with shape ``(num_samples,)`` or
            ``(num_samples, num_targets)``.
        scoring_fn: Function that receives the predicted and true target values
            and returns a numeric score.
        k: Number of cross-validation folds.

    Returns:
        The mean score across all folds.

    Raises:
        ValueError: If ``k`` is smaller than 2, larger than the number of
            samples, or if ``x`` and ``y`` contain different numbers of
            samples.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "x and y must contain the same number of samples, "
            f"but received {x.shape[0]} and {y.shape[0]}."
        )

    if k < 2:
        raise ValueError("k must be at least 2.")

    if k > x.shape[0]:
        raise ValueError(
            "k cannot be larger than the number of samples, "
            f"but received k={k} and {x.shape[0]} samples."
        )

    # Shuffle samples
    permute = np.random.permutation(len(x))
    x = x[permute]
    y = y[permute]

    # Divide dataset into k blocks
    split_x = np.array_split(x, k)
    split_y = np.array_split(y, k)

    scores = []
    for i in range(k):
        train_x = np.concatenate(split_x[:i] + split_x[(i + 1):])
        train_y = np.concatenate(split_y[:i] + split_y[(i + 1):])
        test_x = split_x[i]
        test_y = split_y[i]

        model = model.fit(train_x, train_y)
        test_predictions = model.predict(test_x)
        if isinstance(test_predictions, tuple):
            test_predictions = test_predictions[0]

        fold_score = scoring_fn(test_predictions, test_y)
        scores.append(fold_score)

    return np.mean(scores)
