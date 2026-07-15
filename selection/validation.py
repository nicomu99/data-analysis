from typing import Any, Callable

import numpy as np


def cross_validation(
    model,
    x: np.ndarray,
    y: np.ndarray,
    scoring_fn: Callable,
    k: int = 10
) -> Any:
    # shuffle arrays
    permute = np.random.permutation(len(x))
    x = x[permute]
    y = y[permute]

    # divide dataset into k blocks
    split_x = np.array_split(x, k)
    split_y = np.array_split(y, k)

    res = []
    for i in range(k):
        train_x = np.concatenate(split_x[:i] + split_x[(i + 1):])
        train_y = np.concatenate(split_y[:i] + split_y[(i + 1):])
        test_x = split_x[i]
        test_y = split_y[i]

        model = model.fit(train_x, train_y)
        test_predictions, _ = model.predict(test_x)
        fold_score = scoring_fn(test_predictions, test_y)
        res.append(fold_score)

    return np.mean(res)
