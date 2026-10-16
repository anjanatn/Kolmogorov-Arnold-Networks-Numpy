"""Loss functions returning (scalar_loss, grad_wrt_predictions)."""

import numpy as np


def mse_loss(pred: np.ndarray,
             target: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean Squared Error. Returns (loss, dL/d_pred)."""
    diff = pred - target
    loss = float(np.mean(diff ** 2))
    grad = 2.0 * diff / diff.size
    return loss, grad
