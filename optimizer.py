"""Adam optimizer for lists of NumPy parameter arrays."""

import numpy as np


class Adam:
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}

    def step(self, params_and_grads: list[tuple[np.ndarray, np.ndarray]]):
        """In-place parameter update. Each param array is modified directly."""
        self.t += 1
        for idx, (param, grad) in enumerate(params_and_grads):
            if idx not in self._m:
                self._m[idx] = np.zeros_like(param)
                self._v[idx] = np.zeros_like(param)

            self._m[idx] = self.beta1 * self._m[idx] + (1.0 - self.beta1) * grad
            self._v[idx] = self.beta2 * self._v[idx] + (1.0 - self.beta2) * grad ** 2

            m_hat = self._m[idx] / (1.0 - self.beta1 ** self.t)
            v_hat = self._v[idx] / (1.0 - self.beta2 ** self.t)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
