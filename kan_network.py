"""Multi-layer KAN network (pure NumPy)."""

from __future__ import annotations

import numpy as np

from kan_layer import KANLayer


class KANNetwork:
    """Stack of KANLayer objects forming a complete network.

    layer_sizes: e.g. [2, 5, 1] for 2 inputs -> 5 hidden -> 1 output.
    """

    def __init__(self, layer_sizes: list[int], *,
                 num_intervals: int = 5, degree: int = 3,
                 x_range: tuple[float, float] = (-2.0, 2.0)):
        if len(layer_sizes) < 2:
            raise ValueError("Need at least an input and output layer size.")
        self.layers: list[KANLayer] = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(KANLayer(
                layer_sizes[i], layer_sizes[i + 1],
                num_intervals=num_intervals,
                degree=degree,
                x_range=x_range,
            ))

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dy: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy

    def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        result: list[tuple[np.ndarray, np.ndarray]] = []
        for layer in self.layers:
            result.extend(layer.params_and_grads())
        return result
