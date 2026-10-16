"""Single KAN layer with manual forward and backward passes (pure NumPy).

Edge function:  phi_{j,i}(x) = w_base[j,i] * silu(x) + w_spline[j,i] * sum_m c[j,i,m] B_m(x)
Layer output:   y_j = sum_i phi_{j,i}(x_i)
"""

from __future__ import annotations

import numpy as np

from bspline import bspline_basis, bspline_basis_derivative, make_knots


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ez = np.exp(x[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def _silu(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (silu(x), silu'(x))."""
    sig = _sigmoid(x)
    val = x * sig
    dx = sig * (1.0 + x * (1.0 - sig))
    return val, dx


class KANLayer:
    """One Kolmogorov-Arnold layer.

    Parameters
    ----------
    in_features, out_features : int
    num_intervals : int – grid intervals for B-spline knots
    degree : int – B-spline degree (3 = cubic)
    x_range : knot domain bounds
    """

    def __init__(self, in_features: int, out_features: int, *,
                 num_intervals: int = 5, degree: int = 3,
                 x_range: tuple[float, float] = (-2.0, 2.0)):
        self.in_features = in_features
        self.out_features = out_features
        self.num_intervals = num_intervals
        self.degree = degree

        self.knots = make_knots(num_intervals, degree, *x_range)
        self.num_basis = num_intervals + degree

        # Trainable parameters
        self.coeff = np.random.randn(
            out_features, in_features, self.num_basis
        ).astype(np.float64) / np.sqrt(in_features * self.num_basis)

        self.w_base = (np.random.randn(out_features, in_features).astype(np.float64)
                       / np.sqrt(in_features))

        self.w_spline = np.ones(
            (out_features, in_features), dtype=np.float64
        )

        # Gradient accumulators (set by backward)
        self.d_coeff: np.ndarray | None = None
        self.d_w_base: np.ndarray | None = None
        self.d_w_spline: np.ndarray | None = None

        self._cache: dict | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x: (batch, in) -> y: (batch, out)."""
        silu_val, silu_dx = _silu(x)

        bases, bases_prev = bspline_basis(x, self.knots, self.degree)
        dbases = bspline_basis_derivative(self.knots, self.degree, bases_prev)

        spline_val = np.einsum('oim,bim->boi', self.coeff, bases)

        base_term = np.einsum('oi,bi->bo', self.w_base, silu_val)
        spline_term = np.einsum('oi,boi->bo', self.w_spline, spline_val)
        y = base_term + spline_term

        self._cache = dict(
            x=x, silu_val=silu_val, silu_dx=silu_dx,
            bases=bases, dbases=dbases, spline_val=spline_val,
        )
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Backward pass. dy: (batch, out) -> dx: (batch, in).
        Also sets d_coeff, d_w_base, d_w_spline.
        """
        c = self._cache
        if c is None:
            raise RuntimeError("backward() called before forward()")

        # Parameter gradients
        self.d_w_base = np.einsum('bo,bi->oi', dy, c['silu_val'])
        self.d_w_spline = np.einsum('bo,boi->oi', dy, c['spline_val'])
        self.d_coeff = np.einsum('bo,oi,bim->oim', dy, self.w_spline, c['bases'])

        # Input gradient
        dx_base = np.einsum('bo,oi->bi', dy, self.w_base) * c['silu_dx']

        phi_dx = np.einsum('oim,bim->boi', self.coeff, c['dbases'])
        dx_spline = np.einsum('bo,oi,boi->bi', dy, self.w_spline, phi_dx)

        dx = dx_base + dx_spline
        self._cache = None
        return dx

    def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return [(param, grad), ...] for the optimizer."""
        return [
            (self.coeff, self.d_coeff),
            (self.w_base, self.d_w_base),
            (self.w_spline, self.d_w_spline),
        ]
