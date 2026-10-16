"""B-spline basis functions and derivatives """

import numpy as np


def make_knots(num_intervals: int, degree: int,
               x_min: float = -1.0, x_max: float = 1.0) -> np.ndarray:
    """Create a uniform extended knot vector with degree-padding on each side.

    Returns shape (num_intervals + 2*degree + 1,).
    """
    h = (x_max - x_min) / num_intervals
    knots = x_min + np.arange(-degree, num_intervals + degree + 1) * h
    return knots.astype(np.float64)


def bspline_basis(x: np.ndarray, knots: np.ndarray,
                  degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate B-spline basis functions via Cox-de Boor recursion.

    Parameters
    ----------
    x      : (batch, in_features)
    knots  : (num_knots,)
    degree : B-spline degree

    Returns
    -------
    bases      : (batch, in_features, num_basis)  where num_basis = len(knots) - degree - 1
    bases_prev : (batch, in_features, num_basis+1) degree k-1 bases, needed for derivatives.
                 None when degree == 0.
    """
    n = len(knots)
    x_exp = x[..., np.newaxis]

    # Degree-0: indicator on each knot span
    bases = ((x_exp >= knots[:-1]) & (x_exp < knots[1:])).astype(np.float64)

    # Include right endpoint in last span
    right_mask = (x == knots[-1])
    bases[:, :, -1] = np.where(right_mask, 1.0, bases[:, :, -1])

    if degree == 0:
        return bases, None

    # Cox-de Boor recursion
    bases_prev = None
    for d in range(1, degree + 1):
        bases_prev = bases
        n_new = n - d - 1

        left_denom = knots[d: d + n_new] - knots[:n_new]
        left_num = x_exp - knots[:n_new]
        left = np.where(np.abs(left_denom) > 1e-14,
                        left_num / left_denom, 0.0)
        left = left * bases_prev[:, :, :n_new]

        right_denom = knots[d + 1: d + 1 + n_new] - knots[1: 1 + n_new]
        right_num = knots[d + 1: d + 1 + n_new] - x_exp
        right = np.where(np.abs(right_denom) > 1e-14,
                         right_num / right_denom, 0.0)
        right = right * bases_prev[:, :, 1: 1 + n_new]

        bases = left + right

    return bases, bases_prev


def bspline_basis_derivative(knots: np.ndarray, degree: int,
                             bases_prev: np.ndarray) -> np.ndarray:
    """Analytic derivative dB_{i,k}/dx using degree k-1 bases.

    Returns (batch, in_features, num_basis).
    """
    if degree == 0 or bases_prev is None:
        raise ValueError("Cannot compute derivative of degree-0 B-splines.")

    n = len(knots)
    n_basis = n - degree - 1

    d1 = knots[degree: degree + n_basis] - knots[:n_basis]
    d2 = knots[degree + 1: degree + 1 + n_basis] - knots[1: 1 + n_basis]

    coeff1 = np.where(np.abs(d1) > 1e-14, degree / d1, 0.0)
    coeff2 = np.where(np.abs(d2) > 1e-14, degree / d2, 0.0)

    term1 = coeff1 * bases_prev[:, :, :n_basis]
    term2 = coeff2 * bases_prev[:, :, 1: 1 + n_basis]

    return term1 - term2
