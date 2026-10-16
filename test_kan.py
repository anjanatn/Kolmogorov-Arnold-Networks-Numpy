"""Tests for the NumPy KAN implementation.

Run with:  python -m pytest test_kan.py -v
Or simply:  python test_kan.py
"""

from __future__ import annotations

import sys
import numpy as np

from bspline import bspline_basis, bspline_basis_derivative, make_knots
from kan_layer import KANLayer
from kan_network import KANNetwork
from losses import mse_loss
from optimizer import Adam


# =========================================================================
# 1.  B-spline partition of unity
# =========================================================================

def test_partition_of_unity():
    """Sum of all B-spline bases must equal 1 at every interior point."""
    G, k = 8, 3
    knots = make_knots(G, k, x_min=-1.0, x_max=1.0)

    # Evaluate at random interior points  (stay away from the grid boundary
    # padding region; the POU holds on [x_min, x_max]).
    rng = np.random.default_rng(42)
    x = rng.uniform(-1.0, 1.0, size=(50, 4))  # (batch=50, in=4)

    bases, _ = bspline_basis(x, knots, k)  # (50, 4, G+k)
    sums = bases.sum(axis=-1)              # (50, 4)

    assert np.allclose(sums, 1.0, atol=1e-12), (
        f"Partition of unity violated: max deviation = {np.abs(sums - 1).max():.2e}"
    )
    print("[PASS] Partition of unity")


# =========================================================================
# 2.  B-spline derivative vs. finite differences
# =========================================================================

def test_bspline_derivative():
    """Analytic B-spline derivative should match finite differences."""
    G, k = 6, 3
    knots = make_knots(G, k, x_min=-1.5, x_max=1.5)

    rng = np.random.default_rng(7)
    x = rng.uniform(-1.0, 1.0, size=(20, 3))

    # Analytic
    bases, bases_prev = bspline_basis(x, knots, k)
    dbases_analytic = bspline_basis_derivative(knots, k, bases_prev)

    # Numerical (central finite differences)
    eps = 1e-5
    bases_plus, _ = bspline_basis(x + eps, knots, k)
    bases_minus, _ = bspline_basis(x - eps, knots, k)
    dbases_numeric = (bases_plus - bases_minus) / (2 * eps)

    assert np.allclose(dbases_analytic, dbases_numeric, atol=1e-5), (
        f"B-spline derivative mismatch: max err = "
        f"{np.abs(dbases_analytic - dbases_numeric).max():.2e}"
    )
    print("[PASS] B-spline analytic derivative ~ finite differences")


# =========================================================================
# 3.  KAN layer gradient check (all parameters + input)
# =========================================================================

def _numerical_grad_param(layer: KANLayer, x: np.ndarray,
                          target: np.ndarray, param: np.ndarray,
                          eps: float = 1e-5) -> np.ndarray:
    """Finite-difference gradient of MSE loss w.r.t. a flat parameter array."""
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        old = param[idx]

        param[idx] = old + eps
        pred_p = layer.forward(x)
        loss_p, _ = mse_loss(pred_p, target)

        param[idx] = old - eps
        pred_m = layer.forward(x)
        loss_m, _ = mse_loss(pred_m, target)

        grad[idx] = (loss_p - loss_m) / (2 * eps)
        param[idx] = old
        it.iternext()
    return grad


def _numerical_grad_input(layer: KANLayer, x: np.ndarray,
                          target: np.ndarray,
                          eps: float = 1e-5) -> np.ndarray:
    """Finite-difference gradient of MSE loss w.r.t. input *x*."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]

        x[idx] = old + eps
        pred_p = layer.forward(x)
        loss_p, _ = mse_loss(pred_p, target)

        x[idx] = old - eps
        pred_m = layer.forward(x)
        loss_m, _ = mse_loss(pred_m, target)

        grad[idx] = (loss_p - loss_m) / (2 * eps)
        x[idx] = old
        it.iternext()
    return grad


def test_layer_gradients():
    """Analytic gradients from backward() must match finite differences."""
    rng = np.random.default_rng(123)
    in_f, out_f = 3, 2
    batch = 4

    layer = KANLayer(in_f, out_f, num_intervals=4, degree=3,
                     x_range=(-2.0, 2.0))
    x = rng.uniform(-1.0, 1.0, size=(batch, in_f))
    target = rng.standard_normal((batch, out_f))

    # Analytic forward + backward
    pred = layer.forward(x)
    loss, dy = mse_loss(pred, target)
    dx_analytic = layer.backward(dy)

    atol = 1e-5

    # --- Check parameter gradients ---
    for name, param, ana_grad in [
        ("coeff",    layer.coeff,    layer.d_coeff),
        ("w_base",   layer.w_base,   layer.d_w_base),
        ("w_spline", layer.w_spline, layer.d_w_spline),
    ]:
        num_grad = _numerical_grad_param(layer, x, target, param)
        err = np.abs(ana_grad - num_grad).max()
        ok = err < atol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] d_{name:10s}  max_err = {err:.2e}")
        assert ok, f"Gradient check failed for {name}"

    # --- Check input gradient ---
    num_dx = _numerical_grad_input(layer, x, target)
    err_dx = np.abs(dx_analytic - num_dx).max()
    ok_dx = err_dx < atol
    status = "PASS" if ok_dx else "FAIL"
    print(f"  [{status}] dx            max_err = {err_dx:.2e}")
    assert ok_dx, "Gradient check failed for dx"

    print("[PASS] All layer gradient checks")


# =========================================================================
# 4.  Multi-layer KAN gradient check
# =========================================================================

def test_network_gradients():
    """End-to-end gradient check through a 2-layer KAN."""
    rng = np.random.default_rng(999)
    net = KANNetwork([2, 3, 1], num_intervals=4, degree=3,
                     x_range=(-2.0, 2.0))
    batch = 5
    x = rng.uniform(-1.0, 1.0, size=(batch, 2))
    target = rng.standard_normal((batch, 1))

    # Analytic
    pred = net.forward(x)
    loss, dy = mse_loss(pred, target)
    net.backward(dy)

    eps = 1e-5
    atol = 1e-4

    for li, layer in enumerate(net.layers):
        for pi, (param, ana_grad) in enumerate(layer.params_and_grads()):
            num_grad = np.zeros_like(param)
            it = np.nditer(param, flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index
                old = param[idx]
                param[idx] = old + eps
                lp = float(np.mean((net.forward(x) - target) ** 2))
                param[idx] = old - eps
                lm = float(np.mean((net.forward(x) - target) ** 2))
                num_grad[idx] = (lp - lm) / (2 * eps)
                param[idx] = old
                it.iternext()

            err = np.abs(ana_grad - num_grad).max()
            rel = err / (np.abs(num_grad).max() + 1e-12)
            ok = err < atol
            tag = f"layer{li}_param{pi}"
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {tag:20s}  abs_err={err:.2e}  rel_err={rel:.2e}")
            assert ok, f"Network gradient check failed: {tag}"

    print("[PASS] Multi-layer network gradient checks")


# =========================================================================
# 5.  Training convergence on a trivial function
# =========================================================================

def test_training_convergence():
    """Train a small KAN on y=x^2 and verify loss decreases."""
    rng = np.random.default_rng(0)
    net = KANNetwork([1, 5, 1], num_intervals=5, degree=3,
                     x_range=(-2.0, 2.0))
    opt = Adam(lr=1e-2)

    n = 100
    x_train = rng.uniform(-1.5, 1.5, size=(n, 1))
    y_train = x_train ** 2

    losses = []
    for epoch in range(300):
        pred = net.forward(x_train)
        loss, dy = mse_loss(pred, y_train)
        net.backward(dy)
        opt.step(net.params_and_grads())
        losses.append(loss)

    assert losses[-1] < losses[0] * 0.01, (
        f"Training did not converge: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    )
    print(f"[PASS] Training convergence  "
          f"(initial={losses[0]:.4f} -> final={losses[-1]:.4f})")


# =========================================================================
# Runner
# =========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("KAN NumPy — Test Suite")
    print("=" * 60)

    tests = [
        test_partition_of_unity,
        test_bspline_derivative,
        test_layer_gradients,
        test_network_gradients,
        test_training_convergence,
    ]
    failed = 0
    for t in tests:
        print(f"\n--- {t.__name__} ---")
        try:
            t()
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1

    print("\n" + "=" * 60)
    if failed:
        print(f"  {failed} test(s) FAILED")
        sys.exit(1)
    else:
        print("  All tests PASSED")
        sys.exit(0)
