"""KAN training demo: learns f(x1, x2) = sin(pi*x1) + x2^2."""

from __future__ import annotations

import numpy as np

from kan_network import KANNetwork
from losses import mse_loss
from optimizer import Adam


def generate_data(n, noise_std=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    x = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = np.sin(np.pi * x[:, 0:1]) + x[:, 1:2] ** 2
    y += rng.normal(0, noise_std, size=y.shape)
    return x, y


def main():
    rng = np.random.default_rng(42)
    x_train, y_train = generate_data(500, noise_std=0.05, rng=rng)
    x_test, y_test = generate_data(200, noise_std=0.0, rng=rng)

    net = KANNetwork([2, 5, 1], num_intervals=5, degree=3, x_range=(-2.0, 2.0))
    opt = Adam(lr=5e-3)

    epochs = 500
    print_every = 50
    train_losses: list[float] = []

    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Test Loss':>12}")
    print("-" * 36)

    for epoch in range(1, epochs + 1):
        pred = net.forward(x_train)
        loss, dy = mse_loss(pred, y_train)
        train_losses.append(loss)

        net.backward(dy)
        opt.step(net.params_and_grads())

        if epoch % print_every == 0 or epoch == 1:
            test_pred = net.forward(x_test)
            test_loss = float(np.mean((test_pred - y_test) ** 2))
            print(f"{epoch:6d}  {loss:12.6f}  {test_loss:12.6f}")

    test_pred = net.forward(x_test)
    test_loss = float(np.mean((test_pred - y_test) ** 2))
    print(f"\nFinal test MSE: {test_loss:.6f}")

    # Plot if matplotlib is available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(train_losses)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE Loss")
        axes[0].set_title("Training Loss Curve")
        axes[0].set_yscale("log")
        axes[0].grid(True, alpha=0.3)

        x1_line = np.linspace(-1, 1, 200)
        x_slice = np.column_stack([x1_line, np.zeros_like(x1_line)])
        y_true_slice = np.sin(np.pi * x1_line)
        y_pred_slice = net.forward(x_slice).ravel()

        axes[1].plot(x1_line, y_true_slice, label="True  sin(pi*x1)", lw=2)
        axes[1].plot(x1_line, y_pred_slice, "--", label="KAN prediction", lw=2)
        axes[1].set_xlabel("x1")
        axes[1].set_title("Prediction (x2 = 0)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("kan_demo_results.png", dpi=150)
        print("Plot saved to kan_demo_results.png")
    except ImportError:
        print("(matplotlib not available -- skipping plot)")


if __name__ == "__main__":
    main()
