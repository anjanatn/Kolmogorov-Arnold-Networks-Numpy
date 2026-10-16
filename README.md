# Kolmogorov-Arnold Networks (KAN) in NumPy

A lightweight, from-scratch implementation of **Kolmogorov-Arnold Networks (KANs)** built using pure **NumPy**.

## Overview

Unlike standard Multi-Layer Perceptrons (MLPs) that place fixed activation functions on nodes and learnable weights on edges, Kolmogorov-Arnold Networks place learnable activation functions (parameterized as B-splines) directly on edges with simple sum aggregations on nodes.

This repository provides an educational, dependency-minimal implementation of KAN forward and backward passes from scratch.

## Features

- **Pure NumPy Implementation**: No PyTorch, TensorFlow, or JAX dependencies required.
- **Cox-de Boor B-splines**: Numerically stable B-spline basis evaluation and exact gradient computation.
- **Modular Architecture**:
  - `bspline.py`: B-spline basis functions, knot vector management, and gradient computation.
  - `kan_layer.py`: Single KAN layer with spline weights, base linear/SiLU transformation, and backward propagation.
  - `kan_network.py`: Multi-layer KAN composition.
  - `losses.py`: Mean Squared Error (MSE) loss and gradients.
  - `optimizer.py`: Adam optimizer implemented from scratch.
  - `demo.py`: End-to-end toy function approximation demo ($f(x_1, x_2) = \sin(\pi x_1) + x_2^2$).
  - `test_kan.py`: Numerical gradient checks and unit tests.

## Installation

Clone the repository:
```bash
git clone https://github.com/anjanatn/Kolmogorov-Arnold-Networks-Numpy.git
cd Kolmogorov-Arnold-Networks-Numpy
```

Install dependencies:
```bash
pip install numpy
```

## Quick Start

Run the training demo:
```bash
python demo.py
```

Run test suite:
```bash
python -m unittest test_kan.py
# or if pytest is installed:
pytest
```

## Project Structure

```
├── bspline.py       # B-spline basis functions and gradients
├── kan_layer.py     # KAN layer definition and backward pass
├── kan_network.py   # Multi-layer KAN architecture
├── losses.py        # Loss functions (MSE)
├── optimizer.py     # Adam optimizer from scratch
├── demo.py          # Regression demo
├── test_kan.py      # Gradient validation & tests
├── .gitignore       # Git ignore rules
└── README.md        # Project documentation
```

## References

- Liu, Z., et al. (2024). *KAN: Kolmogorov-Arnold Networks*. [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
