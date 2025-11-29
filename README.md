# MLP-from-Scratch

A small educational implementation of an MLP (multi-layer perceptron) and a tiny autograd engine, inspired by Micrograd. This repo implements a scalar `Value` autograd primitive and modular neural-network layers (Neuron, Layer, MLP) built on top of it.

**Key files**
- `MLP/engine.py`: `Value` class implementing basic operators (+, *, **, relu, tanh, exp) and reverse-mode autodiff (`backward`).
- `MLP/nn.py`: `Neuron`, `Layer`, and `MLP` classes which use `Value` objects to build neural networks.
- `test/test_engine.py`: tests that validate the `Value` implementation by comparing against PyTorch operations.

**Features**
- Minimal autograd for scalar values with a topological backward pass.
- Simple neuron and MLP abstractions (ReLU nonlinearity by default except on final layer).
- Tests that compare forward and backward results to PyTorch for correctness.

## Installation

1. (Recommended) Create and activate a virtual environment or conda environment.

   Example (venv):

   ```pwsh
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install requirements and the package in editable mode:

   ```pwsh
   pip install -r requirements.txt
   pip install -e .
   ```

3. Tests compare outputs to PyTorch. If you want to run the tests, install PyTorch (CPU-only example):

   ```pwsh
   pip install torch
   ```

   (For specific CUDA builds or platform-specific wheels, follow the instructions on https://pytorch.org.)

## Quick Usage

The code is intentionally small and readable. Example usage:

```python
from MLP.nn import MLP
from MLP.engine import Value

# Create a network: 3 inputs -> 4 hidden units -> 1 output
net = MLP(3, [4, 1])

# Sample input as a list of Value objects
x = [Value(1.0), Value(2.0), Value(-3.0)]

# Forward pass
y = net(x)  # returns a single Value when output size == 1

# Backprop
y.backward()

print('output:', y)
print('parameter gradients:')
for p in net.parameters():
    print(p, p.grad)
```

You can inspect and modify `MLP/nn.py` to change layer sizes, activation functions, or to add features like mini-batch support.

## Running Tests

Run the tests with `pytest`. Ensure `torch` is installed first (see Installation above):

```pwsh
pip install torch
python -m pytest -q
```

The tests in `test/test_engine.py` validate that forward and backward computations match PyTorch's results within a small tolerance.

## Project Structure

- `MLP/` - package containing the educational autograd and MLP code
  - `engine.py` - core `Value` autograd class
  - `nn.py` - `Neuron`, `Layer`, `MLP` abstractions
- `test/` - tests comparing to PyTorch
- `demo.ipynb`, `trace_graph.ipynb` - interactive notebooks (demo/visualization)
- `requirements.txt`, `setup.py` - packaging and dependency info

## Notes & Next Steps

- This implementation is scalar-based (each `Value` is a scalar). To train on mini-batches you would typically vectorize operations (e.g., implement a `Tensor` class or use NumPy arrays).
- You can extend the repo with optimizers (SGD, Adam), loss functions, and training loops.

## Author

Debmalya Sett

## License

No license file included. If you want to make this project open-source, add a `LICENSE` file (MIT, Apache-2.0, etc.).
