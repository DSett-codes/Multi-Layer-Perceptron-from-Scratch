# MLP from Scratch

This is a beginner-friendly implementation of:
- a tiny autograd engine (`Tensor` class)
- a simple neural network (`Neuron`, `Layer`, `MLP`)

The demo trains on the XOR dataset using plain Python lists.

## Project Files

- `MLP/engine.py`: core `Tensor` class (data, gradient, backward pass)
- `MLP/nn.py`: simple `Module`, `Neuron`, `Layer`, `MLP` classes
- `demo.py`: easy-to-read XOR training script

## Setup

```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Run Demo

```pwsh
python demo.py
```

You will see:
- training loss every 10 steps
- training accuracy every 10 steps
- final predictions for each XOR input

## How Training Works (in `demo.py`)

1. Forward pass: predict outputs for all training samples.
2. Loss: compute total squared error.
3. Backward pass: call `backward()` to get gradients.
4. Update: use simple SGD (`parameter = parameter - lr * grad`).

## Notes

- This project is for learning, not speed.
- Each `Tensor` is a single scalar, so training is slower than NumPy/PyTorch.
- The code is intentionally written in a clear, step-by-step style.
