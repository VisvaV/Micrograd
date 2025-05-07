# Micrograd MLP - A Neural Network from Scratch Using Scalar Autograd

This repository is an educational implementation of a minimalist neural network framework, inspired by Andrej Karpathy’s [micrograd](https://github.com/karpathy/micrograd). It is written entirely in Python and operates on scalar values using a custom automatic differentiation engine. The purpose of this project is to provide a deep understanding of how forward and backward propagation work at the lowest level.

This implementation includes:
- A scalar-based `Value` class supporting autodiff.
- Neuron, Layer, and MLP classes to construct fully connected feedforward neural networks.
- Manual training loop with gradient descent.
- Graphviz-based visualization of the computation graph.

## Table of Contents

- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Computation Graph Visualization](#computation-graph-visualization)
- [Requirements](#requirements)
- [References](#references)
- [License](#license)

## Project Structure

```
.
├── micrograd_engine.py      # Core autograd engine (Value class)
├── mlp_model.py             # Neural network construction and training loop
├── graph.py                 # Graphviz-based visualization of computation graph
├── requirements.txt         # Python dependencies
└── README.md                # This documentation file
```

## How It Works

### 1. `Value` Class

Defined in `micrograd_engine.py`, this is the core of the autodiff system.

- Supports arithmetic operations: `+`, `-`, `*`, `/`, `**`
- Tracks the computation graph dynamically by recording parent-child relationships between operations.
- Uses backward mode autodiff to compute gradients via `backward()`.

### 2. `Neuron`, `Layer`, and `MLP`

Located in `mlp_model.py`:

- `Neuron`: Computes `tanh(w·x + b)` from input list `x`.
- `Layer`: A fully connected layer consisting of multiple neurons.
- `MLP`: A multi-layer perceptron composed of multiple layers.

These classes are designed to mimic a real neural network using only the scalar `Value` class.

### 3. Training Loop

The model is trained on a toy dataset using mean squared error loss. Gradients are computed via `loss.backward()` and weights are updated using manual gradient descent.

### 4. Visualization

The computation graph of the final loss is visualized using `graph.py`, which uses Graphviz. The result is saved as a `.png` file.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/VisvaV/micrograd-mlp.git
cd micrograd-mlp
pip install -r requirements.txt
```

You also need to install Graphviz system package:

- Windows: Download from https://graphviz.org/download/
- Ubuntu/Debian: `sudo apt install graphviz`
- macOS (brew): `brew install graphviz`

Make sure the `dot` executable is in your system PATH.

## Usage

To train the model and visualize the computation graph, run:

```bash
python mlp_model.py
```

This will:

- Train a 3-layer MLP on a small dataset.
- Print loss at each step.
- Render the computation graph of the final loss to `computation_graph.png`.

## Computation Graph Visualization

At the end of training, a graph of the computation tree for the final scalar loss is rendered. This helps understand how values and gradients flow backward through the model.

The graph shows:

- Nodes for every intermediate `Value`
- Operations like `+`, `*`, `tanh`
- Connections showing dependency and flow

To view the rendered graph:

```bash
open computation_graph.png  # macOS
start computation_graph.png # Windows
xdg-open computation_graph.png # Linux
```

## Requirements

See `requirements.txt`

## References

- Andrej Karpathy’s [micrograd](https://github.com/karpathy/micrograd)
- Graphviz documentation: https://graphviz.org/

## License

This project is open source under the MIT license.
