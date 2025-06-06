# Concept Attention C++

This repository explores a minimal "mini torch" neural network framework implemented in modern C++. The focus is on understanding autograd and model composition without external dependencies. On top of this library we implement a baseline transformer and an experimental "genesis attention" variant.

## Mini Torch

The `mini_torch` directory provides a small set of PyTorch‑like primitives:

- **Tensor** — SIMD backed container supporting element operations, matrix multiplication, autograd and ReLU.
- **Module** base class for composing layers and collecting parameters.
- Layers such as `Linear` and `Embedding`.
- Basic loss functions and an SGD optimizer.
- Lightweight `Dataset` and `DataLoader` utilities for batching samples.

These pieces mimic familiar PyTorch APIs while keeping the implementation concise and CPU only.

## Genesis Attention

A compact transformer implementation demonstrates the novel attention mechanism. Genesis attention splits queries into several low‑dimensional projections and selects the minimum interaction score instead of using a softmax. Additional loss terms regularise the scores.

## Building and Testing

Build the project and run the unit tests with:

```sh
cmake -S . -B build && cmake --build build
ctest -LE TRAIN
```

The optional training demo comparing baseline and genesis models is tagged `[TRAIN]`:

```sh
ctest -L TRAIN
```

