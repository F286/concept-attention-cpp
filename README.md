# Concept Attention C++

This repository experiments with a novel "genesis attention" mechanism for transformer models. The code base provides a minimal CPU-only tensor library inspired by PyTorch, simple transformer implementations, and unit tests verifying basic behaviour.

Genesis attention partitions the token relations into multiple low dimensional projections and computes the minimum relation score across them without softmax normalization. Additional losses constrain attention weights.

The project aims to compare training behaviour of a baseline transformer against this experimental mechanism on tiny datasets.
