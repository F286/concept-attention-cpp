#pragma once
#include "tensor.h"

/// @brief Simple affine transformation layer
class Linear {
public:
    /// @brief Construct layer with input and output dimensions
    Linear(size_t in, size_t out);
    /// @brief Forward pass
    Tensor operator()(const Tensor &input) const;

private:
    Tensor m_weight; ///< weight matrix
    Tensor m_bias;   ///< bias vector
};
