#pragma once
#include "tensor.h"

/// @brief Simple affine transformation layer
class Linear {
public:
    /// @brief Construct layer with input and output dimensions
    Linear(size_t in, size_t out);
    /// @brief Forward pass
    Tensor operator()(const Tensor &input) const;
    /// @brief Gradient descent update using output gradient
    void step(const Tensor &input, const Tensor &grad_output, float lr);

    /// @brief Mutable access to weight matrix
    Tensor &weight();
    /// @brief Const access to weight matrix
    const Tensor &weight() const;
    /// @brief Mutable access to bias vector
    Tensor &bias();
    /// @brief Const access to bias vector
    const Tensor &bias() const;

private:
    Tensor m_weight; ///< weight matrix
    Tensor m_bias;   ///< bias vector
};
