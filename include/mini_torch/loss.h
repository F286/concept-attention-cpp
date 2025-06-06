#pragma once
#include "tensor.h"

/// @brief Mean squared error loss functional
class MSELoss {
public:
    /// @brief Compute loss value
    float operator()(const Tensor &pred, const Tensor &target) const;
    /// @brief Gradient of loss with respect to prediction
    Tensor backward(const Tensor &pred, const Tensor &target) const;
};
