#pragma once
#include "tensor.h"
#include <vector>

/// @brief Mean squared error loss functional
class MSELoss {
public:
    /// @brief Compute loss value
    float operator()(const Tensor &pred, const Tensor &target) const;
    /// @brief Gradient of loss with respect to prediction
    Tensor backward(const Tensor &pred, const Tensor &target) const;
}; 

/// @brief Cross entropy loss on raw logits
class CrossEntropyLoss {
public:
    /// @brief Compute mean loss over batch
    float operator()(const Tensor &logits, const std::vector<size_t> &target) const;
    /// @brief Gradient of loss with respect to logits
    Tensor backward(const Tensor &logits, const std::vector<size_t> &target) const;
};

