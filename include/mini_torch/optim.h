#pragma once
#include "tensor.h"

/// @brief Stochastic gradient descent optimizer
class SGD {
public:
    /// @brief Construct with learning rate and parameter list
    explicit SGD(std::vector<Tensor*> params, float lr);
    /// @brief Apply accumulated gradients
    void step();
    /// @brief Zero gradients of all parameters
    void zero_grad();
private:
    std::vector<Tensor*> m_params; ///< parameters being optimized
    float m_lr; ///< learning rate
};
