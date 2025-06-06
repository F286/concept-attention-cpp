#pragma once
#include "tensor.h"

/// @brief Stochastic gradient descent optimizer
class SGD {
public:
    /// @brief Construct with learning rate
    explicit SGD(float lr);
    /// @brief Update parameter with gradient
    void step(Tensor &param, const Tensor &grad) const;
private:
    float m_lr; ///< learning rate
};
