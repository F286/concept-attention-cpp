#pragma once
#include "tensor.h"

namespace functional {

/// @brief Out of place ReLU
inline Tensor relu(const Tensor &t) { return t.relu(); }

/// @brief Matrix multiplication
inline Tensor matmul(const Tensor &a, const Tensor &b) { return Tensor::matmul(a,b); }

/// @brief Elementwise addition
inline Tensor add(const Tensor &a, const Tensor &b) { return Tensor::add(a,b); }

/// @brief Elementwise subtraction
inline Tensor sub(const Tensor &a, const Tensor &b) { return Tensor::sub(a,b); }

/// @brief Elementwise multiplication
inline Tensor mul(const Tensor &a, const Tensor &b) { return Tensor::mul(a,b); }

} // namespace functional
