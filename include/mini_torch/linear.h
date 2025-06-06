#pragma once
#include "tensor.h"
#include "module.h"

/// @brief Simple affine transformation layer
class Linear : public Module {
public:
    /// @brief Construct layer with input and output dimensions
    Linear(size_t in_features, size_t out_features);
    /// @brief Forward pass
    Tensor operator()(const Tensor &input) const;

    /// @brief List of trainable parameters
    std::vector<Tensor*> parameters() override;

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
