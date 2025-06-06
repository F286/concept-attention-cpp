#pragma once
#include "tensor.h"
#include "module.h"
#include <vector>

/// @brief Normalizes each sample across features
class LayerNorm : public Module {
public:
    /// @brief Construct with feature dimension
    LayerNorm(size_t features, float eps = 1e-5f);
    /// @brief Apply normalization
    Tensor operator()(const Tensor &input) const;
    /// @brief Parameters for optimization
    std::vector<Tensor*> parameters() override;
    /// @brief Mutable access to scale parameter
    Tensor &weight();
    /// @brief Const access to scale parameter
    const Tensor &weight() const;
    /// @brief Mutable access to bias parameter
    Tensor &bias();
    /// @brief Const access to bias parameter
    const Tensor &bias() const;
private:
    size_t m_features; ///< normalized feature count
    float m_eps;       ///< numerical stability constant
    Tensor m_gamma;    ///< scale parameter
    Tensor m_beta;     ///< bias parameter
};
