#pragma once
#include "linear.h"
#include "attention.h"
#include "module.h"

/// @brief Minimal transformer-like model with standard attention
class Model : public Module {
public:
    /// @brief Construct model with embedding dimension
    Model(size_t dim);
    /// @brief Forward pass
    Tensor operator()(const Tensor &input) const;
    /// @brief Train output layer on one sample
    void train_step(const Tensor &input, const Tensor &target, float lr);
    /// @brief List of trainable parameters
    std::vector<Tensor*> parameters() override;

private:
    Linear m_proj_q; ///< query projection
    Linear m_proj_k; ///< key projection
    Linear m_proj_v; ///< value projection
    Linear m_out;    ///< output projection
};

/// @brief Model variant using genesis attention
class GenesisModel : public Module {
public:
    /// @brief Construct model with embedding dimension
    GenesisModel(size_t dim);
    /// @brief Forward pass
    Tensor operator()(const Tensor &input) const;
    /// @brief Train output layer on one sample
    void train_step(const Tensor &input, const Tensor &target, float lr);
    /// @brief List of trainable parameters
    std::vector<Tensor*> parameters() override;

private:
    Linear m_proj_q;         ///< query projection
    Linear m_proj_k;         ///< key projection
    Linear m_proj_v;         ///< value projection
    Linear m_out;            ///< output projection
    GenesisAttention m_attn; ///< genesis attention engine
};
