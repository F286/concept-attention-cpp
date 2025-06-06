#pragma once
#include "tensor.h"
#include "module.h"
#include <vector>
#include <random>

/// @brief Standard scaled dot product attention
class Attention {
public:
    /// @brief Compute attention output
    static Tensor apply(const Tensor &q, const Tensor &k, const Tensor &v);
};

/// @brief Experimental genesis attention implementation
class GenesisAttention : public Module {
public:
    /// @brief Construct with projection counts and dimension
    GenesisAttention(size_t concepts, size_t dim);
    /// @brief Apply attention
    Tensor operator()(const Tensor &q, const Tensor &k, const Tensor &v) const;
    /// @brief List of trainable parameters
    std::vector<Tensor*> parameters() override;

private:
    mutable bool m_initialized;   ///< lazy weight initialization flag
    size_t m_concepts;            ///< number of projections
    size_t m_dim;                 ///< projection dimension
    mutable std::vector<Tensor> m_wq; ///< query projections
    mutable std::vector<Tensor> m_wk; ///< key projections
    mutable std::mt19937 m_rng;       ///< random generator
};
