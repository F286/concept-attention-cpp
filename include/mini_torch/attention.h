#pragma once
#include "tensor.h"

/// @brief Standard scaled dot product attention
class Attention {
public:
    /// @brief Compute attention output
    static Tensor apply(const Tensor &q, const Tensor &k, const Tensor &v);
};

/// @brief Experimental genesis attention implementation
class GenesisAttention {
public:
    /// @brief Construct with projection counts and dimension
    GenesisAttention(size_t concepts, size_t dim);
    /// @brief Apply attention
    Tensor operator()(const Tensor &q, const Tensor &k, const Tensor &v) const;

private:
    size_t m_concepts; ///< number of projections
    size_t m_dim;      ///< projection dimension
};
