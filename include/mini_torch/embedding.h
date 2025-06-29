#pragma once
#include "tensor.h"
#include "module.h"
#include <vector>
#include <random>

/// @brief Lookup table storing embeddings for discrete tokens
class Embedding : public Module {
public:
    /// @brief Construct with vocabulary size and embedding dimension
    Embedding(size_t num_embeddings, size_t embedding_dim);
    /// @brief Retrieve embeddings for given indices
    Tensor operator()(const std::vector<size_t> &indices) const;
    /// @brief List of trainable parameters
    std::vector<Tensor*> parameters() override;
    /// @brief Mutable access to embedding matrix
    Tensor &weight();
    /// @brief Const access to embedding matrix
    const Tensor &weight() const;

private:
    Tensor m_weight; ///< embedding matrix
};

