#include "mini_torch/embedding.h"
#include <random>

Embedding::Embedding(size_t num_embeddings, size_t embedding_dim)
    : m_weight({num_embeddings, embedding_dim}, 0.0f, true) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (size_t i = 0; i < m_weight.size(); ++i)
        m_weight[i] = dist(rng);
}

Tensor Embedding::operator()(const std::vector<size_t> &indices) const {
    size_t n = indices.size();
    size_t dim = m_weight.shape()[1];
    Tensor out({n, dim});
    for (size_t i = 0; i < n; ++i) {
        size_t idx = indices[i];
        for (size_t j = 0; j < dim; ++j)
            out[i * dim + j] = m_weight[idx * dim + j];
    }
    return out;
}

Tensor &Embedding::weight() { return m_weight; }
const Tensor &Embedding::weight() const { return m_weight; }

std::vector<Tensor*> Embedding::parameters() {
    return {&m_weight};
}

