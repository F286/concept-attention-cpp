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
    size_t vocab = m_weight.shape()[0];
    size_t dim = m_weight.shape()[1];
    Tensor one_hot({n, vocab});
    one_hot.fill(0.0f);
    for (size_t i = 0; i < n; ++i)
        one_hot.at(i, indices[i]) = 1.0f;
    return Tensor::matmul(one_hot, m_weight);
}

Tensor &Embedding::weight() { return m_weight; }
const Tensor &Embedding::weight() const { return m_weight; }

std::vector<Tensor*> Embedding::parameters() {
    return {&m_weight};
}

